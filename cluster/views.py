"""聚类视图：只读页（所有人）+ 管理台（仅管理员）。

权限照抄 core.views 的口径：
  管理员 = ``user.is_staff``（重建特征 / 跑聚类 / 命名 / 发布 / 调参）
  其余登录用户 = 只读（看分类结果、看某首曲子的相似作品）
"""
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from core.models import Piece

from .clustering import refresh_vector, run_clustering
from .index import IndexBundle
from .models import Cluster, ClusterMember, ClusterRun, ClusterSetting, PieceVector

# 一次"重建特征"最多处理多少首（解析乐谱较慢，分批避免请求挂太久）
REBUILD_BATCH = 50


def _staff_required(view):
    return user_passes_test(lambda u: u.is_staff,
                            login_url='/accounts/login/')(view)


def _can_view(user, piece: Piece) -> bool:
    return user.is_staff or piece.owner_id == user.id or piece.is_public


def _bundle():
    """当前全部公开作品的特征 → 索引（检索 / 聚类共用）。"""
    vectors = [v for v in PieceVector.objects.filter(
        piece__is_public=True).select_related('piece') if v.axes_ready]
    return IndexBundle(vectors), vectors


def _latest_run(request):
    """总览要展示的那次聚类：默认最新已发布；管理员可预览未发布的。"""
    run = ClusterRun.objects.filter(status=ClusterRun.Status.DONE,
                                    published=True).first()
    if run is None and request.user.is_staff:
        run = ClusterRun.objects.filter(status=ClusterRun.Status.DONE).first()
    return run


# ---------------------------------------------------------------- 只读页

@login_required
def overview(request):
    """聚类总览：这一次分出了哪些类。"""
    run = _latest_run(request)
    clusters = []
    if run:
        clusters = (Cluster.objects.filter(run=run)
                    .prefetch_related('members__piece')
                    .select_related('medoid'))
    return render(request, 'cluster/overview.html', {
        'run': run, 'clusters': clusters,
        'vector_count': PieceVector.objects.filter(piece__is_public=True).count(),
    })


@login_required
def cluster_detail(request, cluster_id):
    """一个类里有哪些作品，各自因为什么被归进来。"""
    cl = get_object_or_404(
        Cluster.objects.select_related('run', 'medoid')
        .prefetch_related('members__piece'), pk=cluster_id)
    if not cl.run.published and not request.user.is_staff:
        raise Http404('该聚类结果尚未发布')
    return render(request, 'cluster/detail.html', {'cluster': cl})


@login_required
def similar(request, piece_id):
    """某首作品的相似作品（三轴各 Top-N → RRF 融合）。"""
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_view(request.user, piece):
        raise Http404('没有查看该作品的权限')

    setting = ClusterSetting.get()
    target = PieceVector.objects.filter(piece=piece).first()
    results = []
    err = ''
    if target is None or not target.axes_ready:
        err = '这首作品还没提取特征（管理员需先在聚类管理台"重建特征"）'
    else:
        bundle, _vectors = _bundle()
        if piece.id not in bundle.by_pid:
            err = '这首作品不在聚类范围内（只对已公开的作品建索引）'
        else:
            rows = bundle.search(piece.id, setting)[:20]
            pieces = {p.id: p for p in Piece.objects.filter(
                id__in=[r['piece_id'] for r in rows])}
            for r in rows:
                r['piece'] = pieces.get(r['piece_id'])
            results = [r for r in rows if r.get('piece')]
    return render(request, 'cluster/similar.html', {
        'piece': piece, 'results': results, 'err': err,
        'target': target, 'setting': setting,
    })


# ---------------------------------------------------------------- 管理台

@_staff_required
def admin_panel(request):
    """管理台：重建特征 / 执行聚类 / 命名 / 发布 / 调参。"""
    runs = ClusterRun.objects.select_related('created_by')[:20]
    setting = ClusterSetting.get()
    total = Piece.objects.filter(is_public=True).exclude(musicxml_file='').count()
    done = PieceVector.objects.filter(
        piece__is_public=True, status=PieceVector.Status.DONE).count()
    return render(request, 'cluster/admin.html', {
        'runs': runs, 'setting': setting,
        'public_pieces': total, 'vectors_done': done,
        'batch': REBUILD_BATCH,
    })


@_staff_required
@require_POST
def admin_rebuild(request):
    """重建特征（分批，每批 REBUILD_BATCH 首）。"""
    qs = (Piece.objects.filter(is_public=True)
          .exclude(musicxml_file='')
          .order_by('id'))
    todo = list(qs[:REBUILD_BATCH])
    ok = fail = 0
    for piece in todo:
        try:
            refresh_vector(piece)
            ok += 1
        except Exception:
            fail += 1
    left = max(qs.count() - len(todo), 0)
    messages.success(request, f'特征重建完成：成功 {ok} 首、失败 {fail} 首'
                              + (f'，还剩 {left} 首可继续' if left else ''))
    return redirect('cluster:cluster_admin')


@_staff_required
@require_POST
def admin_run(request):
    """执行一次聚类。"""
    setting = ClusterSetting.get()
    run = run_clustering(request.user, setting)
    if run.status == ClusterRun.Status.DONE:
        messages.success(
            request, f'聚类完成：{run.piece_count} 首作品分成 {run.cluster_count} 类，'
                     f'记得"发布"后普通用户才能看到')
    else:
        messages.error(request, f'聚类失败：{run.error}')
    return redirect('cluster:cluster_admin')


@_staff_required
@require_POST
def admin_publish(request, run_id):
    """发布 / 撤回一次聚类结果。"""
    run = get_object_or_404(ClusterRun, pk=run_id)
    run.published = not run.published
    run.save(update_fields=['published'])
    messages.success(request, f'已{"发布" if run.published else "撤回"}第 {run.pk} 次聚类')
    return redirect('cluster:cluster_admin')


@_staff_required
@require_POST
def admin_rename(request, cluster_id):
    """给类命名（留空则回到自动标签）。"""
    cl = get_object_or_404(Cluster, pk=cluster_id)
    cl.label = (request.POST.get('label') or '').strip()[:120]
    cl.save(update_fields=['label'])
    messages.success(request, f'已把第 {cl.index} 类改名为「{cl.display_label}」')
    return redirect('cluster:cluster_detail', cluster_id=cl.id)


@_staff_required
@require_POST
def admin_settings(request):
    """调聚类参数。"""
    s = ClusterSetting.get()

    def _f(key, default, lo=0.0, hi=1.0):
        try:
            v = float(request.POST.get(key, default))
        except (TypeError, ValueError):
            v = default
        return lo if v < lo else (hi if v > hi else v)

    def _i(key, default, lo=1, hi=200):
        try:
            v = int(request.POST.get(key, default))
        except (TypeError, ValueError):
            v = default
        return lo if v < lo else (hi if v > hi else v)

    s.w_mode = _f('w_mode', s.w_mode)
    s.w_harmony = _f('w_harmony', s.w_harmony)
    s.w_melody = _f('w_melody', s.w_melody)
    s.top_n = _i('top_n', s.top_n, 1, 200)
    s.rrf_k = _i('rrf_k', s.rrf_k, 1, 200)
    s.phrase_top_m = _i('phrase_top_m', s.phrase_top_m, 1, 50)
    s.melody_agg = (request.POST.get('melody_agg') or s.melody_agg)
    if s.melody_agg not in ('max', 'mean_max'):
        s.melody_agg = 'max'
    s.cluster_threshold = _f('cluster_threshold', s.cluster_threshold, 0.01, 1.0)
    s.save()
    messages.success(request, '聚类参数已保存（下次"执行聚类"生效）')
    return redirect('cluster:cluster_admin')
