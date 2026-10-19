"""核心视图：上传、作品详情、分析触发、状态轮询、管理员设置。"""
import os
import re
from pathlib import Path

from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Count, Q
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .analysis.pipeline import (
    GROUPS, STEP_DEPS, STEP_LABELS, start_convert_async, start_question_async,
    start_step_async, start_steps_async,
)
from .forms import UploadForm
from .models import AnalysisResult, AnalysisRun, Piece, QAQuestion, SystemConfig

# 允许的上传扩展名
ALLOWED_EXT = {'.mid', '.midi', '.musicxml', '.xml', '.mxl',
               '.mp3', '.wav', '.flac', '.ogg', '.m4a'}

EXT_TO_TYPE = {
    '.mid': 'midi', '.midi': 'midi',
    '.musicxml': 'musicxml', '.xml': 'musicxml', '.mxl': 'musicxml',
    '.mp3': 'audio', '.wav': 'audio', '.flac': 'audio',
    '.ogg': 'audio', '.m4a': 'audio',
}


def _staff_required(view):
    return user_passes_test(lambda u: u.is_staff, login_url='/accounts/login/')(view)


def _can_view(user, piece: Piece) -> bool:
    """读取权限：本人 / 管理员 / 已公开。"""
    return user.is_staff or piece.owner_id == user.id or piece.is_public


def _can_edit(user, piece: Piece) -> bool:
    """操作权限（运行分析、改可见性）：本人 / 管理员。"""
    return user.is_staff or piece.owner_id == user.id


# 下载产物的类型 → (模型字段, 默认扩展名, Content-Type)
DOWNLOAD_KINDS = {
    'midi': ('midi_file', '.mid', 'audio/midi'),
    'musicxml': ('musicxml_file', '.musicxml',
                 'application/vnd.recordare.musicxml+xml'),
}


def _safe_filename(title: str, fallback: str) -> str:
    """把作品标题整理成安全的文件名（去路径分隔符与非法字符）。"""
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', '_', (title or '').strip())
    name = name.strip(' .')[:120]
    return name or fallback


@login_required
def dashboard(request):
    """作品列表：本人作品 + 已公开作品（管理员可看全部）。"""
    if request.user.is_staff:
        pieces = Piece.objects.select_related('owner').all()
    else:
        pieces = Piece.objects.select_related('owner').filter(
            Q(owner=request.user) | Q(is_public=True))
    steps_done = {
        p.id: {r.step for r in p.results.filter(status='done')}
        for p in pieces
    }
    return render(request, 'core/dashboard.html', {
        'pieces': pieces,
        'steps_done': steps_done,
    })


@login_required
def upload(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        f = request.FILES['source_file']
        ext = Path(f.name).suffix.lower()
        if ext not in ALLOWED_EXT:
            form.add_error('source_file', f'不支持的文件类型 {ext}。'
                                     '支持：MIDI / MusicXML / 常见音频格式。')
        else:
            piece = Piece.objects.create(
                owner=request.user,
                title=form.cleaned_data['title'] or Path(f.name).stem,
                source_file=f,
                source_type=EXT_TO_TYPE[ext],
                contains_voice=form.cleaned_data['contains_voice'],
                is_public=form.cleaned_data['is_public'],
            )
            start_convert_async(piece)
            messages.success(request, '上传成功，正在转换格式…')
            return redirect('core:piece_detail', piece_id=piece.id)
    return render(request, 'core/upload.html', {'form': form})


@login_required
def piece_detail(request, piece_id: int):
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_view(request.user, piece):
        raise Http404('作品不存在或未公开。')
    convert_result = piece.results.filter(step='convert').first()
    conv_out = (convert_result.output if convert_result else None) or {}
    voice_src = (conv_out.get('voice') or {}).get('source', '')
    if not voice_src:   # 早期转换记录没有该字段：按来源类型兜底说明
        voice_src = 'transcribe' if piece.source_type == 'audio' else 'file'
    voice_parts = (conv_out.get('voice') or {}).get('voice_parts') or []
    voice_note = {
        'file': '按作品文件内的实际声部判定（不随上传时的勾选）',
        'transcribe': '按 MuScriptor 转录结果判定',
    }[voice_src]
    if voice_parts:
        voice_note += '：' + '、'.join(
            f"{p.get('name')}（{p.get('instrument')}）" for p in voice_parts)
    return render(request, 'core/piece_detail.html', {
        'piece': piece,
        'step_labels': STEP_LABELS,
        'can_edit': _can_edit(request.user, piece),
        'convert_status': convert_result.status if convert_result else '',
        'voice_note': voice_note,
    })


# 结构 / 织体 / 风格三个步骤（页面按此顺序竖排，各自单独点击运行）
STS_STEPS = [
    {'key': 'structure', 'title': '① 音乐结构分析',
     'hint': '乐句切分 → 乐段聚类 → 曲式判断（依赖：节奏网格 + 和弦 + 调性）'},
    {'key': 'texture', 'title': '② 织体分析',
     'hint': '逐窗判定单声 / 主调 / 复调 / 支声（依赖：配器 + 声部 + 复调 + 节奏网格）'},
    {'key': 'style', 'title': '③ 风格分析',
     'hint': '流派 / 年代 / 地域推断（依赖：结构 + 织体 + 调性 + 节奏型 + 配器）'},
]


@login_required
def sts_page(request, piece_id: int):
    """结构 / 织体 / 风格分析页。

    三个子窗口竖排、各占页面全宽，彼此独立触发（点哪个跑哪个）。
    """
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_view(request.user, piece):
        raise Http404('作品不存在或未公开。')
    convert_result = piece.results.filter(step='convert').first()
    return render(request, 'core/sts.html', {
        'piece': piece,
        'can_edit': _can_edit(request.user, piece),
        'sts_steps': STS_STEPS,
        'convert_done': bool(convert_result and convert_result.status == 'done'),
    })


@login_required
def qa_page(request, piece_id: int):
    """自定义问答页：对本作品提任意问题。"""
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_view(request.user, piece):
        raise Http404('作品不存在或未公开。')
    convert_result = piece.results.filter(step='convert').first()
    questions = _visible_questions(request.user, piece)
    shown = questions[:QA_LIST_LIMIT]
    return render(request, 'core/qa.html', {
        'piece': piece,
        'can_edit': _can_edit(request.user, piece),
        'questions': shown,
        'qa_total': len(questions),
        # 首屏直接带上历史，避免「加载中…」的空窗；随后仍由轮询刷新
        'qa_initial': [_qa_payload(q) for q in shown],
        'convert_done': bool(convert_result and convert_result.status == 'done'),
    })


# 单步骤历史最多返回多少条（单条详情不含在内）
MAX_HISTORY = 50
# 问答历史：接口一次最多返回多少条（页面保留完整历史，只是分次取）
QA_LIST_LIMIT = 200

# 触发方式 → 人话
TRIGGER_LABELS = {
    'convert': '上传后自动转换',
    'retry-convert': '手动重新转换',
    'group:rhythm': '「分析节奏」按钮',
    'group:melody': '「分析旋律」按钮',
    'group:extraction': '「主旋律提取」按钮',
    'group:sts': '「结构 / 织体 / 风格」一键全跑',
    'legacy': '历史遗留（本功能上线前的结果）',
}


def _trigger_label(trigger: str) -> str:
    if not trigger:
        return '—'
    if trigger in TRIGGER_LABELS:
        return TRIGGER_LABELS[trigger]
    if trigger.startswith('rerun:'):
        step = trigger.split(':', 1)[1]
        return f'单模块重跑（{STEP_LABELS.get(step, step)}）'
    return trigger


def _qa_payload(q: QAQuestion) -> dict:
    """一次提问 → 前端可直接渲染的结构（问答历史是持久的，不随重跑分析消失）。"""
    return {
        'id': q.id,
        'question': q.question,
        'answer': q.answer,
        'status': q.status,
        'error': q.error[:400] if q.error else '',
        'created_at': q.created_at.isoformat(),
        'finished_at': q.finished_at.isoformat() if q.finished_at else '',
        'username': q.user.username,
        # 回答依据了哪些分析、哪些还没跑（前端据此提示"模型不会臆断"）
        'based_on': list((q.details or {}).get('已完成分析', {}).keys()),
        'missing': (q.details or {}).get('尚无数据的方面', []),
    }


def _run_payload(run: AnalysisRun, with_output: bool = False) -> dict:
    item = {
        'id': run.id,
        'seq': run.seq,
        'status': run.status,
        'trigger': run.trigger,
        'trigger_label': _trigger_label(run.trigger),
        'started_at': run.started_at.isoformat() if run.started_at else None,
        'finished_at': run.finished_at.isoformat() if run.finished_at else None,
        'archived_at': run.archived_at.isoformat(),
        'has_output': bool(run.output),
        'error': run.error[:600] if run.error else '',
    }
    if with_output:
        item['output'] = run.output
    return item


@login_required
def step_history(request, piece_id: int):
    """某一步的历史运行记录。

    - 不带 `run` 参数：返回摘要列表（不含大体量 output）；
    - 带 `run=<id>`：返回该次的完整结果（含 output），前端点开时才拉。
    """
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_view(request.user, piece):
        raise Http404('作品不存在或未公开。')   # 与页面一致：不暴露作品存在与否

    step = request.GET.get('step', '')
    runs = piece.runs.filter(step=step)
    run_id = request.GET.get('run')
    if run_id:
        run = runs.filter(pk=run_id).first()
        if run is None:
            return JsonResponse({'ok': False, 'error': '记录不存在。'}, status=404)
        return JsonResponse({'ok': True, 'item': _run_payload(run, True)})
    return JsonResponse({
        'ok': True,
        'step': step,
        'total': runs.count(),
        'items': [_run_payload(r) for r in runs.order_by('-seq')[:MAX_HISTORY]],
    })


def _visible_questions(user, piece: Piece):
    """可见的提问：本人 + 管理员可见全部。"""
    qs = piece.questions.select_related('user').all()
    if not user.is_staff:
        qs = qs.filter(user=user)
    return list(qs)


@login_required
@require_POST
def ask_question(request, piece_id: int):
    """提交一次提问（立即返回，回答在后台线程进行，前端轮询）。"""
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_edit(request.user, piece):
        return JsonResponse({'ok': False, 'error': '无权对该作品提问。'}, status=403)

    question = (request.POST.get('question') or '').strip()
    if not question:
        return JsonResponse({'ok': False, 'error': '请输入问题。'}, status=400)
    if len(question) > 500:
        return JsonResponse({'ok': False, 'error': '问题请控制在 500 字以内。'},
                            status=400)
    if not piece.results.filter(step='convert', status='done').exists():
        return JsonResponse({'ok': False,
                             'error': '格式转换尚未完成（或失败），暂不能提问。'},
                            status=400)

    qa_id = start_question_async(piece, request.user, question)
    return JsonResponse({'ok': True, 'id': qa_id, 'message': '已提交，正在回答…'})


@login_required
def qa_status(request, piece_id: int):
    """问答轮询端点：返回本作品可见的提问与回答。"""
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_view(request.user, piece):
        return JsonResponse({'ok': False, 'error': '无权访问该作品。'}, status=403)
    visible = _visible_questions(request.user, piece)
    items = [_qa_payload(q) for q in visible[:QA_LIST_LIMIT]]
    return JsonResponse({
        'ok': True,
        'items': items,
        'total': len(visible),
        'limit': QA_LIST_LIMIT,
        'any_running': any(i['status'] in ('pending', 'running') for i in items),
    })


@login_required
@require_POST
def retry_convert(request, piece_id: int):
    """重新执行格式转换（转换失败后不必重新上传作品）。

    临时源文件若已被清理，则退化为「用已入库的 MIDI 重建 MusicXML」。
    """
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_edit(request.user, piece):
        return JsonResponse({'ok': False, 'error': '无权操作该作品。'}, status=403)
    running = piece.results.filter(step='convert',
                                   status__in=['running', 'pending']).exists()
    if running:
        return JsonResponse({'ok': False, 'error': '转换正在进行中。'}, status=409)
    if not piece.source_file and not piece.midi_file:
        return JsonResponse(
            {'ok': False,
             'error': '临时源文件已清理且无 MIDI 产物，无法重新转换，请重新上传。'},
            status=400)
    start_convert_async(piece)
    return JsonResponse({'ok': True, 'message': '已重新启动格式转换。'})


@login_required
def download_file(request, piece_id: int, kind: str):
    """下载转换产物 MIDI / MusicXML（强制附件下载，不在浏览器内联打开）。

    权限与查看一致：本人 / 管理员 / 已公开作品。
    音频上传的场景下，原始音频是临时文件会被清理，但转换出的 MIDI 与
    MusicXML 永久保存，始终可下载。
    """
    spec = DOWNLOAD_KINDS.get(kind)
    if spec is None:
        raise Http404('未知的下载类型。')
    field_name, default_ext, content_type = spec

    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_view(request.user, piece):
        raise Http404('作品不存在或未公开。')

    field = getattr(piece, field_name)
    if not field:
        raise Http404('该文件尚未生成（转换未完成或失败）。')
    try:
        if not field.storage.exists(field.name):
            raise Http404('文件已不存在，可能已被清理。')
        fh = field.open('rb')
    except Http404:
        raise
    except Exception:  # noqa: BLE001
        raise Http404('文件不可读取。')

    ext = Path(field.name).suffix or default_ext
    filename = _safe_filename(piece.title, f'piece{piece.id}') + ext
    resp = FileResponse(fh, as_attachment=True, filename=filename,
                        content_type=content_type)
    return resp


@login_required
@require_POST
def toggle_visibility(request, piece_id: int):
    """切换作品公开 / 私有（仅本人或管理员）。"""
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_edit(request.user, piece):
        return JsonResponse({'ok': False, 'error': '无权操作该作品。'}, status=403)
    piece.is_public = not piece.is_public
    piece.save(update_fields=['is_public', 'updated_at'])
    messages.success(request, '作品已设为「公开」，其他登录用户可见。'
                     if piece.is_public else '作品已设为「私有」，仅本人可见。')
    return redirect('core:piece_detail', piece_id=piece.id)


@login_required
@require_POST
def rerun_step(request, piece_id: int):
    """重新执行单个分析步骤（步骤卡右上角的 ⟲ 按钮）。

    依赖已完成的直接复用；依赖缺失会自动补跑，因此即使只跑过 convert
    也能直接单独触发 melody_shape。
    """
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_edit(request.user, piece):
        return JsonResponse({'ok': False, 'error': '无权操作该作品。'}, status=403)

    step = request.POST.get('step', '')
    if step not in STEP_DEPS:
        return JsonResponse({'ok': False, 'error': f'未知步骤：{step}'}, status=400)
    if not piece.results.filter(step='convert', status='done').exists():
        return JsonResponse({'ok': False,
                             'error': '格式转换尚未完成（或失败），暂不能分析。'},
                            status=400)
    running = piece.results.filter(status__in=['running', 'pending']).exists()
    if running:
        return JsonResponse({'ok': False,
                             'error': '已有任务在运行中，请等待完成。'}, status=409)

    steps = start_step_async(piece, step)
    if not steps:
        return JsonResponse({'ok': False, 'error': '该步骤无需重跑。'}, status=400)
    return JsonResponse({'ok': True, 'started': steps,
                         'message': f'已重新执行：'
                                    f'{STEP_LABELS.get(step, step)}'
                                    + (f'（含 {len(steps) - 1} 个依赖步骤）'
                                       if len(steps) > 1 else '')})


@login_required
@require_POST
def run_group(request, piece_id: int):
    """触发一组分析（rhythm / melody / extraction）。"""
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_edit(request.user, piece):
        return JsonResponse({'ok': False, 'error': '无权操作该作品。'}, status=403)

    group = request.POST.get('group', '')
    force = request.POST.get('force') == '1'
    if group not in GROUPS:
        return JsonResponse({'ok': False, 'error': f'未知分析组：{group}'}, status=400)
    if piece.results.filter(step='convert', status='done').exists() is False:
        return JsonResponse({'ok': False,
                             'error': '格式转换尚未完成（或失败），暂不能分析。'}, status=400)
    running = piece.results.filter(status__in=['running', 'pending']).exists()
    if running:
        return JsonResponse({'ok': False,
                             'error': '已有分析在运行中，请等待完成。'}, status=409)
    steps = start_steps_async(piece, group, force=force)
    if not steps:
        return JsonResponse({'ok': True, 'started': [],
                             'message': '该组所有步骤均已完成；如需重跑请使用"重跑"按钮。'})
    return JsonResponse({'ok': True, 'started': steps})


@login_required
def piece_status(request, piece_id: int):
    """状态轮询端点（前端每 2.5s 调用）。"""
    piece = get_object_or_404(Piece, pk=piece_id)
    if not _can_view(request.user, piece):
        return JsonResponse({'ok': False, 'error': '无权访问该作品。'}, status=403)
    results = {}
    for r in piece.results.all():
        results[r.step] = {
            'status': r.status,
            'error': r.error[:600] if r.error else '',
            'output': r.output if r.status == 'done' else None,
            'started_at': r.started_at.isoformat() if r.started_at else None,
            'finished_at': r.finished_at.isoformat() if r.finished_at else None,
        }
    # 每个步骤累计跑过几次（前端据此显示「历史 N」入口）
    run_counts = {row['step']: row['n'] for row in
                  piece.runs.values('step').annotate(n=Count('id'))}
    return JsonResponse({
        'ok': True,
        'results': results,
        'run_counts': run_counts,
        'any_running': any(v['status'] in ('running', 'pending')
                           for v in results.values()),
    })


@_staff_required
def system_settings(request):
    """管理员配置页：复调阈值等参数前端可调。"""
    config = SystemConfig.get_config()
    if request.method == 'POST':
        try:
            config.polyphony_threshold = float(request.POST['polyphony_threshold'])
            config.exchange_min_switches = int(request.POST['exchange_min_switches'])
            config.exchange_similarity_threshold = float(
                request.POST['exchange_similarity_threshold'])
            config.analysis_window_bars = int(request.POST['analysis_window_bars'])
            config.chord_window = request.POST['chord_window']
            config.dsh_profile = request.POST['dsh_profile'].strip() or 'headless'
            config.dsh_timeout = int(request.POST['dsh_timeout'])
            config.max_chord_windows_per_call = int(
                request.POST['max_chord_windows_per_call'])
            config.full_clean()
            config.save()
            messages.success(request, '系统配置已保存。')
            return redirect('core:settings')
        except (KeyError, ValueError) as exc:
            messages.error(request, f'参数有误：{exc}')
    return render(request, 'core/settings.html', {'config': config})
