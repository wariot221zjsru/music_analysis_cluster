"""聚类执行：相似度矩阵 → 层次聚类 → 落库 → 自动标签。

## 为什么用层次聚类 + 阈值，而不是 KMeans

1. **类数不该由人预设**：公开作品是慢慢多起来的，今天 5 类合适、明天可能 8 类。
   层次聚类按"距离阈值"切一刀，类数由数据决定。
2. **能解释"哪两类更近"**：层次结构的合并顺序本身就是可展示的证据，
   KMeans 给不出这个。
3. 小样本（几十首）下 KMeans 对初始中心敏感，层次聚类结果是确定的。

距离用 ``1 - 融合相似度``；相似度 = 三轴原始分数的加权和（不是 RRF 分 ——
RRF 是排名分，尺度不适合当距离）。

## 缺轴怎么处理

某首曲子可能只跑了和声、没跑主旋律线（旋律轴空）。缺的那一轴**记 0 分**：
它在该轴上没有证据，不能当成"中等相似"。这样"三轴都像"的曲子才会被聚到一起，
只有单轴像的会自然落到边缘。
"""
from django.db import transaction

from core.analysis import theme as theme_mod
from core.models import Piece

from . import features
from .index import AXES, IndexBundle, cosine
from .models import Cluster, ClusterMember, ClusterRun, ClusterSetting, PieceVector

# 乐句级精排时每首曲子最多参与几个乐句（防止长曲子把比对量拖爆）
MAX_PHRASES_PER_PAIR = 12


# ---------------------------------------------------------------- 两两分数

def _top_phrases(phrases):
    """取最长的几个乐句参与精排（长乐句信息量大，且能控住计算量）。"""
    if not phrases:
        return []
    ordered = sorted(phrases, key=lambda fp: -(fp.get('count') or 0))
    return ordered[:MAX_PHRASES_PER_PAIR]


def pair_scores(bundle: IndexBundle, pid_a, pid_b, setting=None) -> dict:
    """两首曲子的三轴原始分数（不走 Top-N，直接算这一对）。"""
    va = bundle.by_pid.get(pid_a)
    vb = bundle.by_pid.get(pid_b)
    out = {}
    if va is None or vb is None:
        return out

    if va.mode_vec and vb.mode_vec:
        out['mode'] = round(bundle.mode.score(
            va.mode_vec, vb.mode_vec, va.mode_uncertain, vb.mode_uncertain), 4)
    if va.harmony_vec and vb.harmony_vec:
        out['harmony'] = round(cosine(va.harmony_vec, vb.harmony_vec), 4)
    if va.melody_phrases and vb.melody_phrases:
        best = 0.0
        for fa in _top_phrases(va.melody_phrases):
            for fb in _top_phrases(vb.melody_phrases):
                sc = theme_mod.compare(fa, fb)['score']
                if sc > best:
                    best = sc
        out['melody'] = round(best, 4)
    return out


def _weights(setting):
    return {
        'mode': float(getattr(setting, 'w_mode', 0.3) or 0.0),
        'harmony': float(getattr(setting, 'w_harmony', 0.3) or 0.0),
        'melody': float(getattr(setting, 'w_melody', 0.4) or 0.0),
    }


def similarity_matrix(bundle: IndexBundle, setting):
    """全库两两融合相似度（对称，0~1）。

    每首曲子只在各轴的 Top-N 里留下分数，其余对为 0 —— 与检索同口径，
    也把 O(N²) 的旋律精排压回可控范围。
    """
    pids = bundle.pids
    idx = {pid: i for i, pid in enumerate(pids)}
    n = len(pids)
    weights = _weights(setting)
    wsum = sum(weights.values()) or 1.0
    sim = [[0.0] * n for _ in range(n)]

    for pid in pids:
        target = bundle.by_pid[pid]
        i = idx[pid]
        for axis in AXES:
            w = weights.get(axis, 0.0)
            if w <= 0:
                continue
            for other_pid, sc in bundle.axis_query(axis, target, setting,
                                                   exclude=pid):
                j = idx.get(other_pid)
                if j is None:
                    continue
                sim[i][j] += w * max(sc, 0.0)

    for i in range(n):
        sim[i][i] = wsum
        for j in range(n):
            sim[i][j] /= wsum
    # 对称化：检索是有向的（各自 Top-N 不同），取两边较大值
    for i in range(n):
        for j in range(i + 1, n):
            m = max(sim[i][j], sim[j][i])
            sim[i][j] = sim[j][i] = m
    return pids, sim


# ---------------------------------------------------------------- 层次聚类

def _cluster_dist(c1, c2, dist) -> float:
    """average linkage：两簇间所有成员对的平均距离。"""
    total = 0.0
    for i in c1:
        for j in c2:
            total += dist[i][j]
    return total / max(len(c1) * len(c2), 1)


def hierarchical(dist, threshold: float):
    """自底向上聚合，簇间距离超过阈值就停。返回成员下标列表的列表。"""
    n = len(dist)
    if n == 0:
        return []
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = _cluster_dist(clusters[a], clusters[b], dist)
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None or best[0] > threshold:
            break
        _d, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
    return clusters


def medoid_of(members, dist):
    """簇里最中心的那个（到同簇其他成员平均距离最小）。"""
    if not members:
        return None
    best, best_d = members[0], None
    for i in members:
        d = sum(dist[i][j] for j in members if j != i) / max(len(members) - 1, 1)
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


# ---------------------------------------------------------------- 自动标签

def _mode_family(mode_key: str) -> str:
    """调名 → 调式族（大调 / 小调 / 多利亚…）。"""
    k = str(mode_key or '')
    if not k:
        return '调式未定'
    for kw in ('多利亚', '立底亚', '利底亚', '弗里几亚', '混合利底亚',
               '洛克里亚', '伊奥尼亚', '爱奥尼亚'):
        if kw in k:
            return kw
    low = k.lower()
    if '小调' in k or 'minor' in low:
        return '小调'
    if '大调' in k or 'major' in low:
        return '大调'
    return '其他调式'


def _melody_style(phrases) -> str:
    """乐句音程 → 级进 / 跳进的粗描述。"""
    ivs = []
    for fp in (phrases or []):
        ivs.extend([abs(v) for v in (fp.get('ivs') or [])])
    if not ivs:
        return ''
    avg = sum(ivs) / len(ivs)
    if avg <= 2.0:
        return '级进为主'
    if avg <= 4.0:
        return '级进带跳进'
    return '跳进较多'


def auto_label(vectors: list) -> str:
    """给一类起自动标签：调式族众数 + 和声性质 + 旋律走向。"""
    if not vectors:
        return '空类'
    fams = {}
    quals = {}
    styles = {}
    for v in vectors:
        f = _mode_family(getattr(v, 'mode_key', ''))
        fams[f] = fams.get(f, 0) + 1
        hv = getattr(v, 'harmony_vec', None) or []
        if len(hv) >= 11:
            # 和声向量第 8–11 维是性质分布：maj / min / dom7 / other
            order = ['maj', 'min', 'dom7', 'other']
            k = max(range(4), key=lambda i: hv[7 + i])
            quals[order[k]] = quals.get(order[k], 0) + 1
        st = _melody_style(getattr(v, 'melody_phrases', None))
        if st:
            styles[st] = styles.get(st, 0) + 1

    def _top(d, default=''):
        return max(d, key=d.get) if d else default

    parts = [_top(fams, '调式未定')]
    q = _top(quals)
    if q:
        parts.append({'maj': '大三和弦为主', 'min': '小三和弦为主',
                      'dom7': '属七和弦多', 'other': '其他和弦'}.get(q, q))
    s = _top(styles)
    if s:
        parts.append(s)
    return ' · '.join(parts)


# ---------------------------------------------------------------- 主入口

def eligible_vectors():
    """参与聚类的作品特征：**公开**作品、特征已算好、至少有一个轴。"""
    return (PieceVector.objects
            .filter(piece__is_public=True)
            .filter(models_status_ok())
            .select_related('piece'))


def models_status_ok():
    from django.db.models import Q
    return Q(status=PieceVector.Status.DONE) | Q(status=PieceVector.Status.PARTIAL)


@transaction.atomic
def run_clustering(user, setting: ClusterSetting = None) -> ClusterRun:
    """执行一次聚类并落库（管理员触发）。

    返回 ClusterRun；失败时把 status 置 error 并记 error 信息（不抛给页面）。
    """
    setting = setting or ClusterSetting.get()
    run = ClusterRun.objects.create(
        created_by=user, status=ClusterRun.Status.RUNNING,
        params={
            'w_mode': setting.w_mode, 'w_harmony': setting.w_harmony,
            'w_melody': setting.w_melody, 'top_n': setting.top_n,
            'rrf_k': setting.rrf_k, 'phrase_top_m': setting.phrase_top_m,
            'melody_agg': setting.melody_agg,
            'cluster_threshold': setting.cluster_threshold,
            'feature_version': setting.feature_version,
        })
    try:
        vectors = [v for v in eligible_vectors() if v.axes_ready]
        if len(vectors) < 2:
            run.status = ClusterRun.Status.ERROR
            run.error = f'可参与聚类的公开作品只有 {len(vectors)} 首（至少需要 2 首）'
            run.save(update_fields=['status', 'error'])
            return run

        bundle = IndexBundle(vectors)
        pids, sim = similarity_matrix(bundle, setting)
        n = len(pids)
        dist = [[1.0 - sim[i][j] for j in range(n)] for i in range(n)]
        groups = hierarchical(dist, float(setting.cluster_threshold))

        pieces = {p.id: p for p in Piece.objects.filter(id__in=pids)}
        for gi, members in enumerate(groups, start=1):
            member_pids = [pids[i] for i in members]
            center = medoid_of(members, dist)
            center_pid = pids[center] if center is not None else None
            vecs = [bundle.by_pid[p] for p in member_pids]
            cl = Cluster.objects.create(
                run=run, index=gi, auto_label=auto_label(vecs),
                size=len(member_pids),
                medoid=pieces.get(center_pid) if center_pid else None)
            for i in members:
                pid = pids[i]
                scores = (pair_scores(bundle, pid, center_pid, setting)
                          if center_pid else {})
                d = dist[i][center] if center is not None else 0.0
                ClusterMember.objects.create(
                    cluster=cl, piece=pieces.get(pid),
                    mode_sim=scores.get('mode'),
                    harmony_sim=scores.get('harmony'),
                    melody_sim=scores.get('melody'),
                    dist_to_medoid=round(d, 4),
                    fused_score=round(1.0 - d, 4))
        run.status = ClusterRun.Status.DONE
        run.piece_count = n
        run.cluster_count = len(groups)
        run.save(update_fields=['status', 'piece_count', 'cluster_count'])
        return run
    except Exception as exc:
        run.status = ClusterRun.Status.ERROR
        run.error = str(exc)
        run.save(update_fields=['status', 'error'])
        return run


# ---------------------------------------------------------------- 特征刷新

@transaction.atomic
def refresh_vector(piece: Piece) -> PieceVector:
    """算（或重算）一首作品的三轴特征并存库。"""
    feats = features.extract(piece)
    vec, _created = PieceVector.objects.get_or_create(piece=piece)
    vec.mode_vec = feats['mode_vec']
    vec.mode_key = feats['mode_key']
    vec.mode_tonic_pc = feats['mode_tonic_pc']
    vec.mode_uncertain = feats['mode_uncertain']
    vec.harmony_vec = feats['harmony_vec']
    vec.harmony_chord_count = feats['harmony_chord_count']
    vec.melody_phrases = feats['melody_phrases']
    vec.melody_phrase_count = feats['melody_phrase_count']
    vec.melody_note_count = feats['melody_note_count']
    vec.feature_version = ClusterSetting.get().feature_version
    ready = len(feats['axes'])
    if ready == 3:
        vec.status = PieceVector.Status.DONE
    elif ready == 0:
        vec.status = PieceVector.Status.ERROR
    else:
        vec.status = PieceVector.Status.PARTIAL
    vec.error = feats['error'][:500]
    vec.save()
    return vec


def refresh_all():
    """重算全部公开作品的特征（管理员"重建特征"用）。返回 (成功数, 失败数)。"""
    qs = Piece.objects.filter(is_public=True).exclude(musicxml_file='')
    ok = fail = 0
    for piece in qs:
        try:
            refresh_vector(piece)
            ok += 1
        except Exception:
            fail += 1
    return ok, fail
