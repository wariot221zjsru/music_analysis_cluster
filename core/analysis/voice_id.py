"""声部识别 —— 分析因果链第 4 步（和弦 / 旋律 / 织体 / 交接检测的共同前置）。

输出三组（计划书 4.5.1）：
  ① 各声部"角色候选"（主旋律 / 伴奏 / 低音 / 对位）
  ② per-segment 旋律显著性分数（四项加权：局部最高音 / 最长时值 / 动机重复 / 织体密度倒数）
  ③ 复调标记原始密度（供 polyphony 步判定）
以及中间产物：主旋律归属时间线（供 exchange / melody 步复用）。
"""
from .scoredata import (
    ScoreData, PartInfo, windows_of_measures, build_measure_map,
    sounding_pitches_at, interval_seq,
)


def _part_pitch_seq(part: PartInfo, m_start: float, m_end: float):
    """截取声部在 [m_start, m_end) 四分音符范围内的音高序列。"""
    seq = []
    for e in part.pitched_events:
        if m_start <= e.offset < m_end:
            # 和弦取最高音（旋律显著性的粗粒度近似）
            seq.append(max(e.midi_pitches))
    return seq


def _motif_repetition(pitches):
    """动机重复度：音程 3-gram 中重复项占比。"""
    ivs = interval_seq(pitches)
    if len(ivs) < 3:
        return 0.0
    grams = [tuple(ivs[i:i + 3]) for i in range(len(ivs) - 2)]
    if not grams:
        return 0.0
    unique = len(set(grams))
    return round(1.0 - unique / len(grams), 4)


# 低音区声部的判定（**两条同时满足**才算）：
#   相对：比全曲最高声部的平均音高低 ≥ 2 个八度（24 半音）；
#   绝对：平均音高本身低于 C3（MIDI 48，低音谱表的低音区）。
# 只按相对差会误伤：一首有短笛（≈C6）的曲子里，中音区的吉他伴奏（≈A3）
# 与短笛相差 26 半音，会被当成"低音区"排除掉 —— 它只是伴奏，不是低音。
# 加上绝对门槛后，大提琴主旋律（≈D3，50）也不会被误判。
#
# 为什么必须有这一步：显著性四项在窗内做 min-max 归一，低音声部靠
# "长时值 + 高重复 + 稀疏"三项常常压过真旋律（实测贝斯 0.625 压过人声
# 0.598），必须在**选主旋律**这一步把它排除掉。
LOW_REGISTER_GAP = 24        # 相对门槛（半音）
LOW_REGISTER_CEILING = 48    # 绝对门槛（MIDI 音高，C3）


def part_mean_pitches(data: ScoreData) -> dict:
    """各声部**全曲**平均音高（和弦取最高音），只有发声声部才进字典。"""
    out = {}
    for p in data.parts:
        seq = [max(e.midi_pitches) for e in p.pitched_events if e.midi_pitches]
        if seq:
            out[p.index] = sum(seq) / len(seq)
    return out


def low_register_parts(data: ScoreData) -> set:
    """明显处在低音区的声部索引集合（见 ``LOW_REGISTER_GAP`` / ``LOW_REGISTER_CEILING``）。

    只在**选主旋律归属**时用；显著性分数本身照旧保留，低音声部的
    「伴奏 / 低音」角色判定、复调密度等都不受影响。

    全曲只有一个发声声部时返回空集（独奏 / 独唱没有可排除的对象）。
    """
    mp = part_mean_pitches(data)
    if len(mp) < 2:
        return set()
    top = max(mp.values())
    return {i for i, v in mp.items()
            if top - v >= LOW_REGISTER_GAP and v < LOW_REGISTER_CEILING}


def window_scores(data: ScoreData, m_start: float, m_end: float) -> dict:
    """单个时间窗 [m_start, m_end) 内各发声声部的显著性。

    抽出来是为了让 melody_line 在**声部识别结果缺失 / 过期**时能按同一套
    算法就地补算（见 melody_line 的"稀疏矩阵自愈"），两条路算出来的分数
    必须逐位一致，否则同一首曲子会因为"有没有重跑声部识别"得到不同结果。

    返回 {'per_part': [...], 'scores': {part_index: salience}}；
    scores 只含该窗内真正发声的声部。
    """
    per_part = []
    for p in data.parts:
        seq = _part_pitch_seq(p, m_start, m_end)
        if not seq:
            per_part.append({'part_index': p.index, 'empty': True,
                             'pitch': 0, 'dur': 0, 'motif': 0, 'sparse': 0,
                             'salience': 0.0})
            continue
        mean_pitch = sum(seq) / len(seq)
        total_dur = sum(e.duration for e in p.pitched_events
                        if m_start <= e.offset < m_end)
        n_onsets = len(seq)
        per_part.append({
            'part_index': p.index, 'empty': False,
            'mean_pitch': round(mean_pitch, 2),
            'total_duration': round(total_dur, 3),
            'pitch': mean_pitch, 'dur': total_dur,
            'motif': _motif_repetition(seq),
            'sparse': 1.0 / (1.0 + n_onsets / 8.0),  # 密度倒数（稀疏≈旋律）
        })

    # 逐项 min-max 归一后等权求和
    active = [d for d in per_part if not d.get('empty')]
    if active:
        for key in ('pitch', 'dur', 'motif', 'sparse'):
            vals = [d[key] for d in active]
            lo, hi = min(vals), max(vals)
            for d in active:
                d[key + '_n'] = (d[key] - lo) / (hi - lo) if hi > lo else 0.5
        for d in active:
            d['salience'] = round(
                (d['pitch_n'] + d['dur_n'] + d['motif_n'] + d['sparse_n']) / 4, 4)
    return {
        'per_part': per_part,
        'scores': {d['part_index']: d['salience'] for d in active},
    }


def pick_attribution(per_part: list, low: set = frozenset()):
    """窗内主旋律归属：**先排除低音区声部**，再取显著性最高的那个。

    全部候选都被排掉（例如整窗只有低音在响）时退回不排除，绝不因为
    "没有合格候选"就把整段丢成静默。
    """
    active = [d for d in per_part if not d.get('empty')]
    if not active:
        return None
    cands = [d for d in active if d['part_index'] not in low]
    if not cands:
        cands = active
    return max(cands, key=lambda d: d['salience'])


def salience_matrix(data: ScoreData, config) -> dict:
    """per-segment 旋律显著性分数矩阵 + 主旋律归属时间线。

    时间窗粒度 = config.analysis_window_bars 小节。
    """
    windows = windows_of_measures(data, config.analysis_window_bars)
    measure_offsets, _, bar_ql = build_measure_map(data)
    n_parts = len(data.parts)
    low = low_register_parts(data)

    matrix = []   # [{window, start_measure, end_measure, scores: {part: s}, attribution: part_idx}]
    for w in windows:
        m_start = measure_offsets.get(w['start_measure'], 0.0)
        # 窗口终点 = 下一小节起点（不存在则按窗口长度推算）。
        # 按谱面拍号推算，不能写死 4.0 —— 3/4 拍的曲子那样会多算 1/3 个窗，
        # 把下一窗开头的音算进本窗，末窗归属会系统性偏向"下一窗才进来"的声部。
        m_end = measure_offsets.get(w['end_measure'] + 1)
        if m_end is None:
            m_end = m_start + config.analysis_window_bars * bar_ql

        wsc = window_scores(data, m_start, m_end)
        active = [d for d in wsc['per_part'] if not d.get('empty')]
        best = pick_attribution(wsc['per_part'], low)
        matrix.append({
            'window': w['window'],
            'start_measure': w['start_measure'],
            'end_measure': w['end_measure'],
            'scores': {str(d['part_index']): d['salience'] for d in active},
            'attribution': best['part_index'] if best else None,
        })

    # 主旋律归属时间线（合并相邻同声部窗）
    timeline, cur = [], None
    for w in matrix:
        a = w['attribution']
        if cur and cur['part_index'] == a:
            cur['end_measure'] = w['end_measure']
        else:
            if cur:
                timeline.append(cur)
            cur = {'part_index': a, 'start_measure': w['start_measure'],
                   'end_measure': w['end_measure']}
    if cur:
        timeline.append(cur)

    return {
        'window_bars': config.analysis_window_bars,
        'matrix': matrix,
        'attribution_timeline': timeline,
        'part_names': data.part_names,
        'n_parts': n_parts,
        # 被判为"低音区、不当主旋律"的声部（scores 里仍然有它，只是不参与归属）
        'low_register_parts': sorted(low),
        'low_register_names': [data.part_names[i] for i in sorted(low)],
    }


def polyphony_density(data: ScoreData) -> dict:
    """复调原始密度（按 8 分音符网格采样）。

    同一 Part 在同一时点发声 ≥2 音 → 该时点计入分子
    （MuScriptor 一个乐器只出一条声部，同 Part 多音即复调织体的物理基础）。
    """
    step = 0.5
    t, total, poly = 0.0, 0, 0
    per_part = {p.index: {'poly': 0, 'total': 0} for p in data.parts}
    while t < data.total_ql:
        any_sound = False
        any_poly = False
        for p in data.parts:
            pitches = sounding_pitches_at(p, t)
            if pitches:
                any_sound = True
                per_part[p.index]['total'] += 1
                if len(pitches) >= 2:
                    per_part[p.index]['poly'] += 1
                    any_poly = True
        if any_sound:
            total += 1
            # 分子按"时间点"计（只要有一个声部在此时点发 ≥2 音），
            # 否则分子是声部数、分母是时间点数，ratio 会 >1 导致复调判定恒真。
            if any_poly:
                poly += 1
        t += step
    return {
        'grid_step_quarter': step,
        'sounding_windows': total,
        'polyphonic_windows': poly,
        'ratio': round(poly / total, 4) if total else 0.0,
        'per_part': {
            str(idx): {
                'ratio': round(v['poly'] / v['total'], 4) if v['total'] else 0.0,
                'sounding': v['total'],
            } for idx, v in per_part.items()
        },
    }


def _role_candidates(data: ScoreData, sal: dict, low: set = frozenset()) -> list:
    """各声部角色候选（统计启发式：音高/时值/显著性 → 角色）。"""
    total_sal = {}
    for w in sal['matrix']:
        for idx, s in w['scores'].items():
            total_sal[idx] = total_sal.get(idx, 0) + s
    n_windows = max(len(sal['matrix']), 1)

    ranked = sorted(
        ((idx, s / n_windows) for idx, s in total_sal.items()),
        key=lambda kv: -kv[1])
    # 只有有音高的声部参与角色判定（打击乐会拉低 min，导致"低音"永远判不出来）
    mean_pitches = {str(p.index): p.mean_pitch for p in data.parts if p.pitched_events}

    low_ref = min(mean_pitches.values()) if mean_pitches else 60.0
    # 主旋律候选在**低音区之外**重新排一次：否则贝斯一被判成低音，
    # 整张角色表就没有人当"主旋律（候选）"了（实测：贝斯 0.54 排第一，
    # 人声 0.08 排第三，排除贝斯后应该轮到钢琴 0.46）。
    mel_ranked = [kv for kv in ranked if int(kv[0]) not in low] or ranked
    mel_top = mel_ranked[0][0]
    roles = []
    for rank, (idx, s) in enumerate(ranked):
        mp = mean_pitches.get(idx, 60.0)
        if int(idx) in low and len(ranked) > 1:
            # 低音区声部：显著性再高也只是低音，不能报成"主旋律候选"
            role = '低音'
        elif idx == mel_top and s > 0.35:
            role = '主旋律（候选）'
        elif mp <= low_ref + 2 and rank == len(ranked) - 1:
            role = '低音'
        elif s < 0.2:
            role = '伴奏'
        else:
            role = '对位 / 副旋律'
        roles.append({
            'part_index': int(idx), 'part_name': data.part_names[int(idx)],
            'mean_salience': round(s, 4), 'role_candidate': role,
        })
    return roles


def voice_id(data: ScoreData, config, instrumentation_out: dict) -> dict:
    """声部识别主入口（共同前置）。"""
    sal = salience_matrix(data, config)
    dens = polyphony_density(data)
    roles = _role_candidates(data, sal, set(sal.get('low_register_parts') or []))
    return {
        'roles': roles,
        'salience': sal,
        'polyphony_density': dens,
        'instrumentation_prior': instrumentation_out['parts'],
    }
