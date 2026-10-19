"""织体分析 —— 分析因果链上"配器 / 声部识别 / 复调"之后的汇总层。

    texture ← instrumentation（乐器编制）+ voice_id（声部角色）
              + polyphony（复调判定）+ rhythm_grid（速度 / 小节）

分工：
  本地确定性计算：活跃声部数、节奏同步率、音型化判定、逐窗织体归类、织体时间线；
  LLM：只做织体叙述与伴奏音型命名（软降级）。
"""
from .llm import run_llm_json_soft, JSON_ONLY
from .scoredata import (
    ScoreData, build_measure_map, windows_of_measures, interval_seq, lcs_ratio,
)

# 织体类型（中文名，前端直接展示）
MONOPHONY = '单声织体'
HOMOPHONY = '主调织体'
POLYPHONY = '复调织体'
HETEROPHONY = '支声织体'

# 支声判定：动机相似度与平均音高差（半音）
_HETERO_MOTIF_SIM = 0.85
_HETERO_PITCH_TOL = 3.0
# 节奏同步率阈值（≥ 视为柱式齐奏型伴奏）
_SYNC_THRESHOLD = 0.6
# 音型化判定：伴奏声部音符时值的变异系数（越小越均匀）
_FIGURATION_CV = 0.35
# 持续音判定（四分音符）
_PEDAL_MIN_QL = 4.0


def _role_parts(voice_out: dict) -> tuple:
    """由声部角色候选分出 (旋律声部, 伴奏声部) 的声部序号集合。"""
    melody, accomp = set(), set()
    for r in (voice_out or {}).get('roles', []) or []:
        idx = int(r.get('part_index', -1))
        if idx < 0:
            continue
        role = r.get('role_candidate') or ''
        if '主旋律' in role:
            melody.add(idx)
        elif '对位' in role:
            melody.add(idx)          # 对位声部同样算旋律层
        else:
            accomp.add(idx)
    return melody, accomp


def _window_events(data: ScoreData, s: int, e: int,
                   measure_offsets: dict):
    """窗口内各声部的有音高事件（{part_index: [events]}）。"""
    m_start = measure_offsets.get(s, 0.0)
    m_end = measure_offsets.get(e + 1)
    if m_end is None:
        m_end = m_start + (e - s + 1) * 4.0
    per_part = {}
    for p in data.parts:
        evs = [ev for ev in p.pitched_events
               if m_start <= ev.offset < m_end]
        if evs:
            per_part[p.index] = evs
    return per_part, (m_end - m_start)


def _sync_ratio(per_part: dict, base_idx) -> float:
    """节奏同步率：旋律声部的每个起音点，有多少被其他声部同时起音。"""
    if base_idx not in per_part or len(per_part) < 2:
        return 0.0
    base = {round(ev.offset, 3) for ev in per_part[base_idx]}
    if not base:
        return 0.0
    others = [ {round(ev.offset, 3) for ev in evs}
               for k, evs in per_part.items() if k != base_idx ]
    matched = sum(1 for o in base if any(o in s for s in others))
    return round(matched / len(base), 3)


def _figuration_score(per_part: dict, skip: set) -> float:
    """音型化得分：伴奏声部音符时值越均匀、起音越密集 → 越像分解和弦音型。

    返回 0~1；skip 中的声部（旋律层）不参与统计。
    """
    durs, gaps = [], []
    for idx, evs in per_part.items():
        if idx in skip:
            continue
        durs.extend(ev.duration for ev in evs)
        offs = sorted(ev.offset for ev in evs)
        gaps.extend(b - a for a, b in zip(offs, offs[1:]) if b - a > 1e-6)
    if len(durs) < 4 or not gaps:
        return 0.0
    mean_d = sum(durs) / len(durs)
    mean_g = sum(gaps) / len(gaps)
    if mean_d <= 0 or mean_g <= 0:
        return 0.0
    var_d = sum((d - mean_d) ** 2 for d in durs) / len(durs)
    cv = (var_d ** 0.5) / mean_d
    evenness = max(0.0, 1.0 - cv / _FIGURATION_CV)
    # 起音间隔小于 1 拍 → 音型化（连续跑动）才成立
    density = 1.0 if mean_g <= 1.0 else (0.5 if mean_g <= 2.0 else 0.0)
    return round(evenness * density, 3)


def _has_pedal(per_part: dict, skip: set) -> bool:
    """是否存在持续音（长音铺底 / 踏板音）。"""
    for idx, evs in per_part.items():
        if idx in skip:
            continue
        if any(ev.duration >= _PEDAL_MIN_QL for ev in evs):
            return True
    return False


def _heterophony(per_part: dict, melody_idx: set) -> float:
    """支声可能性：多个旋律层声部演奏近乎相同的音高序列。"""
    cand = [idx for idx in melody_idx if idx in per_part]
    if len(cand) < 2:
        return 0.0
    seqs = []
    for idx in cand[:3]:
        evs = sorted(per_part[idx], key=lambda e: e.offset)
        seqs.append([max(ev.midi_pitches) for ev in evs][:80])
    sims = []
    for a, b in zip(seqs, seqs[1:]):
        if not a or not b:
            continue
        short = min(len(a), len(b))
        sims.append(lcs_ratio(interval_seq(a[:short]), interval_seq(b[:short])))
    return round(max(sims), 3) if sims else 0.0


def _classify(per_part: dict, melody_idx: set, sync: float, fig: float,
              pedal: bool, hetero: float) -> tuple:
    """逐窗织体归类 → (类型, 细节说明)。"""
    active = len(per_part)
    if active == 0:
        return '', '休止'
    if active == 1:
        return MONOPHONY, '单一声部（无伴奏）'
    if hetero >= _HETERO_MOTIF_SIM:
        return HETEROPHONY, '多个声部演奏同一旋律的变体'

    melody_active = [i for i in melody_idx if i in per_part]
    if len(melody_active) >= 2 and sync < _SYNC_THRESHOLD and hetero < _HETERO_MOTIF_SIM:
        return POLYPHONY, f'{len(melody_active)} 个声部各自独立进行（节奏不同步）'

    # 主调：旋律层 + 伴奏层，再细分伴奏形态
    if sync >= _SYNC_THRESHOLD:
        detail = '柱式和弦伴奏（与旋律节奏同步）'
    elif fig > 0:
        detail = '分解和弦 / 音型化伴奏'
    elif pedal:
        detail = '持续音铺底（长音踏板）'
    else:
        detail = '旋律 + 自由伴奏'
    if pedal and '持续音' not in detail:
        detail += ' + 持续音'
    return HOMOPHONY, detail


def _merge_timeline(windows: list) -> list:
    """合并相邻同类型窗口 → 织体时间线。"""
    out = []
    for w in windows:
        if not w['texture']:
            continue
        if out and out[-1]['texture'] == w['texture']:
            out[-1]['end_measure'] = w['end_measure']
            out[-1]['bars'] += w['bars']
        else:
            out.append({'texture': w['texture'], 'detail': w['detail'],
                        'start_measure': w['start_measure'],
                        'end_measure': w['end_measure'], 'bars': w['bars']})
    return out


def _llm_texture(payload: dict, config) -> dict:
    task = (
        '你是专业织体分析师。以下是一部作品的本地织体统计'
        '（逐窗活跃声部数、节奏同步率、音型化程度已由程序算出）。'
        '请据此描述织体形态与变化。\n\n'
        f'{payload}\n\n'
        '输出 JSON：'
        '{"main_texture": "主织体类型（如 主调织体 · 分解和弦伴奏）", '
        '"accompaniment_pattern": "伴奏音型描述（如 阿尔贝蒂低音 / 柱式和弦 / 持续音）", '
        '"timeline_comment": "织体随时间变化的 1-2 句说明", '
        '"density_comment": "声部密度与厚度的 1 句说明", '
        '"highlights": "1-2 句织体亮点点评", '
        '"confidence": 0~1}'
        f'{JSON_ONLY}'
    )
    return run_llm_json_soft(task, profile=config.dsh_profile,
                             timeout=config.dsh_timeout, retries=1)


def texture(data: ScoreData, config, instr_out: dict, voice_out: dict,
            poly_out: dict, rhythm_out: dict) -> dict:
    """织体分析主入口。"""
    measure_offsets, _, _ = build_measure_map(data)
    windows_meta = windows_of_measures(data, config.analysis_window_bars)
    melody_idx, accomp_idx = _role_parts(voice_out)

    windows = []
    for w in windows_meta:
        per_part, span = _window_events(data, w['start_measure'],
                                        w['end_measure'], measure_offsets)
        base = next((i for i in sorted(melody_idx) if i in per_part),
                    next(iter(per_part), None))
        sync = _sync_ratio(per_part, base) if base is not None else 0.0
        fig = _figuration_score(per_part, melody_idx)
        pedal = _has_pedal(per_part, melody_idx)
        hetero = _heterophony(per_part, melody_idx)
        label, detail = _classify(per_part, melody_idx, sync, fig, pedal, hetero)
        notes = sum(len(evs) for evs in per_part.values())
        bars = max(w['end_measure'] - w['start_measure'] + 1, 1)
        windows.append({
            'window': w['window'],
            'start_measure': w['start_measure'],
            'end_measure': w['end_measure'],
            'bars': bars,
            'texture': label,
            'detail': detail,
            'active_parts': len(per_part),
            'melody_parts': len([i for i in melody_idx if i in per_part]),
            'sync_ratio': sync,
            'figuration': fig,
            'has_pedal': pedal,
            'heterophony': hetero,
            'notes_per_bar': round(notes / bars, 2) if span else 0.0,
        })

    # 主织体 = 出现窗口数最多的类型
    counts = {}
    for w in windows:
        if w['texture']:
            counts[w['texture']] = counts.get(w['texture'], 0) + 1
    total = sum(counts.values()) or 1
    ratios = {k: round(v / total, 3) for k, v in counts.items()}
    main_texture = max(counts.items(), key=lambda kv: kv[1])[0] if counts else ''
    main_detail = ''
    if main_texture:
        details = [w['detail'] for w in windows if w['texture'] == main_texture]
        main_detail = max(set(details), key=details.count)

    avg_active = (sum(w['active_parts'] for w in windows) / len(windows)) if windows else 0
    avg_density = (sum(w['notes_per_bar'] for w in windows) / len(windows)) if windows else 0

    payload = {
        'window_bars': config.analysis_window_bars,
        'main_texture': f'{main_texture} · {main_detail}',
        'ratios': ratios,
        'timeline': [{'measures': f"{t['start_measure']}-{t['end_measure']}",
                      'texture': t['texture'], 'detail': t['detail']}
                     for t in _merge_timeline(windows)][:20],
        'families': (instr_out or {}).get('families', []),
        'part_count': (instr_out or {}).get('part_count', 0),
        'is_polyphonic': (poly_out or {}).get('is_polyphonic', False),
        'avg_active_parts': round(avg_active, 2),
        'notes_per_bar': round(avg_density, 2),
    }
    llm = _llm_texture(payload, config)

    return {
        'window_bars': config.analysis_window_bars,
        'windows': windows,
        'timeline': _merge_timeline(windows),
        'ratios': ratios,
        'main_texture': main_texture,
        'main_detail': main_detail,
        'density': {
            'avg_active_parts': round(avg_active, 2),
            'notes_per_bar': round(avg_density, 2),
        },
        'is_polyphonic': (poly_out or {}).get('is_polyphonic', False),
        'llm_analysis': llm,
        'note': ('活跃声部数、节奏同步率、音型化与支声判定为本地确定性计算；'
                 '织体命名与伴奏音型描述由大模型完成（解析失败保留本地统计）。'),
    }
