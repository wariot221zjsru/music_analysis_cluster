"""主旋律线追踪（逐段）—— MIR 口径的 melody line extraction / voice tracking。

与 core.analysis.melody（主旋律**声部识别 / 选轨**）的分工
------------------------------------------------------------

|              | melody（选轨，旧口径）        | melody_line（本模块）            |
|--------------|------------------------------|----------------------------------|
| 回答的问题   | 整曲哪个声部是主旋律          | **每一段**主旋律在哪个声部        |
| 粒度         | 整曲一个声部（或少数交接段）  | 每窗一段，随时可换                |
| 门控         | 受 exchange 的 is_exchange 门控 | **无门控**，恒逐段              |
| 前奏 / 间奏  | 可能整段丢掉                  | 按人声掩膜取最突出的器乐声部      |

关键点：**本模块不读 exchange 的 is_exchange。** 用"是否模仿型交接"去门控
"要不要逐段提取"，是曲式学判据管了 MIR 任务的形态，前奏就是这么丢的。
两条线并列存在、各自重跑，互不干扰。

算法
----
1. 窗级归属：沿用 voice_id 的 per-window 显著性矩阵（音高 / 时值 / 动机重复 /
   稀疏度四项），叠加**人声掩膜**——该窗人声在唱就归人声（config
   开启人声优先时），人声休止才退回显著性最高的器乐声部；
2. 连续同归属的窗合并为段；
3. 碎片合并：短于 melody_segment_min_bars 的段并入相邻段，抑制窗间抖动；
4. 段级置信度 = 窗内 **margin**（第一名 − 第二名显著性）均值。用 margin 而不是
   第一名自身，是因为显著性在窗内做了 min-max 归一：两个声部时第一名最低也有
   0.5，看不出"人声 0.85 压过吉他 0.30"和"0.51 险胜 0.49"的区别。
   只有一个声部在响时无竞争，margin 记为 1.0；
5. 导出两种：
   - absolute：**原位**写入（各段保留真实小节位置，段间空隙留白）——MIR 语义；
   - spliced：首尾相接的连续单声部（方便当一条旋律线听 / 导入 DAW）。
"""
from .melody import save_stream
from .scoredata import ScoreData, build_measure_map, windows_of_measures


# ---------------------------------------------------------------- 工具

def _seg_range(data: ScoreData, measure_offsets: dict, bar_ql: float,
               start_m: int, end_m: int):
    """段 [start_m, end_m] 小节的四分音符区间（含最后一小节的全部音符）。"""
    s = measure_offsets.get(start_m, 0.0)
    e = measure_offsets.get(end_m + 1)
    if e is None:
        e = (data.total_ql + 1e-6 if end_m >= data.measure_count
             else s + (end_m - start_m + 1) * bar_ql)
    return s, e


def _events_in(part, s: float, e: float):
    return [ev for ev in part.pitched_events if s <= ev.offset < e]


def _window_choices(matrix_row: dict, data: ScoreData) -> list:
    """一个窗的 {part_index: salience} → 按显著性降序的列表。"""
    items = []
    for idx, sc in matrix_row.get('scores', {}).items():
        i = int(idx)
        items.append({
            'part_index': i,
            'part_name': data.part_names[i] if i < len(data.part_names) else f'Part {i}',
            'salience': float(sc),
            'is_voice': data.parts[i].is_voice if i < len(data.parts) else False,
        })
    items.sort(key=lambda d: -d['salience'])
    return items


def _pick(choices: list, voice_priority: bool):
    """窗内选轨：人声优先 → 否则显著性最高。返回 (选中项, margin)。"""
    if not choices:
        return None, 0.0
    if voice_priority:
        voices = [c for c in choices if c['is_voice']]
        if voices:
            best = voices[0]        # 多人声时取显著性最高的那条
            # margin 仍按"人声 vs 最强非人声竞争"计，反映人声是否真的突出
            rivals = [c['salience'] for c in choices if not c['is_voice']]
            margin = best['salience'] - (rivals[0] if rivals else 0.0)
            return best, round(max(margin, 0.0), 4)
    best = choices[0]
    second = choices[1]['salience'] if len(choices) > 1 else 0.0
    # 只有一个声部在响：无竞争 → 记为完全确定
    margin = 1.0 if len(choices) == 1 else max(best['salience'] - second, 0.0)
    return best, round(margin, 4)


def _merge_short_runs(runs: list, min_bars: int) -> list:
    """把短于 min_bars 的碎片并入相邻段（优先更长的邻居）。"""
    if min_bars <= 1 or len(runs) <= 1:
        return runs
    segs = [dict(r) for r in runs]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, seg in enumerate(segs):
            bars = seg['end_measure'] - seg['start_measure'] + 1
            if bars >= min_bars:
                continue
            neighbours = []
            if i > 0:
                neighbours.append(i - 1)
            if i < len(segs) - 1:
                neighbours.append(i + 1)
            if not neighbours:
                continue
            j = max(neighbours,
                    key=lambda k: (segs[k]['end_measure'] - segs[k]['start_measure'] + 1,
                                   segs[k]['avg_margin']))
            segs[j]['start_measure'] = min(segs[j]['start_measure'], seg['start_measure'])
            segs[j]['end_measure'] = max(segs[j]['end_measure'], seg['end_measure'])
            segs[j]['absorbed'] = segs[j].get('absorbed', []) + [seg['part_index']]
            segs.pop(i)
            changed = True
            break
    return segs


def _level(conf: float, hi: float, mid: float) -> str:
    if conf >= hi:
        return '高'
    if conf >= mid:
        return '中'
    return '低'


# ---------------------------------------------------------------- 导出

def _note_from_event(ev):
    from music21 import note as m21note
    n = m21note.Note()
    n.pitch.midi = max(ev.midi_pitches)
    n.duration.quarterLength = ev.duration
    return n


def _export_absolute(data: ScoreData, segments: list, measure_offsets: dict,
                     bar_ql: float):
    """原位导出：各段留在真实小节位置，未覆盖区间留白（MIR 语义）。"""
    from music21 import stream

    part = stream.Part()
    part.partName = '主旋律线（逐段追踪 · 原位）'
    for seg in segments:
        s, e = _seg_range(data, measure_offsets, bar_ql,
                          seg['start_measure'], seg['end_measure'])
        for ev in _events_in(data.parts[seg['part_index']], s, e):
            part.insert(ev.offset, _note_from_event(ev))
    return part


def _export_spliced(data: ScoreData, segments: list, measure_offsets: dict,
                    bar_ql: float):
    """拼接导出：各段首尾相接，抹平段间空隙（监听 / 导入 DAW 用）。"""
    from music21 import stream

    part = stream.Part()
    part.partName = '主旋律线（逐段追踪 · 无缝拼接）'
    cursor = 0.0
    for seg in segments:
        s, e = _seg_range(data, measure_offsets, bar_ql,
                          seg['start_measure'], seg['end_measure'])
        span = max(e - s, 0.0)
        for ev in _events_in(data.parts[seg['part_index']], s, e):
            part.insert(cursor + (ev.offset - s), _note_from_event(ev))
        cursor += span
    return part


def _write_part(piece, part, stem: str):
    """写 MIDI + MusicXML；若谱面无小节（原位导出的留白场景）先补齐。"""
    from music21 import stream

    try:
        has_measure = bool(part.recurse().getElementsByClass(stream.Measure))
    except Exception:  # noqa: BLE001
        has_measure = False
    if not has_measure:
        try:
            part = part.makeMeasures(inPlace=False)
            part.makeRests(fillGaps=True, inPlace=True)
        except Exception:  # noqa: BLE001
            pass
    return save_stream(piece, part, stem)


# ---------------------------------------------------------------- 主入口

def melody_line(data: ScoreData, config, piece, voice_out: dict) -> dict:
    """主旋律线追踪主入口（逐段，MIR 口径）。

    依赖 voice_id 的显著性矩阵与谱面的人声标记；**不依赖** polyphony / exchange，
    因此"是否复调""是否交接"都不会改变本步骤的形态。
    """
    sal = voice_out['salience']
    matrix = sal.get('matrix', [])
    measure_offsets, _, bar_ql = build_measure_map(data)
    voice_priority = bool(getattr(config, 'melody_voice_priority', True))
    min_bars = max(int(getattr(config, 'melody_segment_min_bars', 2) or 1), 1)
    hi = float(getattr(config, 'melody_confidence_high', 0.25))
    mid = float(getattr(config, 'melody_confidence_mid', 0.10))

    windows = windows_of_measures(data, config.analysis_window_bars)
    if not windows:
        windows = [{'window': 1, 'start_measure': 1,
                    'end_measure': max(data.measure_count, 1)}]

    # ---- ① 窗级归属（人声掩膜 + 显著性） ----
    win_rows = []
    for i, w in enumerate(windows):
        row = matrix[i] if i < len(matrix) else {'scores': {}}
        choices = _window_choices(row, data)
        best, margin = _pick(choices, voice_priority)
        win_rows.append({
            'window': w['window'],
            'start_measure': w['start_measure'],
            'end_measure': w['end_measure'],
            'part_index': best['part_index'] if best else None,
            'part_name': best['part_name'] if best else '（无发声声部）',
            'is_voice': bool(best['is_voice']) if best else False,
            'voice_sounding': any(c['is_voice'] for c in choices),
            'margin': margin,
            'salience': best['salience'] if best else 0.0,
        })

    # ---- ② 连续同归属合并为 run ----
    runs = []
    for row in win_rows:
        if row['part_index'] is None:
            continue                       # 整窗静默 → 记为间隙，不产生段
        if runs and runs[-1]['part_index'] == row['part_index']:
            runs[-1]['end_measure'] = row['end_measure']
            runs[-1]['_rows'].append(row)
        else:
            runs.append({
                'part_index': row['part_index'],
                'part_name': row['part_name'],
                'is_voice': row['is_voice'],
                'start_measure': row['start_measure'],
                'end_measure': row['end_measure'],
                '_rows': [row],
            })
    for r in runs:
        r['avg_margin'] = round(
            sum(x['margin'] for x in r['_rows']) / len(r['_rows']), 4)

    # ---- ③ 碎片合并 + 段级统计 ----
    runs = _merge_short_runs(runs, min_bars)
    segments = []
    for i, r in enumerate(runs):
        s, e = _seg_range(data, measure_offsets, bar_ql,
                          r['start_measure'], r['end_measure'])
        notes = _events_in(data.parts[r['part_index']], s, e)
        rows = [x for x in win_rows
                if x['part_index'] is not None
                and r['start_measure'] <= x['start_measure'] <= r['end_measure']
                and x['part_index'] == r['part_index']]
        margin = (round(sum(x['margin'] for x in rows) / len(rows), 4)
                  if rows else r.get('avg_margin', 0.0))
        # 人声优先是强先验：同样的 margin，人声段更有把握
        conf = round(min(1.0, margin + 0.15), 4) if r['is_voice'] else margin
        segments.append({
            'seq': i + 1,
            'start_measure': r['start_measure'],
            'end_measure': r['end_measure'],
            'bars': r['end_measure'] - r['start_measure'] + 1,
            'part_index': r['part_index'],
            'part_name': data.parts[r['part_index']].name,
            'instrument': data.parts[r['part_index']].instrument or '（未标注）',
            'is_voice': bool(r['is_voice']),
            'note_count': len(notes),
            'margin': margin,
            'confidence': conf,
            'level': _level(conf, hi, mid),
            'absorbed_from': sorted(set(r.get('absorbed', []))),
        })

    # ---- ④ 覆盖率与掩膜摘要 ----
    covered = sum(s['bars'] for s in segments)
    total_bars = max(data.measure_count, 1)
    voice_parts = [{'index': p.index, 'name': p.name} for p in data.parts if p.is_voice]

    result = {
        'mode': '逐段追踪（人声掩膜 + 显著性）',
        'method': ('每个时间窗按"人声在唱→人声；人声休止→最突出的器乐声部"定归属，'
                   '连续同归属的窗合并为段；不受交接检测门控，'
                   '前奏 / 间奏 / 尾奏同样提取。'),
        'window_bars': config.analysis_window_bars,
        'voice_priority': voice_priority,
        'min_segment_bars': min_bars,
        'confidence_thresholds': {'high': hi, 'mid': mid},
        'has_voice_parts': bool(voice_parts),
        'voice_parts': voice_parts,
        'segments': segments,
        'coverage': {
            'total_bars': total_bars,
            'covered_bars': covered,
            'ratio': round(covered / total_bars, 4),
            'gap_bars': max(total_bars - covered, 0),
        },
        'windows': win_rows,
        'exports': {},
    }

    if not segments:
        result['note'] = '全曲没有可用于追踪的有音高事件（可能整谱静默或全为打击乐）。'
        return result

    # ---- ⑤ 导出（原位 + 无缝拼接） ----
    try:
        result['exports']['absolute'] = _write_part(
            piece, _export_absolute(data, segments, measure_offsets, bar_ql),
            'melody_line_absolute')
        result['exports']['spliced'] = _write_part(
            piece, _export_spliced(data, segments, measure_offsets, bar_ql),
            'melody_line_spliced')
    except Exception as exc:  # noqa: BLE001
        # 导出失败不该让整步失败：分段结果仍然有效
        result['export_error'] = f'{exc}'
        result['note'] = ('分段结果已生成，但 MIDI / MusicXML 导出失败'
                          '（分段表仍可使用，可点右上角 ⟲ 重跑本模块）。')

    # 复调提示：单一旋律线只是其中一条线索
    voices_present = len({s['part_index'] for s in segments})
    if voices_present > 1:
        result['note'] = (
            f'本曲在主旋律线上出现 {voices_present} 个不同声部的接力；'
            '若同时被判定为复调织体，这条线只是其中一条听觉线索，不是唯一主题。')
    return result
