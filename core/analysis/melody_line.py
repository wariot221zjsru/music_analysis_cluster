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
from . import melody_export as mex
from .melody import save_stream
from .scoredata import ScoreData, build_measure_map, windows_of_measures


# ---------------------------------------------------------------- 工具
# 小节区间换算 / 事件截取 / 导出的节奏与音色统一走 melody_export：
#   ① 小节号查不到偏移时按相邻小节推算，不再退回 0.0（旧实现会把段起点当成
#      全曲开头，导出整段错位）；
#   ② 导出写入 tempo 与 instrument，不再是恒 120 BPM + 默认钢琴。


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

def _export_absolute(data: ScoreData, segments: list, measure_offsets: dict,
                     bar_ql: float, options: dict):
    """原位导出：各段留在真实小节位置，未覆盖区间留白（MIR 语义）。"""
    from music21 import stream

    part = stream.Part()
    part.partName = '主旋律线（逐段追踪 · 原位）'
    spans = []
    for seg in segments:
        s, e = mex.seg_range(data, measure_offsets, bar_ql,
                             seg['start_measure'], seg['end_measure'])
        spans.append({'offset': s, 'part_index': seg['part_index']})
        src = data.parts[seg['part_index']]
        for ev in mex.events_in(src, s, e):
            part.insert(ev.offset, mex.make_note(ev))
    # 节奏 / 音色：原位导出保留原曲小节位置，故按真实位置写入
    tempo_desc = mex.apply_tempo(part, data, measure_offsets, bar_ql, options)
    inst_desc = mex.apply_instruments(part, data, spans, options)
    return part, {'tempo': tempo_desc, 'instrument': inst_desc}


def _export_spliced(data: ScoreData, segments: list, measure_offsets: dict,
                    bar_ql: float, options: dict):
    """拼接导出：各段首尾相接，抹平段间空隙（监听 / 导入 DAW 用）。"""
    from music21 import stream

    part = stream.Part()
    part.partName = '主旋律线（逐段追踪 · 无缝拼接）'
    spans = []
    cursor = 0.0
    for seg in segments:
        s, e = mex.seg_range(data, measure_offsets, bar_ql,
                             seg['start_measure'], seg['end_measure'])
        span = max(e - s, 0.0)
        spans.append({'offset': cursor, 'part_index': seg['part_index']})
        src = data.parts[seg['part_index']]
        for ev in mex.events_in(src, s, e):
            # 时值裁剪到段尾，避免上一段的长音压到下一段头上
            part.insert(cursor + (ev.offset - s),
                        mex.make_note(ev, mex.clamped_duration(ev, e)))
        cursor += span
    # 拼接后小节已与原曲错位，变速标记没有意义，退化为单一速度
    if options['tempo_mode'] == 'original':
        from music21 import tempo as m21tempo
        first_bpm = next((t['bpm'] for t in (data.tempos or []) if t.get('bpm')),
                         mex.DEFAULT_BPM)
        part.insert(0.0, m21tempo.MetronomeMark(number=float(first_bpm)))
        tempo_desc = f'原曲速度 {float(first_bpm):.0f} BPM（拼接后小节错位，按单一速度写入）'
    else:
        tempo_desc = mex.apply_tempo(part, data, measure_offsets, bar_ql, options)
    inst_desc = mex.apply_instruments(part, data, spans, options)
    return part, {'tempo': tempo_desc, 'instrument': inst_desc}


def build_export_parts(data: ScoreData, segments: list, options: dict,
                       measure_offsets=None, bar_ql=None) -> list:
    """构建两种版本的导出 Part（**不落盘**）。

    返回 [{'key', 'stem', 'part', 'desc'}]：
      key  = 结果字典里的键（absolute / spliced，保持稳定）
      stem = 落盘文件名主干，带节奏 / 音色后缀（换选项就换文件名，
             避免同名覆盖后浏览器给缓存里的旧文件）

    "按当前设置直接下载"与"应用并重新导出"共用这一份，保证两条路导出一致。
    """
    if measure_offsets is None or bar_ql is None:
        measure_offsets, _, bar_ql = build_measure_map(data)
    built = []
    for kind, fn, stem in (('absolute', _export_absolute, 'melody_line_absolute'),
                           ('spliced', _export_spliced, 'melody_line_spliced')):
        part, desc = fn(data, segments, measure_offsets, bar_ql, options)
        built.append({'key': kind, 'stem': mex.stem_with_options(stem, options),
                      'part': part, 'desc': desc})
    return built


def export_segments(data: ScoreData, piece, segments: list, options: dict,
                    measure_offsets=None, bar_ql=None) -> dict:
    """按段导出两种版本（供主入口与"重新导出"接口共用）。"""
    out = {}
    for b in build_export_parts(data, segments, options, measure_offsets, bar_ql):
        files = save_stream(piece, mex.finalize_part(b['part']), b['stem'])
        files['options'] = dict(options)
        files['tempo_desc'] = b['desc']['tempo']
        files['instrument_desc'] = b['desc']['instrument']
        out[b['key']] = files
    return out


# ---------------------------------------------------------------- 主入口

def melody_line(data: ScoreData, config, piece, voice_out: dict,
                options: dict | None = None) -> dict:
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
    options = mex.resolve_options(config, options)

    windows = windows_of_measures(data, config.analysis_window_bars)
    if not windows:
        windows = [{'window': 1, 'start_measure': 1,
                    'end_measure': max(data.measure_count, 1)}]

    # ---- ① 窗级归属（人声掩膜 + 显著性） ----
    # 按 start_measure 取窗，不按下标：matrix 是 voice_id 用**当时**的窗口长度
    # 算的，本步只依赖 voice_id，改过窗口设置后单步重跑会复用旧矩阵，
    # 按下标取会让"窗边界"和"显著性"对不上，导出取到错误小节的音符。
    by_start = {int(r.get('start_measure', -1)): r for r in matrix}
    win_rows = []
    matrix_mismatch = []
    for w in windows:
        row = by_start.get(w['start_measure'])
        if row is None:
            row = {'scores': {}}
            matrix_mismatch.append(w['start_measure'])
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
        s, e = mex.seg_range(data, measure_offsets, bar_ql,
                             r['start_measure'], r['end_measure'])
        notes = mex.events_in(data.parts[r['part_index']], s, e)
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
        'export_options': options,
        'export_options_label': mex.options_label(options),
        'exports': {},
    }
    if matrix_mismatch:
        # 显著性矩阵与当前窗口划分对不上（改过窗口设置后只重跑了本模块）
        result['window_warning'] = (
            f'有 {len(matrix_mismatch)} 个窗（起始小节 {matrix_mismatch[:6]}）'
            '在声部识别的显著性矩阵里找不到对应记录，已按静默处理。'
            '请在「分析旋律」里重跑声部识别，或把窗口长度改回原值。')

    if not segments:
        result['note'] = '全曲没有可用于追踪的有音高事件（可能整谱静默或全为打击乐）。'
        return result

    # ---- ⑤ 导出（原位 + 无缝拼接，带节奏 / 音色选项） ----
    try:
        result['exports'] = export_segments(
            data, piece, segments, options,
            measure_offsets=measure_offsets, bar_ql=bar_ql)
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
