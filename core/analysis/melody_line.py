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
5. **形态资格层（可选，默认关，在账户总体设置里开）**：形态决定「有没有资格」，
   显著性决定「谁更强」—— 形态只做**否决**，不参与加分、不改 margin。
   段级算出各声部的旋律轮廓形态分（级进比 / 拱形度 / 轮廓平滑度），
   当前归属声部形态不合格时，由段内形态合格且显著性最高的声部顶上。
   用来挡住分解和弦、阿尔贝蒂低音、固定音型这类「周期音型」被误判成主旋律 ——
   它们的动机重复度接近 1.0，单看显著性很容易赢过真正的人声旋律。
   两条安全设计：① 段内音符太少不足以统计轮廓时**不作判定、一律放行**；
   ② 所有声部都不合格时**保留原归属**，绝不把整段丢掉；
6. 导出两种：
   - absolute：**原位**写入（各段保留真实小节位置，段间空隙留白）——MIR 语义；
   - spliced：首尾相接的连续单声部（方便当一条旋律线听 / 导入 DAW）。
"""
from . import melody_export as mex
from .melody import save_stream
from .scoredata import (
    ScoreData, build_measure_map, windows_of_measures, interval_seq,
)


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


# ---------------------------------------------------------------- 形态资格层
# 「形态决定有没有资格，显著性决定谁更强」—— 形态只做**否决**：
# 不参与显著性加分、不改 margin、不改置信度，只在当前归属形态不合格时换人。
#
# 三项指标（都只依赖段内音高序列，不引入新的输入）：
#   级进比 step_ratio      |音程| ≤ 2 的占比。真旋律以级进为主；
#                          分解和弦 / 阿尔贝蒂低音几乎全是跳进 → 接近 0。
#   拱形度 arch            最高音出现位置的居中程度。真旋律的高潮多在乐句中段；
#                          单调爬升或周期音型 → 低。
#   轮廓平滑度 smooth      局部极值点越密越"来回扫"（固定音型）→ 越低。
#
# 级进比权重压倒性最高：它是区分"旋律"与"音型"唯一可靠的一项。拱形与平滑度
# 只作辅助 —— 上下行的分解和弦在轮廓上同样是"平滑大波浪"，单看这两项它很像
# 旋律，只有级进比能把它识破（实测：上下行琶音 step≈0 但 arch≈0.78、
# smooth≈0.73，权重平摊时形态分 0.38 会漏过 0.35 的阈值；按 0.7/0.15/0.15
# 加权后为 0.23，与真人声旋律的 0.74 拉开明显差距）。

SHAPE_MIN_NOTES = 4                 # 少于该音数不判定轮廓（证据不足，一律放行）
SHAPE_WEIGHTS = (0.7, 0.15, 0.15)   # 级进比 / 拱形度 / 轮廓平滑度


def _seg_pitch_seq(data: ScoreData, part_index: int, s: float, e: float) -> list:
    """声部在段 [s, e) 内的音高序列（按起点升序，和弦取最高音）。"""
    evs = sorted(mex.events_in(data.parts[part_index], s, e),
                 key=lambda ev: ev.offset)
    return [max(ev.midi_pitches) for ev in evs if ev.midi_pitches]


def contour_shape(seq: list) -> dict:
    """音高序列 → 旋律轮廓形态分（0–1，越大越像"一条旋律"）。

    ``score`` 为 None = "音符太少，不作判定"，调用方必须按**放行**处理。
    """
    n = len(seq)
    if n < SHAPE_MIN_NOTES:
        return {'note_count': n, 'score': None,
                'reason': f'仅 {n} 个音，不足以判定轮廓'}
    ivs = interval_seq(seq)
    step_ratio = sum(1 for iv in ivs if abs(iv) <= 2) / len(ivs)
    # 拱形度：最高音位置的居中程度。有多个最高点时取**中位**位置 —— 否则
    # 周期性旋律会因为"最高点落在第一个周期"而被系统性低估。
    top = max(seq)
    peaks = [i for i, p in enumerate(seq) if p == top]
    pos = peaks[len(peaks) // 2] / max(n - 1, 1)
    arch = max(0.0, 1.0 - 2.0 * abs(pos - 0.5))
    # 轮廓平滑度：局部极值点越密越像来回扫的固定音型
    ext = 0
    for i in range(1, n - 1):
        a, b = seq[i] - seq[i - 1], seq[i + 1] - seq[i]
        if a and b and (a > 0) != (b > 0):
            ext += 1
    smooth = 1.0 - min(ext / max(n - 2, 1), 1.0)
    score = round(SHAPE_WEIGHTS[0] * step_ratio
                  + SHAPE_WEIGHTS[1] * arch
                  + SHAPE_WEIGHTS[2] * smooth, 4)
    return {
        'note_count': n,
        'step_ratio': round(step_ratio, 4),
        'arch': round(arch, 4),
        'contour_smooth': round(smooth, 4),
        'score': score,
    }


def shape_eligible(shape: dict, min_score: float) -> bool:
    """形态是否够格当主旋律。score 为 None（证据不足）一律放行。"""
    sc = shape.get('score')
    return True if sc is None else sc >= min_score


def _seg_ranked_salience(win_scores: dict, start_m: int, end_m: int,
                         owner: int) -> list:
    """段覆盖窗内各声部的平均显著性，降序 [(part_index, salience)]。

    保证当前归属 owner 一定在列表里（它可能在该段所有窗里都没有分数，
    例如人声掩膜把它选出来但显著性矩阵里没记它）。
    """
    agg = {}
    for sm, scores in win_scores.items():
        if not (start_m <= sm <= end_m):
            continue
        for idx, sc in scores.items():
            agg.setdefault(idx, []).append(sc)
    if not agg:
        return [(owner, 0.0)]
    ranked = sorted(((idx, round(sum(v) / len(v), 4)) for idx, v in agg.items()),
                    key=lambda kv: -kv[1])
    if owner not in agg:
        ranked.append((owner, 0.0))
    return ranked


def apply_shape_gate(data: ScoreData, s: float, e: float, ranked: list,
                     current_idx: int, min_score: float):
    """形态资格层：从按显著性降序的候选里挑第一个**形态合格**的声部。

    返回 ``(chosen_index, shapes, reason)``；reason 为空表示没有换人。
    一个合格的都没有时**保留原归属**，绝不把整段丢掉。
    """
    shapes = {idx: contour_shape(_seg_pitch_seq(data, idx, s, e))
              for idx, _ in ranked}
    for sh in shapes.values():
        sh['eligible'] = shape_eligible(sh, min_score)
    cur = shapes.get(current_idx)
    if cur is None or cur['eligible']:
        return current_idx, shapes, ''
    cur_name = data.parts[current_idx].name
    why = (f"形态分 {cur['score']}（级进 {cur['step_ratio']} / "
           f"拱形 {cur['arch']} / 平滑 {cur['contour_smooth']}）低于下限 {min_score}")
    for idx, _ in ranked:
        if idx == current_idx or not shapes.get(idx, {}).get('eligible'):
            continue
        return idx, shapes, (f'{cur_name}{why}，改由形态合格的 '
                             f'{data.parts[idx].name}'
                             f"（形态分 {shapes[idx]['score']}）承担")
    return current_idx, shapes, (
        f'{cur_name}{why}，但段内没有形态合格的替代声部，保留原归属')


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
                       measure_offsets=None, bar_ql=None, shape_tag: str = '',
                       run_tag: str = '') -> list:
    """构建两种版本的导出 Part（**不落盘**）。

    返回 [{'key', 'stem', 'part', 'desc'}]：
      key  = 结果字典里的键（absolute / spliced，保持稳定）
      stem = 落盘文件名主干，带形态层标记 / 节奏 / 音色 / 运行序号后缀
             （换选项、换口径、换一次运行就换文件名，避免同名覆盖后浏览器
             给缓存里的旧文件）

    "按当前设置直接下载"与"应用并重新导出"共用这一份，保证两条路导出一致。
    """
    if measure_offsets is None or bar_ql is None:
        measure_offsets, _, bar_ql = build_measure_map(data)
    built = []
    for kind, fn, stem in (('absolute', _export_absolute, 'melody_line_absolute'),
                           ('spliced', _export_spliced, 'melody_line_spliced')):
        part, desc = fn(data, segments, measure_offsets, bar_ql, options)
        built.append({'key': kind,
                      'stem': mex.stem_with_options(stem, options, shape_tag,
                                                    run_tag),
                      'part': part, 'desc': desc})
    return built


def export_segments(data: ScoreData, piece, segments: list, options: dict,
                    measure_offsets=None, bar_ql=None, shape_tag: str = '',
                    run_tag: str = '') -> dict:
    """按段导出两种版本（供主入口与"重新导出"接口共用）。"""
    out = {}
    for b in build_export_parts(data, segments, options, measure_offsets, bar_ql,
                                shape_tag, run_tag):
        files = save_stream(piece, mex.finalize_part(b['part']), b['stem'])
        files['options'] = dict(options)
        files['tempo_desc'] = b['desc']['tempo']
        files['instrument_desc'] = b['desc']['instrument']
        out[b['key']] = files
    return out


# ---------------------------------------------------------------- 主入口

def melody_line(data: ScoreData, config, piece, voice_out: dict,
                options: dict | None = None,
                run_seq: int | None = None) -> dict:
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
    # 形态资格层（可选，默认关）：开时才在段级做一次"有没有资格"的否决
    shape_gate = bool(getattr(config, 'melody_shape_gate', False))
    shape_min = float(getattr(config, 'melody_shape_min_score', 0.35) or 0.35)
    shape_min = max(0.0, min(1.0, shape_min))
    options = mex.resolve_options(config, options)
    # 文件名标记：形态层开没开 + 这是第几次运行（每次落盘的文件名都不同，
    # 重跑不覆盖上一次的产物，历史归档里每一次都能下到它自己那份）
    shape_tag = mex.shape_stem_tag(shape_gate, shape_min)
    if run_seq is None:
        run_seq = mex.next_run_seq(piece, 'melody_line')
    run_tag = mex.run_stem_tag(run_seq)

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
    win_scores = {}          # {start_measure: {part_index: salience}}，段级形态层用
    matrix_mismatch = []
    for w in windows:
        row = by_start.get(w['start_measure'])
        if row is None:
            row = {'scores': {}}
            matrix_mismatch.append(w['start_measure'])
        win_scores[w['start_measure']] = {
            int(k): float(v) for k, v in (row.get('scores') or {}).items()}
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
    shape_replacements = []
    for i, r in enumerate(runs):
        s, e = mex.seg_range(data, measure_offsets, bar_ql,
                             r['start_measure'], r['end_measure'])
        # ---- ③b 形态资格层（可选）：形态定"资格"，显著性定"强弱" ----
        owner = r['part_index']
        shape_info = None
        replaced_from = None
        shape_reason = ''
        if shape_gate:
            ranked = _seg_ranked_salience(
                win_scores, r['start_measure'], r['end_measure'], owner)
            owner, shapes, shape_reason = apply_shape_gate(
                data, s, e, ranked, owner, shape_min)
            shape_info = shapes.get(owner) or {}
            if owner != r['part_index']:
                replaced_from = r['part_index']
                shape_replacements.append({
                    'measures': f"{r['start_measure']}–{r['end_measure']}",
                    'from': data.parts[r['part_index']].name,
                    'to': data.parts[owner].name,
                    'reason': shape_reason,
                })
        notes = mex.events_in(data.parts[owner], s, e)
        rows = [x for x in win_rows
                if x['part_index'] is not None
                and r['start_measure'] <= x['start_measure'] <= r['end_measure']
                and x['part_index'] == r['part_index']]
        margin = (round(sum(x['margin'] for x in rows) / len(rows), 4)
                  if rows else r.get('avg_margin', 0.0))
        # 形态层换过人 → 按新声部算人声先验（原来那段的人声 prior 不再适用）
        is_voice = bool(data.parts[owner].is_voice)
        # 人声优先是强先验：同样的 margin，人声段更有把握
        conf = round(min(1.0, margin + 0.15), 4) if is_voice else margin
        segments.append({
            'seq': i + 1,
            'start_measure': r['start_measure'],
            'end_measure': r['end_measure'],
            'bars': r['end_measure'] - r['start_measure'] + 1,
            'part_index': owner,
            'part_name': data.parts[owner].name,
            'instrument': data.parts[owner].instrument or '（未标注）',
            'is_voice': is_voice,
            'note_count': len(notes),
            'margin': margin,
            'confidence': conf,
            'level': _level(conf, hi, mid),
            'absorbed_from': sorted(set(r.get('absorbed', []))),
            'shape': shape_info,
            'shape_replaced_from': replaced_from,
            'shape_reason': shape_reason,
        })

    # ---- ④ 覆盖率与掩膜摘要 ----
    covered = sum(s['bars'] for s in segments)
    total_bars = max(data.measure_count, 1)
    voice_parts = [{'index': p.index, 'name': p.name} for p in data.parts if p.is_voice]

    method = ('每个时间窗按"人声在唱→人声；人声休止→最突出的器乐声部"定归属，'
              '连续同归属的窗合并为段；不受交接检测门控，'
              '前奏 / 间奏 / 尾奏同样提取。')
    if shape_gate:
        method += ('另加一道形态资格筛选：段内声部的旋律轮廓形态分'
                   f'（级进比×0.7 + 拱形度×0.15 + 轮廓平滑度×0.15）低于 {shape_min} '
                   '就失去资格，由形态合格且显著性最高的声部顶上；'
                   '音符太少不足以统计时不作判定，所有声部都不合格时保留原归属。')
    result = {
        'mode': ('逐段追踪（人声掩膜 + 显著性 + 形态资格）' if shape_gate
                 else '逐段追踪（人声掩膜 + 显著性）'),
        'method': method,
        'window_bars': config.analysis_window_bars,
        'voice_priority': voice_priority,
        'min_segment_bars': min_bars,
        'confidence_thresholds': {'high': hi, 'mid': mid},
        'has_voice_parts': bool(voice_parts),
        'voice_parts': voice_parts,
        'shape_gate': shape_gate,
        'shape_gate_threshold': shape_min if shape_gate else None,
        'shape_replacements': len(shape_replacements),
        'shape_replacement_notes': shape_replacements,
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
        'export_run_seq': int(run_seq or 1),
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
            measure_offsets=measure_offsets, bar_ql=bar_ql,
            shape_tag=shape_tag, run_tag=run_tag)
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
