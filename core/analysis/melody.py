"""主旋律声部识别（选轨，旧口径）+ 旋律形态分析。

定位：本模块回答"**整曲（或少数几个交接段）哪个声部是主旋律**"——
即 MIR 里的 melody voice / track identification（选轨）任务。

分支策略（计划书 4.5.2，保持不变）：
  单声   → 唯一主旋律，直接导出 MIDI / MusicXML
  复调   → 多候选 + 置信度（不强行选唯一一条）
  交接   → 先按阈值判定是否为主旋律交接，是则按归属时间线分段拼接导出

与 core.analysis.melody_line 的分工（两张卡并列展示、互不干扰）：
  本模块      = 选轨（整曲 / 交接型），受"是否交接"门控，前奏可能为器乐而丢掉
  melody_line = 逐段追踪（MIR 旋律线），不受门控，前奏 / 间奏同样提取

旋律形态分析（叙述链）按同样分支组织输出。
"""
import copy

from django.utils import timezone

from .. import storage
from . import melody_export as mex
from .llm import run_llm_json_soft, JSON_ONLY
from .scoredata import (
    ScoreData, build_measure_map, interval_seq, midi_to_name,
)


# ---------------- 主旋律声部识别（选轨，并列、可选） ----------------

def _note_or_chord(ev, duration=None):
    """事件 → music21 音符 / 和弦（与显著性口径一致：和弦保留全部音高）。"""
    from music21 import note as m21note, chord as m21chord

    if len(ev.midi_pitches) == 1:
        n = m21note.Note()
        n.pitch.midi = ev.midi_pitches[0]
    else:
        n = m21chord.Chord([m for m in ev.midi_pitches])
    n.duration.quarterLength = (duration if duration is not None else ev.duration)
    return n


def _export_part_as_melody(data: ScoreData, part_index: int, options: dict):
    """把声部抽出为独立旋律 Stream（music21），带节奏 / 音色选项。"""
    from music21 import stream

    measure_offsets, _, bar_ql = build_measure_map(data)
    src = data.parts[part_index]
    part = stream.Part()
    part.partName = f'主旋律 · {src.name}'
    for e in src.pitched_events:
        part.insert(e.offset, _note_or_chord(e))
    tempo_desc = mex.apply_tempo(part, data, measure_offsets, bar_ql, options)
    inst_desc = mex.apply_instruments(
        part, data, [{'offset': 0.0, 'part_index': part_index}], options)
    return part, {'tempo': tempo_desc, 'instrument': inst_desc}


def _export_timeline_spliced(data: ScoreData, timeline: list, options: dict):
    """按主旋律归属时间线拼接各段为一条主旋律（交接分支），带节奏 / 音色选项。"""
    from music21 import stream, tempo as m21tempo

    measure_offsets, _, bar_ql = build_measure_map(data)
    part = stream.Part()
    part.partName = '主旋律（时间线拼接 · 交接）'
    spans = []
    cursor = 0.0
    for seg in timeline:
        idx = seg['part_index']
        if idx is None:
            continue
        # 用 melody_export 的区间换算：小节号查不到时按相邻小节推算，
        # 不再退回 0.0 / 硬编码 4.0 拍（旧实现会让整段错位到曲首）
        s, e = mex.seg_range(data, measure_offsets, bar_ql,
                             seg['start_measure'], seg['end_measure'])
        span = max(e - s, 0.0)
        spans.append({'offset': cursor, 'part_index': idx})
        for ev in mex.events_in(data.parts[idx], s, e):
            part.insert(cursor + (ev.offset - s),
                        _note_or_chord(ev, mex.clamped_duration(ev, e)))
        cursor += span
    if options['tempo_mode'] == 'original':
        first_bpm = next((t['bpm'] for t in (data.tempos or []) if t.get('bpm')),
                         mex.DEFAULT_BPM)
        part.insert(0.0, m21tempo.MetronomeMark(number=float(first_bpm)))
        tempo_desc = f'原曲速度 {float(first_bpm):.0f} BPM（拼接后小节错位，按单一速度写入）'
    else:
        tempo_desc = mex.apply_tempo(part, data, measure_offsets, bar_ql, options)
    inst_desc = mex.apply_instruments(part, data, spans, options)
    return part, {'tempo': tempo_desc, 'instrument': inst_desc}


def _write_melody(piece, part_desc, stem: str, options: dict, run_tag: str = ''):
    """补小节 → 落盘 → 附记本次导出用的节奏 / 音色。

    run_tag 带上运行序号（r1 / r2 …），每次运行的产物各存一份，重跑不覆盖。
    """
    part, desc = part_desc
    files = save_stream(piece, mex.finalize_part(part),
                        mex.stem_with_options(stem, options, run_tag=run_tag))
    files['options'] = dict(options)
    files['tempo_desc'] = desc['tempo']
    files['instrument_desc'] = desc['instrument']
    return files


def save_stream(piece, stream_obj, stem: str):
    """把 music21 Stream 写出为 MIDI + MusicXML，存入永久目录（media/melody/…）。

    注意：原始音频 / 源文件在转换完成后已被清理，因此这里不能依赖
    ``piece.source_file.path``，统一走 core.storage 的永久存储。
    """
    workdir = storage.new_workdir(prefix=f'melody_p{piece.id}_')
    try:
        midi_path = workdir / f'{stem}.mid'
        xml_path = workdir / f'{stem}.musicxml'
        stream_obj.write('midi', fp=str(midi_path))
        stream_obj.write('musicxml', fp=str(xml_path))
        rel_dir = f'melody/{timezone.localdate():%Y%m%d}/p{piece.id}'
        return {
            'midi': storage.store_file(rel_dir, midi_path, f'{stem}.mid'),
            'musicxml': storage.store_file(rel_dir, xml_path, f'{stem}.musicxml'),
        }
    finally:
        storage.cleanup_workdir(workdir)


def build_export_parts(data: ScoreData, prior_output: dict, options: dict) -> list:
    """按上一次的分支结果构建导出 Part（**不落盘**）。

    返回 [{'key', 'stem', 'part', 'desc'}]：
      key  = 结果字典里的键（稳定，不随选项变，页面靠它识别"候选 N"）
      stem = 落盘文件名主干，带节奏 / 音色后缀

    复调 / 交接 / 单声 三种分支共用：分支决定"哪几条声部 / 哪些段"，
    节奏与音色由 options 决定。
    """
    built = []
    if not prior_output:
        return built
    mode = prior_output.get('mode') or ''
    if '交接' in mode:
        timeline = prior_output.get('attribution_timeline') or []
        part, desc = _export_timeline_spliced(data, timeline, options)
        built.append({'key': 'melody_spliced', 'stem': 'melody_spliced',
                      'part': part, 'desc': desc})
    elif '复调' in mode:
        for c in (prior_output.get('candidates') or [])[:5]:
            idx = c.get('part_index')
            if idx is None:
                continue
            part, desc = _export_part_as_melody(data, idx, options)
            key = f'melody_candidate_{idx}'
            built.append({'key': key, 'stem': key, 'part': part, 'desc': desc})
    else:
        chosen = prior_output.get('chosen') or {}
        idx = chosen.get('part_index')
        if idx is not None:
            part, desc = _export_part_as_melody(data, idx, options)
            key = f'melody_{idx}'
            built.append({'key': key, 'stem': key, 'part': part, 'desc': desc})
    return built


def export_again(data: ScoreData, piece, prior_output: dict, options: dict,
                 run_tag: str = '') -> dict:
    """在不重跑分析的前提下，按新的节奏 / 音色重新导出主旋律声部识别（选轨）。

    ``prior_output`` 是上一次 melody_extraction 的 output（含 mode /
    attribution_timeline / chosen / candidates）。
    """
    out = {}
    for b in build_export_parts(data, prior_output, options):
        out[b['key']] = _write_melody(piece, (b['part'], b['desc']),
                                      b['stem'], options, run_tag)
    return out


def melody_extraction(data: ScoreData, config, piece, voice_out: dict,
                      poly_out: dict, exchange_out: dict,
                      options: dict | None = None,
                      run_seq: int | None = None) -> dict:
    """主旋律**声部识别**（选轨）主入口。

    用户单独触发；复调→多候选，交接→时间线分段拼接。
    判定口径与阈值沿用原有实现（exchange 的两个阈值不变）。
    """
    options = mex.resolve_options(config, options)
    # 每次运行一份文件（r1 / r2 …）：重跑不覆盖上一次的产物，
    # 历史归档里每一次运行都能下到它自己那份。
    if run_seq is None:
        run_seq = mex.next_run_seq(piece, 'melody_extraction')
    run_tag = mex.run_stem_tag(run_seq)
    sal = voice_out['salience']
    part_names = sal['part_names']

    # 整曲平均显著性 → 候选置信度
    total_sal = {}
    for w in sal['matrix']:
        for idx, s in w['scores'].items():
            total_sal[idx] = total_sal.get(idx, 0) + s
    n_windows = max(len(sal['matrix']), 1)
    candidates = sorted(
        ({'part_index': int(idx),
          'part_name': part_names[int(idx)],
          'confidence': round(s / n_windows, 4)}
         for idx, s in total_sal.items()),
        key=lambda d: -d['confidence'])
    # 打击乐声部没有旋律意义，不进候选（防御：显著性矩阵理应已排除）
    pitched_idx = {p.index for p in data.parts if p.pitched_events}
    candidates = [c for c in candidates if c['part_index'] in pitched_idx]

    result = {
        'candidates': candidates,
        'export_options': options,
        'export_options_label': mex.options_label(options),
        'export_run_seq': int(run_seq or 1),
        'exports': {},
    }

    if exchange_out['is_exchange']:
        # 交接分支：时间线 + 拼接导出
        timeline = exchange_out['attribution_timeline']
        result['mode'] = '主旋律交接（时间线式归属）'
        result['attribution_timeline'] = timeline
        result['exports'] = _write_melody(
            piece, _export_timeline_spliced(data, timeline, options),
            'melody_spliced', options, run_tag)
    elif poly_out['is_polyphonic']:
        # 复调分支：多候选 + 置信度；导出 top-1（供参考）
        result['mode'] = '复调（多候选 + 置信度，不强行选唯一）'
        if candidates:
            top = _export_part_as_melody(data, candidates[0]['part_index'], options)
            result['exports'] = _write_melody(
                piece, top, f'melody_candidate_{candidates[0]["part_index"]}',
                options, run_tag)
    else:
        # 单声 / 主调：显著性最高声部即主旋律
        result['mode'] = '单声 / 主调（唯一主旋律）'
        if candidates:
            top = _export_part_as_melody(data, candidates[0]['part_index'], options)
            result['chosen'] = candidates[0]
            result['exports'] = _write_melody(
                piece, top, f'melody_{candidates[0]["part_index"]}',
                options, run_tag)

    return result


# ---------------- 旋律形态分析 ----------------

def _melody_features(seq: list) -> dict:
    """单条旋律的本地特征（音高序列 → 统计）。"""
    if len(seq) < 2:
        return {'note_count': len(seq)}
    ivs = interval_seq(seq)
    steps = sum(1 for i in ivs if abs(i) <= 2)
    leaps = sum(1 for i in ivs if abs(i) >= 5)
    # 轮廓方向段
    dirs, cur, run = [], 0, 0
    for i in ivs:
        d = 1 if i > 0 else (-1 if i < 0 else 0)
        if d == cur:
            run += 1
        else:
            if cur != 0:
                dirs.append(cur * run)
            cur, run = d, 1
    if cur != 0:
        dirs.append(cur * run)
    return {
        'note_count': len(seq),
        'pitch_range': [midi_to_name(min(seq)), midi_to_name(max(seq))],
        'range_semitones': max(seq) - min(seq),
        'step_ratio': round(steps / len(ivs), 4),
        'leap_ratio': round(leaps / len(ivs), 4),
        'contour_segments': dirs[:24],   # 正=上行小节数、负=下行
        'mean_pitch': midi_to_name(round(sum(seq) / len(seq)) + 0) if seq else None,
    }


def _llm_melody_analysis(label: str, seq: list, features: dict,
                         config, extra_context: str = '') -> dict:
    pitch_names = [midi_to_name(m) for m in seq[:120]]
    task = (
        f'你是专业旋律分析师，可用 interval_info 工具核验音程。'
        f'请分析 {label} 的旋律形态。\n\n'
        f'【音高序列（前 120 音）】\n{pitch_names}\n\n'
        f'【本地统计特征】\n{features}\n{extra_context}\n'
        '输出 JSON：{"contour_summary": "旋律轮廓概述", '
        '"phrase_analysis": "乐句划分与呼吸分析", '
        '"motive_analysis": "动机识别与发展手法", '
        '"highlights": "1-2 句亮点点评"}'
        f'{JSON_ONLY}'
    )
    # 软降级：保持 reasoningEffort=high 时思考 tokens 与正文共用 max_tokens 额度，
    # 这段分析经常被截断；解析失败只降级本片段，不让整个 melody_shape 步骤变 error。
    return run_llm_json_soft(task, profile=config.dsh_profile,
                             timeout=config.dsh_timeout, retries=1)


def melody_shape(data: ScoreData, config, voice_out: dict, poly_out: dict,
                 exchange_out: dict) -> dict:
    """旋律形态分析主入口（叙述链"旋律"小节的数据来源）。"""
    from .scoredata import build_measure_map
    measure_offsets, _, bar_ql = build_measure_map(data)
    part_names = voice_out['salience']['part_names']
    branch = ('主旋律交接' if exchange_out['is_exchange']
              else ('复调' if poly_out['is_polyphonic'] else '单声 / 主调'))

    analyses = []
    if exchange_out['is_exchange']:
        # 交接分支：逐时间线段落分析
        for seg in exchange_out['attribution_timeline']:
            if seg['part_index'] is None:
                continue
            # 与导出同一套区间换算（不退回 0.0、不硬编码 4.0 拍）
            s, e = mex.seg_range(data, measure_offsets, bar_ql,
                                 seg['start_measure'], seg['end_measure'])
            seq = [max(ev.midi_pitches)
                   for ev in mex.events_in(data.parts[seg['part_index']], s, e)]
            if not seq:
                continue
            label = (f"第{seg['start_measure']}-{seg['end_measure']}小节 · "
                     f"{part_names[seg['part_index']]}")
            features = _melody_features(seq)
            llm = _llm_melody_analysis(label, seq, features, config)
            analyses.append({'segment': label, 'features': features, 'llm_analysis': llm})
    elif poly_out['is_polyphonic']:
        # 复调分支：per-voice 多声部分析（按显著性取前 3 声部）
        total_sal = {}
        for w in voice_out['salience']['matrix']:
            for idx, s_ in w['scores'].items():
                total_sal[idx] = total_sal.get(idx, 0) + s_
        ranked = sorted(total_sal.items(), key=lambda kv: -kv[1])[:3]
        for idx, _ in ranked:
            seq = [max(ev.midi_pitches) for ev in data.parts[int(idx)].pitched_events]
            if not seq:
                continue
            label = f'声部 {part_names[int(idx)]}'
            features = _melody_features(seq)
            llm = _llm_melody_analysis(label, seq, features, config)
            analyses.append({'segment': label, 'features': features, 'llm_analysis': llm})
    else:
        # 单声 / 主调：主旋律声部单份分析
        total_sal = {}
        for w in voice_out['salience']['matrix']:
            for idx, s_ in w['scores'].items():
                total_sal[idx] = total_sal.get(idx, 0) + s_
        if total_sal:
            top_idx = int(max(total_sal.items(), key=lambda kv: kv[1])[0])
            seq = [max(ev.midi_pitches) for ev in data.parts[top_idx].pitched_events]
            if seq:
                label = f'主旋律声部 {part_names[top_idx]}'
                features = _melody_features(seq)
                llm = _llm_melody_analysis(label, seq, features, config)
                analyses.append({'segment': label, 'features': features,
                                 'llm_analysis': llm})

    return {
        'branch': branch,
        'analyses': analyses,
        'note': ('复调作品输出 per-voice 多声部分析；主旋律交接作品输出时间线式归属；'
                 '单声作品输出单条旋律分析（计划书 4.5.2）。'),
    }
