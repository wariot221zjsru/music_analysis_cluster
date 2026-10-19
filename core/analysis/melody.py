"""主旋律提取（可选输出、与旋律分析并列）+ 旋律形态分析。

分支策略（计划书 4.5.2）：
  单声   → 唯一主旋律，直接导出 MIDI / MusicXML
  复调   → 多候选 + 置信度（不强行选唯一一条）
  交接   → 主旋律归属时间线 + 按时间线拼接导出
旋律形态分析（叙述链）按同样分支组织输出。
"""
import copy

from django.utils import timezone

from .. import storage
from .llm import run_llm_json_soft, JSON_ONLY
from .scoredata import (
    ScoreData, build_measure_map, interval_seq, midi_to_name,
)


# ---------------- 主旋律提取（并列、可选） ----------------

def _export_part_as_melody(data: ScoreData, part_index: int):
    """把声部抽出为独立旋律 Stream（music21）。"""
    from music21 import stream, note as m21note, chord as m21chord

    src = data.parts[part_index]
    part = stream.Part()
    part.partName = f'主旋律 · {src.name}'
    for e in src.pitched_events:
        if len(e.midi_pitches) == 1:
            n = m21note.Note()
            n.pitch.midi = e.midi_pitches[0]
        else:
            n = m21chord.Chord([m + 0 for m in e.midi_pitches])
        n.duration.quarterLength = e.duration
        part.insert(e.offset, n)
    return part


def _export_timeline_spliced(data: ScoreData, timeline: list):
    """按主旋律归属时间线拼接各段为一条主旋律（交接分支）。"""
    from music21 import stream, note as m21note, chord as m21chord

    measure_offsets, _, _ = build_measure_map(data)
    part = stream.Part()
    part.partName = '主旋律（时间线拼接 · 交接）'
    cursor = 0.0
    for seg in timeline:
        idx = seg['part_index']
        if idx is None:
            continue
        s = measure_offsets.get(seg['start_measure'], 0.0)
        e = measure_offsets.get(seg['end_measure'] + 1,
                                s + (seg['end_measure'] - seg['start_measure'] + 1) * 4.0)
        for ev in data.parts[idx].pitched_events:
            if s <= ev.offset < e:
                if len(ev.midi_pitches) == 1:
                    n = m21note.Note()
                    n.pitch.midi = ev.midi_pitches[0]
                else:
                    n = m21chord.Chord([m + 0 for m in ev.midi_pitches])
                n.duration.quarterLength = ev.duration
                part.insert(cursor + (ev.offset - s), n)
        cursor += (e - s)
    return part


def _save_stream(piece, stream_obj, stem: str):
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


def melody_extraction(data: ScoreData, config, piece, voice_out: dict,
                      poly_out: dict, exchange_out: dict) -> dict:
    """主旋律提取主入口（用户单独触发；复调→多候选，交接→时间线）。"""
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
        'exports': {},
    }

    if exchange_out['is_exchange']:
        # 交接分支：时间线 + 拼接导出
        timeline = exchange_out['attribution_timeline']
        spliced = _export_timeline_spliced(data, timeline)
        result['mode'] = '主旋律交接（时间线式归属）'
        result['attribution_timeline'] = timeline
        result['exports'] = _save_stream(piece, spliced, 'melody_spliced')
    elif poly_out['is_polyphonic']:
        # 复调分支：多候选 + 置信度；导出 top-1（供参考）
        result['mode'] = '复调（多候选 + 置信度，不强行选唯一）'
        if candidates:
            top = _export_part_as_melody(data, candidates[0]['part_index'])
            result['exports'] = _save_stream(
                piece, top, f'melody_candidate_{candidates[0]["part_index"]}')
    else:
        # 单声 / 主调：显著性最高声部即主旋律
        result['mode'] = '单声 / 主调（唯一主旋律）'
        if candidates:
            top = _export_part_as_melody(data, candidates[0]['part_index'])
            result['chosen'] = candidates[0]
            result['exports'] = _save_stream(
                piece, top, f'melody_{candidates[0]["part_index"]}')

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
    measure_offsets, _, _ = build_measure_map(data)
    part_names = voice_out['salience']['part_names']
    branch = ('主旋律交接' if exchange_out['is_exchange']
              else ('复调' if poly_out['is_polyphonic'] else '单声 / 主调'))

    analyses = []
    if exchange_out['is_exchange']:
        # 交接分支：逐时间线段落分析
        for seg in exchange_out['attribution_timeline']:
            if seg['part_index'] is None:
                continue
            s = measure_offsets.get(seg['start_measure'], 0.0)
            e = measure_offsets.get(
                seg['end_measure'] + 1,
                s + (seg['end_measure'] - seg['start_measure'] + 1) * 4.0)
            seq = [max(ev.midi_pitches)
                   for ev in data.parts[seg['part_index']].pitched_events
                   if s <= ev.offset < e]
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
