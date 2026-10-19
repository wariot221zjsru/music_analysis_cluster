"""复调检测（分析因果链第 5 步）与主旋律交接检测（第 6 步）。

复调（equal polyphony）：同一时间窗多声部同时承担旋律地位（"一起响"）。
主旋律交接（fugato / 模仿）：每个时间窗仍有唯一主旋律，但归属随时间
在不同声部间切换（"接力响"）。二者判定逻辑不同、可同时存在。
阈值均来自 SystemConfig（管理员前端可调）。
"""
from .scoredata import (
    ScoreData, build_measure_map, interval_seq, lcs_ratio, pitched_parts,
)


def polyphony(data: ScoreData, config, voice_out: dict) -> dict:
    """复调自动判定（单声 / 复调）。"""
    dens = voice_out['polyphony_density']
    threshold = config.polyphony_threshold
    is_poly = dens['ratio'] >= threshold

    per_part_flag = {
        idx: d['ratio'] >= threshold
        for idx, d in dens['per_part'].items() if d['sounding'] > 0
    }
    return {
        'is_polyphonic': is_poly,
        'ratio': dens['ratio'],
        'threshold': threshold,
        'per_part': dens['per_part'],
        'polyphonic_parts': [int(i) for i, f in per_part_flag.items() if f],
        # "单声"看的是有音高的声部数（打击乐不算独立声部）
        'branch': '复调' if is_poly
        else ('单声' if len(pitched_parts(data)) <= 1 else '主调'),
        'confidence': round(min(dens['ratio'] / threshold, 2.0)
                            if is_poly else max(0.0, 1 - dens['ratio'] / max(threshold, 1e-6)), 4),
    }


def _segment_pitch_seq(data: ScoreData, part_index: int, start_m: int, end_m: int):
    measure_offsets, _, _ = build_measure_map(data)
    s = measure_offsets.get(start_m, 0.0)
    e = measure_offsets.get(end_m + 1, s + (end_m - start_m + 1) * 4.0)
    return [max(ev.midi_pitches) for ev in data.parts[part_index].pitched_events
            if s <= ev.offset < e]


def exchange(data: ScoreData, config, voice_out: dict) -> dict:
    """主旋律交接检测：基于归属时间线的切换次数 + 声部间动机相似性。"""
    timeline = voice_out['salience']['attribution_timeline']
    part_names = voice_out['salience']['part_names']

    switches = []
    for a, b in zip(timeline, timeline[1:]):
        if a['part_index'] is None or b['part_index'] is None:
            continue
        if a['part_index'] != b['part_index']:
            seq_a = _segment_pitch_seq(data, a['part_index'],
                                       a['start_measure'], a['end_measure'])
            seq_b = _segment_pitch_seq(data, b['part_index'],
                                       b['start_measure'], b['end_measure'])
            sim = lcs_ratio(interval_seq(seq_a[:24]), interval_seq(seq_b[:24]))
            switches.append({
                'from_part': a['part_index'],
                'from_name': part_names[a['part_index']],
                'to_part': b['part_index'],
                'to_name': part_names[b['part_index']],
                'at_measure': b['start_measure'],
                'similarity': round(sim, 4),
            })

    min_switches = config.exchange_min_switches
    sim_threshold = config.exchange_similarity_threshold
    valid_switches = [s for s in switches if s['similarity'] >= sim_threshold]
    is_exchange = len(valid_switches) >= min_switches

    return {
        'is_exchange': is_exchange,
        'switch_count': len(valid_switches),
        'min_switches': min_switches,
        'similarity_threshold': sim_threshold,
        'switches': valid_switches,
        'attribution_timeline': [
            {
                'part_index': seg['part_index'],
                'part_name': part_names[seg['part_index']] if seg['part_index'] is not None else None,
                'start_measure': seg['start_measure'],
                'end_measure': seg['end_measure'],
            } for seg in timeline
        ],
        'branch': '主旋律交接（时间线式归属）' if is_exchange else '无交接 / 固定归属',
        'note': ('民乐 / 交响乐合奏常见的"接力轮奏"模式：'
                 '每个时间窗仍有唯一主旋律声部，但归属随时间切换。'),
    }
