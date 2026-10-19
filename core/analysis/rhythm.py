"""节奏网格与速度曲线 + 节奏型与律动分析（分析因果链第 1、2 步）。

rhythm_grid：拍号 / 小节 / 速度 —— 纯本地元数据解析（时间网格是地基）。
rhythm_pattern：节奏型统计 + 律动特征 + LLM 描述性总结。
"""
from .llm import run_llm_json_soft, JSON_ONLY
from .scoredata import ScoreData


def rhythm_grid(data: ScoreData, config) -> dict:
    """节奏网格与速度曲线（本地确定性计算）。"""
    from music21 import tempo as m21tempo  # noqa: F401 保留说明：速度来自 ScoreData.tempos

    default_bpm = next((t['bpm'] for t in data.tempos if t['bpm']), None)
    return {
        'time_signatures': data.time_signatures or
            [{'measure': 1, 'numerator': 4, 'denominator': 4, 'offset': 0.0}],
        'tempos': data.tempos,
        'default_bpm': default_bpm,
        'measure_count': data.measure_count,
        'total_quarter_length': round(data.total_ql, 3),
        'part_count': len(data.parts),
        'notes': [
            {'part': p.name, 'instrument': p.instrument,
             'is_percussion': p.is_percussion,
             'note_count': p.note_count, 'total_duration': round(p.total_duration, 3)}
            for p in data.parts
        ],
    }


def rhythm_pattern(data: ScoreData, config) -> dict:
    """节奏型与律动分析：本地统计 + LLM 总结（叙述链的"节奏"小节）。"""
    all_events = [e for p in data.parts for e in p.events]

    # 1) 时值直方图
    dur_hist = {}
    for e in all_events:
        dur_hist[e.duration] = dur_hist.get(e.duration, 0) + 1
    dur_top = sorted(dur_hist.items(), key=lambda kv: -kv[1])[:8]

    # 2) 拍位分布（onset 相对小节起点的位置）
    from .scoredata import build_measure_map
    measure_offsets, _, bar_ql = build_measure_map(data)
    beat_positions = {}
    syncopation = 0
    for e in all_events:
        m_start = measure_offsets.get(e.measure, 0.0)
        pos = e.offset - m_start
        beat = round(pos, 3)
        beat_positions[beat] = beat_positions.get(beat, 0) + 1
        # 弱拍 / 反拍起音（不在整拍上）计为切分倾向
        if abs(beat - round(beat)) > 1e-6:
            syncopation += 1
    beat_dist = sorted(beat_positions.items())[:16]
    syncopation_ratio = round(syncopation / max(len(all_events), 1), 4)

    # 3) 每小节 onset 模式（"节奏型"）Top-N
    patterns = {}
    for m in range(1, data.measure_count + 1):
        m_start = measure_offsets.get(m, None)
        if m_start is None:
            continue
        onsets = sorted({round(e.offset - m_start, 3)
                         for e in all_events
                         if e.measure == m and 0 <= e.offset - m_start < bar_ql})
        if onsets:
            key = ' '.join(str(o) for o in onsets)
            patterns[key] = patterns.get(key, 0) + 1
    top_patterns = sorted(patterns.items(), key=lambda kv: -kv[1])[:5]

    stats = {
        'duration_histogram': [{'quarter_length': d, 'count': c} for d, c in dur_top],
        'beat_position_distribution': [{'beat_offset': b, 'count': c} for b, c in beat_dist],
        'syncopation_ratio': syncopation_ratio,
        'top_measure_patterns': [{'pattern': k, 'count': v} for k, v in top_patterns],
        'total_note_events': len(all_events),
    }

    # LLM 律动总结（要求模型用工具箱验证时值换算）
    # 注意：保持 reasoningEffort=high 时思考 tokens 与正文共用额度，
    # 长回答极易被截断 → 走软降级，解析失败也不让本步骤报错。
    task = (
        '你是专业节奏分析师，可以使用音乐理论工具箱（note_info 等）核验时值换算。'
        '以下是一首乐曲的节奏统计特征（JSON）：\n'
        f'{stats}\n\n'
        '请给出节奏的描述性分析，JSON 字段：\n'
        '{"groove": "律动概括（如 行进感/舞蹈性/自由Rubato）",'
        '"character": "节奏性格描述",'
        '"syncopation": "切分/强弱错位的具体说明",'
        '"main_patterns": "主要节奏型及其音乐效果",'
        '"description": "2-4 句整体节奏小结"}'
        f'{JSON_ONLY}'
    )
    llm = run_llm_json_soft(task, profile=config.dsh_profile,
                            timeout=config.dsh_timeout, retries=1)

    return {'stats': stats, 'llm_analysis': llm}
