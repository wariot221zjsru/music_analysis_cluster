"""调性布局分析（分析因果链第 8 步）。

因果方向（计划书 4.3.1）：调性判定依赖和弦序列证据 ——
本地 Krumhansl-Schmuckler 音级时值统计给出每段候选调，
LLM 结合和弦进行确定主调、调性区域与转调点。
"""
from .llm import run_llm_json_soft, JSON_ONLY
from .scoredata import ScoreData, ks_key_candidates, midi_to_pc

_SECTION_BARS = 8  # KS 统计分段长度（小节）


def tonality(data: ScoreData, config, chords_out: dict, voice_out: dict) -> dict:
    """调性布局分析主入口。"""
    # 1) 分段 KS 候选
    sections = []
    for start in range(1, data.measure_count + 1, _SECTION_BARS):
        end = min(start + _SECTION_BARS - 1, data.measure_count)
        pc_duration = [0.0] * 12
        for p in data.parts:
            for e in p.pitched_events:
                if start <= e.measure <= end:
                    for pitch in e.midi_pitches:
                        pc_duration[midi_to_pc(pitch)] += e.duration
        if sum(pc_duration) > 0:
            sections.append({
                'start_measure': start, 'end_measure': end,
                'ks_candidates': ks_key_candidates(pc_duration, top=2),
            })

    # 2) 压缩和弦序列（给 LLM 的证据）
    prog = chords_out.get('progression', [])
    prog_brief = [
        {'label': r.get('label'), 'chord': r.get('chord'), 'function': r.get('function')}
        for r in prog if r.get('chord')
    ]

    task = (
        '你是专业调性分析师，可用 scale_generate / scale_harmonize 工具核验'
        '音阶拼写与和声化。请基于以下证据分析全曲调性布局：\n\n'
        f'【分段 KS 调性候选（按音级时值统计）】\n{sections}\n\n'
        f'【和弦进行序列】\n{prog_brief[:120]}\n\n'
        '综合两份证据（和弦证据优先于统计候选），确定：主调、各调性区域、'
        '转调点。输出 JSON：'
        '{"main_key": "主调（如 C 大调）",'
        '"regions": [{"start_measure": 1, "end_measure": 16, "key": "调名", '
        '"type": "大调/小调", "evidence": "判断依据"}],'
        '"modulations": [{"measure": 17, "from_key": "旧调", "to_key": "新调", '
        '"technique": "转调手法（如属准备 / 平行转换）"}],'
        '"confidence": 0~1, "description": "2-3 句调性布局小结"}'
        f'{JSON_ONLY}'
    )
    llm = run_llm_json_soft(task, profile=config.dsh_profile,
                            timeout=config.dsh_timeout, retries=1)
    if not isinstance(llm, dict):
        llm = {'main_key': None, 'description': str(llm)}

    return {
        'main_key': llm.get('main_key'),
        'regions': llm.get('regions', []),
        'modulations': llm.get('modulations', []),
        'confidence': llm.get('confidence'),
        'description': llm.get('description', ''),
        'ks_sections': sections,
        # 解析失败时保留诊断信息（步骤不报错，前端提示重跑本步骤）
        'llm_error': llm.get('_error', ''),
        'llm_raw_text': llm.get('_raw_text', ''),
    }
