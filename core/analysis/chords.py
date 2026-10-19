"""和弦进行分析（分析因果链第 7 步）。

流程（计划书 4.5.1 / 4.4）：
  声部识别 → 按窗口聚合音高集合（节拍强弱加权）→ LLM 逐窗用
  dsh-musictheory 的 chord_identify 做确定性识别 + 功能语义标注。
"""
from .llm import run_llm_json_soft, JSON_ONLY
from .scoredata import (
    ScoreData, build_measure_map, sounding_pitches_at, midi_to_name, midi_to_pc,
)

# 功能标记参考（提示词里给模型的和声常识锚点）
_FUNCTION_HINT = (
    '常见功能标记：T（主）、S（下属）、D（属）；'
    '罗马数字 I ii iii IV V vi vii°（大调）/ i ii° III iv v VI VII（小调）；'
    '终止式如 V-I、IV-V-I、Phrygian 半终止等可用文字补充。'
)


def _aggregate_windows(data: ScoreData, config) -> list:
    """按窗口（小节 / 拍）聚合音高集合，节拍强弱加权。"""
    measure_offsets, measures, bar_ql = build_measure_map(data)

    windows = []
    if config.chord_window == 'beat':
        for m in measures:
            m_start = measure_offsets.get(m, 0.0)
            for b in range(int(bar_ql)):
                windows.append({
                    'label': f'第{m}小节 第{b + 1}拍',
                    'measure': m, 'beat': b + 1,
                    't_start': m_start + b, 't_end': m_start + b + 1,
                })
    else:
        for m in measures:
            m_start = measure_offsets.get(m, 0.0)
            windows.append({
                'label': f'第{m}小节', 'measure': m,
                't_start': m_start, 't_end': m_start + bar_ql,
            })

    for w in windows:
        pc_weight = {}
        bass_midi = None
        # 以 1/4 拍为采样步长遍历窗口
        t = w['t_start']
        while t < w['t_end'] - 1e-9:
            strong = abs((t - w['t_start']) % 1.0) < 1e-6  # 整拍强权
            for p in data.parts:
                for pitch in sounding_pitches_at(p, t):
                    pc = midi_to_pc(pitch)
                    pc_weight[pc] = pc_weight.get(pc, 0.0) + (2.0 if strong else 1.0)
                    if bass_midi is None or pitch < bass_midi:
                        bass_midi = pitch
            t += 0.25
        # 按权重排序取前 6 个音级，映射到 4 个八度的拼写交给模型
        top_pcs = sorted(pc_weight.items(), key=lambda kv: -kv[1])[:6]
        w['pitch_classes'] = [pc for pc, _ in top_pcs]
        w['pitch_names'] = [midi_to_name(60 + pc) for pc, _ in top_pcs]
        w['bass'] = midi_to_name(bass_midi) if bass_midi is not None else None
    return windows


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def chords(data: ScoreData, config, voice_out: dict) -> dict:
    """和弦进行分析主入口。"""
    windows = _aggregate_windows(data, config)
    windows = [w for w in windows if w['pitch_classes']]  # 丢弃空窗口（休止）

    results = []
    errors = []          # 解析失败的批次（本步骤不因此整体失败）
    for chunk in _chunked(windows, config.max_chord_windows_per_call):
        payload = [
            {'label': w['label'], 'pitches': w['pitch_names'], 'bass': w['bass']}
            for w in chunk
        ]
        task = (
            '你是专业和声分析师，必须使用 chord_identify 工具对每个窗口的音高集合'
            '做确定性识别（先算后说、凡算必验），再结合低音给转位判断，'
            '最后做功能标记。\n'
            f'窗口音高数据（JSON，pitches 按权重从强到弱排序）：\n{payload}\n\n'
            f'{_FUNCTION_HINT}\n'
            '输出 JSON 数组，每项：'
            '{"label": "窗口标签", "chord": "和弦记号（如 Cmaj/E、G7、Am）",'
            '"quality": "性质", "inversion": "转位说明或 null",'
            '"function": "功能标记（如 I、V7、ii）", '
            '"confidence": 0~1}。'
            f'{JSON_ONLY}'
        )
        llm = run_llm_json_soft(task, profile=config.dsh_profile,
                                timeout=config.dsh_timeout, retries=1)
        if isinstance(llm, list):
            results.extend(llm)
        elif isinstance(llm, dict):
            if 'chords' in llm:
                results.extend(llm['chords'])
            elif '_error' in llm:
                # 该批次整体解析失败：记账后继续（已有 were_error 标记）
                errors.append(llm)

    # 本地兜底：LLM 漏标的窗口补 null
    labels_llm = {r.get('label') for r in results}
    for w in windows:
        if w['label'] not in labels_llm:
            results.append({'label': w['label'], 'chord': None,
                            'note': 'LLM 未返回，可重跑本步骤'})

    return {
        'window_type': config.chord_window,
        'window_count': len(windows),
        'progression': results,
        'llm_errors': errors,
        'raw_windows': [
            {'label': w['label'], 'pitch_names': w['pitch_names'], 'bass': w['bass']}
            for w in windows
        ],
    }
