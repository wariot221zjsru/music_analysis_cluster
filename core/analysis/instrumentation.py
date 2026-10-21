"""配器提取（静态乐器清单）——分析因果链第 3 步。

配器是谱面元数据（MusicXML Part / Instrument 标签），直接读取、
不经大模型（计划书 4.5.3：让大模型读配器既浪费 token 又增加幻觉风险）。
输出作为声部识别（voice_id）的强先验。
"""

_INSTRUMENT_FAMILY = {
    'violin': '弦乐', 'viola': '弦乐', 'cello': '弦乐', 'double bass': '弦乐',
    'flute': '木管', 'piccolo': '木管', 'clarinet': '木管', 'oboe': '木管',
    'bassoon': '木管', 'saxophone': '木管',
    'trumpet': '铜管', 'trombone': '铜管', 'horn': '铜管', 'tuba': '铜管',
    'piano': '键盘', 'harpsichord': '键盘', 'organ': '键盘',
    'guitar': '弹拨', 'erhu': '拉弦（民乐）', '二胡': '拉弦（民乐）',
    'guzheng': '弹拨（民乐）', '古筝': '弹拨（民乐）',
    'pipa': '弹拨（民乐）', '琵琶': '弹拨（民乐）',
    'dizi': '吹管（民乐）', '笛': '吹管（民乐）',
    'suona': '吹管（民乐）', 'sheng': '吹管（民乐）',
    'percussion': '打击乐', 'drum': '打击乐',
}


def _family(instrument: str, is_percussion: bool = False,
            is_pitched_percussion: bool = False) -> str:
    # 有音高打击乐（颤音琴 / 钟琴 / 马林巴 / 木琴 / 定音鼓 …）音色上属打击乐
    # 家族，但有确定音高、能奏旋律，单独标出来 —— 否则会被误以为它和鼓组一样
    # 不参与旋律分析（事实上这类声部常常就是主题声部）。
    if is_pitched_percussion:
        return '打击乐（有音高）'
    if is_percussion:
        return '打击乐'
    low = (instrument or '').lower()
    for k, v in _INSTRUMENT_FAMILY.items():
        if k in low:
            return v
    return '其他'


def instrumentation(data, config) -> dict:
    """静态配器提取（零 LLM 调用）。"""
    parts = []
    for p in data.parts:
        parts.append({
            'part_index': p.index,
            'name': p.name,
            'instrument': p.instrument or ('打击乐（鼓组）' if p.is_percussion
                                           else '（未标注）'),
            'family': _family(p.instrument, p.is_percussion,
                              p.is_pitched_percussion),
            'is_voice': p.is_voice,
            'is_percussion': p.is_percussion,
            'is_pitched_percussion': p.is_pitched_percussion,
            'note_count': p.note_count,
            'pitched_note_count': p.pitched_note_count,
        })
    mallet = [p['name'] for p in parts if p['is_pitched_percussion']]
    note = ('静态配器直接读取谱面元数据，未经大模型（零幻觉、零延迟）。'
            '无音高打击乐（鼓组 / 节奏打击乐）只参与节奏统计，'
            '不参与和弦 / 调性 / 旋律等音高类分析')
    if mallet:
        note += ('；有音高打击乐（' + '、'.join(mallet) +
                 '）有确定音高，按旋律声部参与全部音高类分析')
    return {
        'parts': parts,
        'part_count': len(parts),
        'has_voice': any(p['is_voice'] for p in parts),
        'has_percussion': any(p['is_percussion'] or p['is_pitched_percussion']
                              for p in parts),
        'families': sorted({p['family'] for p in parts}),
        'note': note + '。',
    }
