"""风格分析 —— 建立在结构 / 织体 / 调性 / 节奏 / 配器之上的最高层判断。

    style ← structure + texture + tonality + rhythm_pattern
            + instrumentation + chords + rhythm_grid

分工：
  本地确定性计算：风格指纹（和声词汇、音阶材料、律动、编制、密度）；
  LLM：基于指纹做流派 / 年代 / 地域推断并给出证据（软降级）。
"""
from .llm import run_llm_json_soft, JSON_ONLY
from .scoredata import ScoreData, midi_to_pc

# 五声音阶音级（宫商角徵羽的相对音级）
_PENTATONIC_PCS = (0, 2, 4, 7, 9)
# 扩展和弦标记关键词（出现即视为高叠和弦 → 爵士 / 后期浪漫 / 印象倾向）
_EXT_CHORD_KEYS = ('7', '9', '11', '13', 'maj7', 'm7', 'dim', 'aug', 'sus')
# 民族乐器分组关键词（instrumentation 的 family 字段）
_FOLK_FAMILIES = ('弹拨（民乐）', '拉弦（民乐）', '吹管（民乐）')


def _pc_profile(data: ScoreData) -> list:
    """全曲音级时值分布（12 维，归一化）。"""
    pc = [0.0] * 12
    for p in data.parts:
        for e in p.pitched_events:
            for pitch in e.midi_pitches:
                pc[midi_to_pc(pitch)] += e.duration
    total = sum(pc) or 1.0
    return [round(x / total, 4) for x in pc]


def _harmonic_vocabulary(chords_out: dict) -> dict:
    """和声词汇统计：高叠和弦占比、功能和声占比、和弦总数。"""
    prog = (chords_out or {}).get('progression', []) or []
    named = [r for r in prog if r.get('chord')]
    if not named:
        return {'chord_count': 0, 'extended_ratio': 0.0, 'triad_ratio': 0.0}
    ext = 0
    for r in named:
        blob = f"{r.get('chord') or ''} {r.get('quality') or ''}".lower()
        if any(k in blob for k in _EXT_CHORD_KEYS):
            ext += 1
    return {
        'chord_count': len(named),
        'extended_ratio': round(ext / len(named), 3),
        'triad_ratio': round(1 - ext / len(named), 3),
    }


def _pentatonic_ratio(pc_profile: list) -> float:
    """五声性：五声音级时值占全部时值的比例。"""
    return round(sum(pc_profile[i] for i in _PENTATONIC_PCS), 3)


def _mode_of(tonality_out: dict) -> str:
    main = (tonality_out or {}).get('main_key') or ''
    if '小调' in main:
        return '小调'
    if '大调' in main:
        return '大调'
    return '未确定'


def _tempo_of(rhythm_grid_out: dict) -> str:
    tempos = (rhythm_grid_out or {}).get('tempos', []) or []
    bpms = [t['bpm'] for t in tempos if t.get('bpm')]
    if bpms:
        return f'{round(sum(bpms) / len(bpms))} BPM'
    texts = [t.get('text') for t in tempos if t.get('text')]
    return '、'.join(texts[:3]) if texts else '未标注'


def _local_hints(fp: dict) -> list:
    """把指纹翻译成给模型的文字线索（模型据此判断，不再自己统计）。"""
    hints = []
    if fp['extended_ratio'] >= 0.5:
        hints.append(f"高叠和弦占比 {fp['extended_ratio'] * 100:.0f}%（七和弦及以上为主）")
    elif fp['extended_ratio'] <= 0.2:
        hints.append(f"以三和弦为主（高叠和弦仅 {fp['extended_ratio'] * 100:.0f}%）")
    if fp['pentatonic_ratio'] >= 0.85:
        hints.append(f"五声性 {fp['pentatonic_ratio'] * 100:.0f}%（音高材料高度五声化）")
    if fp['syncopation_ratio'] >= 0.3:
        hints.append(f"切分比例 {fp['syncopation_ratio'] * 100:.0f}%（强弱错位明显）")
    if fp['folk']:
        hints.append(f"编制含民族乐器：{'、'.join(fp['families'])}")
    if fp['has_voice']:
        hints.append('编制含人声')
    if fp['has_percussion']:
        hints.append('编制含打击乐')
    if fp['is_polyphonic']:
        hints.append('织体判定为复调')
    if not hints:
        hints.append('各项特征均不突出，风格倾向需结合整体判断')
    return hints


def _llm_style(payload: dict, config) -> dict:
    task = (
        '你是专业音乐风格分析师，熟悉西方艺术音乐、爵士、流行与中国民族音乐的'
        '风格语汇。以下是一部作品的本地风格指纹（已由程序算出，'
        '你不需要重新统计，只需据此推断）。\n\n'
        f'{payload}\n\n'
        '请给出风格判断，输出 JSON：'
        '{"genre": "流派 / 风格（如 古典主义、浪漫主义、印象主义、爵士、流行、'
        '中国民族器乐、巴洛克 等；不确定就写最可能的一个并标注存疑）", '
        '"period": "年代或时期（如 19 世纪中叶 / 20 世纪初 / 当代）", '
        '"region": "地域 / 文化语境（如 德奥、法国、中国、美国）", '
        '"evidence": ["证据1", "证据2", "证据3"], '
        '"harmonic_language": "和声语言 1 句说明", '
        '"rhythm_feel": "律动与节奏特征 1 句说明", '
        '"instrumentation_comment": "编制与音色 1 句说明", '
        '"description": "3-4 句整体风格小结", '
        '"confidence": 0~1}'
        f'{JSON_ONLY}'
    )
    return run_llm_json_soft(task, profile=config.dsh_profile,
                             timeout=config.dsh_timeout, retries=1)


def style(data: ScoreData, config, structure_out: dict, texture_out: dict,
          tonality_out: dict, rhythm_pattern_out: dict, instrumentation_out: dict,
          chords_out: dict, rhythm_grid_out: dict) -> dict:
    """风格分析主入口。"""
    pc = _pc_profile(data)
    harm = _harmonic_vocabulary(chords_out)
    stats = (rhythm_pattern_out or {}).get('stats', {}) or {}
    fams = (instrumentation_out or {}).get('families', []) or []

    fp = {
        'main_key': (tonality_out or {}).get('main_key'),
        'mode': _mode_of(tonality_out),
        'tempo': _tempo_of(rhythm_grid_out),
        'time_signatures': (rhythm_grid_out or {}).get('time_signatures', [])[:3],
        'measure_count': data.measure_count,
        'form': (structure_out or {}).get('form', {}).get('name'),
        'form_local': (structure_out or {}).get('form', {}).get('local_guess'),
        'main_texture': (texture_out or {}).get('main_texture'),
        'texture_detail': (texture_out or {}).get('main_detail'),
        'is_polyphonic': (texture_out or {}).get('is_polyphonic', False),
        'families': fams,
        'part_count': (instrumentation_out or {}).get('part_count', 0),
        'has_voice': (instrumentation_out or {}).get('has_voice', False),
        'has_percussion': (instrumentation_out or {}).get('has_percussion', False),
        'folk': any(f in _FOLK_FAMILIES for f in fams),
        'chord_count': harm['chord_count'],
        'extended_ratio': harm['extended_ratio'],
        'triad_ratio': harm['triad_ratio'],
        'syncopation_ratio': stats.get('syncopation_ratio', 0.0),
        'top_patterns': [p['pattern'] for p in
                         (stats.get('top_measure_patterns') or [])[:3]],
        'pentatonic_ratio': _pentatonic_ratio(pc),
        'pc_profile': pc,
    }

    payload = {
        'main_key': fp['main_key'], 'mode': fp['mode'], 'tempo': fp['tempo'],
        'time_signatures': fp['time_signatures'], 'measure_count': fp['measure_count'],
        'form': fp['form'], 'main_texture': f"{fp['main_texture']} · {fp['texture_detail']}",
        'instrumentation': {'families': fp['families'], 'part_count': fp['part_count'],
                            'has_voice': fp['has_voice'],
                            'has_percussion': fp['has_percussion']},
        'harmonic': {'chord_count': fp['chord_count'],
                     'extended_ratio': fp['extended_ratio'],
                     'triad_ratio': fp['triad_ratio']},
        'rhythm': {'syncopation_ratio': fp['syncopation_ratio'],
                   'top_patterns': fp['top_patterns']},
        'scale': {'pentatonic_ratio': fp['pentatonic_ratio']},
        'local_hints': _local_hints(fp),
    }
    llm = _llm_style(payload, config)

    return {
        'fingerprint': fp,
        'local_hints': _local_hints(fp),
        'genre': llm.get('genre') if isinstance(llm, dict) else None,
        'period': llm.get('period') if isinstance(llm, dict) else None,
        'region': llm.get('region') if isinstance(llm, dict) else None,
        'evidence': llm.get('evidence', []) if isinstance(llm, dict) else [],
        'description': llm.get('description', '') if isinstance(llm, dict) else '',
        'llm_analysis': llm,
        'note': ('调性 / 和声词汇 / 音阶材料 / 律动 / 编制等指纹为本地确定性统计；'
                 '流派、年代与地域由大模型基于指纹推断（解析失败保留指纹供人工判断）。'),
    }
