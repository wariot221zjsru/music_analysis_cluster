"""自定义问答：把"已算出的谱面事实"汇总成上下文，再让 dsh 回答用户提问。

设计要点：
  1. 问答不重新分析——先跑过的步骤全部复用，只把结论压缩进上下文。
    没跑过的步骤直接说明"该方面尚无数据"，不谎报。
  2. 答案是**自然语言**而非 JSON：固定结构会限制表达，
    且输出被 max_tokens 截断时 JSON 直接整段报废，纯文本最多少几段。
  3. 沿用 dsh-musictheory 的"先算后说、凡算必验"：提示词里要求涉及
    音程 / 和弦 / 音阶时必须用工具核验，而不是凭印象作答。
"""
from .llm import run_llm
from .scoredata import ScoreData, midi_to_name

# 给模型的主旋律片段长度（音）
_MELODY_PREVIEW = 60


def _brief(key: str, out: dict) -> dict:
    """按步骤压缩成简短摘要（问答只需要结论，不需要中间数据）。"""
    if not out:
        return {}
    if key == 'rhythm_grid':
        return {'小节数': out.get('measure_count'),
                '拍号': [f"{t['numerator']}/{t['denominator']}"
                       for t in (out.get('time_signatures') or [])][:3],
                '速度': [t.get('bpm') or t.get('text')
                       for t in (out.get('tempos') or [])][:4]}
    if key == 'rhythm_pattern':
        s = (out.get('stats') or {})
        return {'切分比例': s.get('syncopation_ratio'),
                '主要节奏型': [p['pattern'] for p in (s.get('top_measure_patterns') or [])][:3]}
    if key == 'instrumentation':
        return {'声部数': out.get('part_count'),
                '乐器分组': out.get('families'),
                '含人声': out.get('has_voice'),
                '含打击乐': out.get('has_percussion')}
    if key == 'voice_id':
        return {'声部角色': [(r['part_name'], r['role_candidate'])
                          for r in (out.get('roles') or [])][:8]}
    if key == 'polyphony':
        return {'复调判定': out.get('branch'), '是否复调': out.get('is_polyphonic'),
                '密度': out.get('ratio')}
    if key == 'exchange':
        return {'交接判定': out.get('branch'),
                '切换次数': out.get('switch_count')}
    if key == 'chords':
        return {'和弦进行': [{'label': r.get('label'), 'chord': r.get('chord'),
                            'function': r.get('function')}
                          for r in (out.get('progression') or []) if r.get('chord')][:48]}
    if key == 'tonality':
        return {'主调': out.get('main_key'),
                '调性区域': [(r.get('start_measure'), r.get('end_measure'), r.get('key'))
                          for r in (out.get('regions') or [])][:8],
                '转调': [(m.get('measure'), m.get('from_key'), m.get('to_key'))
                       for m in (out.get('modulations') or [])][:6]}
    if key == 'melody_shape':
        return {'形态分析': [a.get('segment') for a in (out.get('analyses') or [])][:8],
                '旋律轮廓摘要': [
                    (a.get('llm_analysis') or {}).get('contour_summary')
                    for a in (out.get('analyses') or [])][:4]}
    if key == 'melody_extraction':
        return {'选轨模式': out.get('mode'),
                '候选声部': [(c['part_name'], c['confidence'])
                          for c in (out.get('candidates') or [])][:5]}
    if key == 'melody_line':
        return {
            '逐段追踪模式': out.get('mode'),
            '分段': [{'小节': f"{s['start_measure']}-{s['end_measure']}",
                    '声部': s.get('part_name'), '人声': s.get('is_voice'),
                    '置信度': s.get('confidence')}
                   for s in (out.get('segments') or [])][:12],
            '覆盖率': (out.get('coverage') or {}).get('ratio'),
        }
    if key == 'structure':
        f = out.get('form') or {}
        return {'曲式': f.get('name'), '骨架': f.get('scheme'),
                '乐段': [{'label': s['label'],
                        'measures': f"{s['start_measure']}-{s['end_measure']}",
                        'cadence': s.get('cadence'), 'key': s.get('key')}
                       for s in (out.get('sections') or [])][:16],
                '终止式': [(c['measure'], c['type'])
                         for c in (out.get('cadences') or [])][:12]}
    if key == 'texture':
        return {'主织体': out.get('main_texture'), '细节': out.get('main_detail'),
                '织体比例': out.get('ratios'),
                '时间线': [{'measures': f"{t['start_measure']}-{t['end_measure']}",
                         'texture': t['texture']}
                        for t in (out.get('timeline') or [])][:12]}
    if key == 'style':
        return {'流派判断': out.get('genre'), '年代': out.get('period'),
                '地域': out.get('region'), '证据': (out.get('evidence') or [])[:6],
                '风格小结': out.get('description')}
    return {}


#: 问答依赖的步骤 → 展示名（顺序即上下文组织顺序）
CONTEXT_STEPS = [
    ('rhythm_grid', '节奏网格'), ('rhythm_pattern', '节奏型'),
    ('instrumentation', '配器'), ('voice_id', '声部'),
    ('polyphony', '复调'), ('exchange', '主旋律交接'),
    ('chords', '和弦'), ('tonality', '调性'),
    ('melody_shape', '旋律形态'), ('melody_extraction', '主旋律声部识别（选轨）'),
    ('melody_line', '主旋律线追踪（逐段）'),
    ('structure', '结构'), ('texture', '织体'), ('style', '风格'),
]


def build_context(piece, data: ScoreData, results: dict) -> dict:
    """汇总作品事实 → 问答上下文。"""
    ctx = {
        '作品': {
            '标题': piece.title,
            '源类型': piece.get_source_type_display(),
            '含人声': piece.contains_voice,
        },
        '已完成分析': {},
        '尚无数据的方面': [],
    }
    for key, label in CONTEXT_STEPS:
        out = results.get(key)
        if out:
            ctx['已完成分析'][label] = _brief(key, out)
        else:
            ctx['尚无数据的方面'].append(label)

    # 谱面直读的补充事实（不依赖任何分析步骤）
    if data is not None:
        ctx['谱面'] = {
            '小节数': data.measure_count,
            '声部数': len(data.parts),
            '声部名': data.part_names[:12],
        }
        # 主旋律预览：显著性最高的声部的前 N 个音（无显著性时用第一个有音高声部）
        lead = None
        voice_out = results.get('voice_id') or {}
        roles = voice_out.get('roles') or []
        if roles:
            lead = int(roles[0].get('part_index', 0))
        if lead is None:
            lead = next((p.index for p in data.parts if p.pitched_events), None)
        if lead is not None and 0 <= lead < len(data.parts):
            seq = [max(e.midi_pitches) for e in data.parts[lead].pitched_events]
            ctx['谱面']['主旋律声部'] = data.parts[lead].name
            ctx['谱面']['主旋律音高序列'] = [midi_to_name(m) for m in seq[:_MELODY_PREVIEW]]
    return ctx


def build_prompt(question: str, ctx: dict) -> str:
    """问答提示词。"""
    missing = ctx.get('尚无数据的方面') or []
    missing_line = ('（尚未分析：' + '、'.join(missing) + '，这些方面不要臆断，'
                    '直接告诉用户需要先跑对应分析。）' if missing else '')
    return (
        '你是这部作品的专属音乐分析助手。下面是这部作品已经算出的客观事实'
        '（来自程序分析，不是用户的描述），请你基于这些事实回答用户的提问。\n\n'
        f'【作品事实】\n{ctx}\n\n'
        f'【用户提问】\n{question}\n'
        f'{missing_line}\n\n'
        '回答要求：\n'
        '1. 用中文，先给结论，再给 2-4 条依据（可引用上面事实里的具体小节 / 和弦 / 调性）；\n'
        '2. 涉及音程、和弦、音阶时必须用音乐理论工具核验后再说，不要凭印象；\n'
        '3. 事实不足时明确说"现有分析不足以判断"，并指出还需要哪个分析步骤；\n'
        '4. 不要输出 JSON，直接用自然语言分段，控制在 900 字以内。'
    )


def answer(question: str, ctx: dict, config) -> str:
    """调用 dsh 回答（纯文本；失败抛 LLMError，由调用方落库为 error）。"""
    return run_llm(build_prompt(question, ctx), profile=config.dsh_profile,
                   timeout=config.dsh_timeout)
