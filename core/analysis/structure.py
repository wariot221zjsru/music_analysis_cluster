"""音乐结构分析：乐句 → 乐段 → 曲式。

因果链位置（计划书 4.3.1 的延伸）：
    structure ← rhythm_grid（小节/速度）+ chords（终止式）+ tonality（调性区域）

分工沿用本项目一贯原则：
  本地确定性计算：小节密度剖面、终止式落点、乐句切分、乐段聚类、曲式候选；
  LLM：只做语义化命名与叙述（软降级，失败不影响结构骨架）。
"""
import math
import re

from .llm import run_llm_json_soft, JSON_ONLY
from .scoredata import ScoreData, interval_seq, lcs_ratio

# 乐句长度约束（小节）
_MIN_PHRASE_BARS = 2
_MAX_PHRASE_BARS = 16
# 乐段聚类相似度阈值（动机 LCS 与音级分布余弦各占一半）
_SECTION_SIM_THRESHOLD = 0.62

_MEASURE_RE = re.compile(r'第\s*(\d+)\s*小节')
_SECTION_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


# ---------------- 本地确定性计算 ----------------

def _measure_profile(data: ScoreData) -> dict:
    """每小节的统计剖面：音符数、总时值、最长音、音高集合。"""
    prof = {}
    for p in data.parts:
        for e in p.pitched_events:
            d = prof.setdefault(e.measure, {'count': 0, 'dur': 0.0,
                                            'pitches': [], 'max_dur': 0.0})
            d['count'] += len(e.midi_pitches)
            d['dur'] += e.duration
            d['pitches'].extend(e.midi_pitches)
            d['max_dur'] = max(d['max_dur'], e.duration)
    return prof


def _measure_chords(chords_out: dict) -> dict:
    """{小节号: {'chord':..., 'function':...}}（和弦窗按拍时取该小节首个）。"""
    out = {}
    for r in (chords_out or {}).get('progression', []) or []:
        m = _MEASURE_RE.search(r.get('label') or '')
        if not m:
            continue
        mm = int(m.group(1))
        slot = out.setdefault(mm, {'chord': None, 'function': None})
        if slot['chord'] is None and r.get('chord'):
            slot['chord'] = r['chord']
        if slot['function'] is None and r.get('function'):
            slot['function'] = r['function']
    return out


def _norm_function(text: str) -> str:
    """功能标记规整：'V7' → 'V'，'ii6' → 'ii'，方便终止式比对。"""
    s = (text or '').strip().upper()
    s = s.replace('7', '').replace('6', '').replace('°', '')
    return s


def _cadence_points(mchords: dict) -> dict:
    """终止式落点：{小节号: 终止式类型}。

    只看最典型的四种：V→I 完全正格、IV→I 变格、V→vi 阻碍、停在 V 半终止。
    """
    points = {}
    for m in sorted(mchords):
        cur = _norm_function(mchords[m].get('function'))
        prev = _norm_function(mchords.get(m - 1, {}).get('function'))
        if not cur:
            continue
        if prev.startswith('V') and cur.startswith('I'):
            points[m] = '完全正格终止（V-I）'
        elif prev.startswith('IV') and cur.startswith('I'):
            points[m] = '变格终止（IV-I）'
        elif prev.startswith('V') and cur.startswith('VI'):
            points[m] = '阻碍终止（V-vi）'
        elif cur.startswith('V'):
            # 停在 V 才是半终止；若下一小节就解决了 I，那是正格终止的前半，
            # 落点已记在下一小节，此处不重复记（否则乐句会被切碎）。
            nxt = _norm_function(mchords.get(m + 1, {}).get('function'))
            if not nxt.startswith('I'):
                points[m] = '半终止（停在 V）'
    return points


def _phrase_bounds(data: ScoreData, cadences: dict, measure_count: int) -> list:
    """乐句切分：终止式落点 + 稀疏小节 + 长音收束，再按长度整饬。"""
    if measure_count <= 0:
        return []
    prof = _measure_profile(data)
    if not prof:
        return [(1, measure_count)]

    mean_count = sum(d['count'] for d in prof.values()) / len(prof)
    candidates = set(cadences)
    for m in range(1, measure_count + 1):
        d = prof.get(m)
        if d is None:                       # 整小节休止 → 呼吸点
            candidates.add(m)
            continue
        if d['max_dur'] >= 2.0:             # 二分音符及以上的长音收束
            candidates.add(m)
        if d['count'] <= mean_count * 0.5:  # 明显稀疏 → 句末留白
            candidates.add(m)

    bounds, start = [], 1
    for b in sorted(x for x in candidates if 1 <= x <= measure_count):
        if b - start + 1 < _MIN_PHRASE_BARS:
            continue
        bounds.append((start, b))
        start = b + 1
    if measure_count >= start:
        bounds.append((start, measure_count))

    # 整饬：过长的句按 8 小节硬切；切完仍为空则整曲等分
    out = []
    for s, e in bounds:
        cur = s
        while e - cur + 1 > _MAX_PHRASE_BARS:
            out.append((cur, cur + 7))
            cur += 8
        out.append((cur, e))
    if not out:
        out = [(1, measure_count)]
    return [(s, e) for s, e in out if e >= s]


def _phrase_features(data: ScoreData, s: int, e: int) -> dict:
    """乐句特征：音级时值分布（和声指纹）+ 旋律音程序列（动机指纹）。"""
    pc_dur = [0.0] * 12
    mel = []
    for p in data.parts:
        for ev in p.pitched_events:
            if s <= ev.measure <= e:
                for pitch in ev.midi_pitches:
                    pc_dur[pitch % 12] += ev.duration
                mel.append((ev.offset, max(ev.midi_pitches)))
    total = sum(pc_dur) or 1.0
    pc_norm = [x / total for x in pc_dur]
    mel.sort(key=lambda t: t[0])
    seq = [m for _, m in mel]
    return {
        'pc': pc_norm,
        'ivs': interval_seq(seq)[:80],
        'note_count': len(seq),
        'pitch_range': (max(seq) - min(seq)) if seq else 0,
    }


def _similarity(a: dict, b: dict) -> float:
    """乐句相似度：动机 LCS 0.5 + 音级分布余弦 0.5。"""
    if not a['ivs'] or not b['ivs']:
        return 0.0
    mot = lcs_ratio(a['ivs'], b['ivs'])
    na = math.sqrt(sum(x * x for x in a['pc'])) or 1.0
    nb = math.sqrt(sum(x * x for x in b['pc'])) or 1.0
    cos = sum(x * y for x, y in zip(a['pc'], b['pc'])) / (na * nb)
    return 0.5 * mot + 0.5 * cos


def _key_of(measure: int, tonality_out: dict) -> str:
    """取该小节所属调性区域的调名（无则回落主调）。"""
    for r in (tonality_out or {}).get('regions', []) or []:
        try:
            if r.get('start_measure', 0) <= measure <= r.get('end_measure', 0):
                return r.get('key') or ''
        except TypeError:
            continue
    return (tonality_out or {}).get('main_key') or ''


def _infer_form(base_seq: list, has_variant: bool):
    """由乐段标签序列推断曲式（返回 (名称, 骨架, 置信度)）。"""
    comp = []
    for x in base_seq:                      # 压缩连续重复：A A → A
        b = x.rstrip("'")
        if not comp or comp[-1] != b:
            comp.append(b)
    if not comp:
        return '未识别', '', 0.0
    scheme = '-'.join(f'{c}({i + 1})' for i, c in enumerate(comp))
    n = len(comp)
    if n == 1:
        if has_variant:
            return '变奏曲式（主题与变奏）', comp[0] + " + 变奏", 0.55
        return '一部曲式（单乐段 / 乐段反复）', comp[0], 0.5
    first = comp[0]
    if n >= 5 and comp.count(first) >= 3:
        return f'回旋曲式（{"".join(comp)}）', ''.join(comp), 0.70
    if n == 3 and comp[0] == comp[2]:
        return '单三部曲式（ABA，带再现）', ''.join(comp), 0.72
    if n == 2:
        return '单二部曲式（AB）', ''.join(comp), 0.60
    if n == 3:
        return f'三段并列结构（{"".join(comp)}，无再现）', ''.join(comp), 0.45
    return f'自由 / 复合结构（{"".join(comp)}）', ''.join(comp), 0.40


# ---------------- LLM 语义化 ----------------

def _llm_structure(payload: dict, config) -> dict:
    task = (
        '你是专业曲式分析师。以下是一部作品的本地结构分析结果'
        '（小节密度、终止式、调性区域已由程序算出，不需要你重新统计）。'
        '请据此命名曲式并做简要叙述。\n\n'
        f'{payload}\n\n'
        '输出 JSON：'
        '{"form_name": "曲式名称", '
        '"section_plan": [{"label": "A", "measures": "1-16", '
        '"function": "呈示 / 对比 / 展开 / 再现 / 连接 之一", '
        '"comment": "该段 1-2 句说明"}], '
        '"phrase_comment": "乐句组织与呼吸的 1-2 句说明", '
        '"highlights": "1-2 句结构亮点点评", '
        '"confidence": 0~1}'
        f'{JSON_ONLY}'
    )
    return run_llm_json_soft(task, profile=config.dsh_profile,
                             timeout=config.dsh_timeout, retries=1)


# ---------------- 主入口 ----------------

def structure(data: ScoreData, config, rhythm_out: dict, chords_out: dict,
              tonality_out: dict) -> dict:
    """音乐结构分析主入口。"""
    measure_count = data.measure_count or (rhythm_out or {}).get('measure_count', 0)
    mchords = _measure_chords(chords_out)
    cadences = _cadence_points(mchords)
    bounds = _phrase_bounds(data, cadences, measure_count)

    feats = [_phrase_features(data, s, e) for s, e in bounds]

    # ---- 乐段聚类：与已有段落代表句相似 → 视为同段的变体 ----
    groups = []
    for i, f in enumerate(feats):
        best, best_sim = None, 0.0
        for g in groups:
            sim = _similarity(f, g['feat'])
            if sim > best_sim:
                best_sim, best = sim, g
        if best is not None and best_sim >= _SECTION_SIM_THRESHOLD:
            best['members'].append(i)
            best['sim'] = round(best_sim, 3)
        else:
            groups.append({'order': len(groups), 'members': [i], 'feat': f,
                           'sim': 1.0})

    # ---- 按出现顺序展开成段落清单（同一段再次出现标 A' / A''）----
    seen = {}
    sections, seq_labels, variants = [], [], False
    for i, (s, e) in enumerate(bounds):
        g = next(g for g in groups if i in g['members'])
        base = _SECTION_LABELS[g['order']] if g['order'] < len(_SECTION_LABELS) \
            else f'段{g["order"] + 1}'
        times = seen.get(base, 0)
        seen[base] = times + 1
        label = base + ("'" * times)      # 第二次出现 → A'
        if times:
            variants = True
        seq_labels.append(label)
        cad = cadences.get(e, '')
        sections.append({
            'label': label,
            'start_measure': s,
            'end_measure': e,
            'bars': e - s + 1,
            'cadence': cad,
            'key': _key_of(s, tonality_out),
        })

    form_name, scheme, confidence = _infer_form(
        [x.rstrip("'") for x in seq_labels], variants)

    phrases = []
    for i, (s, e) in enumerate(bounds):
        phrases.append({
            'index': i + 1,
            'start_measure': s,
            'end_measure': e,
            'bars': e - s + 1,
            'note_count': feats[i]['note_count'],
            'pitch_range': feats[i]['pitch_range'],
            'cadence': cadences.get(e, ''),
        })

    # 给 LLM 的载荷刻意压缩：只传段落骨架，不传音符
    payload = {
        'measure_count': measure_count,
        'main_key': (tonality_out or {}).get('main_key'),
        'tempos': (rhythm_out or {}).get('tempos', [])[:6],
        'time_signatures': (rhythm_out or {}).get('time_signatures', [])[:4],
        'sections': [{'label': s['label'],
                      'measures': f"{s['start_measure']}-{s['end_measure']}",
                      'cadence': s['cadence'], 'key': s['key']}
                     for s in sections][:24],
        'local_form_guess': form_name,
    }
    llm = _llm_structure(payload, config)

    return {
        'measure_count': measure_count,
        'phrase_count': len(phrases),
        'phrases': phrases,
        'sections': sections,
        'cadences': [{'measure': m, 'type': t} for m, t in sorted(cadences.items())],
        'form': {
            'name': (llm.get('form_name') if isinstance(llm, dict) else None)
            or form_name,
            'scheme': scheme,
            'local_guess': form_name,
            'confidence': confidence,
        },
        'llm_analysis': llm,
        'note': ('乐句切分、终止式落点与乐段聚类为本地确定性计算（未经大模型）；'
                 '曲式命名与段落功能叙述由大模型完成，解析失败时保留本地骨架。'),
    }
