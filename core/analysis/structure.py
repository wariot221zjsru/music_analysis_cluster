"""音乐结构分析：乐句 → 主题 → 曲式（三层，单向依赖 + 一轮反馈）。

因果链位置（计划书 4.3.1 的延伸）：
    structure ← rhythm_grid（小节/速度）+ chords（终止式）+ tonality（调性区域）
                + melody_line（可选：主旋律线归属，用来只取主旋律做主题指纹）

## 三层各自回答什么

| 层 | 问题 | 判据 |
|---|---|---|
| 乐句 ``phrases`` | 边界在哪（where） | 终止式落点 / 长音收束 / 稀疏小节 |
| 主题 ``themes``  | 这段材料是谁（what） | 动机指纹 + 节奏型 + 音级分布（见 ``theme.py``） |
| 曲式 ``form``    | 这段在干什么（why） | 主题出现次序 + 调性布局 + 终止式等级 |

**依赖是单向的**：曲式消费主题，主题不消费曲式。反过来做（拿奏鸣曲式模板去套、
硬找出两个主题）会凭空造出结构。所以这里先把三层的中间产物都留下来，
曲式层只在上层做全局打分。

## 为什么曲式和前两层"做在一起"，但不是"合成一个阈值"

老版本用**一个**相似度阈值同时判身份和功能（``_SECTION_SIM_THRESHOLD = 0.62``），
那不是联合，是省略中间层：主题变形稍大 → 掉到阈值下 → 判成新段 → 曲式从 ABA
变 ABC，一次误判连坐两级，而且中间没有任何可检查的产物。

这里的做法是 **分层产出候选 + 全局重打分 + 一轮反馈**：

1. 乐句层留下全部边界候选（落点 + 理由），不只留一个切法；
2. 主题层留下**全部两两证据**（相似度 / 变形类型 / 覆盖比），不急着定死；
3. 曲式层在「主题序列 + 调性 + 终止式 + 长度均衡」四轴上枚举模板打分，
   并输出**每个候选模板的分数**（可解释，不是黑箱）；
4. **一轮反馈**：若胜出的曲式假设含"再现"，就把"相似度够不上强阈值、但调性
   回到了该主题首次出现的调"的段提升为同一主题，重算一次。

反馈只做一轮、且**只放宽不收紧**（否则会振荡不收敛）；每一次改判都记理由。

分工沿用本项目一贯原则：
  本地确定性计算：乐句切分、主题聚类、曲式候选打分；
  LLM：只做语义化命名与叙述（软降级，失败不影响结构骨架）。
"""
import math
import re

from . import theme as theme_mod
from .llm import run_llm_json_soft, JSON_ONLY
from .scoredata import ScoreData

# 乐句长度约束（小节）
_MIN_PHRASE_BARS = 2
_MAX_PHRASE_BARS = 16

# 曲式层的四轴权重：材料序列 / 调性布局 / 终止式等级 / 长度均衡
FORM_W = (0.45, 0.25, 0.18, 0.12)
# 最优模板与第二名差小于此值 → 判为难判（页面上提示"曲式接近两解"）
FORM_CLOSE = 0.06

_MEASURE_RE = re.compile(r'第\s*(\d+)\s*小节')
_TERMINAL_RANK = {
    '完全正格终止（V-I）': 1.0,
    '变格终止（IV-I）': 0.80,
    '阻碍终止（V-vi）': 0.55,
    '半终止（停在 V）': 0.45,
    '': 0.30,
}

# 曲式模板：(名称, 主题序列, 是否含再现, 全曲应当有几个不同主题)
# 最后一个字段是硬约束：变奏曲式与一部曲式的前提都是"全曲只有一个主题"。
# 不加它，['A','B','A'] 与 ['A','A','A'] 的编辑距离只有 1，变奏会靠"对比型
# 调性满分"反超真正的三部曲式 —— 用一个语义上根本不成立的模板解释结构。
_TEMPLATES = [
    ('一部曲式（单乐段 / 乐段反复）', ['A'], False, 1),
    ('单二部曲式（AB）', ['A', 'B'], False, 2),
    ('单三部曲式（ABA，带再现）', ['A', 'B', 'A'], True, 2),
    ('回旋曲式（ABACA）', ['A', 'B', 'A', 'C', 'A'], True, 3),
    ('变奏曲式（主题与变奏）', ['A', 'A', 'A'], False, 1),
]


# ---------------- ① 乐句层 ----------------

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


def _boundary_candidates(data: ScoreData, cadences: dict, measure_count: int) -> list:
    """**全部**乐句边界候选，附理由 —— 不是切法，是切法的原料。

    留着它是为了可检查：用户可以看出来"这里为什么断句"，也能看出
    程序手里其实还有另一个落点没被选中（切分本身有歧义，8 小节一句
    和 4+4 两小句在谱面上常常都说得通）。
    """
    prof = _measure_profile(data)
    reasons = {}
    for m, t in (cadences or {}).items():
        reasons.setdefault(m, []).append(t)
    if prof:
        mean_count = sum(d['count'] for d in prof.values()) / len(prof)
        for m in range(1, measure_count + 1):
            d = prof.get(m)
            if d is None:
                reasons.setdefault(m, []).append('整小节休止（呼吸点）')
                continue
            if d['max_dur'] >= 2.0:
                reasons.setdefault(m, []).append('二分音符及以上的长音收束')
            if d['count'] <= mean_count * 0.5:
                reasons.setdefault(m, []).append('小节密度明显稀疏')
    return [{'measure': m, 'reasons': r}
            for m, r in sorted(reasons.items()) if 1 <= m <= measure_count]


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


# ---------------- ② 主题层（音高取自哪里） ----------------

def _melody_owner_map(melody_out: dict) -> dict:
    """{小节: 声部序号}：主旋律线追踪给出的逐段归属。

    有它的时候，主题指纹只取**主旋律那个声部**的音；没有它（还没跑过
    主旋律线追踪）才退回"全体声部的最高音" —— 后者正是老实现的口径，
    伴奏的顶音会混进动机指纹里，把主题认串。
    """
    out = {}
    if not isinstance(melody_out, dict):
        return out
    for seg in melody_out.get('segments') or []:
        pi = seg.get('part_index')
        if pi is None:
            continue
        try:
            s, e = int(seg['start_measure']), int(seg['end_measure'])
        except (KeyError, TypeError, ValueError):
            continue
        for m in range(s, e + 1):
            out[m] = pi
    return out


def _phrase_notes(data: ScoreData, s: int, e: int, owner: dict) -> list:
    """乐句内的音符 ``[(offset, midi, duration)]``，按时间排序。

    ``owner`` 有该小节的归属时只取那一条声部；否则取全体声部的最高音
    （老口径，作为兜底）。

    还有一种情况要兜：主旋律声部在这一句里几乎不响（长休止 / 只有零星几个
    音）。此时按声部取会得到一个几乎空的指纹，那一句会凭空变成"新主题"。
    所以取到的音少于本句总音的 40% 时，退回全体最高音 —— 宁可指纹被伴奏
    污染一点，也不能让整句判定失效。
    """
    rows, every = [], []
    for p in data.parts:
        for ev in p.pitched_events:
            if not (s <= ev.measure <= e):
                continue
            item = (ev.offset, max(ev.midi_pitches), ev.duration)
            every.append(item)
            want = owner.get(ev.measure)
            if want is None or p.index == want:
                rows.append(item)
    if len(rows) < max(3, 0.4 * len(every)):
        rows = every
    rows.sort(key=lambda t: t[0])
    return rows


def _key_of(measure: int, tonality_out: dict) -> str:
    """取该小节所属调性区域的调名（无则回落主调）。"""
    for r in (tonality_out or {}).get('regions', []) or []:
        try:
            if r.get('start_measure', 0) <= measure <= r.get('end_measure', 0):
                return r.get('key') or ''
        except TypeError:
            continue
    return (tonality_out or {}).get('main_key') or ''


# ---------------- ③ 曲式层 ----------------

def _sections_of(bounds: list, labels: list, cadences: dict,
                 tonality_out: dict, notenums: list) -> tuple:
    """连续同主题的乐句合并成一个乐段（乐段 = 同一主题的一次连续陈述）。"""
    sections = []
    for i, (s, e) in enumerate(bounds):
        base = labels[i].rstrip("'")
        if sections and sections[-1]['theme'] == base:
            sec = sections[-1]
            sec['end_measure'] = e
            sec['bars'] = e - sec['start_measure'] + 1
            sec['phrases'].append(i + 1)
            sec['note_count'] += notenums[i]
            if cadences.get(e):
                sec['cadence'] = cadences[e]
            continue
        sections.append({
            'theme': base,
            'start_measure': s,
            'end_measure': e,
            'bars': e - s + 1,
            'phrases': [i + 1],
            'note_count': notenums[i],
            'cadence': cadences.get(e, ''),
            'key': _key_of(s, tonality_out),
        })

    # 标签按出现次序：A、A'、A''；B、B'
    seen, seq = {}, []
    for sec in sections:
        base = sec['theme']
        times = seen.get(base, 0)
        seen[base] = times + 1
        sec['label'] = base + ("'" * times)
        seq.append(base)
    return sections, seq


def _edit_distance(a: list, b: list) -> int:
    """两条主题序列的编辑距离（插入 / 删除 / 替换都算 1）。"""
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(dp[j - 1] if x == y
                       else 1 + min(dp[j - 1], dp[j], cur[j - 1]))
        dp = cur
    return dp[-1]


def _seq_fit(actual: list, tpl: list) -> float:
    """材料序列匹配度：编辑距离 + 长度比 + **段数是否对得上**。

    最后一项是必须的：``ABA`` 与 ``AB`` 的编辑距离只有 1，光按长度归一化
    会算成 0.67，于是"假再现"（材料回来了、调性没回来）会让 ABA 输给 AB
    —— 那等于用一个段数都不对的模板去解释三段结构。段数不匹配要当硬伤罚。
    """
    if not actual:
        return 0.0
    d = _edit_distance(actual, tpl)
    denom = max(len(actual), len(tpl), 1)
    base = max(0.0, 1.0 - d / denom)
    len_pen = min(len(actual), len(tpl)) / denom
    shape_pen = 1.0 if len(actual) == len(tpl) else 0.75
    return round(base * (0.55 + 0.45 * len_pen) * shape_pen, 4)


def _key_fit(has_return: bool, sections: list) -> float:
    """调性布局：再现型模板要求末段回主调，对比型模板要求中段换调。

    "假再现"（材料回来了、调性没回来）在音乐学上不算再现 —— 所以说
    "曲式功能必须和前两层一起判"不是偏好，是这条合取条件逼出来的。
    """
    keys = [s.get('key') or '' for s in sections]
    if not keys:
        return 0.5
    first = keys[0]
    if has_return and len(keys) >= 3:
        if not keys[-1]:
            return 0.5
        return 1.0 if keys[-1] == first else 0.25
    if len(keys) >= 2:
        other = [k for k in keys[1:] if k]
        if not other:
            return 0.5
        diff = sum(1 for k in other if k != first) / len(other)
        return 0.35 + 0.65 * diff
    return 1.0 if first else 0.5


def _cadence_fit(sections: list) -> float:
    """终止式等级：段末越"终止"越支持一个完整的结构单位。"""
    if not sections:
        return 0.3
    vals = [_TERMINAL_RANK.get(s.get('cadence') or '', 0.30) for s in sections]
    if len(vals) >= 2:      # 末段收束的分量更重
        return round(0.5 * (sum(vals[:-1]) / len(vals[:-1]))
                     + 0.5 * vals[-1], 4)
    return round(vals[0], 4)


def _balance_fit(has_return: bool, sections: list) -> float:
    """长度均衡：再现段应与首段长度相近；无再现则各部分不宜悬殊。"""
    bars = [max(s.get('bars', 0), 1) for s in sections]
    if len(bars) < 2:
        return 0.6
    if has_return:
        return round(1.0 - min(1.0, abs(bars[-1] - bars[0]) / max(bars[0], 1)), 4)
    mean = sum(bars) / len(bars)
    var = sum((x - mean) ** 2 for x in bars) / len(bars)
    cv = math.sqrt(var) / mean if mean else 0.0
    return round(max(0.0, 1.0 - min(1.0, cv)), 4)


def _infer_form(seq: list, sections: list) -> dict:
    """在四轴上枚举模板打分，返回**全部候选**的分数（可解释，不是黑箱）。"""
    cands = []
    uniq = len(set(seq))
    for name, tpl_seq, has_return, tpl_uniq in _TEMPLATES:
        mat = _seq_fit(seq, tpl_seq)
        if tpl_uniq is not None and uniq != tpl_uniq:
            # 语义上不成立：变奏 / 一部要求全曲一个主题，这里却出现了几个
            mat = round(mat * 0.60, 4)
        key = _key_fit(has_return, sections)
        cad = _cadence_fit(sections)
        bal = _balance_fit(has_return, sections)
        total = round(sum(w * x for w, x in zip(FORM_W, (mat, key, cad, bal))), 4)
        cands.append({
            'name': name,
            'seq': ''.join(tpl_seq),
            'has_return': has_return,
            'score': total,
            'parts': {'material': round(mat, 4), 'key': round(key, 4),
                      'cadence': round(cad, 4), 'balance': round(bal, 4)},
        })
    cands.sort(key=lambda c: -c['score'])
    best = cands[0]
    margin = round(best['score'] - (cands[1]['score'] if len(cands) > 1 else 0.0), 4)
    confidence = round(min(0.95, best['score'] * 0.70 + min(margin, 0.30)), 2)
    return {
        'name': best['name'],
        'scheme': ''.join(seq),
        'template': best['seq'],
        'has_return': best['has_return'],
        'confidence': confidence,
        'margin': margin,
        'close': margin < FORM_CLOSE,
        'candidates': cands,
    }


def _apply_feedback(links: list, n: int, labels: list, keys: list,
                    leader_map: dict) -> tuple:
    """一轮反馈：曲式假设说"这里应该有再现"，就用弱阈值 + 调性回归重判。

    只放宽、不收紧；只在**调性真的回来了**的时候放宽 —— 材料像但调性
    没回来的（假再现）不算再现。返回 ``(新标签, 提升记录)``。
    """
    weak = theme_mod.build_labels(links, n, theme_mod.THEME_WEAK)
    weak_leader = {}
    for g in weak['groups']:
        for m in g['members']:
            weak_leader[m] = g['leader']

    new_labels = list(labels)
    promoted = []
    for i in range(n):
        sl, wl = leader_map.get(i), weak_leader.get(i)
        if sl is None or wl is None or sl == wl:
            continue
        # 如果第 i 句自己是一组的组长、组里还不止它一句，提升它会把它后面
        # 那几句变成孤儿（等于顺手"收紧"了别人）。反馈只许放宽，跳过。
        if sl == i and sum(1 for v in leader_map.values() if v == sl) > 1:
            continue
        # 弱阈值把第 i 句并到了更早的第 wl 句那一组 → 看调性是否回归
        if not (keys[i] and keys[i] == keys[wl]):
            continue
        base = new_labels[wl].rstrip("'")
        new_labels[i] = base
        link = next((l for l in links if l['i'] == wl and l['j'] == i), {})
        promoted.append({
            'phrase': i + 1,
            'to_phrase': wl + 1,
            'theme': base,
            'sim': link.get('score'),
            'kind': link.get('kind'),
            'reason': (f'材料相似度 {link.get("score")} 未达强阈值，'
                       f'但调性回到了主题 {base} 首次出现的「{keys[i]}」'
                       '—— 按"材料回归 + 调性回归"合取判定，这是再现而非新主题'),
        })
    return new_labels, promoted


# ---------------- LLM 语义化 ----------------


def _theme_rows(labels: list, links: list, promoted: list, bounds: list) -> list:
    """按（反馈后的）标签生成主题清单 —— 标签是唯一真相。

    反馈回路改的是 ``labels``，清单必须跟着它重算；否则会出现"乐句上写着 A、
    主题清单里却没有这一句"这种自相矛盾的输出。
    """
    idx = {(l['i'], l['j']): l for l in links}
    order, rows = {}, []
    for i, lab in enumerate(labels):
        base = lab.rstrip("'")
        if base not in order:
            order[base] = len(rows)
            rows.append({
                'id': theme_mod.letter(len(rows)),
                'first_phrase': i + 1,
                'occurrences': [i + 1],
                'count': 1,
                'bars': bounds[i][1] - bounds[i][0] + 1,
                'variants': [],
            })
            continue
        row = rows[order[base]]
        row['occurrences'].append(i + 1)
        row['count'] += 1
        row['bars'] += bounds[i][1] - bounds[i][0] + 1
        lk = idx.get((row['first_phrase'] - 1, i))
        if lk:
            sim, kind, frag = lk.get('score'), lk.get('kind'), lk.get('fragment')
        else:                      # 反馈回路提升的：证据在 promoted 里
            hit = next((p for p in promoted if p.get('phrase') == i + 1), {})
            sim, kind, frag = hit.get('sim'), hit.get('kind'), False
        row['variants'].append({'phrase': i + 1, 'sim': sim,
                                'kind': kind or '—', 'fragment': bool(frag)})
    return rows

def _llm_structure(payload: dict, config) -> dict:
    task = (
        '你是专业曲式分析师。以下是一部作品的本地结构分析结果'
        '（小节密度、终止式、主题聚类、调性区域已由程序算出，不需要你重新统计）。'
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
              tonality_out: dict, melody_out: dict = None) -> dict:
    """音乐结构分析主入口。

    ``melody_out`` 是 ``melody_line`` 那一步的输出（可为 None）：有它
    就只取主旋律声部做主题指纹，没有就退回"全体最高音"的老口径。
    """
    measure_count = data.measure_count or (rhythm_out or {}).get('measure_count', 0)
    mchords = _measure_chords(chords_out)
    cadences = _cadence_points(mchords)
    owner = _melody_owner_map(melody_out)

    bounds = _phrase_bounds(data, cadences, measure_count)
    cands = _boundary_candidates(data, cadences, measure_count)

    notes = [_phrase_notes(data, s, e, owner) for s, e in bounds]
    fps = [theme_mod.fingerprint(n) for n in notes]
    notenums = [f['count'] for f in fps]

    # ---- ② 主题层：算全部证据 + 强阈值聚一次 ----
    th = theme_mod.analyze(fps)
    labels = th['labels']
    leader_map = {}
    for g in th['groups']:
        for m in g['members']:
            leader_map[m] = g['leader']

    sections, seq = _sections_of(bounds, labels, cadences, tonality_out, notenums)
    keys = [_key_of(s, tonality_out) for s, _e in bounds]

    # ---- ③ 曲式层（页面开关控制）----
    form_enabled = bool(getattr(config, 'structure_form_enabled', True))
    feedback = {'applied': False, 'promoted': [], 'reason': ''}

    if form_enabled:
        form = _infer_form(seq, sections)
        # 一轮反馈：假设含再现 → 用弱阈值 + 调性回归重判；只做一次
        if form['has_return'] and len(bounds) >= 3:
            new_labels, promoted = _apply_feedback(
                th['links'], len(bounds), labels, keys, leader_map)
            if promoted:
                n_sections, n_seq = _sections_of(
                    bounds, new_labels, cadences, tonality_out, notenums)
                n_form = _infer_form(n_seq, n_sections)
                feedback = {
                    'applied': True,
                    'promoted': promoted,
                    'reason': (f'曲式假设「{form["name"]}」含再现，据此把 '
                               f'{len(promoted)} 段"材料相似度不足但调性回归"的'
                               '段落提升为同一主题，再判一次'),
                }
                labels, sections, seq, form = new_labels, n_sections, n_seq, n_form
    else:
        form = {
            'name': '未分析曲式（开关关闭）',
            'scheme': ''.join(seq),
            'template': '',
            'has_return': False,
            'confidence': None,
            'margin': None,
            'close': False,
            'candidates': [],
        }

    # 主题清单**必须跟着标签走**：反馈回路可能刚改过 labels，
    # 若这里还用反馈前的 groups，页面会出现"乐句写着 A、主题清单里没这句"。
    themes = _theme_rows(labels, th['links'], feedback['promoted'], bounds)

    phrases = []
    for i, (s, e) in enumerate(bounds):
        phrases.append({
            'index': i + 1,
            'start_measure': s,
            'end_measure': e,
            'bars': e - s + 1,
            'note_count': notenums[i],
            'pitch_range': fps[i]['span'],
            'cadence': cadences.get(e, ''),
            'theme': labels[i],
            'theme_base': labels[i].rstrip("'"),
        })

    # 给 LLM 的载荷刻意压缩：只传段落骨架与主题序列，不传音符
    payload = {
        'measure_count': measure_count,
        'main_key': (tonality_out or {}).get('main_key'),
        'tempos': (rhythm_out or {}).get('tempos', [])[:6],
        'time_signatures': (rhythm_out or {}).get('time_signatures', [])[:4],
        'sections': [{'label': s['label'],
                      'measures': f"{s['start_measure']}-{s['end_measure']}",
                      'cadence': s['cadence'], 'key': s['key'],
                      'theme': s['theme']}
                     for s in sections][:24],
        'theme_sequence': ''.join(seq),
        'local_form_guess': form['name'],
        'candidate_scores': [{'name': c['name'], 'score': c['score']}
                             for c in form['candidates'][:3]],
    }
    llm = _llm_structure(payload, config) if form_enabled else None

    note = ('乐句切分、主题指纹与曲式候选打分均为本地确定性计算（未经大模型）；'
            '曲式命名与段落功能叙述由大模型完成，解析失败时保留本地骨架。')
    if not form_enabled:
        note = ('页面上的「分析曲式」开关关闭：只做了乐句切分与主题识别，'
                '不做曲式推断，也不调用大模型叙述。')

    return {
        'measure_count': measure_count,
        'phrase_count': len(phrases),
        'phrases': phrases,
        'phrase_boundaries': cands,
        'themes': themes,
        'sections': sections,
        'cadences': [{'measure': m, 'type': t} for m, t in sorted(cadences.items())],
        'form': dict(form, enabled=form_enabled, feedback=feedback),
        'theme_params': {
            'threshold': th['threshold'],
            'weak_threshold': theme_mod.THEME_WEAK,
            'frag_ratio': theme_mod.THEME_FRAG_RATIO,
            'frag_cover': theme_mod.THEME_FRAG_COVER,
            'weights': th['weights'],
            'melody_voice': bool(owner),
            # 主旋律线的段数签名：melody_line 重跑过、段数变了 → 主题指纹的
            # 取样口径就变了，structure 该跟着重算（见 pipeline._stale_deps）
            'melody_sig': len((melody_out or {}).get('segments') or []) or None,
        },
        'llm_analysis': llm,
        'note': note,
    }
