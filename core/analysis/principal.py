"""主音判定：在一个「同时响多个音」的事件里，挑出最可能是主旋律的那个音。

为什么需要这一层
----------------
导出主旋律线时，每个发声事件以前一律取 ``max(ev.midi_pitches)``。这是 MIR
里最老的基线（**skyline**），在两种织体上它是错的：

- **双音 / 加厚旋律**：平行三度、加八度、加和声音的主题，同一个发声事件
  有两个以上的音在响，最高音未必就是"唱"的那条 —— 有时伴奏骑在上面；
- **voicing 中途重排**：同一条线的和音排法变了（三度变六度、转位），
  最高音会在某个瞬间跳到填充音上，导出的线出现一个莫名的锯齿。

判定在**层的份上**做，不在**单音的份上**做
------------------------------------------
盯住单个音问"你是不是和声填料"是问不出来的 —— 一个 G 到底是旋律音还是五度
填充音，只有把它放回前后几拍的线里才看得出来。所以这里把每个事件按音高
**排位（rank）**拆成若干条候选线：第 0 层是每事件的最高音、第 1 层是次高音
……双音平行进行时，第 0 层就是上声部、第 1 层就是下声部，两条都是完整连贯的
线；柱式和弦伴奏拆出来则是几条一起平移、毫无个性的线。**每条候选线整体打分，
选总分最高的那条**——挑线而不是挑音，这是本模块唯一的一条根本思路。

五条本地证据（都不看"同时响几个音"；第五条只在少样本口径里参与打分）
------------------------------------------------------------------
1. **轮廓** ``shape``：候选线本身的旋律形态分（级进比 / 拱形度 / 平滑度），
   直接沿用 :func:`melody_line.contour_shape`。周期音型在这一项上会露馅；
2. **和弦成员度** ``nonmember``：这一层的音有多少不落在当前和声的音级集合内。
   旋律有大量外音（经过音 / 邻音 / 先现音 / 留音），和声填料几乎全是和弦内音。
   **注意反例**：平行三度旋律的和弦音率也很高，所以这一项必须配第三条；
3. **相对低音的位置漂移** ``bass_spread``：这一层的音相对当期低音的音程有多
   少种取值。填料钉在根 / 五 / 三轮换上，取值屈指可数；平行三度旋律虽然都是
   和弦音，却会因为三度六度交替而在"三音 / 五音"之间漂移；
4. **顶层先验** ``top_prior``：人跟唱时跟的是最上层那条外轮廓。这是很弱的
   先验（权重最小），只在前面三条打不出高下时起决定作用 —— 它其实就是
   skyline，留着它是因为**对严格平行的双音，它就是正确答案**；
5. **连接平滑度** ``link``：只看多音事件那几处，这一层的音跟前一声、后一声
   接不接得上（见下面"多音事件再少也要判"一节）。样本充足时权重 0，
   因为它和轮廓高度相关；样本少时它才是唯一还说话的那条。

实在分不出高下（最优层与第二名差距 ≤ :data:`PRINCIPAL_LLM_MARGIN`）时，把这一
段的量化证据打包交给 DeepSeek（dsh）判定，并把它的**判断与理由**一并返回，
由页面在右侧面板里原样展示。LLM 不可用 / 返回不合规则退回本地最优层 —— 这一层
永远不许让整段导出失败。

**多音事件再少也要判，不许退回 skyline**
----------------------------------------
早先这里有一条"段里多音事件少于 :data:`PRINCIPAL_MIN_EVENTS` 个就不拟合、退回
最高音"的短路。它是错的：整段就一两处双音时，恰恰是最高音最容易翻车的地方
—— 那一个叠上去的音经常是邻声部侵入的、或是一次临时的加厚，而"取最高音"这条
规矩会把整段线在这一瞬间顶歪。样本少**不是不判的理由，只是要换一套判法**：

- 轮廓（``shape``）在少样本下几乎失效：两条候选线只差一两个音，整条线的
  形态分被几十个单音稀释掉了，量不出差别；
- 于是补第五条证据 **连接平滑度** ``link``：只看多音事件那几个瞬间，这一层的音
  跟**前一声、后一声**接不接得上。级进接上来的是同一条线，硬跳上来的多半是
  填料。样本越少它越重要，所以少样本口径把它的权重给到 0.30（见
  :data:`PRINCIPAL_WEIGHTS_SMALL`）；
- 顶层先验（其实就是 skyline）在少样本下**降权**到 0.12：整段只有一两处双音时
  根本没有"两条线谁在上面"的统计可言，凭什么默认高的那个在唱；
- 统计不可靠 → 更难判 → 更该问模型：少样本的交界放宽到
  :data:`PRINCIPAL_LLM_MARGIN_SMALL`。开关关着就还是本地那几条说了算。

样本充足时 :data:`PRINCIPAL_WEIGHTS` 里 ``link`` 的权重是 0，行为与此前逐位一致。

启用条件：仅当「主旋律线保留双音」这个开关**关掉**时才跑（默认开着，原样保留
双音，不需要挑主音）。
"""
from .scoredata import ScoreData, midi_to_pc, midi_to_name

PRINCIPAL_MAX_LAYERS = 3        # 最多考虑几条候选层（第 0/1/2 高音）
PRINCIPAL_MIN_EVENTS = 3        # 多音事件少于这个数 → 少样本口径（照样判，不许退回 skyline）
# 轮廓 / 非和弦成员 / 低音位置漂移 / 顶层先验 / 连接平滑度
PRINCIPAL_WEIGHTS = (0.40, 0.22, 0.23, 0.15, 0.00)        # 样本充足：连接平滑度不参与
PRINCIPAL_WEIGHTS_SMALL = (0.30, 0.12, 0.16, 0.12, 0.30)  # 少样本：看衔接，不看"谁在上面"
PRINCIPAL_LLM_MARGIN = 0.05     # 最优层与第二名差距在此以内 → 交给 dsh 判定
PRINCIPAL_LLM_MARGIN_SMALL = 0.12   # 少样本时统计不可靠，界线放宽 → 更愿意问模型
PRINCIPAL_LLM_MAX_EVENTS = 10   # 一次 LLM 调用里最多问几个多音事件
HRM_TOP_PCS = 6                 # 每窗和声骨架最多取几个音级（与 chords.py 同口径）
HRM_STEP = 0.5                  # 和声采样步长（四分音符）
HRM_MAX_SAMPLES = 24            # 每窗最多采样几次（防止长窗拖成几十万次循环）

_LOCAL_KEYS = ('shape', 'nonmember', 'bass_spread', 'top_prior', 'link')


# ---------------------------------------------------------------- 和声骨架

def harmonic_background(data: ScoreData, measure_offsets: dict, bar_ql: float,
                        s: float, e: float) -> list:
    """段内的和声骨架：按小节聚合**全曲**在该时刻发响的音级与最低音。

    与 :func:`chords._aggregate_windows` 同口径（整拍加权、按权重取前
    :data:`HRM_TOP_PCS` 个音级），这里就地再算一遍而不去读 chords 的产物，
    是为了**不把 melody_line 的依赖链拉长** —— 和弦进行分析要跑 LLM，
    而主音判定只是在问"这一刻的和声背景是什么"，几条乘法足够。
    """
    wins = []
    ms = sorted(m for m, off in measure_offsets.items()
                if s - 1e-9 <= off < e - 1e-9)
    for m in ms:
        start = measure_offsets[m]
        end = measure_offsets.get(m + 1)
        end = (start + bar_ql) if end is None else min(end, e)
        if end <= start:
            continue
        span = min(end - start, HRM_MAX_SAMPLES * HRM_STEP)
        weight, bass = {}, None
        step = HRM_STEP * max(1.0, (end - start) / max(span, 1e-9))
        t = start
        guard = 0
        while t < end - 1e-9 and guard < HRM_MAX_SAMPLES:
            strong = abs((t - start) % 1.0) < 1e-6
            for p in data.parts:
                for ev in p.pitched_events:
                    if ev.offset - 1e-9 <= t < ev.offset + ev.duration - 1e-9:
                        for pitch in ev.midi_pitches:
                            weight[midi_to_pc(pitch)] = (
                                weight.get(midi_to_pc(pitch), 0.0)
                                + (2.0 if strong else 1.0))
                            if bass is None or pitch < bass:
                                bass = pitch
            t += step
            guard += 1
        if not weight:
            continue
        top = [pc for pc, _ in sorted(weight.items(), key=lambda kv: -kv[1])][:HRM_TOP_PCS]
        wins.append({'measure': m, 'start': start, 'end': end,
                     'pcs': frozenset(top),
                     'pcs_named': [midi_to_name(60 + pc) for pc in sorted(top)],
                     'bass': bass})
    return wins


def harmony_at(harmony: list, t: float):
    """t 时刻的和声背景 ``(pcs, bass)``；查不到（休止）返回 ``(空集, None)``。"""
    for w in harmony:
        if w['start'] - 1e-9 <= t < w['end'] - 1e-9:
            return w['pcs'], w['bass']
    return frozenset(), None


# ---------------------------------------------------------------- 候选层

def layer_events(evs: list, rank: int) -> list:
    """第 ``rank`` 层的 (事件, 该层取到的音) 列表。

    音数不足排位的事件取它自己的最高音 —— 一段谱里单音与多音事件混着走时，
    不能因为某个瞬间只响一个音就让这条线断点。
    """
    out = []
    for ev in evs:
        ps = sorted(ev.midi_pitches, reverse=True)
        if ps:
            out.append((ev, ps[min(rank, len(ps) - 1)]))
    return out


def _link_score(seq: list, idxs: list) -> float:
    """连接平滑度：候选层在**多音事件**处跟前一声、后一声接不接得上。

    整段只有一两处双音时，"整条线的轮廓"里那一个音的差异会被几十个单音稀释
    掉，唯一还能说话的就是它跟前后接不接得起来 —— 级进接上来的是同一条线，
    硬跳上来的多半是和声填料或邻声部临时侵入的音。所以这一条只在多音事件那
    几个位置上看，不看整条线。
    """
    if not idxs:
        return 0.5
    vals = []
    for i in idxs:
        ds = []
        if i - 1 >= 0:
            ds.append(abs(seq[i] - seq[i - 1]))
        if i + 1 < len(seq):
            ds.append(abs(seq[i] - seq[i + 1]))
        # 0 半音（同音反复）最像同一条线，12 半音（八度以上）基本不像
        vals.append(0.5 if not ds else
                    sum(max(0.0, 1.0 - min(d, 12) / 12.0) for d in ds) / len(ds))
    return sum(vals) / len(vals)


def _evidence(layer: list, harmony: list, rank: int, multi_idx=None) -> dict:
    """一条候选层的五条本地证据（各 0–1，越大越像主旋律那条线）。

    ``multi_idx`` 是段里**多音事件**在序列中的下标（``link`` 只看这几处）；
    传 ``None`` 表示不区分，此时 ``link`` 退成 0.5 的中性值。
    """
    from .melody_line import contour_shape        # 懒加载：避免模块级循环依赖

    seq = [p for _, p in layer]
    sh = contour_shape(seq)
    shape = 0.5 if sh.get('score') is None else float(sh['score'])

    members, ivs = 0, set()
    for ev, pitch in layer:
        pcs, bass = harmony_at(harmony, ev.offset)
        if pcs and midi_to_pc(pitch) in pcs:
            members += 1
        if bass is not None:
            ivs.add((pitch - bass) % 12)
    n = max(len(layer), 1)
    # 非和弦成员度：外音率越高越像旋律
    nonmember = 1.0 - members / n
    # 相对低音的位置漂移：取值种类越多越像在自由运动
    bass_spread = min(len(ivs) / max(min(n, 8), 1), 1.0)
    # 顶层先验：这一层在每个事件里排到第几位（0 = 最高音），反过来看。
    # 严格平行的双音里，两条线的轮廓、外音率、位置漂移全都一模一样，最后分
    # 出高下的就是这一条 —— 而人跟着唱的确实是最上面那条，方向是对的。
    # 权重给得最小，只在前面几条咬成平手时才由它落槌；少样本口径下再降到
    # 0.12 —— 整段只有一两处双音时，根本没有"哪条线在上面"的统计可言。
    priors = []
    for ev, _p in layer:
        size = len(ev.midi_pitches)
        priors.append(1.0 if size <= 1
                      else 1.0 - min(rank, size - 1) / (size - 1))
    top_prior = sum(priors) / len(priors) if priors else 1.0
    link = _link_score(seq, multi_idx or [])
    return {'shape': shape, 'nonmember': round(nonmember, 4),
            'bass_spread': round(bass_spread, 4),
            'top_prior': round(top_prior, 4),
            'link': round(link, 4)}


def score_layer(evidence: dict, small: bool = False) -> float:
    """五条证据的加权总分。

    ``small=True``（多音事件少于 :data:`PRINCIPAL_MIN_EVENTS`）时换一套权重：
    轮廓降、连接平滑度升、顶层先验压到最低 —— 少样本下不许靠"谁音高谁在唱"
    这种没统计依据的规矩落槌。
    """
    ws = PRINCIPAL_WEIGHTS_SMALL if small else PRINCIPAL_WEIGHTS
    return round(sum(w * evidence.get(k, 0.0)
                     for w, k in zip(ws, _LOCAL_KEYS)), 4)


# ---------------------------------------------------------------- LLM 兜底

_PRINCIPAL_TASK = """你是专业配器分析师。下面这段被判定为「主旋律」的声部里有多个音同时在响，
需要在每个瞬间判断**哪一个音是主旋律在唱的那个**，其余是和声填料。

声部：{name}（乐器 {inst}），第 {sm}–{em} 小节，共 {n} 个待判瞬间。

和声背景（按小节，音级集合与低音）：
{harmony}

本地已经算过五条证据（每条候选层 = 按音高排位拆出来的一条线）：
{layers}

待判瞬间（offset 为四分音符位置，音名按从高到低给出）：
{events}

{smallnote}
判定要求：
1. 只看这几个瞬间的前后关系、和声背景和旋律走向，不要因为某个音更高就选它；
2. 平行三度 / 六度的双音进行里，唱的那条通常是最上面那条，但若低音那条明显更具
   旋律独立性（有经过音、有轮廓），也可以选它；
3. 和声填料的特征：音几乎全落在当前和弦里、相对低音的位置固定在根/五/三上；
4. 个别瞬间叠上来的音，若与前后一声接不上（跳进很硬）而另一个音是顺着级进接上
   来的，那么级进的那个才是旋律在唱的。

输出 JSON：
{{"decisions":[{{"offset":12.0,"principal":"G4","reason":"一句话说明（60 字以内）"}}],
  "summary":"这一段整体按什么原则判的（80 字以内）"}}
"""


def _build_llm_task(name, inst, sm, em, harmony, layer_info, targets,
                    small: bool = False) -> str:
    return _PRINCIPAL_TASK.format(
        smallnote=('注意：这一段里同时响多个音的瞬间只有 '
                   f'{len(targets)} 处，本地统计样本极小，'
                   '请以这几处与前后单音的衔接（级进还是硬跳）为主来判，'
                   '不要默认取最高的那个音。'
                   if small else ''),
        name=name, inst=inst or '（未标注）', sm=sm, em=em, n=len(targets),
        harmony='\n'.join(
            f"  第{w['measure']}小节：{'、'.join(w['pcs_named'])}"
            f"（低音 {midi_to_name(w['bass']) if w['bass'] is not None else '—'}）"
            for w in harmony) or '  （无）',
        layers='\n'.join(
            f"  第{l['rank']}层：总分 {l['total']}（轮廓 {l['shape']} / "
            f"外音率 {l['nonmember']} / 低音位置漂移 {l['bass_spread']} / "
            f"顶层先验 {l['top_prior']} / 连接平滑度 {l.get('link')}）"
            for l in layer_info) or '  （无）',
        events='\n'.join(
            f"  offset {t['offset']:g}（第{t['measure']}小节）："
            f"{'、'.join(midi_to_name(p) for p in t['pitches'])}"
            f"　时值 {t['duration']:g} 拍" +
            (f"　力度 {t['velocity']}" if t.get('velocity') is not None else '')
            for t in targets),
    )


def _judge_by_llm(task: str, cfg) -> dict:
    """调 dsh 判定；**任何失败都不抛异常**，返回带 ``_error`` 的诊断 dict。"""
    from .llm import run_llm_json_soft, JSON_ONLY
    return run_llm_json_soft(
        task + JSON_ONLY,
        profile=getattr(cfg, 'dsh_profile', 'headless') or 'headless',
        timeout=int(getattr(cfg, 'dsh_timeout', 600) or 600))


def _apply_llm(raw: dict, targets: list) -> dict:
    """把 LLM 的返回值落到各待判事件上；不合法的一律退回本地结果。"""
    out = {'summary': '', 'applied': [], 'rejected': []}
    if not isinstance(raw, dict) or raw.get('_error'):
        return out
    out['summary'] = str(raw.get('summary') or '')[:400]
    decisions = raw.get('decisions')
    if not isinstance(decisions, list):
        return out
    by_off = {}
    for d in decisions:
        if not isinstance(d, dict):
            continue
        try:
            off = float(d.get('offset'))
        except (TypeError, ValueError):
            continue
        by_off[round(off, 3)] = d
    for t in targets:
        d = by_off.get(round(float(t['offset']), 3))
        if not d:
            continue
        want = str(d.get('principal') or '').strip()
        names = {midi_to_name(p): p for p in t['pitches']}
        hit = names.get(want)
        if hit is None:            # 模型说了一个不在候选里的音 → 不采纳
            out['rejected'].append({'offset': t['offset'], 'said': want})
            continue
        t['principal'] = hit
        t['source'] = 'deepseek'
        t['reason'] = str(d.get('reason') or '')[:200]
        out['applied'].append({'offset': t['offset'], 'pitch': hit,
                               'name': want,
                               'reason': t['reason']})
    return out


# ---------------------------------------------------------------- 主入口

def resolve_principal(data: ScoreData, part_index: int, s: float, e: float,
                      measure_offsets: dict, bar_ql: float,
                      cfg=None) -> dict:
    """挑出段内每个多音事件的主音，返回选定的层与逐事件判定。

    返回的 ``per_event`` 只在**单音导出**时用得上；开关「保留双音」打开时本模块
    根本不会被调用（见 :func:`melody_line.melody_line`）。
    """
    from . import melody_export as mex

    src = data.parts[part_index]
    evs = sorted(mex.events_in(src, s, e), key=lambda ev: ev.offset)
    result = {
        'part_index': part_index,
        'part_name': src.name,
        'measures': None,
        'chord_event_count': 0,
        'layers': [],
        'chosen_rank': 0,
        'margin': None,
        'resolved_by': 'single-notes',
        'small_sample': False,
        'per_event': [],
        'llm': None,
    }
    if not evs:
        return result

    multi_idx = [i for i, ev in enumerate(evs) if len(ev.midi_pitches) >= 2]
    result['chord_event_count'] = len(multi_idx)
    # 先把每个事件的候选音列出来。多音事件的主音**待定**（不预填最高音）——
    # 预填就等于默认 skyline，而 skyline 正是这里要推翻的东西。
    for ev in evs:
        ps = sorted(ev.midi_pitches, reverse=True)
        result['per_event'].append({
            'offset': round(float(ev.offset), 4),
            'measure': ev.measure,
            'pitches': list(ps),
            'principal': ps[0] if ps else None,
            'source': 'single' if len(ps) < 2 else 'pending',
            'reason': '',
        })
    if not multi_idx:
        # 一个多音事件都没有 → 主音就是它自己，没有"选哪一层"这回事
        result['resolved_by'] = 'single-notes'
        return result
    # 多音事件少于 PRINCIPAL_MIN_EVENTS → 少样本口径：照样判，只是换一套权重
    # （轮廓降权、连接平滑度升权、顶层先验压到最低）。**不再退回最高音** ——
    # 整段只有一两处双音时，"取最高音"恰恰最容易把线顶歪。
    small = len(multi_idx) < PRINCIPAL_MIN_EVENTS
    result['small_sample'] = small

    harmony = harmonic_background(data, measure_offsets, bar_ql, s, e)
    max_rank = min(PRINCIPAL_MAX_LAYERS,
                   max(len(evs[i].midi_pitches) for i in multi_idx))
    layers = []
    for rank in range(max_rank):
        layer = layer_events(evs, rank)
        ev = _evidence(layer, harmony, rank, multi_idx)
        ev.update({'rank': rank, 'total': score_layer(ev, small)})
        layers.append(ev)
    layers.sort(key=lambda d: -d['total'])
    best, second = layers[0], (layers[1] if len(layers) > 1 else None)
    margin = best['total'] - (second['total'] if second else 0.0)
    result['layers'] = layers
    result['chosen_rank'] = best['rank']
    result['margin'] = round(margin, 4)

    # 先把每个事件按**选定层**落到具体音高上，再决定要不要问 LLM
    rank = best['rank']
    chosen = layer_events(evs, rank)
    picked = {round(float(ev.offset), 4): p for ev, p in chosen}
    for row in result['per_event']:
        p = picked.get(row['offset'])
        if len(row['pitches']) >= 2 and p is not None:
            row['principal'] = p
            row['source'] = 'local'
    result['resolved_by'] = 'local'

    # 少样本时统计本来就不可靠，难判的界线放宽 → 更愿意交给模型
    gate = PRINCIPAL_LLM_MARGIN_SMALL if small else PRINCIPAL_LLM_MARGIN
    if second is None or margin >= gate or cfg is None:
        return result

    # 第一名第二名咬得太紧 → 交给 DeepSeek 判定。
    # 开关「主旋律线判定困难时交给 DeepSeek」关着 → 一次网络请求都不发，
    # 结果与开启 DeepSeek 之前的版本逐位一致（比大小的本事还是本地那四条）。
    if not getattr(cfg, 'melody_llm_judge', True):
        result['resolved_by'] = 'local'
        result['llm'] = {'called': False, 'skipped': 'setting-off'}
        return result

    targets = [r for r in result['per_event'] if len(r['pitches']) >= 2]
    # 只问最难判的那几个：全部塞进去会让模型在几百个音里敷衍，还容易超输出上限
    targets = targets[:PRINCIPAL_LLM_MAX_EVENTS]
    task = _build_llm_task(
        src.name, src.instrument,
        evs[0].measure, evs[-1].measure, harmony, layers,
        [{'offset': r['offset'], 'measure': r['measure'],
          'pitches': r['pitches'],
          'duration': next(ev.duration for ev in evs
                           if round(float(ev.offset), 4) == r['offset']),
          'velocity': next(ev.velocity for ev in evs
                           if round(float(ev.offset), 4) == r['offset'])}
         for r in targets], small)
    raw = _judge_by_llm(task, cfg)
    applied = _apply_llm(raw, result['per_event'])
    if applied['summary'] or applied['applied']:
        result['resolved_by'] = 'deepseek'
        for row in result['per_event']:
            if row.get('source') == 'deepseek':
                result['chosen_rank'] = None      # 不再是"整段选一层"
                break
    result['llm'] = {
        'called': True,
        'asked': len(targets),
        'applied': len(applied['applied']),
        'rejected': len(applied['rejected']),
        'summary': applied['summary'],
        'decisions': applied['applied'],
        'raw_error': (str(raw.get('_error')) if isinstance(raw, dict)
                      and raw.get('_error') else None),
    }
    return result


def principal_map(principal: dict) -> dict:
    """``{offset: 主音 MIDI 号}`` —— 导出时按它把和弦压成单音。"""
    return {round(float(r['offset']), 4): r['principal']
            for r in (principal or {}).get('per_event', [])
            if r.get('principal') is not None}
