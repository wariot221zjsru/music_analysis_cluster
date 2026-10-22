"""主旋律线追踪（逐段）—— MIR 口径的 melody line extraction / voice tracking。

与 core.analysis.melody（主旋律**声部识别 / 选轨**）的分工
------------------------------------------------------------

|              | melody（选轨，旧口径）        | melody_line（本模块）            |
|--------------|------------------------------|----------------------------------|
| 回答的问题   | 整曲哪个声部是主旋律          | **每一段**主旋律在哪个声部        |
| 粒度         | 整曲一个声部（或少数交接段）  | 每窗一段，随时可换                |
| 门控         | 受 exchange 的 is_exchange 门控 | **无门控**，恒逐段              |
| 前奏 / 间奏  | 可能整段丢掉                  | 按人声掩膜取最突出的器乐声部      |

关键点：**本模块不读 exchange 的 is_exchange。** 用"是否模仿型交接"去门控
"要不要逐段提取"，是曲式学判据管了 MIR 任务的形态，前奏就是这么丢的。
两条线并列存在、各自重跑，互不干扰。

算法
----
1. 窗级归属：沿用 voice_id 的 per-window 显著性矩阵（音高 / 时值 / 动机重复 /
   稀疏度四项），叠加**人声掩膜**——该窗人声在唱就归人声（config
   开启人声优先时），人声休止才退回显著性最高的器乐声部；
2. 连续同归属的窗合并为段；
3. 碎片合并：短于 melody_segment_min_bars 的段并入相邻段，抑制窗间抖动；
4. 段级置信度 = 窗内 **margin**（第一名 − 第二名显著性）均值。用 margin 而不是
   第一名自身，是因为显著性在窗内做了 min-max 归一：两个声部时第一名最低也有
   0.5，看不出"人声 0.85 压过吉他 0.30"和"0.51 险胜 0.49"的区别。
   只有一个声部在响时无竞争，margin 记为 1.0；
5. **形态资格层（可选，默认关，在账户总体设置里开）**：形态决定「有没有资格」，
   显著性决定「谁更强」—— 形态只做**否决**，不参与加分、不改 margin。
   段级算出各声部的旋律轮廓形态分（级进比 / 拱形度 / 轮廓平滑度），
   当前归属声部形态不合格时，由段内形态合格且显著性最高的声部顶上。
   用来挡住分解和弦、阿尔贝蒂低音、固定音型这类「周期音型」被误判成主旋律 ——
   它们的动机重复度接近 1.0，单看显著性很容易赢过真正的人声旋律。
   两条安全设计：① 段内音符太少不足以统计轮廓时**不作判定、一律放行**；
   ② 所有声部都不合格时**保留原归属**，绝不把整段丢掉；
6. 导出两种：
   - absolute：**原位**写入（各段保留真实小节位置，段间空隙留白）——MIR 语义；
   - spliced：首尾相接的连续单声部（方便当一条旋律线听 / 导入 DAW）。
"""
from . import melody_export as mex
from . import melody_judge as judge_mod
from . import principal as principal_mod
from .melody import save_stream
from .scoredata import (
    ScoreData, build_measure_map, windows_of_measures, interval_seq,
)


# ---------------------------------------------------------------- 工具
# 小节区间换算 / 事件截取 / 导出的节奏与音色统一走 melody_export：
#   ① 小节号查不到偏移时按相邻小节推算，不再退回 0.0（旧实现会把段起点当成
#      全曲开头，导出整段错位）；
#   ② 导出写入 tempo 与 instrument，不再是恒 120 BPM + 默认钢琴。


def part_label(data: ScoreData, idx) -> str:
    """声部索引 → 名字（越界也不炸，给个占位）。"""
    try:
        i = int(idx)
        if 0 <= i < len(data.part_names):
            return data.part_names[i]
    except (TypeError, ValueError):
        pass
    return f'Part {idx}'


def _window_choices_label(data: ScoreData, idx) -> str:
    """兼容旧名：DeepSeek 裁定里原本用它取本地声部名。"""
    return part_label(data, idx)


def _window_choices(matrix_row: dict, data: ScoreData,
                    low: set = frozenset(), doubles: dict | None = None) -> list:
    """一个窗的 {part_index: salience} → 按显著性降序的候选列表。

    ``low`` 是低音区声部（比全曲最高声部低 ≥ 2 个八度），在这里**只打标记**
    不剔除：主旋律在高音区，选人时排除它们（见 :func:`_pick`），但算置信度
    的 margin 仍把它们算作竞争对手 —— 否则"贝斯被排除"会让置信度虚高到
    满分，反而看不出这段到底有多确定。

    ``doubles`` 是齐奏加倍分组（见 :func:`voice_id.doubling_groups`）：同一条
    旋律被多件乐器同奏时它们算**同一条线索**，互相之间不构成竞争。
    """
    doubles = doubles or {}
    items = []
    for idx, sc in matrix_row.get('scores', {}).items():
        i = int(idx)
        group = doubles.get(i, i)
        peers = sorted(j for j, g in doubles.items() if g == group and j != i)
        items.append({
            'part_index': i,
            'part_name': data.part_names[i] if i < len(data.part_names) else f'Part {i}',
            'salience': float(sc),
            'is_voice': data.parts[i].is_voice if i < len(data.parts) else False,
            'low_register': i in low,
            # 同组 = 同一条旋律（齐奏加倍），算 margin 时互相排除
            'group': group,
            'doubled_with': peers,
        })
    items.sort(key=lambda d: -d['salience'])
    return items


# --------------------------------------------------- 低音区保护（"整首都是低音"）
# 「低音区声部不竞选主旋律」是一条**经验先验**（主旋律通常在更高的音区），
# 不是物理定律。遇到下面两种情形它必须让路：
#
#   ① **整首曲子都在低音区**。这一条相对差判据自己就处理了 —— 没有人比它
#      高两个八度，`low_register_parts` 返回空集，谁都不排除（实测：大号 /
#      大提琴 / 低音提琴三声部曲子，最高声部平均 51.8，无人被判低音区）。
#
#   ② **某个低音声部听感上绝对占优**：贝斯主奏、主题本来就在低音区、上方
#      只剩一个持续音铺底 / 零星装饰。这时候再把它排除，报出来的主旋律就是
#      那个每小节打两下的踏板音 —— 比报错覆盖率更糟。
#
# 翻盘要**两条独立证据同时成立**才生效 —— 单看任何一条都会误伤：
#
#   证据 ① 形态轮廓（contour_shape）：低音线条的级进比 / 拱形度是否明显比
#          同窗**所有**非低音候选更像一条旋律。光看这条不够 —— 走动贝斯的
#          轮廓常常比稀疏的长笛旋律还漂亮（实测形态差 0.199，却不该翻盘）。
#   证据 ② 实质占比（substance）：把显著性里的"平均音高"这一项**拿掉**，
#          只比时值 / 动机重复 / 密度三项。拿掉音高项是因为它恰好就是
#          "高音区更像主旋律"这条先验本身，留着比等于没比（低音永远输在这一项）。
#
# 再加一条**力度**（谱面标了力度才作数，见 :data:`DYNAMICS_MIN_SUPPORT`）：
#
#   证据 ③ 力度（dynamics）：这条线听着是不是真的压过上方。谱面把它写成
#          更弱（力度差 < −4）就**不许翻盘** —— 力度记号是作曲家写下的聆听
#          指示，它说这条线是退让的；反过来明显更响（≥ +8）时，另两条门槛
#          按比例放宽到一半 —— 力度已经替这条线把话说了一半。
#
# 前两条同时达标才说明**不是音区在说话，是这条线本身更能听**；第三条决定
# 这份"更能听"要不要打折、要不要干脆否决：
#   低音主奏 + 高音长音铺底     形态 +0.145、实质 +0.334  → 翻盘 ✓
#   贝斯主奏 + 柱式和弦伴奏     形态 +0.789、实质 +0.334  → 翻盘 ✓
#   走动贝斯 + 稀疏长笛旋律     形态 +0.199、实质  0.000  → 不翻盘 ✓
#   naik 的贝斯（基线）         形态 +0.019、实质 -0.057  → 不翻盘 ✓
#   两条都达标、但低音写得比上方 pp → 不翻盘 ✓（力度否决）
#   轮廓 / 实质领先不足一半、低音写得比上方 ff → 翻盘 ✓（力度放宽门槛）
LOW_REGISTER_SHAPE_LEAD = 0.10      # 证据①：形态分要高出多少
LOW_REGISTER_SUBSTANCE_LEAD = 0.15  # 证据②：去音高项的显著性要高出多少
LOW_MELODY_MIN_PITCHES = 3          # 至少几个不同音高
LOW_MELODY_MIN_RANGE = 4            # 至少跨多少半音（一个大三度）


# -------------------------------------------------- 力度（"听感"的直接证据）
# 力度不进显著性四项：那四项是所有模块共用的尺度，动它会连带改写已标定的置信度
# 分级、历史产物与全部实测结论。它只在两个地方说话：
#
#   ① 低音区翻盘（本节第三/四条证据）：明显更弱的低音不许翻盘 —— 谱面都把它
#      写成 pp 了，它就不可能是"听感占优"的那条；明显更响的低音，则把另两条
#      门槛按比例放宽 —— 力度已经替它把话说一半了；
#   ② 显著性分不出高下（≤ DYNAMICS_TIEBAND）时，响的那个赢（见 :func:`_pick`）。
#
# **谱面没标力度时整条链路完全不生效**（velocity 为 None → 力度差为 None →
# 不作判定），没有力度数据的曲子结果与以前逐位一致。
DYNAMICS_MIN_SUPPORT = 0.5   # 参与比较的声部，窗内至少一半事件要有力度数据
DYNAMICS_TIEBAND = 0.03      # 显著性差距在此以内视为"分不出高下"
DYNAMICS_TOLERANCE = 4       # 力度差在 ±4/127 以内算打平（耳朵听不出）
DYNAMICS_LEAD = 8            # 差这么多才算"明显更响"（约一个力度级别）
DYNAMICS_RELAX = 0.5         # 力度明显占优时，翻盘两条门槛按此比例放宽
DYNAMICS_CONF_DIVISOR = 64   # 打平裁决的置信度 = 力度领先 / 该值（封顶 1.0）


def looks_like_line(seq: list) -> bool:
    """这条音高序列够不够格叫"一条旋律线"（不同音高数与跨度下限）。

    少了这道闸不行：只在一个音上打点的踏板音，级进比 / 拱形度 / 平滑度
    三项全都满分（形态分 1.0），比真旋律还高。
    """
    if len(set(seq)) < LOW_MELODY_MIN_PITCHES:
        return False
    return (max(seq) - min(seq)) >= LOW_MELODY_MIN_RANGE


def _part_dyn(per_part: list):
    """per_part → ({part_index: 平均力度}, {part_index: 力度数据覆盖率})。

    老产物（改这条规则之前存的 dict）里没有 ``dyn`` / ``dyn_support`` 键，
    ``.get`` 会得到 None / 0.0 —— 与"谱面没标力度"同一处理路径，自动失效。
    """
    dyn = {d['part_index']: d.get('dyn')
           for d in (per_part or []) if not d.get('empty')}
    sup = {d['part_index']: float(d.get('dyn_support') or 0.0)
           for d in (per_part or []) if not d.get('empty')}
    return dyn, sup


def dynamics_lead(per_part: list, cand: int, rivals: list):
    """候选相对一组竞争对手的平均力度差（velocity 单位，1–127）。

    返回 ``None`` = **不作判定**，调用方必须原样放行：
      * 候选或对手任何一方没有力度数据（谱面没标）；
      * 参与比较的声部里有谁窗内力度覆盖率 < :data:`DYNAMICS_MIN_SUPPORT`
        （零星几个音的均值说明不了整段的响度）；
      * 根本没有对手可参照（没有参照就谈不上"领先"）。
    """
    if per_part is None or not rivals:
        return None
    dyn, sup = _part_dyn(per_part)
    base = dyn.get(cand)
    if base is None:
        return None
    others = [dyn.get(r) for r in rivals]
    others = [v for v in others if isinstance(v, (int, float))]
    if not others:
        return None
    involved = [cand] + [r for r in rivals
                         if isinstance(dyn.get(r), (int, float))]
    if any(sup.get(i, 0.0) < DYNAMICS_MIN_SUPPORT for i in involved):
        return None
    return round(base - max(others), 2)


def low_register_overrides(data: ScoreData, items: list, s: float, e: float,
                           per_part: list | None = None) -> dict:
    """窗内哪些低音区声部该因"听感占优"回来当主旋律。

    返回 ``{part_index: {'shape': 形态分, 'shape_lead': 领先多少,
    'substance_lead': 去音高项领先多少, 'dyn_lead': 平均力度差（None=没数据）}}``；
    不该翻盘时返回空字典。

    ``items`` 是窗内候选（同 :func:`_window_choices` 的形状）；``per_part``
    可直接传 :func:`voice_id.window_scores` 的结果，省掉一次重算。
    没有任何非低音候选时返回空 —— 那种情形由"整窗只剩低音在响"兜底处理。
    """
    lows = [c for c in items if c.get('low_register')]
    others = [c for c in items if not c.get('low_register')]
    if not lows or not others:
        return {}
    from . import voice_id as voice_mod    # 懒加载：避免模块级循环依赖
    if per_part is None:
        per_part = voice_mod.window_scores(data, s, e)['per_part']
    sub = {d['part_index']: d.get('substance', 0.0)
           for d in per_part if not d.get('empty')}
    seqs = {c['part_index']: _seg_pitch_seq(data, c['part_index'], s, e)
            for c in items}
    # 上方候选的音符太少（形态分为 None，不作判定）→ **不翻盘**。
    # 证据不足应当维持原判，不能反过来让低音占便宜（naik 的尾声就是这样：
    # 最后两小节只有钢琴两三个音，贝斯的显著性反而更高）。
    shapes = {}
    for c in others:
        sc = contour_shape(seqs[c['part_index']])['score']
        if sc is None:
            return {}
        shapes[c['part_index']] = sc
    # 上方候选里真正"像一条旋律线"的那些（不同音高够多、跨度够大）
    line_like = [c for c in others if looks_like_line(seqs[c['part_index']])]
    out = {}
    for c in lows:
        seq = seqs[c['part_index']]
        sh = contour_shape(seq)
        if sh['score'] is None or not looks_like_line(seq):
            continue
        # 参照对象：有"成线的"上方候选就只比它们（避免拿踏板当标尺），
        # 否则退一步跟所有上方候选比 —— 力度问的是"谁听着更响"，与
        # 对手成不成线无关。
        rivals = [o['part_index'] for o in (line_like or others)]
        lead_dyn = dynamics_lead(per_part, c['part_index'], rivals)
        # 力度明显弱于对手 → 不许翻盘。谱面都把它写成更弱了，它就不可能是
        # "听感占优"的那条；这条优先于下面所有放行的分支。
        if lead_dyn is not None and lead_dyn < -DYNAMICS_TOLERANCE:
            continue
        # 力度明显占优 → 另两条门槛按比例放宽（力度已经替它把话说了一半）
        relax = (DYNAMICS_RELAX
                 if (lead_dyn is not None and lead_dyn >= DYNAMICS_LEAD)
                 else 1.0)
        if not line_like:
            # 上方留下的全是持续音铺底 / 踏板 / 两个音来回的摇摆，只是因为
            # **音区高**才被留下 —— 它压不住低音区那条真正在走的线。
            out[c['part_index']] = {'shape': sh['score'], 'shape_lead': None,
                                    'substance_lead': None, 'dyn_lead': lead_dyn,
                                    'dyn_relaxed': relax < 1.0,
                                    'reason': 'no_line_like_counterpart'}
            continue
        lead_shape = sh['score'] - max(shapes[o['part_index']] for o in line_like)
        lead_sub = (sub.get(c['part_index'], 0.0)
                    - max(sub.get(o['part_index'], 0.0) for o in line_like))
        if (lead_shape >= LOW_REGISTER_SHAPE_LEAD * relax
                and lead_sub >= LOW_REGISTER_SUBSTANCE_LEAD * relax):
            out[c['part_index']] = {
                'shape': sh['score'],
                'shape_lead': round(lead_shape, 4),
                'substance_lead': round(lead_sub, 4),
                'dyn_lead': lead_dyn,
                'dyn_relaxed': relax < 1.0,
            }
    return out


def best_override(overrides: dict):
    """多个低音声部都够格时取形态分最高的那个；没有则返回 None。"""
    if not overrides:
        return None
    return max(overrides, key=lambda k: overrides[k]['shape'])


def _pick(choices: list, voice_priority: bool, override: dict | None = None,
          per_part: list | None = None):
    """窗内选轨：低音翻盘 → 人声优先 → 显著性最高 → 力度裁决。返回 (选中项, margin)。

    低音区声部**不参与竞选**（主旋律在高音区），但仍是 margin 的竞争对手；
    整窗只剩低音在响时才让它上，绝不把整窗记成静默。

    ``override`` 是 :func:`low_register_overrides` 认可的例外 —— 低音线条在
    "轮廓更像旋律"与"实质占比更高"两条证据上都压过上方候选时**直接当选**，
    不再按显著性排队：显著性里的"平均音高"和"密度倒数"两项本身就是
    "高音区更像主旋律"这条先验，留着比等于没比，低音永远输在这两项上。
    置信度取"显著性领先"与"形态领先"里的较大者 —— 翻盘是因为形态证据强，
    不是因为显著性分数高，否则会一边翻盘一边报"置信度 0"。

    ``per_part``（:func:`voice_id.window_scores` 的结果，带 ``dyn``）用于最后一步
    **力度裁决**：显著性差 ≤ :data:`DYNAMICS_TIEBAND` 时两条线在四项上其实分不出
    高下，这时谱面写得更响（力度差 ≥ :data:`DYNAMICS_LEAD`）的那条顶上来 —— 耳朵
    会跟着响的那条走。没有力度数据时（``None``）这一步整个跳过，结果不变。
    """
    if not choices:
        return None, 0.0
    override = override or {}

    def _ok(c) -> bool:
        return not c.get('low_register') or c['part_index'] in override

    eligible = [c for c in choices if _ok(c)] or choices

    def _margin(best: dict) -> float:
        # 竞争对手只算**有资格当主旋律**的声部：
        #   ① 齐奏加倍的"对手"其实是同一条旋律，不算竞争 —— 否则两者的显著性
        #      逐位相同，margin 恒为 0，明明稳稳选对了却报"置信度 0"；
        #   ② 低音区声部同样不算 —— 它靠"长时值 + 高重复 + 稀疏"在窗内
        #      min-max 归一里系统性偏高，拿它当基准会把所有段的置信度压成 0
        #      （这正是当初要在选主旋律时排除它的同一个理由）。
        #      例外：已经按"听感占优"恢复资格的低音，它配得上当基准。
        rivals = [c['salience'] for c in choices
                  if c['group'] != best['group'] and _ok(c)]
        # 没有合格的对手：整窗只有它（和它的齐奏加倍）在竞选 → 完全确定
        if not rivals:
            return 1.0
        return max(best['salience'] - max(rivals), 0.0)

    win = best_override(override)
    if win is not None:
        best = next((c for c in choices if c['part_index'] == win), None)
        if best is not None:
            info = override[win]
            # 上方没有像旋律的候选时，参考的就是它自己的形态分；其余按领先量
            lead = info.get('shape_lead')
            lead = info['shape'] if lead is None else lead
            return best, round(max(_margin(best), lead), 4)

    if voice_priority:
        voices = [c for c in eligible if c['is_voice']]
        if voices:
            best = max(voices, key=lambda c: c['salience'])   # 多人声取最显著那条
            return best, round(_margin(best), 4)
    # 力度裁决：显著性分不出高下时，听感上更响的那条赢（要有力度数据才作数）。
    # 只在这一种情形下改判 —— 差距明显时不插手，显著性的判断优先。
    if per_part and len(eligible) >= 2:
        top, second = eligible[0], eligible[1]
        if abs(top['salience'] - second['salience']) <= DYNAMICS_TIEBAND:
            lead_dyn = dynamics_lead(per_part, second['part_index'],
                                     [top['part_index']])
            if lead_dyn is not None and lead_dyn >= DYNAMICS_LEAD:
                return second, round(
                    max(_margin(second),
                        min(1.0, lead_dyn / DYNAMICS_CONF_DIVISOR)), 4)
    best = eligible[0]
    return best, round(_margin(best), 4)


def _merge_short_runs(runs: list, min_bars: int) -> list:
    """把短于 min_bars 的碎片并入相邻段（优先更长的邻居）。"""
    if min_bars <= 1 or len(runs) <= 1:
        return runs
    segs = [dict(r) for r in runs]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, seg in enumerate(segs):
            bars = seg['end_measure'] - seg['start_measure'] + 1
            if bars >= min_bars:
                continue
            neighbours = []
            if i > 0:
                neighbours.append(i - 1)
            if i < len(segs) - 1:
                neighbours.append(i + 1)
            if not neighbours:
                continue
            j = max(neighbours,
                    key=lambda k: (segs[k]['end_measure'] - segs[k]['start_measure'] + 1,
                                   segs[k]['avg_margin']))
            segs[j]['start_measure'] = min(segs[j]['start_measure'], seg['start_measure'])
            segs[j]['end_measure'] = max(segs[j]['end_measure'], seg['end_measure'])
            segs[j]['absorbed'] = segs[j].get('absorbed', []) + [seg['part_index']]
            segs.pop(i)
            changed = True
            break
    return segs


def _level(conf: float, hi: float, mid: float) -> str:
    if conf >= hi:
        return '高'
    if conf >= mid:
        return '中'
    return '低'


# ---------------------------------------------------------------- 形态资格层
# 「形态决定有没有资格，显著性决定谁更强」—— 形态只做**否决**：
# 不参与显著性加分、不改 margin、不改置信度，只在当前归属形态不合格时换人。
#
# 三项指标（都只依赖段内音高序列，不引入新的输入）：
#   级进比 step_ratio      |音程| ≤ 2 的占比。真旋律以级进为主；
#                          分解和弦 / 阿尔贝蒂低音几乎全是跳进 → 接近 0。
#   拱形度 arch            最高音出现位置的居中程度。真旋律的高潮多在乐句中段；
#                          单调爬升或周期音型 → 低。
#   轮廓平滑度 smooth      局部极值点越密越"来回扫"（固定音型）→ 越低。
#
# 级进比权重压倒性最高：它是区分"旋律"与"音型"唯一可靠的一项。拱形与平滑度
# 只作辅助 —— 上下行的分解和弦在轮廓上同样是"平滑大波浪"，单看这两项它很像
# 旋律，只有级进比能把它识破（实测：上下行琶音 step≈0 但 arch≈0.78、
# smooth≈0.73，权重平摊时形态分 0.38 会漏过 0.35 的阈值；按 0.7/0.15/0.15
# 加权后为 0.23，与真人声旋律的 0.74 拉开明显差距）。

SHAPE_MIN_NOTES = 4                 # 少于该音数不判定轮廓（证据不足，一律放行）
SHAPE_WEIGHTS = (0.7, 0.15, 0.15)   # 级进比 / 拱形度 / 轮廓平滑度


def _seg_pitch_seq(data: ScoreData, part_index: int, s: float, e: float) -> list:
    """声部在段 [s, e) 内的音高序列（按起点升序，和弦取最高音）。"""
    evs = sorted(mex.events_in(data.parts[part_index], s, e),
                 key=lambda ev: ev.offset)
    return [max(ev.midi_pitches) for ev in evs if ev.midi_pitches]


def contour_shape(seq: list) -> dict:
    """音高序列 → 旋律轮廓形态分（0–1，越大越像"一条旋律"）。

    ``score`` 为 None = "音符太少，不作判定"，调用方必须按**放行**处理。
    """
    n = len(seq)
    if n < SHAPE_MIN_NOTES:
        return {'note_count': n, 'score': None,
                'reason': f'仅 {n} 个音，不足以判定轮廓'}
    ivs = interval_seq(seq)
    step_ratio = sum(1 for iv in ivs if abs(iv) <= 2) / len(ivs)
    # 拱形度：最高音位置的居中程度。有多个最高点时取**中位**位置 —— 否则
    # 周期性旋律会因为"最高点落在第一个周期"而被系统性低估。
    top = max(seq)
    peaks = [i for i, p in enumerate(seq) if p == top]
    pos = peaks[len(peaks) // 2] / max(n - 1, 1)
    arch = max(0.0, 1.0 - 2.0 * abs(pos - 0.5))
    # 轮廓平滑度：局部极值点越密越像来回扫的固定音型
    ext = 0
    for i in range(1, n - 1):
        a, b = seq[i] - seq[i - 1], seq[i + 1] - seq[i]
        if a and b and (a > 0) != (b > 0):
            ext += 1
    smooth = 1.0 - min(ext / max(n - 2, 1), 1.0)
    score = round(SHAPE_WEIGHTS[0] * step_ratio
                  + SHAPE_WEIGHTS[1] * arch
                  + SHAPE_WEIGHTS[2] * smooth, 4)
    return {
        'note_count': n,
        'step_ratio': round(step_ratio, 4),
        'arch': round(arch, 4),
        'contour_smooth': round(smooth, 4),
        'score': score,
    }


def shape_eligible(shape: dict, min_score: float) -> bool:
    """形态是否够格当主旋律。score 为 None（证据不足）一律放行。"""
    sc = shape.get('score')
    return True if sc is None else sc >= min_score


def _seg_ranked_salience(win_scores: dict, start_m: int, end_m: int,
                         owner: int) -> list:
    """段覆盖窗内各声部的平均显著性，降序 [(part_index, salience)]。

    保证当前归属 owner 一定在列表里（它可能在该段所有窗里都没有分数，
    例如人声掩膜把它选出来但显著性矩阵里没记它）。
    """
    agg = {}
    for sm, scores in win_scores.items():
        if not (start_m <= sm <= end_m):
            continue
        for idx, sc in scores.items():
            agg.setdefault(idx, []).append(sc)
    if not agg:
        return [(owner, 0.0)]
    ranked = sorted(((idx, round(sum(v) / len(v), 4)) for idx, v in agg.items()),
                    key=lambda kv: -kv[1])
    if owner not in agg:
        ranked.append((owner, 0.0))
    return ranked


def apply_shape_gate(data: ScoreData, s: float, e: float, ranked: list,
                     current_idx: int, min_score: float):
    """形态资格层：从按显著性降序的候选里挑第一个**形态合格**的声部。

    返回 ``(chosen_index, shapes, reason)``；reason 为空表示没有换人。
    一个合格的都没有时**保留原归属**，绝不把整段丢掉。
    """
    shapes = {idx: contour_shape(_seg_pitch_seq(data, idx, s, e))
              for idx, _ in ranked}
    for sh in shapes.values():
        sh['eligible'] = shape_eligible(sh, min_score)
    cur = shapes.get(current_idx)
    if cur is None or cur['eligible']:
        return current_idx, shapes, ''
    cur_name = data.parts[current_idx].name
    why = (f"形态分 {cur['score']}（级进 {cur['step_ratio']} / "
           f"拱形 {cur['arch']} / 平滑 {cur['contour_smooth']}）低于下限 {min_score}")
    for idx, _ in ranked:
        if idx == current_idx or not shapes.get(idx, {}).get('eligible'):
            continue
        return idx, shapes, (f'{cur_name}{why}，改由形态合格的 '
                             f'{data.parts[idx].name}'
                             f"（形态分 {shapes[idx]['score']}）承担")
    return current_idx, shapes, (
        f'{cur_name}{why}，但段内没有形态合格的替代声部，保留原归属')


# ---------------------------------------------------------------- 导出

def _pitches_for(ev, seg: dict):
    """这次导出该事件留下哪几个音（见 :func:`principal.resolve_principal`）。

    - 保留双音（默认）：**全部留下** —— 平行三度、加八度这类加厚旋律整条留住；
    - 压成单音：按该段的主音判定表挑一个。表里查不到（事件本身是单音 / 老结果里
      根本没有主音表）时才退回最高音，不在产物里插空洞 —— 多音事件**一律有判定
      结果**，哪怕这一段只有一两处双音（少样本口径照样判，不再"跳过"）。

    表里查不到还有一种常见情形：这份 result 是**重导出**读回来的，段的 map 在
    JSON 往返里键变成了字符串，这里一并兜住。
    """
    if seg.get('keep_doubles', True):
        return list(ev.midi_pitches)
    raw = (seg.get('principal') or {}).get('map') or {}
    try:
        pmap = {round(float(k), 4): v for k, v in raw.items() if v is not None}
    except (TypeError, ValueError, AttributeError):
        pmap = {}
    hit = pmap.get(round(float(ev.offset), 4))
    return [hit] if hit is not None else [max(ev.midi_pitches)]


def _export_absolute(data: ScoreData, segments: list, measure_offsets: dict,
                     bar_ql: float, options: dict):
    """原位导出：各段留在真实小节位置，未覆盖区间留白（MIR 语义）。"""
    from music21 import stream

    part = stream.Part()
    part.partName = '主旋律线（逐段追踪 · 原位）'
    spans = []
    for seg in segments:
        s, e = mex.seg_range(data, measure_offsets, bar_ql,
                             seg['start_measure'], seg['end_measure'])
        spans.append({'offset': s, 'part_index': seg['part_index']})
        src = data.parts[seg['part_index']]
        for ev in mex.events_in(src, s, e):
            part.insert(ev.offset,
                        mex.make_note(ev, pitches=_pitches_for(ev, seg)))
    # 节奏 / 音色：原位导出保留原曲小节位置，故按真实位置写入
    tempo_desc = mex.apply_tempo(part, data, measure_offsets, bar_ql, options)
    inst_desc = mex.apply_instruments(part, data, spans, options)
    return part, {'tempo': tempo_desc, 'instrument': inst_desc}


def _export_spliced(data: ScoreData, segments: list, measure_offsets: dict,
                    bar_ql: float, options: dict):
    """拼接导出：各段首尾相接，抹平段间空隙（监听 / 导入 DAW 用）。"""
    from music21 import stream

    part = stream.Part()
    part.partName = '主旋律线（逐段追踪 · 无缝拼接）'
    spans = []
    cursor = 0.0
    for seg in segments:
        s, e = mex.seg_range(data, measure_offsets, bar_ql,
                             seg['start_measure'], seg['end_measure'])
        span = max(e - s, 0.0)
        spans.append({'offset': cursor, 'part_index': seg['part_index']})
        src = data.parts[seg['part_index']]
        for ev in mex.events_in(src, s, e):
            # 时值裁剪到段尾，避免上一段的长音压到下一段头上
            part.insert(cursor + (ev.offset - s),
                        mex.make_note(ev, mex.clamped_duration(ev, e),
                                      pitches=_pitches_for(ev, seg)))
        cursor += span
    # 拼接后小节已与原曲错位，变速标记没有意义，退化为单一速度
    if options['tempo_mode'] == 'original':
        from music21 import tempo as m21tempo
        first_bpm = next((t['bpm'] for t in (data.tempos or []) if t.get('bpm')),
                         mex.DEFAULT_BPM)
        part.insert(0.0, m21tempo.MetronomeMark(number=float(first_bpm)))
        tempo_desc = f'原曲速度 {float(first_bpm):.0f} BPM（拼接后小节错位，按单一速度写入）'
    else:
        tempo_desc = mex.apply_tempo(part, data, measure_offsets, bar_ql, options)
    inst_desc = mex.apply_instruments(part, data, spans, options)
    return part, {'tempo': tempo_desc, 'instrument': inst_desc}


def build_export_parts(data: ScoreData, segments: list, options: dict,
                       measure_offsets=None, bar_ql=None, shape_tag: str = '',
                       run_tag: str = '', double_tag: str = '') -> list:
    """构建两种版本的导出 Part（**不落盘**）。

    返回 [{'key', 'stem', 'part', 'desc'}]：
      key  = 结果字典里的键（absolute / spliced，保持稳定）
      stem = 落盘文件名主干，带形态层标记 / 双音口径 / 节奏 / 音色 / 运行序号
             后缀（换选项、换口径、换一次运行就换文件名，避免同名覆盖后浏览器
             给缓存里的旧文件）

    "按当前设置直接下载"与"应用并重新导出"共用这一份，保证两条路导出一致。
    """
    if measure_offsets is None or bar_ql is None:
        measure_offsets, _, bar_ql = build_measure_map(data)
    built = []
    for kind, fn, stem in (('absolute', _export_absolute, 'melody_line_absolute'),
                           ('spliced', _export_spliced, 'melody_line_spliced')):
        part, desc = fn(data, segments, measure_offsets, bar_ql, options)
        built.append({'key': kind,
                      'stem': mex.stem_with_options(stem, options, shape_tag,
                                                    run_tag, double_tag),
                      'part': part, 'desc': desc,
                      'time_sig': mex.source_time_sig(data)})
    return built


def export_segments(data: ScoreData, piece, segments: list, options: dict,
                    measure_offsets=None, bar_ql=None, shape_tag: str = '',
                    run_tag: str = '', double_tag: str = '') -> dict:
    """按段导出两种版本（供主入口与"重新导出"接口共用）。"""
    out = {}
    for b in build_export_parts(data, segments, options, measure_offsets, bar_ql,
                                shape_tag, run_tag, double_tag):
        files = save_stream(piece, mex.finalize_part(b['part'], b.get('time_sig')),
                            b['stem'])
        files['options'] = dict(options)
        files['tempo_desc'] = b['desc']['tempo']
        files['instrument_desc'] = b['desc']['instrument']
        out[b['key']] = files
    return out


# ---------------------------------------------------------------- 主入口

def melody_line(data: ScoreData, config, piece, voice_out: dict,
                options: dict | None = None,
                run_seq: int | None = None) -> dict:
    """主旋律线追踪主入口（逐段，MIR 口径）。

    依赖 voice_id 的显著性矩阵与谱面的人声标记；**不依赖** polyphony / exchange，
    因此"是否复调""是否交接"都不会改变本步骤的形态。
    """
    sal = voice_out['salience']
    matrix = sal.get('matrix', [])
    measure_offsets, _, bar_ql = build_measure_map(data)
    voice_priority = bool(getattr(config, 'melody_voice_priority', True))
    min_bars = max(int(getattr(config, 'melody_segment_min_bars', 2) or 1), 1)
    hi = float(getattr(config, 'melody_confidence_high', 0.25))
    mid = float(getattr(config, 'melody_confidence_mid', 0.10))
    # 形态资格层（可选，默认关）：开时才在段级做一次"有没有资格"的否决
    shape_gate = bool(getattr(config, 'melody_shape_gate', False))
    shape_min = float(getattr(config, 'melody_shape_min_score', 0.35) or 0.35)
    shape_min = max(0.0, min(1.0, shape_min))
    # 双音口径（默认开）：开 → 一个事件几个音就导出几个音，双音 / 加厚旋律
    # 原样保留；关 → 每个多音事件只留主音那一个音，由 principal 层挑。
    keep_doubles = bool(getattr(config, 'melody_keep_doubles', True))
    # DeepSeek 兜底裁定（默认开）：只在本地判据咬得太紧时才发问；关了就一次
    # 网络请求都不发。budget 跨窗累计，用完就不再问 —— 大曲子几十个窗全去问
    # 既慢又贵，还容易让模型在后半程敷衍（见 melody_judge 模块顶部第 3 条）。
    llm_judge = bool(getattr(config, 'melody_llm_judge', True))
    judge_budget = [judge_mod.JUDGE_MAX_CALLS] if llm_judge else [0]
    judge_cases = []
    options = mex.resolve_options(config, options)
    # 文件名标记：形态层开没开 + 双音口径 + 这是第几次运行（每次落盘的文件名
    # 都不同，重跑不覆盖上一次的产物，历史归档里每一次都能下到它自己那份）
    shape_tag = mex.shape_stem_tag(shape_gate, shape_min)
    double_tag = mex.double_stem_tag(keep_doubles)
    if run_seq is None:
        run_seq = mex.next_run_seq(piece, 'melody_line')
    run_tag = mex.run_stem_tag(run_seq)

    windows = windows_of_measures(data, config.analysis_window_bars)
    if not windows:
        windows = [{'window': 1, 'start_measure': 1,
                    'end_measure': max(data.measure_count, 1)}]

    # ---- ① 窗级归属（人声掩膜 + 显著性） ----
    # 按 start_measure 取窗，不按下标：matrix 是 voice_id 用**当时**的窗口长度
    # 算的，本步只依赖 voice_id，改过窗口设置后单步重跑会复用旧矩阵，
    # 按下标取会让"窗边界"和"显著性"对不上，导出取到错误小节的音符。
    #
    # 更麻烦的是矩阵可能**整体偏短**：声部识别早先是按"小节结构只有 32 小节"
    # 的旧谱解析跑出来的（首声部提前收尾导致小节表偏短），本步重新解析得到
    # 66 小节，后半截的窗在矩阵里一个都查不到。以前遇到这种情况直接按静默
    # 处理 → 半首曲子凭空消失（覆盖率 32/66）。现在改成**就地补算**：
    # 用 voice_id 同一套 window_scores 现算，算法一致、结果可比，
    # 用户也不必记得"先去重跑声部识别"。
    from . import voice_id as voice_mod      # 懒加载：避免模块级循环依赖
    low = voice_mod.low_register_parts(data)
    by_note = voice_mod.density_by_note_count(config)   # 密度口径（默认按发音数）
    # 力度要不要算进"听感"（默认开）：关掉之后整条链路都看不见力度数据 ——
    # 不是"不让它投票"，而是把它置成活在"谱面没标力度"的状态，
    # 这样低音区保护、显著性裁决两处的判定会自然跳过，不会漏掉某处。
    dyn_on = voice_mod.dynamics_aware(config)
    matrix_bars = sal.get('window_bars')
    # 矩阵是不是按**当前设置**算出来的：窗长、密度口径任一对不上就整表重算
    # （老产物没有 density_by_note_count 字段 → 视为 True，正是当时的算法）。
    stale_all = (matrix_bars is not None
                 and int(matrix_bars) != int(config.analysis_window_bars))
    if bool(sal.get('density_by_note_count', True)) != by_note:
        stale_all = True
    by_start = ({} if stale_all
                else {int(r.get('start_measure', -1)): r for r in matrix})
    missing = [w['start_measure'] for w in windows
               if w['start_measure'] not in by_start]
    # 只缺个别窗也**整表重算**：留下一半旧值一半新值更糟 —— 旧值是按另一份
    # 谱面解析（小节表偏短）算出来的，同一个声部在两套口径下的窗边界不同，
    # 拼在一起会出现"这一段归它、下一段也归它、中间却断了"的怪结果。
    if missing:
        by_start = {}
    win_rows = []
    win_scores = {}          # {start_measure: {part_index: salience}}，段级形态层用
    matrix_filled = sorted({w['start_measure'] for w in windows}
                           if (missing or stale_all) else set())
    for w in windows:
        row = by_start.get(w['start_measure'])
        m_start = measure_offsets.get(w['start_measure'], 0.0)
        m_end = measure_offsets.get(w['end_measure'] + 1)
        if m_end is None:
            m_end = m_start + (w['end_measure'] - w['start_measure'] + 1) * bar_ql
        per_part = None
        if row is None:
            # 矩阵里没有这个窗 → 按同一套算法就地补算
            per_part = voice_mod.window_scores(data, m_start, m_end,
                                               by_note_count=by_note)['per_part']
            row = {'scores': {str(d['part_index']): d['salience']
                              for d in per_part if not d.get('empty')}}
        win_scores[w['start_measure']] = {
            int(k): float(v) for k, v in (row.get('scores') or {}).items()}
        # 齐奏加倍：同一条旋律由多件乐器同奏（颤音琴 + 钟琴、小提琴 I + 长笛…）
        # 在听觉上是一条线索，不该让它们互相"竞争"把置信度压成 0。
        doubles = voice_mod.doubling_groups(
            voice_mod.window_signatures(data, m_start, m_end))
        choices = _window_choices(row, data, low, doubles)
        # 低音区保护与力度裁决要看窗级的原始字典（substance / dyn）。
        # 必须**按当前密度口径现算**，不能让它在别处按默认口径再算一遍 ——
        # 两条路算出两套 substance，同一个窗会有两种归属。
        if per_part is None and (low or len(choices) > 1):
            per_part = voice_mod.window_scores(data, m_start, m_end,
                                               by_note_count=by_note)['per_part']
        # 关掉力度开关 → 把 dyn 抹成"谱面没标力度"，下游两条判定自动跳过。
        # 改成新 dict 而不是就地改 per_part：那份还要原样留在窗级结果里。
        dyn_part = per_part if dyn_on else (
            None if per_part is None
            else [{**d, 'dyn': None, 'dyn_support': 0.0} for d in per_part])
        # 低音区保护：整窗上方没有更像旋律的候选时，低音线条恢复竞选资格
        ovr = (low_register_overrides(data, choices, m_start, m_end,
                                      per_part=dyn_part) if low else {})
        best, margin = _pick(choices, voice_priority, ovr, per_part=dyn_part)
        # ---- DeepSeek 兜底裁定（可选）：本地判据咬得太紧时才问 ----
        judge = None
        # 开关关着连 `is_hard` 都不查：一次网络请求都不发，也不必记一堆空案例
        if llm_judge and judge_mod.is_hard(choices, margin):
            local_margin = round(margin, 4)
            judge = judge_mod.judge_window(
                data, config, w, choices, dyn_part, m_start, m_end,
                best['part_index'], margin, judge_budget)
            if judge.get('applied') and judge.get('pick') != best['part_index']:
                new = next(c for c in choices
                           if c['part_index'] == judge['pick'])
                conf = judge.get('confidence')
                # 置信度：置信度仍在"I 还记得我有多不确定"上 —— 改判的依据是
                # 模型的一句音乐判断，不是新的算术领先，所以保守地取本地领先
                # 与模型自报置信度的较大者，不把它抬到"很确定"。
                if isinstance(conf, (int, float)):
                    margin = round(max(margin, max(0.0, min(1.0, float(conf)))), 4)
                best = new
            judge_cases.append({
                'window': w['window'],
                'start_measure': w['start_measure'],
                'end_measure': w['end_measure'],
                'local_part_index': judge['prev'],
                'local_part_name': _window_choices_label(data, judge['prev']),
                'part_index': best['part_index'],
                'part_name': best['part_name'],
                'margin': round(margin, 4),
                'local_margin': local_margin,
                'called': judge['called'],
                'applied': judge['applied'],
                'reason': judge['reason'],
                'confidence': judge['confidence'],
                'candidates': [{
                    'part_index': c['part_index'],
                    'part_name': c['part_name'],
                    'salience': c['salience'],
                    'is_voice': c['is_voice'],
                    'low_register': c['low_register'],
                    'note_count': next(
                        (d.get('note_count') for d in (dyn_part or [])
                         if int(d.get('part_index', -1)) == c['part_index']),
                        None),
                    'mean_pitch': next(
                        (d.get('mean_pitch') for d in (dyn_part or [])
                         if int(d.get('part_index', -1)) == c['part_index']),
                        None),
                } for c in choices[:judge_mod.JUDGE_MAX_PARTS]],
            })
        win_rows.append({
            'window': w['window'],
            'start_measure': w['start_measure'],
            'end_measure': w['end_measure'],
            'part_index': best['part_index'] if best else None,
            'part_name': best['part_name'] if best else '（无发声声部）',
            'is_voice': bool(best['is_voice']) if best else False,
            'voice_sounding': any(c['is_voice'] for c in choices),
            'doubled_with': (best.get('doubled_with') or []) if best else [],
            'low_override': bool(best and best['part_index'] in ovr),
            'low_override_shape': ovr.get(best['part_index']) if best else None,
            'margin': margin,
            'salience': best['salience'] if best else 0.0,
        })

    # ---- ② 连续同归属合并为 run ----
    runs = []
    for row in win_rows:
        if row['part_index'] is None:
            continue                       # 整窗静默 → 记为间隙，不产生段
        if runs and runs[-1]['part_index'] == row['part_index']:
            runs[-1]['end_measure'] = row['end_measure']
            runs[-1]['_rows'].append(row)
        else:
            runs.append({
                'part_index': row['part_index'],
                'part_name': row['part_name'],
                'is_voice': row['is_voice'],
                'start_measure': row['start_measure'],
                'end_measure': row['end_measure'],
                '_rows': [row],
            })
    for r in runs:
        r['avg_margin'] = round(
            sum(x['margin'] for x in r['_rows']) / len(r['_rows']), 4)

    # ---- ③ 碎片合并 + 段级统计 ----
    runs = _merge_short_runs(runs, min_bars)
    segments = []
    shape_replacements = []
    principal_cases = []
    for i, r in enumerate(runs):
        s, e = mex.seg_range(data, measure_offsets, bar_ql,
                             r['start_measure'], r['end_measure'])
        # ---- ③b 形态资格层（可选）：形态定"资格"，显著性定"强弱" ----
        owner = r['part_index']
        shape_info = None
        replaced_from = None
        shape_reason = ''
        if shape_gate:
            ranked = _seg_ranked_salience(
                win_scores, r['start_measure'], r['end_measure'], owner)
            owner, shapes, shape_reason = apply_shape_gate(
                data, s, e, ranked, owner, shape_min)
            shape_info = shapes.get(owner) or {}
            if owner != r['part_index']:
                replaced_from = r['part_index']
                shape_replacements.append({
                    'measures': f"{r['start_measure']}–{r['end_measure']}",
                    'from': data.parts[r['part_index']].name,
                    'to': data.parts[owner].name,
                    'reason': shape_reason,
                })
        notes = mex.events_in(data.parts[owner], s, e)
        # 主音判定（只在"不保留双音"时做）：这一段每个同时响多个音的事件，
        # 主旋律在唱的是哪一个。保留双音时整停用 —— 一个事件的音一个不少。
        pr = (principal_mod.resolve_principal(
                  data, owner, s, e, measure_offsets, bar_ql, config)
              if not keep_doubles else None)
        pr_brief = None
        if pr is not None:
            pr_brief = {
                'resolved_by': pr.get('resolved_by'),
                'chosen_rank': pr.get('chosen_rank'),
                'margin': pr.get('margin'),
                'chord_event_count': pr.get('chord_event_count'),
                'small_sample': pr.get('small_sample'),
                'map': {str(r['offset']): r['principal']
                        for r in pr.get('per_event', [])
                        if r.get('principal') is not None},
            }
            principal_cases.append({
                'segment': f"{r['start_measure']}–{r['end_measure']}",
                'part_name': data.parts[owner].name,
                'resolved_by': pr.get('resolved_by'),
                'chosen_rank': pr.get('chosen_rank'),
                'margin': pr.get('margin'),
                'chord_event_count': pr.get('chord_event_count'),
                'small_sample': pr.get('small_sample'),
                'layers': pr.get('layers') or [],
                'llm': pr.get('llm'),
                'events': [row for row in pr.get('per_event', [])
                           if len(row.get('pitches') or []) >= 2],
            })
        rows = [x for x in win_rows
                if x['part_index'] is not None
                and r['start_measure'] <= x['start_measure'] <= r['end_measure']
                and x['part_index'] == r['part_index']]
        margin = (round(sum(x['margin'] for x in rows) / len(rows), 4)
                  if rows else r.get('avg_margin', 0.0))
        # 形态层换过人 → 按新声部算人声先验（原来那段的人声 prior 不再适用）
        is_voice = bool(data.parts[owner].is_voice)
        # 人声优先是强先验：同样的 margin，人声段更有把握
        conf = round(min(1.0, margin + 0.15), 4) if is_voice else margin
        segments.append({
            'seq': i + 1,
            'start_measure': r['start_measure'],
            'end_measure': r['end_measure'],
            'bars': r['end_measure'] - r['start_measure'] + 1,
            'part_index': owner,
            'part_name': data.parts[owner].name,
            'instrument': data.parts[owner].instrument or '（未标注）',
            'is_voice': is_voice,
            'note_count': len(notes),
            'margin': margin,
            'confidence': conf,
            'level': _level(conf, hi, mid),
            'absorbed_from': sorted(set(r.get('absorbed', []))),
            'shape': shape_info,
            'shape_replaced_from': replaced_from,
            'shape_reason': shape_reason,
            # 双音口径写进每段：重导出时不重新判定，沿用这一次跑出来的主音表
            'keep_doubles': keep_doubles,
            'principal': pr_brief,
        })

    # ---- ④ 覆盖率与掩膜摘要 ----
    covered = sum(s['bars'] for s in segments)
    total_bars = max(data.measure_count, 1)
    voice_parts = [{'index': p.index, 'name': p.name} for p in data.parts if p.is_voice]

    method = ('每个时间窗按"人声在唱→人声；人声休止→最突出的器乐声部"定归属，'
              '连续同归属的窗合并为段；不受交接检测门控，'
              '前奏 / 间奏 / 尾奏同样提取。')
    if shape_gate:
        method += ('另加一道形态资格筛选：段内声部的旋律轮廓形态分'
                   f'（级进比×0.7 + 拱形度×0.15 + 轮廓平滑度×0.15）低于 {shape_min} '
                   '就失去资格，由形态合格且显著性最高的声部顶上；'
                   '音符太少不足以统计时不作判定，所有声部都不合格时保留原归属。')
    # 齐奏加倍：整条旋律线索由多件乐器同奏时说清楚，免得置信度看着"凭空变高"
    doubled = {}
    for r in win_rows:
        for j in r.get('doubled_with') or []:
            doubled.setdefault(r['part_name'], set()).add(data.parts[j].name)
    if doubled:
        method += ('齐奏加倍（' + '；'.join(
            f'{k} 与 {"、".join(sorted(v))} 同奏同一条旋律' for k, v in doubled.items())
            + '）视为同一条听觉线索，不计入竞争，置信度不因加倍被低估。')
    low_names = [data.parts[i].name for i in sorted(low)] if low else []
    # 力度：谱面标没标力度，决定了"听感"这条证据能不能用。把它写进结果，
    # 页面上才能交代清楚 —— 否则用户会以为系统压根没考虑力度。
    dyn_parts = []
    for p in data.parts:
        cov = p.dynamics_coverage
        if cov > 0:
            dyn_parts.append({'index': p.index, 'name': p.name,
                              'coverage': round(cov, 3),
                              'mean_velocity': p.mean_velocity})
    if low_names:
        # 主旋律在高音区：低音区声部不参与竞选
        method += (f'低音区声部（{"、".join(low_names)}：平均音高低于 C3 且比全曲'
                   '最高声部低 ≥2 个八度）不参与主旋律竞选；'
                   '整窗只剩低音在响时才保留它。')
    # 低音区保护：哪些低音声部是靠"确实更像一条旋律"翻盘进来的，必须写明白
    overrides = {}
    for r in win_rows:
        if r.get('low_override') and r.get('low_override_shape'):
            overrides.setdefault(r['part_name'], []).append(
                r['low_override_shape'])
    if overrides:
        detail = []
        for nm, lst in overrides.items():
            inf = lst[0]
            lead_dyn = inf.get('dyn_lead')
            dyn_tail = ''
            if lead_dyn is not None and lead_dyn >= DYNAMICS_LEAD:
                dyn_tail = f'、力度领先 {lead_dyn:.0f}/127（门槛相应放宽）'
            elif lead_dyn is not None and lead_dyn < -DYNAMICS_TOLERANCE:
                dyn_tail = (f'、但力度弱于对手 {abs(lead_dyn):.0f}/127'
                            '（不参与翻盘）')
            if inf.get('shape_lead') is None:
                # "上方全是踏板 / 持续音"这条分支不看轮廓门槛，但**照样看力度** ——
                # 写得比踏板还弱的线不可能靠"没有对手"就当上主旋律
                detail.append(f'{nm}：上方只剩踏板 / 持续音，没有成线的对手'
                              + dyn_tail)
                continue
            tail = (f'{nm}：轮廓领先 {inf["shape_lead"]:.2f}、'
                    f'去音高项领先 {inf["substance_lead"]:.2f}')
            detail.append(tail + dyn_tail)
        method += (f'低音区保护：{"、".join(overrides)} 虽在低音区，但'
                   + '；'.join(detail) + '，按"听感占优"承担了这些窗的主旋律。')
    result = {
        'mode': ('逐段追踪（人声掩膜 + 显著性 + 形态资格）' if shape_gate
                 else '逐段追踪（人声掩膜 + 显著性）'),
        'method': method,
        'window_bars': config.analysis_window_bars,
        'voice_priority': voice_priority,
        'min_segment_bars': min_bars,
        'confidence_thresholds': {'high': hi, 'mid': mid},
        'has_voice_parts': bool(voice_parts),
        'voice_parts': voice_parts,
        # 被判为"低音区、不当主旋律"的声部（不参与逐段竞选）
        'low_register_parts': sorted(low),
        'low_register_names': low_names,
        # 低音区里靠"更像一条旋律"被重新接纳的声部（见 low_register_overrides）
        'low_register_overridden': sorted(overrides),
        'dynamics': {
            # 开关（系统设置「主旋律判定考虑力度」，默认开）
            'enabled': dyn_on,
            # 谱面标没标力度：没标时 notated=False，整条力度链路自动失效，
            # 既不帮谁也不拦谁（不能把"没标"当成"力度相同"或"力度为零"）
            'notated': bool(dyn_parts) and dyn_on,
            'parts': dyn_parts if dyn_on else [],
            'min_support': DYNAMICS_MIN_SUPPORT,
            'lead': DYNAMICS_LEAD,
            'tolerance': DYNAMICS_TOLERANCE,
        },
        'shape_gate': shape_gate,
        'shape_gate_threshold': shape_min if shape_gate else None,
        'shape_replacements': len(shape_replacements),
        'shape_replacement_notes': shape_replacements,
        # 双音口径（系统设置「主旋律线保留双音」，默认开）
        #   keep_doubles=True  → 产物里一个事件几个音就几个音（和弦）
        #   keep_doubles=False → 只留主音那一个音，cases 里逐段记着怎么挑的
        'double': {
            'keep': keep_doubles,
            'tag': mex.double_label(keep_doubles),
            'enabled': not keep_doubles,      # 主音判定层这次有没有真的启用
            'cases': principal_cases,
            'llm_called': sum(1 for c in principal_cases
                              if (c.get('llm') or {}).get('called')),
        },
        # DeepSeek 兜底裁定（系统设置「判定困难时交给 DeepSeek」，默认开）：
        #   called  —— 真的发问了几个窗
        #   applied —— 其中被采纳、改动了归属的有几个
        'llm_judge': {
            'enabled': llm_judge,
            'asked': len(judge_cases),
            'called': sum(1 for c in judge_cases if c['called']),
            'applied': sum(1 for c in judge_cases if c['applied']),
            'budget': judge_mod.JUDGE_MAX_CALLS if llm_judge else 0,
            'margin': judge_mod.JUDGE_MARGIN,
            'windows': judge_cases,
        },
        'segments': segments,
        'coverage': {
            'total_bars': total_bars,
            'covered_bars': covered,
            'ratio': round(covered / total_bars, 4),
            'gap_bars': max(total_bars - covered, 0),
        },
        'windows': win_rows,
        'export_options': options,
        'export_options_label': mex.options_label(options),
        'export_run_seq': int(run_seq or 1),
        'exports': {},
    }
    # 显著性矩阵里查不到的窗已由本步就地补算（同一套算法），这里只做告知，
    # 不再让这些窗"凭空消失"。以前是"已按静默处理"，半首曲子会直接丢掉。
    result['matrix_filled'] = matrix_filled
    if matrix_filled:
        why = ('窗口长度已从 %s 小节改成 %s 小节' % (matrix_bars, config.analysis_window_bars)
               if stale_all else '声部识别早于最近一次谱面解析')
        result['window_warning'] = (
            f'声部识别的显著性矩阵与当前谱面对不上（{why}），'
            f'{len(matrix_filled)} 个窗的显著性已按同一套算法在本步就地重算，'
            '分段结果不受影响；若不放心可在「分析旋律」里重跑声部识别。')

    if not segments:
        result['note'] = '全曲没有可用于追踪的有音高事件（可能整谱静默或全为打击乐）。'
        return result

    # ---- ⑤ 导出（原位 + 无缝拼接，带节奏 / 音色选项） ----
    try:
        result['exports'] = export_segments(
            data, piece, segments, options,
            measure_offsets=measure_offsets, bar_ql=bar_ql,
            shape_tag=shape_tag, run_tag=run_tag, double_tag=double_tag)
    except Exception as exc:  # noqa: BLE001
        # 导出失败不该让整步失败：分段结果仍然有效
        result['export_error'] = f'{exc}'
        result['note'] = ('分段结果已生成，但 MIDI / MusicXML 导出失败'
                          '（分段表仍可使用，可点右上角 ⟲ 重跑本模块）。')

    # 复调提示：单一旋律线只是其中一条线索
    voices_present = len({s['part_index'] for s in segments})
    if voices_present > 1:
        result['note'] = (
            f'本曲在主旋律线上出现 {voices_present} 个不同声部的接力；'
            '若同时被判定为复调织体，这条线只是其中一条听觉线索，不是唯一主题。')
    return result
