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
5. **平线否决（恒生效，不是可选项）**：窗级候选先过「这是不是一条线」
   （:func:`is_flat_line`）—— 踏板音 / 柱式和弦伴奏取最高音后常常是一条平线，
   而它在显著性里靠"和弦取最高音"音高虚高、在形态里靠"音程全 0"拿满分
   （级进比 / 拱形度 / 平滑度三项齐满分 = 形态分 1.0），两条都是假证据。
   **但平线有两种，必须分开**（:func:`flat_kind`，判据是"一次响几个音"）：

   - **柱式和弦 / 踏板铺底**（多音事件占了一半以上）→ **剥夺资格**，不当主旋律；
   - **同音反复的单音线条** → 这很可能是**吟诵式旋律**：旋律性弱的流行歌、民歌
     的收句（非洲民歌《巴努瓦》最后一句整句唱 do）一大串音都在同一个音高上，
     它就是主旋律在唱。**只降级为存疑候选**（排到毫无疑点的候选之后），
     不剥夺资格 —— 没有更像旋律的候选时仍由它演唱。

   **人声豁免**：人在唱就是主旋律，这是掩膜级的强先验，比形态硬。
   这一条与下面第 6 条分得清清楚楚：第 6 条是可开关的加分项，
   本条是恒生效的资格闸，关掉形态层它照样在；
   本地在 chord / chant 上拿不准时（两者显著性咬得紧），如果开了
   「判定困难时交给 DeepSeek」，会把这一整窗连同这个分歧一起发给模型
   （见 :func:`melody_judge.line_dispute`，第二种发问场合）；
6. **形态资格层（可选，默认关，在账户总体设置里开）**：形态决定「有没有资格」，
   显著性决定「谁更强」—— 形态只做**否决**，不参与加分、不改 margin。
   段级算出各声部的旋律轮廓形态分（级进比 / 拱形度 / 轮廓平滑度），
   当前归属声部形态不合格时，由段内形态合格且显著性最高的声部顶上。
   用来挡住分解和弦、阿尔贝蒂低音、固定音型这类「周期音型」被误判成主旋律 ——
   它们的动机重复度接近 1.0，单看显著性很容易赢过真正的人声旋律。
   两条安全设计：① 段内音符太少不足以统计轮廓时**不作判定、一律放行**；
   ② 所有声部都不合格时**保留原归属**，绝不把整段丢掉；
   柱式和弦那种假满分（第 5 条）在这里同样被否决，堵住"形态 1.0 的踏板把真
   旋律顶掉"的漏洞；而同音反复的吟诵**不在这里否决** —— 它形态满分说的是
   "一直在同一个音上"，不是"它是伴奏"；
   形态层换人时若接替者的显著性比原归属差一大截，这一换是全凭形态孤证，
   开了 DeepSeek 就复核一句（:func:`melody_judge.shape_dispute`）；
7. 导出两种：
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
                    low: set = frozenset(), doubles: dict | None = None,
                    seqs: dict | None = None,
                    ratios: dict | None = None) -> list:
    """一个窗的 {part_index: salience} → 按显著性降序的候选列表。

    ``low`` 是低音区声部（比全曲最高声部低 ≥ 2 个八度），在这里**只打标记**
    不剔除：主旋律在高音区，选人时排除它们（见 :func:`_pick`），但算置信度
    的 margin 仍把它们算作竞争对手 —— 否则"贝斯被排除"会让置信度虚高到
    满分，反而看不出这段到底有多确定。

    ``doubles`` 是齐奏加倍分组（见 :func:`voice_id.doubling_groups`）：同一条
    旋律被多件乐器同奏时它们算**同一条线索**，互相之间不构成竞争。

    ``seqs`` 是 {part_index: 音高序列}（和弦取最高音，同 :func:`_seg_pitch_seq`）；
    ``ratios`` 是 {part_index: 多音事件占比}（同 :func:`chord_ratio`）。
    两个都传就给每个候选打上 ``flat_line`` 与 ``flat_kind`` 标记
    （见 :func:`is_flat_line` / :func:`flat_kind`）—— 分开两种平线：
    **柱式和弦伴奏** 失去资格，**同音反复的吟诵** 只是降级为存疑候选。
    **不传则一律不打标记** —— 只调这一个函数、没算序列的场合（判定层、测试）
    行为与以前逐位一致，不会凭空多出一道闸。
    """
    doubles = doubles or {}
    seqs = seqs or {}
    ratios = ratios or {}
    items = []
    for idx, sc in matrix_row.get('scores', {}).items():
        i = int(idx)
        group = doubles.get(i, i)
        peers = sorted(j for j, g in doubles.items() if g == group and j != i)
        seq = seqs.get(i) or []
        kind = flat_kind(seq, ratios.get(i, 0.0)) if seq else 'none'
        items.append({
            'part_index': i,
            'part_name': data.part_names[i] if i < len(data.part_names) else f'Part {i}',
            'salience': float(sc),
            'is_voice': data.parts[i].is_voice if i < len(data.parts) else False,
            'low_register': i in low,
            # 平线（踏板音 / 柱式持续音 / 同音反复）：见 _pick 的三档处理
            'flat_line': kind != 'none',
            # 'chord' 确定是伴奏（多音事件为主）→ 出局；
            # 'chant' 很可能是吟诵式旋律（单音反复）→ 只降级，见 _pick
            'flat_kind': kind,
            'chant_suspect': kind == 'chant',
            'chord_ratio': ratios.get(i, 0.0),
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


# ------------------------------------------ 平线的**性质**：和弦铺底 vs 吟诵
# 「连着好几个音都是同一个音高」有两种完全不同的东西，平线闸必须分开对待：
#
#   * **柱式和弦 / 踏板铺底（伴奏）**：一个起音同时响好几个音，取最高音之后
#     变成一条平线。que sera sera 的 PIZZICATO 就是它（三连音柱式和弦，
#     最高音恒为 E5）。这种**确定**是伴奏，没有资格当主旋律。   * **吟诵 / 同音反复（旋律）**：旋律性弱的流行歌、民歌的收句，
#     一大串音唱在同一个音高上 —— 非洲民歌《巴努瓦》的最后一句全是 do。
#     这条线虽然"平"，**它就是主旋律**，不能让给伴奏。
#
# 区分两者的判据是**织体**，不是轮廓：柱式和弦是"一次响几个音"，吟诵是"一个音
# 一个音地唱"。所以用「多音事件占全部发音事件的比例」来判 —— 这个比例 requiere
# 不需要知道和弦里哪个音在唱，也不依赖 ":func:`principal`" 那一层。
FLAT_CHORD_RATIO = 0.5      # 多音事件占比 ≥ 此值 → 判为和弦铺底（不是吟诵）


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

    **只在"要不要给好处"的方向上用**（低音区翻盘）：音数不足时它也判 False，
    也就是"证据不足 = 不许占便宜"。反过来要"剥夺资格"时必须用
    :func:`is_flat_line`，那里的证据不足是**放行**。
    """
    if len(set(seq)) < LOW_MELODY_MIN_PITCHES:
        return False
    return (max(seq) - min(seq)) >= LOW_MELODY_MIN_RANGE


# ------------------------------------------------ "是不是一条线"（平线否决）
# 与低音区保护对称的另一半。低音区早就有"太低的别来竞选"，高音区却一直缺
# "只在一个音上打点的别来竞选" —— 于是柱式和弦伴奏 / 踏板音一路畅通：
#
#   * 显著性四项里，和弦一律 **取最高音**（voice_id.window_scores 与
#     _seg_pitch_seq 都是 max()），柱式和弦因此音高虚高一个八度；
#   * 取最高音后若和声没换，序列是 E5, E5, E5… —— 音程全 0，**动机重复满分**；
#   * 更糟的是它对形态三项同样满分：级进比 1.0（全是 0）、拱形度 1.0（每个点
#     都是最高点）、平滑度 1.0（没有极值点）→ **形态分 1.0，比真旋律还高**。
#     形态闸门本该挡住伴奏，实际会反过来把主旋律让给它。
#
# 所以这里把"是不是一条线"从低音区翻盘那条**支路**提到**主选路径**上：
# 窗级候选先过这道闸，平线失去当主旋律的资格；形态闸门（shape_eligible）
# 也接同一道闸，堵住上面那个 1.0 分的公式漏洞。
#
# 证据不足时**不否决**：窗内只有两三个音的短句本来就可能只有两个音高，凭这
# 条把它划掉是误伤。音数门槛与轮廓判定同口径（:data:`SHAPE_MIN_NOTES`）。
#
# **只看"有几个不同音高"，不看跨度** —— 这一条是踩出来的：跨度门槛（4 半音）
# 在"给好处"的方向上是合适的，在"剥夺资格"的方向上太激进。实测
# `smoke_low_register_test` 用例 A 的大提琴只有四个音（50, 52, 53, 52）：
# 三个不同音高、跨度 3 半音 —— 它是全曲最高的那条线、本来就该当选，加上跨度
# 条件后它被判成平线出局，主旋律反而落到恰好跨度 4 的低音提琴上。
# 踏板音与柱式和弦的特征本来就是"翻来覆去只有一两个音"，**不同音高数**这一条
# 已经把它们说死了，不需要跨度再来补一刀。
def is_flat_line(seq: list) -> bool:
    """这条序列**确定**不是一条旋律线（踏板音 / 柱式持续音 / 单音打点）。

    ``False`` 有两种含义 —— 像一条线，**或者证据不足（音太少）**：两者都放行，
    本闸只做否决、不做好处。这正是它与 :func:`looks_like_line` 的分工：
    后者在音少时也判 False（低音区翻盘宁可错杀，不许靠证据不足占便宜），
    而且它还要再看跨度 —— 门槛更严，因为它决定的是"要不要给好处"。
    """
    if len(seq) < SHAPE_MIN_NOTES:
        return False
    return len(set(seq)) < LOW_MELODY_MIN_PITCHES


def chord_ratio(data: ScoreData, part_index: int, s: float, e: float) -> float:
    """该声部在 [s, e) 内**多音事件占全部发音事件的比例**（0–1）。

    柱式和弦伴奏一次起音响好几个音 → 接近 1；一个音一个音地唱的吟诵 → 接近 0。
    这个比值只看"一次响几个音"，不需要先知道"和弦里哪个音在唱"，
    所以它能在**选轨层**（还没挑主音之前）就把两种平线分开。
    """
    evs = mex.events_in(data.parts[part_index], s, e)
    if not evs:
        return 0.0
    multi = sum(1 for ev in evs if len(ev.midi_pitches or []) > 1)
    return round(multi / len(evs), 4)


def flat_kind(seq: list, ratio: float) -> str:
    """平线是**哪一种**平线：``'none'`` / ``'chord'`` / ``'chant'``。

    ``ratio`` 见 :func:`chord_ratio`（多音事件占比）：

    - ``'chord'`` —— 柱式和弦 / 踏板铺底，**确定**是伴奏，失去主旋律资格；
    - ``'chant'`` —— 同音反复的旋律（吟诵 / 弱旋律流行歌 / 民歌收句）。
      这是主旋律在唱，不许剥夺它的资格；它只是"看起来像伴奏"，
      所以降级为**存疑候选**（排在没有疑点的候选之后），
      咬得紧时交给模型裁决（见 :func:`melody_judge.line_dispute`）。

    判据落在**织体**上（一次响几个音），不落在轮廓上 —— 轮廓会把两者等量齐观。
    """
    if not is_flat_line(seq):
        return 'none'
    return 'chord' if (ratio or 0.0) >= FLAT_CHORD_RATIO else 'chant'


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


def line_counterpart(seq: list, ratio: float = 0.0) -> bool:
    """这条线配不配当"低音区翻盘"要超越的**对手**。

    低音能翻盘的前提之一是"上面那条线不成线"。但"不成线"有两种平法，
    只有一种是真的没人在唱：

    * **柱式和弦 / 踏板铺底**（``'chord'``）—— **不算对手**。上方只剩和声铺底
      时，低音主奏本来就该翻盘（这条路径设计出来就是为了抓这个）；
    * **同音反复的吟诵**（``'chant'``）—— **算对手**。它在唱歌，没有给谁让路。
      不认它的话，低音会因为"上面只有一条平线"自动翻盘，把旋律挤走 ——
      《巴努瓦》式收句就栽在这里：旋律整句唱 do，低音靠"没有对手"上位。

    ``ratio`` 见 :func:`chord_ratio`。
    """
    kind = flat_kind(seq, ratio)
    if kind == 'chord':
        return False
    if kind == 'chant':
        return True
    return looks_like_line(seq)


def low_register_overrides(data: ScoreData, items: list, s: float, e: float,
                           per_part: list | None = None,
                           seqs: dict | None = None,
                           ratios: dict | None = None) -> dict:
    """窗内哪些低音区声部该因"听感占优"回来当主旋律。

    返回 ``{part_index: {'shape': 形态分, 'shape_lead': 领先多少,
    'substance_lead': 去音高项领先多少, 'dyn_lead': 平均力度差（None=没数据）}}``；
    不该翻盘时返回空字典。

    ``items`` 是窗内候选（同 :func:`_window_choices` 的形状）；``per_part``
    可直接传 :func:`voice_id.window_scores` 的结果，省掉一次重算；``seqs``
    可直接传主入口已经算好的 {part_index: 音高序列}，``ratios`` 同理（多音事件
    占比，判"对手"的时候用得到）。不传就地算，结果一致。
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
    seqs = seqs or {}
    ratios = ratios or {}
    seqs = {c['part_index']: (seqs[c['part_index']]
                              if c['part_index'] in seqs
                              else _seg_pitch_seq(data, c['part_index'], s, e))
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
    # 上方候选里真正"是一条线在唱"的那些 —— 见 :func:`line_counterpart`：
    # 柱式和弦铺底**不算**对手，同音反复的吟诵**算**
    line_like = [c for c in others
                 if line_counterpart(seqs[c['part_index']],
                                     ratios.get(c['part_index'], 0.0))]
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


def candidate_tier(c: dict, override: dict | None = None) -> int:
    """候选在窗内的**资格档位**：0 正常竞选 / 1 存疑（降级）/ 2 出局。

    两道否决一起来的时候要分得清清楚楚：

    * **低音区只由低音区排除管**，平线闸不插手 —— 低音区排除挡的是"音区太低"
      （哪怕它是一条漂亮的走动贝斯），平线闸挡的是"根本不成线"（哪怕它音区很
      高）。让平线闸也去管低音，等于把踏板贝斯挡两遍，还会把
      :func:`low_register_overrides` 辛苦翻盘回来的低音线条又否掉（翻盘已经验过
      它"成线"了，不需要再验一次）。
    * **平线要先问是哪一种平线**（:func:`flat_kind`）。柱式和弦铺底（多音事件为
      主）出局；而**单音的同音反复**很可能就是主旋律 —— 吟诵、旋律性弱的流行歌、
      《巴努瓦》那样全唱 do 的收句 —— 它只是"看起来像伴奏"，降级为存疑候选，
      排到毫无疑点的候选之后，**不剥夺资格**。
    * **人声豁免**：人在唱就是主旋律，这是掩膜级的强先验，比形态硬
      （吟诵式的一段同音也是人在唱，不该让给伴奏）。

    缺 ``flat_kind`` 的老口径候选（测试里手工构造的 dict）按柱式和弦处理 ——
    与 2026-10-22 那一版逐位一致。

    档位 2 **不一定真的出局**：整窗候选全被否掉时 :func:`_pick` 会按显著性退回，
    绝不把整窗记成静默。
    """
    override = override or {}
    if c.get('low_register'):
        return 0 if c['part_index'] in override else 2
    if c.get('is_voice'):
        return 0
    if not c.get('flat_line'):
        return 0
    kind = c.get('flat_kind') or 'chord'
    return 2 if kind == 'chord' else 1


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

    def _tier(c) -> int:
        return candidate_tier(c, override)

    def _ok(c) -> bool:
        return _tier(c) < 2

    # 先由「毫无疑点」的候选竞选。**存疑的同音反复只在它压不住别人时才往后
    # 排** —— 显著性已经把话说清楚的时候不该拿"它平不平"去推翻：
    #   吟诵 0.75 对上一条显著性 0.25 的支声旋律，让 0.25 赢只因为它"更像旋律"，
    #   等于把"降级"用成了"剥夺"（用户提的《巴努瓦》收句、弱旋律流行歌正是这种
    #   局面）。差距 ≤ JUDGE_FLAT_BAND 才是本地真的分不清 —— 那也是
    #   :func:`melody_judge.line_dispute` 判"该去问模型"的同一条线，两处必须同口径。
    clean = [c for c in choices if _tier(c) == 0]
    demoted = [c for c in choices if _tier(c) == 1]
    blocked = [c for c in choices if _tier(c) == 2]
    if clean:
        ceiling = max(c['salience'] for c in clean) + judge_mod.JUDGE_FLAT_BAND
        keeps = [c for c in demoted if c['salience'] >= ceiling]
        eligible = sorted(clean + keeps, key=lambda c: -c['salience'])
    else:
        # 一个毫无疑点的都没有：退到存疑档；连存疑档也空（整窗全是柱式和弦
        # 平线）才退回全体 —— 绝不把整窗记成静默。
        eligible = demoted or blocked or choices

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


def contour_shape(seq: list, ratio: float = 0.0) -> dict:
    """音高序列 → 旋律轮廓形态分（0–1，越大越像"一条旋律"）。

    ``ratio`` 是同一段的多音事件占比（:func:`chord_ratio`），用来把平线分成
    "柱式和弦铺底"与"单音同音反复"，两种的处理方式完全不同（见 :func:`shape_eligible`）。
    不传则按单音处理（= 不剥夺资格）。

    ``score`` 为 None = "音符太少，不作判定"，调用方必须按**放行**处理。
    """
    n = len(seq)
    if n < SHAPE_MIN_NOTES:
        return {'note_count': n, 'score': None, 'flat': False,
                'flat_kind': 'none', 'reason': f'仅 {n} 个音，不足以判定轮廓'}
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
    okind = flat_kind(seq, ratio)
    return {
        'note_count': n,
        # 平线要分清是哪一种：chord（柱式和弦铺底）的形态满分是**公式漏洞**，
        # 由 shape_eligible 直接否决；chant（单音同音反复）的形态满分则要看
        # 它的织体是不是在唱歌，不能凭这一项剥夺资格。
        'flat': okind != 'none',
        'flat_kind': okind,
        'step_ratio': round(step_ratio, 4),
        'arch': round(arch, 4),
        'contour_smooth': round(smooth, 4),
        'score': score,
    }


def shape_eligible(shape: dict, min_score: float) -> bool:
    """形态是否够格当主旋律。

    两道闸，**平线那道在前**，但只针对**柱式和弦铺底**（``flat_kind == 'chord'``）：
    它的音程全是 0 → 级进比 1.0、拱形度近满分、平滑度 1.0 → 形态分接近满分。
    让它去比分数，它会赢过真旋律，形态闸门就成了"把主旋律让给伴奏"的通道。

    **单音的同音反复（``'chant'``）不在这里否决**：吟诵 / 弱旋律流行歌 / 《巴努瓦》
    那样全唱 do 的收句，形态三项同样是满分 —— 但那次满分说的不是"它是伴奏"，
    而是"它一直在同一个音上"，证据指向不同的结论。它的形态分本来就高于阈值，
    按分数线自然通过，不需要额外否决。

    score 为 None（音太少，证据不足）一律放行 —— 这道闸只做否决。
    """
    sc = shape.get('score')
    if sc is None:
        return True
    kind = shape.get('flat_kind')
    if kind is None:                      # 老产物没有这个字段
        kind = 'chord' if shape.get('flat') else 'none'
    if kind == 'chord':
        return False
    return sc >= min_score


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
    # 形态分要用**窗级同一套口径**：序列取最高音、平线再按多音事件占比分
    # chord / chant（contour_shape 不传 ratio 时会把吟诵当成柱式和弦处理，
    # 平线闸的效果就在段级悄悄换了一套）。
    shapes = {idx: contour_shape(_seg_pitch_seq(data, idx, s, e),
                                 chord_ratio(data, idx, s, e))
              for idx, _ in ranked}
    for sh in shapes.values():
        sh['eligible'] = shape_eligible(sh, min_score)
    cur = shapes.get(current_idx)
    if cur is None or cur['eligible']:
        return current_idx, shapes, ''
    cur_name = data.parts[current_idx].name
    if cur.get('flat_kind') == 'chord':
        # 平线的形态分是 1.0，写"低于下限 0.35"会自相矛盾。这里必须说清
        # 它是**假满分**：音程全是 0 → 级进比 / 拱形度 / 平滑度三项齐满分。
        seq = _seg_pitch_seq(data, current_idx, s, e)
        why = (f"是一条平线（{cur['note_count']} 个音里只有 {len(set(seq))} 个"
               f"不同音高、跨度 {max(seq) - min(seq)} 半音）—— 柱式和弦 / 踏板"
               f"持续音的音程全是 0，级进比 {cur['step_ratio']} / 拱形 "
               f"{cur['arch']} / 平滑 {cur['contour_smooth']} 三项齐满分，"
               f"形态分 {cur['score']} 是假满分")
    else:
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


def _seg_judge_choices(data: ScoreData, ranked: list) -> list:
    """段级形态闸门要问模型时，把 :func:`_seg_ranked_salience` 的结果摊成
    :func:`melody_judge.judge_window` 认得的 choices 形状。"""
    out = []
    for idx, sc in ranked:
        p = data.parts[idx] if idx < len(data.parts) else None
        out.append({
            'part_index': idx,
            'part_name': (data.part_names[idx] if idx < len(data.part_names)
                          else f'Part {idx}'),
            'salience': float(sc),
            'is_voice': bool(getattr(p, 'is_voice', False)),
            'low_register': False,
            'group': idx,
            'doubled_with': [],
        })
    out.sort(key=lambda c: -c['salience'])
    return out


def _seg_judge_stats(data: ScoreData, ranked: list, s: float, e: float) -> list:
    """段级候选的极简统计。

    给模型看的数字必须有出处 —— 段级没有窗级的 per_part，全部写成 0 等于
    让它在一片空白里猜。这里补齐发音数 / 平均音高 / 多音事件占比三项。
    """
    rows = []
    for idx, _sc in ranked[:judge_mod.JUDGE_MAX_PARTS]:
        seq = _seg_pitch_seq(data, idx, s, e)
        if not seq:
            continue
        rows.append({'part_index': idx, 'note_count': len(seq),
                     'mean_pitch': round(sum(seq) / len(seq), 1),
                     'chord_ratio': chord_ratio(data, idx, s, e)})
    return rows


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
        # 窗内各声部的音高序列（和弦取最高音）：平线闸与低音区翻盘**共用同一份**。
        # 两个地方各算一遍就会得到两套结果（与"显著性必须按同一口径现算"是同一
        # 个道理）—— 一边说它是平线、一边拿它当标尺，是最难查的那种错。
        seqs = {int(k): _seg_pitch_seq(data, int(k), m_start, m_end)
                for k in (row.get('scores') or {})}
        # 多音事件占比：把"柱式和弦伴奏的平线"与"吟诵式旋律的平线"分开的依据，
        # 与 seqs 同一套口径现算（理由同 seqs）。
        ratios = {int(k): chord_ratio(data, int(k), m_start, m_end)
                  for k in (row.get('scores') or {})}
        choices = _window_choices(row, data, low, doubles, seqs=seqs,
                                  ratios=ratios)
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
                                      per_part=dyn_part, seqs=seqs,
                                      ratios=ratios)
               if low else {})
        best, margin = _pick(choices, voice_priority, ovr, per_part=dyn_part)
        # ---- DeepSeek 兜底裁定（可选）：本地判据咬得太紧时才问 ----
        # 两种问法（见 melody_judge.build_judge_task）：
        #   kind='tie'  —— 两条候选显著性咬住，模型给它们重排；
        #   kind='line' —— **本地判据自己打起来了**："是不是一条线"这道闸把某个
        #                  候选剔除 / 降级，而它的显著性与当选者咬得很近。这是
        #                  2026-10-23 加的 —— 平线分得再细也有认缴的地带
        #                  （旋律性弱的流行歌、民歌收句整句同音），这时候问人比
        #                  问公式可靠。
        # 开关关着连 `is_hard` / `line_dispute` 都不查：一次网络请求都不发，
        # 也不必记一堆空案例。
        judge = None
        disp = None
        kind = None
        if llm_judge and best is not None:
            if judge_mod.is_hard(choices, margin):
                kind = 'tie'
            else:
                disp = judge_mod.line_dispute(choices, best['part_index'], ovr)
                kind = 'line' if disp else None
        if kind:
            local_margin = round(margin, 4)
            judge = judge_mod.judge_window(
                data, config, w, choices, dyn_part, m_start, m_end,
                best['part_index'], margin, judge_budget, dispute=disp)
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
                'kind': judge['kind'],
                'dispute': judge['dispute_desc'],
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
                    'flat_kind': c.get('flat_kind'),
                    'chord_ratio': c.get('chord_ratio'),
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
            # 被"是不是一条线"这道闸挡掉的声部（柱式和弦铺底 / 踏板音）。
            # 写出来是为了让人一眼看出"这段为什么不是音最高的那个在唱"。
            'flat_excluded': [c['part_name'] for c in choices
                              if c.get('flat_kind') == 'chord'
                              and not c.get('is_voice')],
            # 降级但**没有剥夺资格**的那些（单音同音反复：吟诵 / 弱旋律收句）。
            # 它们只是排到了后面，没人可选时照样当选。
            'chant_suspects': [c['part_name'] for c in choices
                               if c.get('flat_kind') == 'chant'
                               and not c.get('is_voice')],
            # 这一窗归的正好是一条同音反复的线：页面上一眼能看出这是"唱出来的
            # 一条线"而不是"提取失败"。
            'chant_owner': bool(best and best.get('flat_kind') == 'chant'),
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
        shape_llm = None
        if shape_gate:
            ranked = _seg_ranked_salience(
                win_scores, r['start_measure'], r['end_measure'], owner)
            gated, shapes, shape_reason = apply_shape_gate(
                data, s, e, ranked, owner, shape_min)
            # 形态层换了人 → 这是本地判据自己的又一次"冒险"（它只说原归属形态
            # 不合格，没说接替者有多合适）。接替者的显著性比原归属差一大截时，
            # 这就是一次"两害相权"，交给模型复核一句（2026-10-23 加的第二种
            # 发问场合，见 melody_judge.shape_dispute）。
            if gated != owner:
                sal_map = dict(ranked)
                disp = (judge_mod.shape_dispute(
                            gated, sal_map.get(gated), owner,
                            sal_map.get(owner), len(ranked))
                        if llm_judge else None)
                if disp:
                    seg_choices = _seg_judge_choices(data, ranked)
                    j = judge_mod.judge_window(
                        data, config,
                        {'start_measure': r['start_measure'],
                         'end_measure': r['end_measure']},
                        seg_choices, _seg_judge_stats(data, ranked, s, e),
                        s, e, gated, 0.0, judge_budget, dispute=disp)
                    shape_llm = {
                        'called': j['called'], 'applied': j['applied'],
                        'pick': j['pick'], 'prev': j['prev'],
                        'reason': j['reason'], 'confidence': j['confidence'],
                        'kind': j['kind'], 'dispute': j['dispute_desc'],
                    }
                    pick = j.get('pick')
                    if j.get('applied') and pick is not None and pick != gated:
                        why2 = f'（{j.get("reason") or "未说明理由"}）'
                        if pick == owner:
                            shape_reason += f'；模型复核后改回原归属{why2}'
                        else:
                            shape_reason += (
                                f'；模型复核后改判给 '
                                f'{data.parts[pick].name}{why2}')
                        gated = pick
            owner = gated
            shape_info = shapes.get(owner) or {}
            if owner != r['part_index']:
                replaced_from = r['part_index']
                shape_replacements.append({
                    'measures': f"{r['start_measure']}–{r['end_measure']}",
                    'from': data.parts[r['part_index']].name,
                    'to': data.parts[owner].name,
                    'reason': shape_reason,
                    'llm': shape_llm,
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
            'shape_llm': shape_llm,
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
    # 平线否决：哪些声部是"只在一个音上打点"被挡在竞选之外的。必须写明白 ——
    # 否则页面上会看到"音最高的那个声部没被选上"，看着像算错了。
    # 柱式和弦（剥夺资格）与同音反复（只降级）分开说，两件事完全不同。
    flat_hits, chant_hits = {}, {}
    for r in win_rows:
        start = r.get('start_measure')
        for nm in (r.get('flat_excluded') or []):
            flat_hits.setdefault(nm, set()).add(start)
        for nm in (r.get('chant_suspects') or []):
            chant_hits.setdefault(nm, set()).add(start)
    if flat_hits or chant_hits:
        def _m_list(ms):
            ms = sorted(m for m in ms if isinstance(m, int))
            if not ms:
                return ''
            if len(ms) <= 4:
                return '、'.join(str(m) for m in ms)
            return f'{ms[0]}–{ms[-1]} 等 {len(ms)} 个窗'
    if flat_hits:
        method += ('平线否决：' + '；'.join(
            f'{nm}（第 {_m_list(ms)} 小节）' for nm, ms in flat_hits.items())
            + '在窗内几乎不换音，且**一个起音同时响好几个音**（柱式和弦 / 踏板'
            '铺底）：显著性里"和弦取最高音"让它音高虚高、动机重复满分，'
            '形态里音程全 0 让级进比 / 拱形度 / 平滑度三项齐满分，'
            '两项都是假证据 —— 不参与主旋律竞选。')
    if chant_hits:
        method += ('同音反复（保留资格）：' + '；'.join(
            f'{nm}（第 {_m_list(ms)} 小节）' for nm, ms in chant_hits.items())
            + '这条线连着许多音都在同一个音高上，但它是**一条单音线条**'
            '（一个音一个音地唱），很可能是吟诵式旋律 —— 旋律性弱的流行歌、'
            '民歌的收句常有这种写法（如非洲民歌《巴努瓦》最后一句整句唱 do）。'
            '它只是排到后面，没有失去资格：没有更像旋律的候选时仍由它演唱，'
            '与别的候选咬得紧时交给模型裁决。')
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
        # 被"是不是一条线"挡在竞选之外的声部（见 is_flat_line）：
        # 柱式和弦 / 踏板铺底 —— 显著性虚高 + 形态假满分，两条证据都是假的
        'flat_excluded_parts': sorted(flat_hits),
        'flat_excluded_by_part': {nm: sorted(ms) for nm, ms in flat_hits.items()},
        # **同音反复但保留资格**的声部（见 flat_kind）：单音线条一段下来都在同一
        # 个音高上，很可能是吟诵式旋律，只降级不剥夺
        'chant_suspect_parts': sorted(chant_hits),
        'chant_suspect_by_part': {nm: sorted(ms) for nm, ms in chant_hits.items()},
        # 最终真的由一条同音反复的线在唱的窗（这类段要让人一眼看出不是提取失败）
        'chant_owner_windows': sorted(
            r.get('start_measure') for r in win_rows if r.get('chant_owner')),
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
