"""主旋律线提取的 **DeepSeek 兜底裁定**（选轨层）。

为什么要有这一层
----------------
本地判据是四条显著性（平均音高 / 总时值 / 动机重复 / 密度倒数）加形态、力度
两票。**它们都是统计量。** 统计量在大多数曲子上够用，但总有那么几窗：

- 两条候选线的四项分数互相咬住（差距不到几个百分点），换一个曲子可能就翻
  过来；
- 或者三票对一票、几条判据各执一词；
- 或者这一窗恰好是过渡、织体在换，谁都不像主角。

这时候"谁更像主旋律"其实已经是一个**音乐判断**，不是算术题 —— 而人听一遍
就能答。所以把这些难判的窗交给 DeepSeek：把这一窗里每个候选声部的关键特征
摊开给它，让它选，并要求给出理由。

与 :mod:`principal` 的分工
--------------------------
``principal`` 管的是**选音**（一个多音事件里哪个音是主旋律在唱的），本模块
管的是**选轨**（这一窗归给哪个声部）。两条路共用同一个开关
``SystemConfig.melody_llm_judge``，也共用同一套"问了不采纳也没关系"的降级。

设计原则（重要）
----------------
1. **模型只参与裁决，不参与计算。** 显著性、形态、力度照旧本地算，模型拿到
   的是这些已经算好的数；它的输出只有"选谁"，改不了其它任何东西。
2. **任何失败都必须静默退回本地。** 未安装 dsh、超时、返回不可解析、说了个
   不存在的声部 —— 一律当作没问过，窗归属仍是本地结果。跑一次分析要几分钟，
   不能因为一次网络波动把整步变成 error。
3. **只问最难的几窗。** 大曲子几十个窗，全部去问既慢又贵，还容易让模型在后
   半程敷衍。``JUDGE_MAX_CALLS`` 封顶。
4. **理由要带回来。** 页面上左右对照展示，读者得能判断这一次裁定值不值得信。
"""
from .scoredata import midi_to_name

# ---- 常量（改动请同步 项目文档/主旋律提取/03-参数与阈值.md） ----
JUDGE_MAX_CALLS = 24        # 一次分析里最多问几次（防止大曲子拖死）
JUDGE_MARGIN = 0.06         # 第一名与第二名显著性差 ≤ 此值 → 判为"难判"
JUDGE_MIN_CANDIDATES = 2    # 候选不足两个就没有"难判"这回事
JUDGE_MAX_PARTS = 6         # 一次最多摊开几个候选声部
JUDGE_MAX_SEQ = 16          # 每个声部最多给出几个代表性音高
# ---- 「是不是一条线」这道闸自己拿不准时，同样交给模型 ----
# 平线闸分两种判法（见 melody_line.flat_kind）：多音事件为主的判成柱式和弦（剔
# 除），单音同音反复的判成吟诵（降级）。**两种判法都会错** —— 前者可能把一首
# 旋律性弱的歌当成伴奏，后者可能把高音区的零星点缀当成旋律。本地只是在"看起
# 来平不平"上打转，这时候最该问人。
JUDGE_FLAT_BAND = 0.12      # 被平线闸处理过的候选与实际当选者差距在此以内 → 争议
JUDGE_SHAPE_GAP = 0.10      # 段级形态换人时，替代者与 owner 的显著性落差 ≥ 此值 → 争议

_LOCAL_KEYS = ('pitch_n', 'dur_n', 'motif_n', 'sparse_n')

_TASK = """你是专业配器分析师。下面这一窗要给「主旋律」选归属声部，候选之间咬得很紧，
本地的四项显著性（平均音高 / 总时值 / 动机重复 / 密度倒数）、形态分与力度
都分不出高下，请你按听觉判断定夺。

曲目：{title}　第 {sm}–{em} 小节（第 {wi} 窗）

候选声部（按本地显著性降序）：
{cands}

本地已经算出的结果：第 {local} 声部暂时领先 {margin} 分。

判定要求：
1. 判断标准是"听上去哪一条最像人在唱的主旋律"，不是"哪一条音最高""哪一条
   音最多" —— 柱式和弦伴奏、分解和弦 ostinato、和声填补都不算主旋律，哪怕
   它们音区高或者音很密；
2. 有 vocals / 人声声部时，歌词通常跟着主旋律走，但它同样可能是伴唱，
   照样要按第 1 条判断；
3. 长时值 + 高度重复 + 极度稀疏 = 低音 / 持续音的特征，不是主旋律；
4. 力度（velocity）只在两边实在分不出时才当参考。

输出 JSON：
{{"pick":<number>,"reason":"一句话说明（60 字以内）","confidence":<0-1>}}
"""

# 「是不是一条线」这道闸内部起了分歧时的问法（见 :func:`line_dispute`）。
# 与上面那套 task 的区别：这里**不要求模型重排高下**，只要求它回答一个二选一
# 的问题 —— 那条被本地处理过的平线，是伴奏织体还是主旋律在唱。
_TASK_LINE = """你是专业配器分析师。这一窗本地判据**自己打起来了**，需要你裁决。

曲目：{title}　第 {sm}–{em} 小节（第 {wi} 窗）

**分歧所在**：{dispute}

候选声部（按本地显著性降序）：
{cands}

本地当前归属：第 {local} 声部（领先 {margin} 分）。

裁决要点：
1. 柱式和弦伴奏（一个起音同时响好几个音、跟着和声走）、踏板音、持续音 = **伴奏**，
   哪怕它音区最高、响得最久、在统计上显得最"突出"；
2. **吟诵式的旋律必须认下来**：旋律性弱的流行歌、民歌的收句（例如非洲民歌《巴努瓦》
   最后一句整句都唱 do），一大串音都是同一个音高 —— 只要它是一条**单音线条**、
   带着歌唱 / 歌词的节奏，它就是主旋律，不许让给伴奏。
   **音区不够高、音数不够多、缺少变化都不能作为反证**；
3. 上面两条冲突时看**织体**：一次响好几个音的倾向和声伴奏，
   一个音一个音咬出来的倾向旋律；
4. 力度只在两边实在分不出时才当参考。

输出 JSON：
{{"pick":<number>,"reason":"一句话说明（60 字以内）","confidence":<0-1>}}
"""

CAND_MARKERS = ('①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧')


def is_hard(choices: list, margin: float) -> bool:
    """这一窗算不算"难判"（本地判据咬得太紧）。

    ``choices`` 是 :func:`melody_line._window_choices` 的结果；``margin`` 是
    :func:`melody_line._pick` 返回的领先量。只看这两样就够了 —— 它们已经是
    本地全部证据汇总后的结果。
    """
    if len(choices) < JUDGE_MIN_CANDIDATES:
        return False
    return margin <= JUDGE_MARGIN


def line_dispute(choices: list, winner_idx, override: dict | None = None):
    """「是不是一条线」这道闸**自己**和显著性打起来了吗。

    返回争议描述 dict，没有争议则返回 ``None``。有三种情形值得发问：

    * 排在最前面的候选被判成**柱式和弦**（``flat_kind == 'chord'``）而被剔除 ——
      万一它其实是一首旋律性弱的歌的主旋律？
    * 某个候选被判成**同音反复**（``'chant'``）而降级 —— 万一它只是高音区里
      一下一下敲的点缀，真正的旋律还在别人手上？
    * 两种候选的显著性与实际当选者咬在 :data:`JUDGE_FLAT_BAND` 以内 —— 本地
      压根分不出该信谁。

    ``None`` 的两种常见情形也必须诚实：**整窗只剩被否的那一个**（问了也没有
    第二个答案，直接按显著性退回）、**差距悬殊**（本地判据其实很清楚）。
    """
    if len(choices) < JUDGE_MIN_CANDIDATES:
        return None
    from .melody_line import candidate_tier     # 懒加载：避免模块级循环依赖
    tiers = {c['part_index']: candidate_tier(c, override) for c in choices}
    clean = [c for c in choices if tiers[c['part_index']] == 0]
    demoted = [c for c in choices if tiers[c['part_index']] > 0]
    if not clean or not demoted:
        return None
    win = next((c for c in choices if c['part_index'] == winner_idx), None)
    if win is None:
        return None
    blk = max(demoted, key=lambda c: c['salience'])
    gap = round(blk['salience'] - win['salience'], 4)
    if abs(gap) > JUDGE_FLAT_BAND:
        return None
    kind = blk.get('flat_kind') or 'chord'
    ratio = blk.get('chord_ratio') or 0.0
    if kind == 'chord':
        desc = (f'声部 {blk["part_index"]}「{blk["part_name"]}」取最高音后几乎不换音'
                f'（窗内多音事件占 {ratio:.0%}），本地把它当成柱式和弦 / 踏板铺底，'
                f'剥夺了它的主旋律资格')
    else:
        desc = (f'声部 {blk["part_index"]}「{blk["part_name"]}」是一条同音反复的'
                f'单音线（多音事件只占 {ratio:.0%}），本地拿不准它是吟诵式旋律'
                f'还是零星点缀，先把它降级处理')
    return {'part_index': blk['part_index'], 'part_name': blk['part_name'],
            'gap': gap, 'kind': kind, 'chord_ratio': ratio,
            'tier': tiers[blk['part_index']], 'desc': desc}


def shape_dispute(new_idx, new_sal, prev_idx, prev_sal, rank_len: int):
    """段级形态闸门换人，但换得**很不情愿** → 值得问一句吗。

    形态层只说"owner 的形态不合格"，没说"接替它的人有多合适"。接替者的显著性
    比原来差一大截（≥ :data:`JUDGE_SHAPE_GAP`）时，这其实是一次"两害相权"，
    本地并没有真的把握 —— 这类才是要让模型看一眼的。
    """
    if not isinstance(new_sal, (int, float)) or new_idx == prev_idx:
        return None
    if isinstance(prev_sal, (int, float)) and prev_sal >= 0:
        gap = round(prev_sal - new_sal, 4)
        if gap < JUDGE_SHAPE_GAP:
            return None
        desc = (f'形态层判定归属声部的形态不合格，改由声部 {new_idx} 承担，'
                f'但接替者的显著性低了 {gap} —— 本地只是"两害相权"，'
                f'并没有真的把握')
    elif rank_len >= 2:
        desc = (f'形态层判定原归属的形态不合格，改由声部 {new_idx} 承担，'
                f'原归属在本段没有显著性记录，无从比较强弱')
    else:
        return None
    return {'desc': desc, 'gap': None if not isinstance(prev_sal, (int, float))
                          else round(prev_sal - new_sal, 4)}


def _cand_lines(data, choices: list, per_part: list, m_start: float,
                m_end: float) -> str:
    """把候选声部的特征摊成给模型看的文本。"""
    from . import melody_export as mex

    by_idx = {int(d.get('part_index', -1)): d for d in (per_part or [])}
    lines = []
    for i, c in enumerate(choices[:JUDGE_MAX_PARTS]):
        idx = c['part_index']
        d = by_idx.get(idx, {})
        src = data.parts[idx] if idx < len(data.parts) else None
        seq = []
        if src is not None:
            for ev in sorted(mex.events_in(src, m_start, m_end),
                             key=lambda e: e.offset)[:JUDGE_MAX_SEQ]:
                seq.append(midi_to_name(max(ev.midi_pitches)))
        tags = []
        if c.get('is_voice'):
            tags.append('人声')
        if c.get('low_register'):
            tags.append('低音区')
        # 「是不是一条线」这道闸的处理结果必须摊给模型看 —— 它要判的就是这件事
        if c.get('flat_kind') == 'chord':
            tags.append('柱式和弦平线（本地已剥夺资格）')
        elif c.get('flat_kind') == 'chant':
            tags.append('同音反复的单音线（本地降级为存疑）')
        elif c.get('flat_line'):
            tags.append('几乎不换音')
        if c.get('doubled_with'):
            names = [data.parts[j].name for j in c['doubled_with']
                     if j < len(data.parts)]
            tags.append('与 ' + '、'.join(names) + ' 齐奏')
        dyn = d.get('dyn')
        lines.append(
            f"  {CAND_MARKERS[i] if i < len(CAND_MARKERS) else i + 1}"
            f" 声部 {idx}「{c['part_name']}」"
            f"（{getattr(src, 'instrument', None) or '未标乐器'}"
            f"{'，' + '/'.join(tags) if tags else ''}）\n"
            f"      显著性 {c['salience']}（音高 {round(d.get('pitch_n', 0), 3)}"
            f" / 时值 {round(d.get('dur_n', 0), 3)}"
            f" / 动机 {round(d.get('motif_n', 0), 3)}"
            f" / 稀疏 {round(d.get('sparse_n', 0), 3)}）\n"
            f"      平均音高 {d.get('mean_pitch', '—')}"
            f"　发音数 {d.get('note_count', '—')}"
            f"　起音数 {d.get('onsets', '—')}"
            f"　密度 {d.get('density', '—')}"
            + (f"　力度 {dyn}（覆盖 {round(d.get('dyn_support', 0), 2)}）"
               if dyn is not None else '　力度 —')
            + (f"\n      走线：{' → '.join(seq)}" if seq else ''))
    return '\n'.join(lines)


def build_judge_task(data, title, wi, sm, em, choices, per_part, m_start,
                     m_end, local_idx, margin, dispute=None) -> str:
    """拼出给模型的问句。

    ``dispute`` 是 :func:`line_dispute` / :func:`shape_dispute` 给的分歧描述，
    传了就走**第二种问法**（"本地判据自己打起来了，这条线到底是不是主旋律"），
    不传走原来的"两条候选咬得紧，请你重排"。
    """
    cands = _cand_lines(data, choices, per_part, m_start, m_end)
    if dispute is not None:
        desc = dispute.get('desc') if isinstance(dispute, dict) else str(dispute)
        return _TASK_LINE.format(
            title=title or '（未命名）', sm=sm, em=em, wi=wi, dispute=desc,
            cands=cands, local=local_idx, margin=round(margin, 4))
    return _TASK.format(
        title=title or '（未命名）', sm=sm, em=em, wi=wi, cands=cands,
        local=local_idx, margin=round(margin, 4))


def judge_window(data, cfg, window: dict, choices: list, per_part: list,
                 m_start: float, m_end: float, local_idx, margin,
                 budget: list, dispute=None) -> dict:
    """把一窗的归属交给 DeepSeek 裁决；**任何失败都不抛异常**。

    ``budget`` 是可变计数器 ``[剩余可问次数]`` —— 由调用方持有，跨窗累计，
    用完就不再问。

    ``dispute`` 传了就是「本地判据起分歧」这一种问法（见 :func:`build_judge_task`）；
    ``None`` 是原来的「显著性咬得紧」。两者共用同一个 budget 与降级路径。

    返回 dict，永远带 ``called`` 键：

    - ``called=False`` —— 没问（预算用完 / 开关关着）；
    - ``called=True`` 且 ``applied=True`` —— 采纳了模型的意见，``pick`` 是新的
      声部索引，``prev`` 是本地原本选的那个；
    - ``called=True`` 且 ``applied=False`` —— 问了但没采纳（模型说了一个不在
      候选里的声部 / 返回不可解析 / 调用失败），``reason`` 记着原因。
    """
    out = {'called': False, 'applied': False, 'pick': None, 'prev': local_idx,
           'reason': '', 'summary': '', 'confidence': None, 'window': wi(window),
           'kind': 'line' if dispute is not None else 'tie',
           'dispute_desc': (dispute or {}).get('desc', '') if isinstance(dispute, dict)
                           else ('' if dispute is None else str(dispute))}
    try:
        if budget[0] <= 0:
            out['reason'] = '已用完本次分析的 DeepSeek 询问预算'
            return out
        if not getattr(cfg, 'melody_llm_judge', True):
            out['reason'] = '系统设置里关掉了 DeepSeek 兜底'
            return out
        from .llm import run_llm_json_soft, JSON_ONLY
        task = build_judge_task(data, getattr(cfg, '_piece_title', None),
                                wi(window), window.get('start_measure'),
                                window.get('end_measure'), choices, per_part,
                                m_start, m_end, local_idx, margin,
                                dispute=dispute)
        budget[0] -= 1
        out['called'] = True
        raw = run_llm_json_soft(
            task + JSON_ONLY,
            profile=getattr(cfg, 'dsh_profile', 'headless') or 'headless',
            timeout=int(getattr(cfg, 'dsh_timeout', 600) or 600))
        if not isinstance(raw, dict) or raw.get('_error'):
            out['reason'] = str((raw or {}).get('_error') or '未返回结果')
            return out
        try:
            pick = int(raw.get('pick'))
        except (TypeError, ValueError):
            out['reason'] = f"模型没有给出合法的声部序号：{raw.get('pick')!r}"
            return out
        if pick not in {c['part_index'] for c in choices}:
            out['reason'] = f'模型选的声部 {pick} 不在候选里'
            return out
        out.update({'applied': True, 'pick': pick,
                    'reason': str(raw.get('reason') or '')[:200],
                    'confidence': raw.get('confidence')})
        return out
    except Exception as exc:                      # noqa: BLE001 —— 兜住一切
        out['reason'] = f'{type(exc).__name__}: {exc}'
        return out


def wi(window: dict):
    return window.get('window') or window.get('start_measure')
