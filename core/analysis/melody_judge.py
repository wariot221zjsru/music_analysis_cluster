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
                     m_end, local_idx, margin) -> str:
    return _TASK.format(
        title=title or '（未命名）', sm=sm, em=em, wi=wi,
        cands=_cand_lines(data, choices, per_part, m_start, m_end),
        local=local_idx, margin=round(margin, 4))


def judge_window(data, cfg, window: dict, choices: list, per_part: list,
                 m_start: float, m_end: float, local_idx, margin,
                 budget: list) -> dict:
    """把一窗的归属交给 DeepSeek 裁决；**任何失败都不抛异常**。

    ``budget`` 是可变计数器 ``[剩余可问次数]`` —— 由调用方持有，跨窗累计，
    用完就不再问。

    返回 dict，永远带 ``called`` 键：

    - ``called=False`` —— 没问（预算用完 / 开关关着）；
    - ``called=True`` 且 ``applied=True`` —— 采纳了模型的意见，``pick`` 是新的
      声部索引，``prev`` 是本地原本选的那个；
    - ``called=True`` 且 ``applied=False`` —— 问了但没采纳（模型说了一个不在
      候选里的声部 / 返回不可解析 / 调用失败），``reason`` 记着原因。
    """
    out = {'called': False, 'applied': False, 'pick': None, 'prev': local_idx,
           'reason': '', 'summary': '', 'confidence': None, 'window': wi(window)}
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
                                m_start, m_end, local_idx, margin)
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
