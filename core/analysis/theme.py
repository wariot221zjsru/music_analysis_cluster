"""主题层：材料身份 —— 「这段材料是谁」。

因果链位置：
    theme ← melody_line（可选：主旋律线归属）+ rhythm_grid（小节表）
    structure ← theme（曲式层消费主题序列）

**这一层只回答身份，不回答功能。**

  身份 = 两段材料是不是同一个主题（含变奏 / 变形 / 碎片化的同一个）
  功能 = 这段在曲式里干什么（呈示 / 对比 / 展开 / 再现）—— 归曲式层

## 为什么不把身份和功能写成一个阈值

老版本的 ``structure.py`` 用**一个**相似度阈值同时判"是不是同一段"和"所以是什么曲式"。
后果是误判连坐：主题变形稍大 → 掉到阈值下 → 判成新段 → 曲式从 ABA 变 ABC。
本模块的职责就是把这条链断开：这里只产出**证据**（每对乐句的相似度、变形类型、
覆盖比），不落槌；落槌交给上层，上层可以按曲式假设换阈值重算 —— 因为证据留着，
重算不用再碰音符。

## 指纹的三条轴

1. **音高动机** ``ivs``：有向音程序列。移调不变（模进自动容忍），
   并在比对时展开四种形态 —— 原型 / 倒影 / 逆行 / 逆行倒影。
2. **节奏型** ``rhythm``：相邻时值的比值取 log2 后量化。整体时值缩放不变
   （全乘 2 不改变比值），所以"放慢一倍的主题"仍认得出是同一个。
3. **音级分布** ``pc``：12 维时值占比的余弦。它是兜底轴 —— 前两条都算不出来
   （句子太短、只有一个音）时，至少还有调式色彩可比。

缺了第 2 条是老实现最大的漏洞：音程 LCS 与音级分布都是**音高侧**的，
于是"同音高、节奏完全不同"的两句会被判成同一主题。
"""
import math

from .scoredata import interval_seq, lcs_ratio

# 同一主题（原型级重复）：直接用这个阈值聚类
THEME_STRONG = 0.72
# 变形级重复下限：曲式层若判定"这里应该有再现"，可用它放宽后重算
THEME_WEAK = 0.52
# 碎片：长度比不超过这个值，且短的那条几乎整条落在长的那条里
THEME_FRAG_RATIO = 0.62
THEME_FRAG_COVER = 0.80
# 三条轴的权重：音高动机 / 节奏型 / 音级分布
# 节奏给到 0.34 是刻意的：老实现完全没有节奏轴，两条同音高、节奏完全不同的
# 句子会拿满分并被并成同一主题。给到 0.34 之后，那种情形掉到 0.66，
# 够不上强阈值（0.72），必须另有音高证据才并得起来。
THEME_W = (0.46, 0.34, 0.20)
# 每条轴最多取多少个音（长乐句尾部对主题识别贡献很小，砍掉省时间）
THEME_MAX_NOTES = 64
# 快速排除：音级分布余弦低于此值且长度比过于悬殊 → 不必做四条 LCS
THEME_PRUNE_PC = 0.35
THEME_PRUNE_RATIO = 0.25

_VARIANTS = ('原型', '倒影', '逆行', '逆行倒影')


def _lcs_len(a: list, b: list) -> int:
    """最长公共子序列长度（``lcs_ratio`` 只给比值，碎片判定要长度）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _pc_cos(a: list, b: list) -> float:
    """音级分布的余弦相似度，**对移调取最大**（12 次循环移位）。

    为什么必须移调不变：音级分布本身对移调是敏感的。不做移位取最大的话，
    同一个主题移到属调（最常见的模进）会因为"用的音不同"被扣分，
    而音高动机那一边明明是满分 —— 两条轴互相打架。
    移位取最大等于问："忽略调高，这两个音级集合像不像"。
    """
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    best = 0.0
    for k in range(12):
        c = sum(a[i] * b[(i + k) % 12] for i in range(12))
        if c > best:
            best = c
    return best / (na * nb)


def _rhythm_shape(durs: list) -> list:
    """节奏型：相邻时值比取 log2 后量化到 −2..2。

    比值只跟"长短关系"有关，与绝对时值无关 —— 所以整段放慢一倍、
    或把四分音符写成八分音符，比值都不变，认得出是同一个主题。
    """
    out = []
    for a, b in zip(durs, durs[1:]):
        if a <= 0 or b <= 0:
            continue
        r = math.log2(b / a)
        out.append(max(-2, min(2, int(round(r)))))
    return out[:THEME_MAX_NOTES]


def fingerprint(notes: list) -> dict:
    """一个乐句的指纹。

    ``notes`` = ``[(offset, midi, duration)]``，调用方保证按时间排好序。
    """
    notes = notes[:THEME_MAX_NOTES]
    seq = [n[1] for n in notes]
    durs = [n[2] for n in notes]
    pc = [0.0] * 12
    for _off, p, d in notes:
        pc[p % 12] += d
    total = sum(pc) or 1.0
    return {
        'count': len(seq),
        'span': (max(seq) - min(seq)) if seq else 0,
        'ivs': interval_seq(seq),
        'rhythm': _rhythm_shape(durs),
        'pc': [x / total for x in pc],
    }


def compare(a: dict, b: dict) -> dict:
    """两段材料的相似证据（各轴 0–1，外加变形类型与碎片标记）。"""
    if not a['count'] or not b['count']:
        return {'score': 0.0, 'motif': 0.0, 'rhythm': 0.0, 'pc': 0.0,
                'kind': '—', 'cover': 0.0,
                'ratio': 0.0, 'fragment': False}

    ratio = min(a['count'], b['count']) / max(a['count'], b['count'])
    pc = _pc_cos(a["pc"], b["pc"])

    # 音级分布差得远、长度又悬殊 → 不可能同一主题，省掉四条 LCS
    if pc < THEME_PRUNE_PC and ratio < THEME_PRUNE_RATIO:
        return {'score': round(THEME_W[2] * max(pc, 0.0), 4), 'motif': 0.0,
                'rhythm': 0.0, 'pc': round(pc, 4), 'kind': '—', 'cover': 0.0,
                'ratio': round(ratio, 4), 'fragment': False}

    if a['ivs'] and b['ivs']:
        # 四种形态的换算关系（v = b 的音程序列）：
        #   倒影        = −v          （音程取反，顺序不变）
        #   逆行        = reverse(−v)  （音高序列反转 → 音程既取反又反序）
        #   逆行倒影    = reverse(v)   （倒影之后再逆行）
        # 早先这里把"逆行"和"逆行倒影"的式子写反了，于是逆行被认成倒影 ——
        # 只影响变形类型的命名，不影响相似度数值，但会把分析结论说错。
        cand = {
            '原型': lcs_ratio(a['ivs'], b['ivs']),
            '倒影': lcs_ratio(a['ivs'], [-x for x in b['ivs']]),
            '逆行': lcs_ratio(a['ivs'], [-x for x in reversed(b['ivs'])]),
            '逆行倒影': lcs_ratio(a['ivs'], list(reversed(b['ivs']))),
        }
        kind = max(cand, key=lambda k: cand.get(k, 0.0))
        motif = cand[kind]
        # 覆盖率按**短的那条**算：碎片是"短句几乎整条落进长句"，
        # 而 lcs_ratio 的分母是长句，用来判碎片会永远偏低。
        over = max(_lcs_len(a['ivs'], [-x for x in b['ivs']]),
                   _lcs_len(a['ivs'], b['ivs']),
                   _lcs_len(a['ivs'], [-x for x in reversed(b['ivs'])]),
                   _lcs_len(a['ivs'], list(reversed(b['ivs']))))
        cover = over / max(min(len(a['ivs']), len(b['ivs'])), 1)
    else:
        kind, motif, cover = '—', 0.0, 0.0

    # 节奏轴：两边都是等时值（整句全四分音符）时这条轴**没有信息**，
    # 退回中性 0.5 —— 不能因为"两边都没节奏"就判满分，那等于把"都没说话"
    # 当成"说的是一样的话"，音高稍有差别也会被并成同一主题。
    if a['rhythm'] and b['rhythm']:
        flat = not any(a['rhythm']) and not any(b['rhythm'])
        rhy = 0.5 if flat else lcs_ratio(a['rhythm'], b['rhythm'])
    else:
        rhy = 0.5
    score = (THEME_W[0] * motif + THEME_W[1] * rhy
             + THEME_W[2] * max(pc, 0.0))
    fragment = ratio <= THEME_FRAG_RATIO and cover >= THEME_FRAG_COVER
    return {
        'score': round(score, 4),
        'motif': round(motif, 4),
        'rhythm': round(rhy, 4),
        'pc': round(pc, 4),
        'kind': kind if motif >= 0.35 else '—',
        'cover': round(cover, 4),
        'ratio': round(ratio, 4),
        'fragment': bool(fragment),
    }


def links_of(fps: list) -> list:
    """所有乐句两两之间的证据（i < j）。**不判**，只算。"""
    out = []
    n = len(fps)
    for i in range(n):
        for j in range(i + 1, n):
            c = compare(fps[i], fps[j])
            out.append(dict(i=i, j=j, **c))
    return out


def build_labels(links: list, n: int, threshold: float = THEME_STRONG,
                 allow_fragment: bool = True) -> dict:
    """按证据把乐句聚成主题（纯函数：换阈值重算不必再碰音符）。

    ``allow_fragment=False`` 时碎片不认（严格口径：长度差太多的不并）。
    返回 ``{'groups': ..., 'labels': ..., 'themes': ...}``。
    """
    idx = {(l['i'], l['j']): l for l in links}
    groups = []
    picked = []
    for i in range(n):
        best = None
        best_sim, best_kind, best_frag = 0.0, '', False
        for g in groups:
            lk = idx.get((g['leader'], i))
            if lk is None:
                continue
            if allow_fragment and lk['fragment']:
                sim, frag = lk['cover'], True
            else:
                sim, frag = lk['score'], False
            if sim > best_sim:
                best, best_sim, best_kind, best_frag = g, sim, lk['kind'], frag
        if best is not None and best_sim >= threshold:
            best['members'].append(i)
            best['links'].append({'to': i, 'sim': round(best_sim, 4),
                                  'kind': best_kind, 'fragment': best_frag})
            picked.append((best['id'], round(best_sim, 4), best_kind, best_frag))
        else:
            gid = len(groups)
            groups.append({'id': gid, 'leader': i, 'members': [i], 'links': []})
            picked.append((gid, 1.0, '首次出现', False))

    labels, seen, themes = [], {}, []
    for gid, _sim, _kind, _frag in picked:
        base = _letter(gid)
        times = seen.get(base, 0)
        seen[base] = times + 1
        labels.append(base + ("'" * times))

    for g in groups:
        members = g['members']
        themes.append({
            'id': _letter(g['id']),
            'first_phrase': members[0] + 1,
            'occurrences': [m + 1 for m in members],
            'count': len(members),
            'variants': [{'phrase': l['to'] + 1, 'sim': l['sim'],
                          'kind': l['kind'],
                          'fragment': l['fragment']} for l in g['links']],
        })
    return {'groups': groups, 'labels': labels, 'themes': themes}


def letter(i: int) -> str:
    """主题序号 → 字母（A、B、C…；超过 26 个回落「主题N」）。"""
    return chr(ord('A') + i) if i < 26 else f'主题{i + 1}'


_letter = letter


def analyze(fps: list) -> dict:
    """默认口径：算全部证据 + 按强阈值聚一次。"""
    links = links_of(fps)
    got = build_labels(links, len(fps), THEME_STRONG)
    return {
        'links': links,
        'groups': got['groups'],
        'labels': got['labels'],
        'themes': got['themes'],
        'threshold': THEME_STRONG,
        'weights': {'motif': THEME_W[0], 'rhythm': THEME_W[1], 'pc': THEME_W[2]},
    }
