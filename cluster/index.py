"""向量索引 + 三轴检索 + RRF 融合。

## 三个索引

  调式 mode     一曲一向量（12 维，已按主音对齐）→ 直接余弦，**不循环移位**
  和声 harmony  一曲一向量（12 维）→ 余弦
  旋律 melody   **乐句级**：一乐句一条向量，粗排命中后再做乐句级精排

## 旋律为什么是乐句级

整曲揉成一个向量会把"这首曲子中间有 4 小节和另一首很像"这件事抹平。
所以旋律轴存每首曲子的**乐句指纹列表**，打分时问的是
"这两首之间最像的那一对乐句有多像"（可切换成"各乐句最佳匹配的均值"）。
这能抓出"共享一个主题"的曲子，代价是计算量 —— 所以走粗排 + 精排：

  粗排  查询曲每个乐句的 42 维向量 → 在全库乐句上取 Top-M
  精排  只对粗排命中的候选曲，在命中的那几对乐句上跑 theme.compare
        （含原型/倒影/逆行/逆行倒影四种 LCS），取最大值

直接全比对是 |A|×|B| 对乐句，粗排把它压到每首候选曲最多 M 对。

## 后端

纯 Python 暴力余弦。几百首曲子、几千条乐句向量，毫秒级；不引入任何新依赖。
``VectorIndex`` 是抽象基类，日后要换 pgvector / FAISS 只需再实现一个子类，
上层（检索、RRF、聚类）一行都不用改。
"""
from . import features

AXES = ('mode', 'harmony', 'melody')


def cosine(a, b) -> float:
    """余弦相似度。"""
    return features._cosine(a, b)


class VectorIndex:
    """单向量轴：一曲一向量，余弦检索（暴力）。"""

    axis = 'generic'

    def __init__(self):
        self.pids = []
        self.vecs = []
        # 调式轴专用：主音没定下来的那些（比对时降级为循环移位）
        self.uncertain = []

    def add(self, pid, vec, uncertain: bool = False):
        if not vec:
            return
        self.pids.append(pid)
        self.vecs.append(vec)
        self.uncertain.append(bool(uncertain))

    def __len__(self):
        return len(self.pids)

    def score(self, vec, other, flag_a: bool = False, flag_b: bool = False) -> float:
        """两向量相似度。

        调式轴任一方主音不确定时降级为"循环移位取最大"（此时只能比调式色彩，
        分不出具体调式）。其余情况直接余弦 —— 调式向量已按主音对齐，
        **不能再移位**，否则多利亚会和自然大调混在一起。
        """
        if self.axis == 'mode' and (flag_a or flag_b):
            return features._pc_cos_max_shift(vec, other)
        return cosine(vec, other)

    def query(self, vec, top_n: int, exclude=None, uncertain: bool = False):
        """检索 Top-N。返回 [(piece_id, score)] 降序。"""
        out = []
        for pid, v, flag in zip(self.pids, self.vecs, self.uncertain):
            if exclude is not None and pid == exclude:
                continue
            sc = self.score(vec, v, uncertain, flag)
            out.append((pid, round(sc, 4)))
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out[:top_n]


class ModeIndex(VectorIndex):
    axis = 'mode'


class HarmonyIndex(VectorIndex):
    axis = 'harmony'


class MelodyIndex:
    """旋律轴：乐句级索引（一乐句一条）。"""

    axis = 'melody'

    def __init__(self):
        self.pids = []        # 每条乐句所属作品
        self.fps = []         # 每条乐句的指纹（含 vec）

    def add_piece(self, pid, phrases):
        for fp in phrases or []:
            if not fp.get('vec'):
                continue
            self.pids.append(pid)
            self.fps.append(fp)

    def __len__(self):
        return len(self.pids)

    def query(self, target_phrases, top_n: int, exclude=None,
              phrase_top_m: int = 10, agg: str = 'max'):
        """旋律轴检索。

        ``target_phrases`` 是查询曲的乐句指纹列表。返回 [(piece_id, score)] 降序，
        score 是**精排后**的乐句级相似度（theme.compare 的 score）。
        """
        if not target_phrases:
            return []
        from core.analysis import theme as theme_mod

        # ---- 粗排：每个目标乐句，在全库乐句上取 Top-M ----
        # cand[pid] = {'best': 粗排最高分, 'pairs': [(目标乐句, 候选乐句), ...]}
        cand = {}
        for tf in target_phrases:
            tvec = tf.get('vec')
            if not tvec:
                continue
            hits = []
            for pid, cf in zip(self.pids, self.fps):
                if exclude is not None and pid == exclude:
                    continue
                sc = cosine(tvec, cf.get('vec'))
                hits.append((sc, pid, cf))
            hits.sort(key=lambda x: (-x[0], x[1]))
            for sc, pid, cf in hits[:phrase_top_m]:
                slot = cand.setdefault(pid, {'best': 0.0, 'pairs': []})
                slot['best'] = max(slot['best'], sc)
                slot['pairs'].append((tf, cf))

        # ---- 精排：在粗排命中的乐句对上跑完整 compare，取最大 / 均值 ----
        out = []
        for pid, slot in cand.items():
            if agg == 'mean_max':
                # 每个目标乐句各自找最佳匹配，再取均值（更稳，抗偶然相似）
                per_target = {}
                for tf, cf in slot['pairs']:
                    sc = theme_mod.compare(tf, cf)['score']
                    key = id(tf)
                    if sc > per_target.get(key, -1.0):
                        per_target[key] = sc
                scores = list(per_target.values())
                final = sum(scores) / len(scores) if scores else 0.0
            else:
                final = max((theme_mod.compare(tf, cf)['score']
                             for tf, cf in slot['pairs']), default=0.0)
            out.append((pid, round(final, 4)))
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out[:top_n]


class IndexBundle:
    """三个索引打包，建一次、查多次。"""

    def __init__(self, vectors):
        """``vectors`` 是 PieceVector 的可迭代对象（只要三轴字段齐全）。"""
        self.mode = ModeIndex()
        self.harmony = HarmonyIndex()
        self.melody = MelodyIndex()
        self.by_pid = {}
        for v in vectors:
            pid = v.piece_id
            self.by_pid[pid] = v
            if v.mode_vec:
                self.mode.add(pid, v.mode_vec, v.mode_uncertain)
            if v.harmony_vec:
                self.harmony.add(pid, v.harmony_vec)
            if v.melody_phrases:
                self.melody.add_piece(pid, v.melody_phrases)
        self.pids = sorted(self.by_pid)

    def axis_query(self, axis: str, target, setting, exclude=None):
        """单轴检索 Top-N。返回 [(piece_id, score)]。"""
        top_n = int(getattr(setting, 'top_n', 20) or 20)
        if axis == 'mode':
            vec = target.mode_vec
            if not vec:
                return []
            return self.mode.query(vec, top_n, exclude=exclude,
                                   uncertain=target.mode_uncertain)
        if axis == 'harmony':
            vec = target.harmony_vec
            if not vec:
                return []
            return self.harmony.query(vec, top_n, exclude=exclude)
        if axis == 'melody':
            if not target.melody_phrases:
                return []
            return self.melody.query(
                target.melody_phrases, top_n, exclude=exclude,
                phrase_top_m=int(getattr(setting, 'phrase_top_m', 10) or 10),
                agg=getattr(setting, 'melody_agg', 'max'))
        return []

    def search(self, target_pid, setting):
        """查询一首作品的相似作品：三轴各 Top-N → RRF 融合。

        返回按融合分降序的列表，每项带各轴的分数与排名（页面要解释"为什么像"）。
        """
        target = self.by_pid.get(target_pid)
        if target is None:
            return []
        ranks = {}
        for axis in AXES:
            ranks[axis] = self.axis_query(axis, target, setting,
                                          exclude=target_pid)
        return fuse(ranks, setting)


def fuse(ranks: dict, setting) -> list:
    """倒数排名融合（RRF）。

    ``score = Σ_axis w_axis / (k + rank_axis)``。
    没进某一轴 Top-N 的作品**不从该轴拿分** —— 这是 RRF 的语义：
    各轴只在自己的候选里投票，不是全库补一个默认排名。
    """
    k = int(getattr(setting, 'rrf_k', 60) or 60)
    weights = {
        'mode': float(getattr(setting, 'w_mode', 0.3) or 0.0),
        'harmony': float(getattr(setting, 'w_harmony', 0.3) or 0.0),
        'melody': float(getattr(setting, 'w_melody', 0.4) or 0.0),
    }
    fused = {}
    detail = {}
    for axis, lst in ranks.items():
        w = weights.get(axis, 0.0)
        if w <= 0:
            continue
        for rank, (pid, sc) in enumerate(lst, start=1):
            fused[pid] = fused.get(pid, 0.0) + w / (k + rank)
            d = detail.setdefault(pid, {'axes': {}, 'rrf': 0.0})
            d['axes'][axis] = {'rank': rank, 'score': sc}
    out = []
    for pid, val in fused.items():
        d = detail[pid]
        d['rrf'] = round(val, 6)
        out.append({
            'piece_id': pid,
            'rrf': round(val, 6),
            'axes': d['axes'],
            'mode_sim': d['axes'].get('mode', {}).get('score'),
            'harmony_sim': d['axes'].get('harmony', {}).get('score'),
            'melody_sim': d['axes'].get('melody', {}).get('score'),
        })
    out.sort(key=lambda d: (-d['rrf'], d['piece_id']))
    return out
