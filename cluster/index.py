"""向量索引 + 三轴检索 + RRF 融合。

## 三个索引

  调式 mode     一曲一向量（12 维，已按主音对齐）→ 余弦，**不循环移位**
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

## 后端：纯 Python（默认）+ pgvector（可选加速）

**权威数据是 PieceVector 的 JSON 字段**，任何环境都能跑，纯 Python 暴力余弦
（几百首、几千条乐句向量，毫秒级，零新依赖）。

若 PostgreSQL 装了 pgvector 扩展且加速表存在（见 0002 迁移），下面的
``mode`` / ``harmony`` 轴检索与 ``melody`` 粗排会自动改走原生 ``<=>`` 余弦查询
（走 IVFFlat 索引），不再在 Python 里循环。用不了的场景自动回退：

  * 数据库不是 PostgreSQL / 没装 vector 扩展 / 加速表没建
  * 调式轴 ``uncertain=True``（主音没定，需要"循环移位取最大"，
    pgvector 的 <=> 只能比固定向量，做不了移位 → 必须回退）

``VectorIndex`` 是抽象基类，上层（检索、RRF、聚类）接口完全不变 ——
换后端只需动本文件。
"""
from . import features

AXES = ('mode', 'harmony', 'melody')

# pgvector 加速表（由 0002 迁移在检测到扩展时创建）
PG_PIECE_TABLE = 'cluster_pgvec_piece'
PG_PHRASE_TABLE = 'cluster_pgvec_phrase'

_PG_OK = None


# ---------------------------------------------------------------- pgvector 探测

def pg_available() -> bool:
    """PostgreSQL + pgvector 扩展 + 加速表是否可用（结果缓存，只查一次）。"""
    global _PG_OK
    if _PG_OK is None:
        _PG_OK = False
        try:
            from django.db import connection
            with connection.cursor() as c:
                c.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
                if c.fetchone() is None:
                    return False
                c.execute('SELECT to_regclass(%s)', [PG_PIECE_TABLE])
                row = c.fetchone()
                _PG_OK = bool(row and row[0])
        except Exception:
            _PG_OK = False
    return _PG_OK


def _fmt_vec(vec) -> str:
    """Python 列表 → pgvector 字面量 '[x,y,z]'。"""
    return '[' + ','.join(f'{float(x):.6f}' for x in vec) + ']'


def cosine(a, b) -> float:
    """余弦相似度。"""
    return features._cosine(a, b)


# ---------------------------------------------------------------- 单向量轴

class VectorIndex:
    """单向量轴：一曲一向量，余弦检索。可用 pgvector 时自动走 SQL。"""

    axis = 'generic'
    # pgvector 加速表里的列名（mode/harmony 子类覆盖）
    pg_col = None

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

    # ---- pgvector 路径 ----

    def _pg_query(self, vec, top_n: int, exclude=None):
        """用原生 <=> 查 Top-N（余弦距离 → 相似度 1 - dist）。"""
        from django.db import connection

        col = self.pg_col
        head = [f'SELECT piece_id, 1 - ({col} <=> %s::vector) AS sim']
        where = [f'FROM {PG_PIECE_TABLE}', f'WHERE {col} IS NOT NULL']
        params = [_fmt_vec(vec)]
        if exclude is not None:
            where.append('AND piece_id != %s')
            params.append(exclude)
        tail = [f'ORDER BY {col} <=> %s::vector', 'LIMIT %s']
        params += [_fmt_vec(vec), int(top_n)]
        sql = ' '.join(head + where + tail)
        with connection.cursor() as c:
            c.execute(sql, params)
            return [(r[0], round(float(r[1]), 4)) for r in c.fetchall()]

    # ---- 对外查询 ----

    def query(self, vec, top_n: int, exclude=None, uncertain: bool = False):
        """检索 Top-N。返回 [(piece_id, score)] 降序。

        pgvector 走不通（或调式轴需要循环移位）时自动回退纯 Python，
        两条路径结果一致（都是余弦，只是 pg 那边由索引近似加速）。
        """
        if not vec:
            return []
        # 主音不确定 → 需要循环移位，pgvector 的 <=> 做不到，只能回退
        shift = self.axis == 'mode' and bool(uncertain)
        if pg_available() and not shift and self.pg_col:
            try:
                return self._pg_query(vec, top_n, exclude)
            except Exception:
                pass  # 任何 pg 层异常都静默回退，保证功能可用
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
    pg_col = 'mode_vec'


class HarmonyIndex(VectorIndex):
    axis = 'harmony'
    pg_col = 'harmony_vec'


# ---------------------------------------------------------------- 旋律轴

class MelodyIndex:
    """旋律轴：乐句级索引（一乐句一条）。粗排可用 pgvector。"""

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

    def _pg_coarse(self, target_phrases, exclude, phrase_top_m):
        """粗排：每个目标乐句用 pgvector 取 Top-M。返回 cand 字典。"""
        from django.db import connection

        cand = {}
        with connection.cursor() as c:
            for tf in target_phrases:
                tvec = tf.get('vec')
                if not tvec:
                    continue
                qv = _fmt_vec(tvec)
                params = [qv]
                excl_sql = ''
                if exclude is not None:
                    excl_sql = ' AND piece_id != %s'
                    params.append(exclude)
                sql = (f'SELECT piece_id, payload, 1 - (vec <=> %s::vector) AS sim '
                       f'FROM {PG_PHRASE_TABLE} WHERE vec IS NOT NULL{excl_sql} '
                       f'ORDER BY vec <=> %s::vector LIMIT %s')
                params += [qv, int(phrase_top_m)]
                c.execute(sql, params)
                for piece_id, payload, _sim in c.fetchall():
                    cf = dict(payload) if payload else {}
                    slot = cand.setdefault(piece_id, {'pairs': []})
                    slot['pairs'].append((tf, cf))
        return cand

    def _coarse(self, target_phrases, exclude, phrase_top_m):
        """粗排（纯 Python）：每个目标乐句在全库乐句上取 Top-M。"""
        cand = {}
        for tf in target_phrases:
            tvec = tf.get('vec')
            if not tvec:
                continue
            hits = []
            for pid, cf in zip(self.pids, self.fps):
                if exclude is not None and pid == exclude:
                    continue
                hits.append((cosine(tvec, cf.get('vec')), pid, cf))
            hits.sort(key=lambda x: (-x[0], x[1]))
            for _sc, pid, cf in hits[:phrase_top_m]:
                slot = cand.setdefault(pid, {'pairs': []})
                slot['pairs'].append((tf, cf))
        return cand

    @staticmethod
    def _fine(cand, agg: str):
        """精排：在粗排命中的乐句对上跑 theme.compare，取最大 / 各乐句均值。"""
        from core.analysis import theme as theme_mod

        out = []
        for pid, slot in cand.items():
            results = [(tf, cf, theme_mod.compare(tf, cf)['score'])
                       for tf, cf in slot['pairs']]
            if agg == 'mean_max':
                # 每个目标乐句各自找最佳匹配，再取均值（更稳，抗偶然相似）
                per_target = {}
                for tf, _cf, sc in results:
                    key = id(tf)
                    if sc > per_target.get(key, -1.0):
                        per_target[key] = sc
                scores = list(per_target.values())
                final = sum(scores) / len(scores) if scores else 0.0
            else:
                final = max((sc for _tf, _cf, sc in results), default=0.0)
            out.append((pid, round(final, 4)))
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out

    def query(self, target_phrases, top_n: int, exclude=None,
              phrase_top_m: int = 10, agg: str = 'max'):
        """旋律轴检索。

        ``target_phrases`` 是查询曲的乐句指纹列表。返回 [(piece_id, score)] 降序，
        score 是**精排后**的乐句级相似度（theme.compare 的 score）。
        """
        if not target_phrases:
            return []
        cand = None
        if pg_available():
            try:
                cand = self._pg_coarse(target_phrases, exclude, phrase_top_m)
            except Exception:
                cand = None  # 静默回退
        if cand is None:
            cand = self._coarse(target_phrases, exclude, phrase_top_m)
        return self._fine(cand, agg)[:top_n]


# ---------------------------------------------------------------- 打包

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


# ---------------------------------------------------------------- RRF 融合

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


# ---------------------------------------------------------------- pgvector 同步

def sync_pg(vec_row):
    """把一首作品的三个轴写进 pgvector 加速表（无 pg 环境则什么也不做）。

    在 clustering.refresh_vector 里调用。加速表是**纯缓存**：
    JSON 字段才是权威数据，这里失败不影响聚类 / 检索正确性。
    """
    if not pg_available():
        return
    from django.db import connection
    from psycopg.types.json import Json

    pid = vec_row.piece_id
    try:
        with connection.cursor() as c:
            c.execute(
                f'INSERT INTO {PG_PIECE_TABLE} '
                '(piece_id, mode_vec, harmony_vec, updated_at) '
                'VALUES (%s, %s::vector, %s::vector, now()) '
                'ON CONFLICT (piece_id) DO UPDATE SET '
                'mode_vec = EXCLUDED.mode_vec, '
                'harmony_vec = EXCLUDED.harmony_vec, updated_at = now()',
                [pid,
                 _fmt_vec(vec_row.mode_vec) if vec_row.mode_vec else None,
                 _fmt_vec(vec_row.harmony_vec) if vec_row.harmony_vec else None])

            c.execute(f'DELETE FROM {PG_PHRASE_TABLE} WHERE piece_id = %s', [pid])
            rows = []
            for no, fp in enumerate(vec_row.melody_phrases or []):
                v = fp.get('vec')
                if not v:
                    continue
                rows.append((pid, no, _fmt_vec(v), Json(fp)))
            if rows:
                c.executemany(
                    f'INSERT INTO {PG_PHRASE_TABLE} '
                    '(piece_id, phrase_no, vec, payload) '
                    'VALUES (%s, %s, %s::vector, %s)', rows)
    except Exception:
        pass  # 加速表不可用时静默降级，主流程照走


def drop_pg(piece_id):
    """删除某首作品的加速表行（作品删除 / 特征清空时用）。"""
    if not pg_available():
        return
    from django.db import connection
    try:
        with connection.cursor() as c:
            c.execute(f'DELETE FROM {PG_PIECE_TABLE} WHERE piece_id = %s', [piece_id])
            c.execute(f'DELETE FROM {PG_PHRASE_TABLE} WHERE piece_id = %s', [piece_id])
    except Exception:
        pass
