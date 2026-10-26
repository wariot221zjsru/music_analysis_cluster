"""pgvector 打通脚本（幂等，可重复运行）。

按顺序做完下面 6 件事，任何一步失败都明确报出原因、不静默跳过：

    1. 服务端是否识别 vector 扩展（pg_available_extensions）
    2. music_agent 库里是否已 CREATE EXTENSION（没有就试着装）
    3. 回滚并重跑 cluster 0002 迁移（建 cluster_pgvec_piece / _phrase + 索引）
    4. 校验表、列、索引是否真的建出来
    5. 重建特征（refresh_vector → sync_pg 把三轴灌进加速表）
    6. 真跑一次三轴检索，确认走的是 pgvector 而不是纯 Python

用法：

    python check_pgvector.py            # 全流程
    python check_pgvector.py --no-rebuild   # 跳过第 5 步（只想看表/索引）
"""
import os
import sys

try:  # Windows 控制台默认 GBK，避免中文输出乱码
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'musicagent.settings')

import django  # noqa: E402

django.setup()

from django.core.management import call_command  # noqa: E402
from django.db import connection  # noqa: E402

NO_REBUILD = '--no-rebuild' in sys.argv
SKIP_MIGRATE = '--skip-migrate' in sys.argv  # 表已建好时别再 DROP 重建（会清数据）

ok = True


def report(step, passed, detail=''):
    global ok
    if not passed:
        ok = False
    print(f'[{"PASS" if passed else "FAIL"}] {step}' + (f' — {detail}' if detail else ''))


def q(sql, args=None):
    with connection.cursor() as c:
        c.execute(sql, args)
        return c.fetchall()


# ---------------------------------------------------------------- 1. 扩展可识别
print('\n=== 1. 服务端是否识别 vector ===')
with connection.cursor() as c:
    c.execute('SHOW server_version')
    version = c.fetchone()[0]
print(f'PostgreSQL {version}')
avail = q("SELECT default_version FROM pg_available_extensions WHERE name='vector'")
report('1 服务端已加载 vector（dll + control 就位）', bool(avail),
       f'default_version={avail[0][0]}' if avail else
       'lib\\vector.dll 或 share\\extension\\vector.control 没放对位置')
if not avail:
    sys.exit(1)

# ---------------------------------------------------------------- 2. 库内已启用
print('\n=== 2. music_agent 库是否已启用 vector ===')
installed = q("SELECT extversion FROM pg_extension WHERE extname='vector'")
if not installed:
    try:
        q('CREATE EXTENSION vector')
        installed = q("SELECT extversion FROM pg_extension WHERE extname='vector'")
    except Exception as e:
        report('2 CREATE EXTENSION vector', False,
               f'{type(e).__name__}: {str(e).splitlines()[0]}')
        print('\n  该库未启用扩展，且当前账号不是超级用户。')
        print('  请任选一条（管理员执行），然后重跑本脚本：')
        print('    a) psql -U postgres -h localhost -d music_agent -c "CREATE EXTENSION vector;"')
        print('    b) 管理员 cmd: echo trusted = true >> '
              '"D:\\Program Files\\PostgreSQL\\18\\share\\extension\\vector.control"')
        sys.exit(1)
report('2 库内已启用 vector', True, f'extversion={installed[0][0]}')

# ---------------------------------------------------------------- 3. 重跑迁移
print('\n=== 3. 回滚并重跑 cluster 0002（建加速表 + IVFFlat 索引）===')
if SKIP_MIGRATE:
    print('   （--skip-migrate：保留现有表与数据，跳过重建）')
else:
    call_command('migrate', 'cluster', '0001', verbosity=0)
    call_command('migrate', 'cluster', verbosity=2)

# ---------------------------------------------------------------- 4. 校验表/索引
print('\n=== 4. 加速表与索引 ===')
tables = {r[0] for r in q("SELECT tablename FROM pg_tables "
                          "WHERE tablename LIKE 'cluster_pgvec%'")}
report('4a cluster_pgvec_piece 存在', 'cluster_pgvec_piece' in tables)
report('4b cluster_pgvec_phrase 存在', 'cluster_pgvec_phrase' in tables)
if 'cluster_pgvec_piece' not in tables:
    sys.exit(1)

cols = q("""SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_name IN ('cluster_pgvec_piece', 'cluster_pgvec_phrase')
            ORDER BY table_name, column_name""")
print('   列：')
for name, dt, udt in cols:
    print(f'     {name:<14} {udt if udt == "vector" else dt}')

idxs = q("""SELECT indexname, indexdef FROM pg_indexes
            WHERE tablename LIKE 'cluster_pgvec%' ORDER BY indexname""")
print('   索引：')
for name, ddl in idxs:
    print(f'     {name}')
    print(f'       {ddl}')
report('4c 三个余弦向量索引已建',
       sum('vector_cosine_ops' in d for _n, d in idxs) >= 3,
       f'共 {len(idxs)} 个索引')

# ---------------------------------------------------------------- 5. 重建特征
if NO_REBUILD:
    print('\n=== 5. 跳过重建（--no-rebuild）===')
else:
    print('\n=== 5. 重建特征 → 灌入加速表 ===')
    from core.models import Piece  # noqa: E402
    from cluster.clustering import refresh_vector  # noqa: E402

    pieces = list(Piece.objects.filter(is_public=True).order_by('id'))
    print(f'   公开作品 {len(pieces)} 首')
    for p in pieces:
        v = refresh_vector(p)
        print(f'   #{p.id} {p.title[:24]:<24} status={v.status} '
              f'mode={"Y" if v.mode_vec else "-"} '
              f'harm={"Y" if v.harmony_vec else "-"} '
              f'phr={v.melody_phrase_count}')
    report('5 特征重建完成', True)

print('\n=== 6. 加速表数据 + 检索是否走 pgvector ===')
n_piece = q('SELECT count(*) FROM cluster_pgvec_piece')[0][0]
n_phrase = q('SELECT count(*) FROM cluster_pgvec_phrase')[0][0]
print(f'   cluster_pgvec_piece  = {n_piece} 行')
print(f'   cluster_pgvec_phrase = {n_phrase} 行')
report('6a 加速表有数据', n_piece > 0, f'piece={n_piece}, phrase={n_phrase}')

from cluster import index as cindex  # noqa: E402

cindex._PG_OK = None
report('6b index.pg_available() 为 True', cindex.pg_available())

if n_piece:
    from cluster.models import PieceVector  # noqa: E402

    bundle = cindex.IndexBundle(PieceVector.objects.exclude(status='error'))
    print(f'   参与对照的作品向量：{len(bundle.pids)} 条')

    probe = q('SELECT piece_id FROM cluster_pgvec_piece '
              'WHERE mode_vec IS NOT NULL ORDER BY piece_id LIMIT 1')
    if probe and probe[0][0] in bundle.by_pid:
        pid = probe[0][0]
        mv = q('SELECT mode_vec FROM cluster_pgvec_piece WHERE piece_id=%s', [pid])[0][0]
        hits = cindex.ModeIndex()._pg_query(
            [float(x) for x in str(mv).strip('[]').split(',')], 5, exclude=pid)
        print(f'   <=> 原生查询 Top5（排除自身 #{pid}）: {hits}')
        report('6c 原生 <=> 余弦查询可用', bool(hits))

    # 6d：逐首对照 pg 路径 vs 纯 Python 路径的完整排序
    #     IVFFlat 是近似索引，lists=100 而只有几条数据时可能漏召回 → 必须测
    full = max(len(bundle.pids) - 1, 1)
    mismatch = []
    for pid in bundle.pids:
        tgt = bundle.by_pid[pid]
        if not tgt.mode_vec or tgt.mode_uncertain:
            continue
        pg = cindex.ModeIndex()._pg_query(tgt.mode_vec, full, exclude=pid)
        # 纯 Python 精确余弦（不走 pg），作为对照基准
        py = sorted(
            [(op, round(cindex.cosine(tgt.mode_vec, bundle.by_pid[op].mode_vec), 4))
             for op in bundle.pids if op != pid and bundle.by_pid[op].mode_vec],
            key=lambda kv: (-kv[1], kv[0]))[:full]
        same_ids = [p for p, _ in pg] == [p for p, _ in py]
        same_score = all(abs(a - b) < 1e-3
                         for (_p1, a), (_p2, b) in zip(pg, py))
        if not (same_ids and same_score):
            mismatch.append((pid, pg, py))
        print(f'   #{pid}: pg={pg}')
        print(f'        py={py}  {"一致" if same_ids and same_score else "★不一致"}')
    report('6d pg 路径与纯 Python 结果完全一致（IVFFlat 未漏召回）',
           not mismatch, f'{len(mismatch)} 首不一致' if mismatch else '全部一致')

print('\n' + ('全部通过 ✅' if ok else '存在失败项 ❌（见上方 FAIL）'))
sys.exit(0 if ok else 1)
