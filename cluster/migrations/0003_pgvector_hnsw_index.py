"""把三轴的 IVFFlat 索引换成 HNSW。

为什么要换
----------
0002 建的是 IVFFlat（lists = 100）。IVFFlat 查询只扫 probes 个聚类中心
（默认 probes = 1），当表里只有几十行、却被分成 100 个 list 时，大量 list 是空的
——实测后果是**漏召回**：7 首曲子里每首都少召回 1 条，其中一首（#5）甚至一条都查不到，
pg 路径与纯 Python 精确余弦对不上。

HNSW 不需要按数据量调 probes，配合连接参数 ``hnsw.ef_search=100``
（见 settings.py 的 DATABASES OPTIONS），在现有规模上召回与精确余弦一致；
将来曲库涨到几万条也能直接用。

没有 pgvector 的环境整体跳过（与 0002 同策略）。
"""
from django.db import migrations

PIECE_TABLE = 'cluster_pgvec_piece'
PHRASE_TABLE = 'cluster_pgvec_phrase'

DROP_SQL = [
    'DROP INDEX IF EXISTS pgvec_piece_mode',
    'DROP INDEX IF EXISTS pgvec_piece_harmony',
    'DROP INDEX IF EXISTS pgvec_phrase_vec',
]

CREATE_SQL = [
    f'CREATE INDEX IF NOT EXISTS pgvec_piece_mode ON {PIECE_TABLE} '
    f'USING hnsw (mode_vec vector_cosine_ops) '
    f'WITH (m = 16, ef_construction = 64)',
    f'CREATE INDEX IF NOT EXISTS pgvec_piece_harmony ON {PIECE_TABLE} '
    f'USING hnsw (harmony_vec vector_cosine_ops) '
    f'WITH (m = 16, ef_construction = 64)',
    f'CREATE INDEX IF NOT EXISTS pgvec_phrase_vec ON {PHRASE_TABLE} '
    f'USING hnsw (vec vector_cosine_ops) '
    f'WITH (m = 16, ef_construction = 64)',
]


def _vector_usable(schema_editor) -> bool:
    conn = schema_editor.connection
    try:
        with conn.cursor() as c:
            c.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if c.fetchone() is None:
                return False
            c.execute('SELECT to_regclass(%s)', [PIECE_TABLE])
            row = c.fetchone()
            return bool(row and row[0])
    except Exception:
        return False


def forwards(apps, schema_editor):
    if not _vector_usable(schema_editor):
        return
    with schema_editor.connection.cursor() as c:
        for sql in DROP_SQL + CREATE_SQL:
            c.execute(sql)


def backwards(apps, schema_editor):
    """退回 IVFFlat（保持 0002 的形态）。"""
    if not _vector_usable(schema_editor):
        return
    with schema_editor.connection.cursor() as c:
        for sql in DROP_SQL:
            c.execute(sql)
        for sql in [
            f'CREATE INDEX IF NOT EXISTS pgvec_piece_mode ON {PIECE_TABLE} '
            f'USING ivfflat (mode_vec vector_cosine_ops) WITH (lists = 100)',
            f'CREATE INDEX IF NOT EXISTS pgvec_piece_harmony ON {PIECE_TABLE} '
            f'USING ivfflat (harmony_vec vector_cosine_ops) WITH (lists = 100)',
            f'CREATE INDEX IF NOT EXISTS pgvec_phrase_vec ON {PHRASE_TABLE} '
            f'USING ivfflat (vec vector_cosine_ops) WITH (lists = 100)',
        ]:
            c.execute(sql)


class Migration(migrations.Migration):

    dependencies = [
        ('cluster', '0002_pgvector_accel_tables'),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
