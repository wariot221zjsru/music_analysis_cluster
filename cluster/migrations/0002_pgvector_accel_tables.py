"""pgvector 加速表（可选层，缺失自动跳过）。

设计理念
--------
三轴向量的**权威数据**是 PieceVector 的 JSONField，任何环境都完整可用。
本迁移只是给 PostgreSQL 加一层"原生向量"加速：建两张带 vector 列的表 + IVFFlat
余弦索引，由 index.py 的 pgvector 路径查询（见 index.pg_available）。

因此整段 forwards 包在 try/except 里：
  * 数据库不是 PostgreSQL          → 跳过
  * 服务端没装 pgvector 扩展        → 跳过
  * CREATE EXTENSION 需要 superuser 但当前账号没权限 → 跳过
不论哪种情况，**聚类 / 检索功能都不受影响**（自动回退纯 Python 余弦）。

表结构
------
  cluster_pgvec_piece   一曲一行：mode_vec(12) + harmony_vec(12)
  cluster_pgvec_phrase  一乐句一行：vec(42) + payload(jsonb 指纹，精排用)
"""
from django.db import migrations

PIECE_TABLE = 'cluster_pgvec_piece'
PHRASE_TABLE = 'cluster_pgvec_phrase'


def _vector_usable(schema_editor) -> bool:
    """服务端是否真的能用 vector 类型。"""
    conn = schema_editor.connection
    try:
        with conn.cursor() as c:
            # 扩展必须"已安装"且"可安装"（available）；只装了才算真能用
            c.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            installed = c.fetchone() is not None
            if not installed:
                c.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
                return c.fetchone() is not None
            return True
    except Exception:
        return False


def forwards(apps, schema_editor):
    if not _vector_usable(schema_editor):
        return  # 没有 pgvector → 纯 Python 后端照常工作，什么都不做
    conn = schema_editor.connection
    try:
        with conn.cursor() as c:
            c.execute('CREATE EXTENSION IF NOT EXISTS vector')

            c.execute(f'''
                CREATE TABLE IF NOT EXISTS {PIECE_TABLE} (
                    piece_id    integer PRIMARY KEY,
                    mode_vec    vector(12),
                    harmony_vec vector(12),
                    updated_at  timestamptz
                )
            ''')
            c.execute(f'''
                CREATE TABLE IF NOT EXISTS {PHRASE_TABLE} (
                    id         bigserial PRIMARY KEY,
                    piece_id   integer NOT NULL,
                    phrase_no  integer,
                    vec        vector(42),
                    payload    jsonb
                )
            ''')
            c.execute(f'CREATE INDEX IF NOT EXISTS pgvec_piece_mode '
                      f'ON {PIECE_TABLE} USING ivfflat (mode_vec vector_cosine_ops) '
                      f'WITH (lists = 100)')
            c.execute(f'CREATE INDEX IF NOT EXISTS pgvec_piece_harmony '
                      f'ON {PIECE_TABLE} USING ivfflat (harmony_vec vector_cosine_ops) '
                      f'WITH (lists = 100)')
            c.execute(f'CREATE INDEX IF NOT EXISTS pgvec_phrase_vec '
                      f'ON {PHRASE_TABLE} USING ivfflat (vec vector_cosine_ops) '
                      f'WITH (lists = 100)')
            c.execute(f'CREATE INDEX IF NOT EXISTS pgvec_phrase_piece '
                      f'ON {PHRASE_TABLE} (piece_id)')
    except Exception:
        # 没权限 / 扩展装不上 / 版本不支持 → 静默降级，不影响主流程
        pass


def backwards(apps, schema_editor):
    conn = schema_editor.connection
    try:
        with conn.cursor() as c:
            c.execute(f'DROP TABLE IF EXISTS {PHRASE_TABLE}')
            c.execute(f'DROP TABLE IF EXISTS {PIECE_TABLE}')
    except Exception:
        pass


class Migration(migrations.Migration):

    dependencies = [
        ('cluster', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
