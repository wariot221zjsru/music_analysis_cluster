"""聚类回归专用设置：独立 SQLite + 独立 media（与其它冒烟脚本互不干扰）。

从 musicagent.settings 继承（因此包含 cluster app），只覆盖库与媒体目录。
"""
from musicagent.settings import *  # noqa: F401,F403

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / '_smoke_cluster_db.sqlite3',  # noqa: F405
    }
}

MEDIA_ROOT = BASE_DIR / '_smoke_cluster_media'  # noqa: F405
