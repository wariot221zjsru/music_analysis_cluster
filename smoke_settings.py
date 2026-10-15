"""冒烟测试专用设置：SQLite（不依赖 PostgreSQL 凭据）。"""
from musicagent.settings import *  # noqa: F401,F403

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / '_smoke_db.sqlite3',  # noqa: F405
    }
}
