"""冒烟测试专用设置：SQLite + **独立** media 目录（不依赖 PostgreSQL 凭据）。

关键：MEDIA_ROOT 必须隔离。以前它继承 musicagent.settings 的 `BASE_DIR/media`，
于是冒烟测试会
  ① 把测试产物写进真实 media（污染用户数据）；
  ② 收尾清理时 walk 整个 media/melody/ 删文件，把**真实作品的导出产物一起删掉**
     —— 用户点「主旋律 MIDI ⬇」就 404（文件在库里有路径，磁盘上没了）。
改成独立的 _smoke_media 后，两类问题一起消失。
"""
from musicagent.settings import *  # noqa: F401,F403

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / '_smoke_db.sqlite3',  # noqa: F405
    }
}

# 独立的媒体目录：与真实项目完全隔离，测试可随意清理
MEDIA_ROOT = BASE_DIR / '_smoke_media'  # noqa: F405
