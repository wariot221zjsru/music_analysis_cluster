"""清理 media/tmp 下的过期临时文件。

原始音频是临时文件，转换成功后会被自动删除；
转换失败时源文件会保留，方便重试，此命令用于定期清理这些残留。

用法：
    python manage.py purge_tmp            # 清理 24 小时前的临时文件
    python manage.py purge_tmp --hours 1  # 清理 1 小时前的临时文件
"""
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = '清理 media/tmp 下超过指定小时数的临时文件（音频源残留）。'

    def add_arguments(self, parser):
        parser.add_argument('--hours', type=int, default=24,
                            help='保留最近 N 小时内的临时文件（默认 24）')

    def handle(self, *args, **options):
        hours = options['hours']
        cutoff = time.time() - hours * 3600
        root = Path(settings.MEDIA_ROOT) / 'tmp'
        removed = 0
        if root.exists():
            for f in root.rglob('*'):
                if f.is_file() and f.stat().st_mtime < cutoff:
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass
            # 自下而上清理空目录
            for d in sorted((p for p in root.rglob('*') if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
                try:
                    d.rmdir()
                except OSError:
                    pass
        self.stdout.write(self.style.SUCCESS(
            f'已清理 {removed} 个临时文件（早于 {hours} 小时）。'))
