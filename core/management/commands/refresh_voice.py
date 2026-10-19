"""按作品文件重新判定「是否含人声」并回写 Piece.contains_voice。

口径（与 convert_piece 一致）：
  - MIDI / MusicXML：只认文件里的实际声部，覆盖上传时的勾选；
  - 音频：只有谱面检出人声才置为 True（Muscriptor 转录结果不在此重算）。

用途：本规则上线前已入库的作品，勾选值可能与实际不符，跑一次即可对齐。

用法：
    python manage.py refresh_voice            # 只处理需要修正的
    python manage.py refresh_voice --all      # 全部重算（并打印明细）
"""
from django.core.management.base import BaseCommand

from core.analysis.scoredata import detect_voice
from core.models import Piece


class Command(BaseCommand):
    help = '按文件内实际声部重新判定并回写「是否含人声」。'

    def add_arguments(self, parser):
        parser.add_argument('--all', action='store_true', dest='all',
                            help='全部重算并打印明细（默认只打印被修正的）')

    def handle(self, *args, **options):
        show_all = options['all']
        checked = fixed = failed = 0
        for piece in Piece.objects.exclude(musicxml_file='').order_by('id'):
            try:
                info = detect_voice(piece.musicxml_file.path)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                self.stdout.write(self.style.WARNING(
                    f'#{piece.id} {piece.title}：谱面读取失败（{exc}），跳过。'))
                continue
            checked += 1
            has_voice = bool(info['has_voice'])
            if piece.source_type == 'audio' and not has_voice:
                # 音频：谱面没检出人声时不动（转录管线的结果不能靠重解析推翻）
                if show_all:
                    self.stdout.write(
                        f'#{piece.id} {piece.title}：音频源，谱面未检出人声，'
                        f'保持 {piece.contains_voice}。')
                continue
            if has_voice != piece.contains_voice:
                old = piece.contains_voice
                piece.contains_voice = has_voice
                piece.save(update_fields=['contains_voice', 'updated_at'])
                fixed += 1
                self.stdout.write(self.style.SUCCESS(
                    f'#{piece.id} {piece.title}：{old} → {has_voice}'
                    f'（人声声部：'
                    f'{"、".join(p["name"] for p in info["voice_parts"]) or "—"}）'))
            elif show_all:
                self.stdout.write(
                    f'#{piece.id} {piece.title}：{piece.contains_voice}（一致）'
                    f'共 {info["part_count"]} 个声部。')
        self.stdout.write(self.style.SUCCESS(
            f'检查 {checked} 个作品，修正 {fixed} 个，失败 {failed} 个。'))
