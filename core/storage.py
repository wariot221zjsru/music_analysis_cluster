"""文件存储策略（对应需求）。

约定：
  - 上传的**原始音频**只作临时文件存放（``media/tmp/``），转换成功后立即删除，
    不做长期保存；转换失败时保留，便于重试，可用 ``manage.py purge_tmp`` 清理。
  - **MIDI 与 MusicXML 永久保存，且分目录存放**：
        media/midi/<日期>/p<作品id>/xxx.mid          永久 MIDI
        media/musicxml/<日期>/p<作品id>/xxx.musicxml  永久 MusicXML
  - 上传的 MIDI / MusicXML 源文件同样会被复制进对应的永久目录，
    随后原始临时源一并清理。

所有写入统一走 Django 的 ``default_storage``（FileSystemStorage → MEDIA_ROOT），
返回值即 ``FileField.name``（存储相对路径），可直接赋给模型字段。
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from django.core.files import File
from django.core.files.storage import default_storage
from django.utils import timezone

logger = logging.getLogger(__name__)

# 存储子目录名
MIDI_DIR = 'midi'
MUSICXML_DIR = 'musicxml'
TMP_DIR = 'tmp'


def _safe_name(name: str) -> str:
    """只保留文件名，避免路径逃逸。"""
    return Path(str(name)).name


def target_dir(kind: str, piece_id: int) -> str:
    """构造某个作品某类产物的存储目录，例如 ``midi/20261015/p17``。

    按作品 id 再分一层，保证同名文件互不覆盖，同时保持文件名可读。
    """
    return f'{kind}/{timezone.localdate():%Y%m%d}/p{piece_id}'


def store_file(rel_dir: str, src_path, filename: str) -> str:
    """把本地文件复制进永久存储目录，返回存储相对路径。"""
    dest = f'{rel_dir.strip("/")}/{_safe_name(filename)}'
    if default_storage.exists(dest):
        default_storage.delete(dest)
    with Path(src_path).open('rb') as fh:
        return default_storage.save(dest, File(fh))


def store_midi(piece_id: int, src_path, filename: str) -> str:
    """永久保存 MIDI，返回存储相对路径。"""
    return store_file(target_dir(MIDI_DIR, piece_id), src_path, filename)


def store_musicxml(piece_id: int, src_path, filename: str) -> str:
    """永久保存 MusicXML，返回存储相对路径。"""
    return store_file(target_dir(MUSICXML_DIR, piece_id), src_path, filename)


def delete_if_exists(storage_name: str) -> None:
    """按存储相对路径删除文件（不存在则忽略）。"""
    if not storage_name:
        return
    try:
        if default_storage.exists(storage_name):
            default_storage.delete(storage_name)
    except Exception:  # noqa: BLE001
        logger.warning('删除存储文件失败：%s', storage_name, exc_info=True)


def purge_source(piece) -> bool:
    """清理作品的临时源文件（音频 / 已入库的 MIDI、MusicXML 源）。

    返回是否执行了删除。删除后清空 ``source_file`` 并置 ``source_purged=True``。
    """
    if not piece.source_file:
        return False
    name = piece.source_file.name
    try:
        piece.source_file.delete(save=False)     # 删除 MEDIA 中的物理文件
    except Exception:  # noqa: BLE001
        logger.warning('删除临时源文件失败：%s', name, exc_info=True)
        delete_if_exists(name)
    piece.source_file = ''
    piece.source_purged = True
    piece.save(update_fields=['source_file', 'source_purged'])
    return True


def new_workdir(prefix: str) -> Path:
    """转换过程的临时工作目录（系统临时区，不落在 media 下）。"""
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_workdir(path) -> None:
    """删除临时工作目录。"""
    shutil.rmtree(str(path), ignore_errors=True)
