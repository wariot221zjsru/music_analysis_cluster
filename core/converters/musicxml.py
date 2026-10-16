"""MIDI / MusicXML 转换：接入层的统一出口。

三条输入路径在 MusicXML 处汇合（计划书第三章总体设计）：
    audio   → [MuScriptor 预留接口] → MIDI → music21 → MusicXML
    midi    →                          music21 → MusicXML
    musicxml→                          直接使用

存储策略（core/storage.py）：
  - 上传的原始音频是**临时文件**，转换成功后删除；
  - MIDI 与 MusicXML **永久保存且分目录**（media/midi/…、media/musicxml/…）；
  - 上传的 MIDI / MusicXML 源文件也复制进对应永久目录，随后清理临时源。
"""
from pathlib import Path

from .. import storage
from . import muscriptor


def midi_to_musicxml(midi_path: str | Path, output_path: str | Path) -> Path:
    """MIDI → MusicXML（music21）。"""
    from music21 import converter
    score = converter.parse(str(midi_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write('musicxml', fp=str(output_path))
    return output_path


def convert_piece(piece) -> dict:
    """把 Piece 统一转换到 MusicXML，并把 MIDI / MusicXML 写入永久目录。

    - 回写 ``piece.midi_file`` / ``piece.musicxml_file``（永久存储相对路径）；
    - 转换成功后清理临时源文件（音频 / 已入库的 MIDI、MusicXML 源）；
    - 转换失败时**保留**临时源，便于重试（可用 ``manage.py purge_tmp`` 清理）。

    返回摘要 dict（供 convert 步骤的 output 字段使用）。
    抛出的异常信息由 pipeline 捕获并写入 AnalysisResult.error。
    """
    summary = {'source_type': piece.source_type, 'notes': []}
    src = Path(piece.source_file.path)
    stem = src.stem
    workdir = storage.new_workdir(prefix=f'convert_p{piece.id}_')

    try:
        if piece.source_type == 'musicxml':
            # 直接入库为永久 MusicXML
            rel_xml = storage.store_musicxml(piece.id, src, src.name)
            piece.musicxml_file = rel_xml
            piece.save(update_fields=['musicxml_file'])
            summary['notes'].append('MusicXML 已入库永久保存（musicxml/）。')

        elif piece.source_type == 'midi':
            # 上传的 MIDI 即永久 MIDI
            rel_midi = storage.store_midi(piece.id, src, src.name)
            piece.midi_file = rel_midi
            # MIDI → MusicXML
            local_xml = workdir / f'{stem}.musicxml'
            midi_to_musicxml(src, local_xml)
            rel_xml = storage.store_musicxml(piece.id, local_xml, f'{stem}.musicxml')
            piece.musicxml_file = rel_xml
            piece.save(update_fields=['midi_file', 'musicxml_file'])
            summary['notes'].append(
                'MIDI 已入库永久保存（midi/），并经 music21 转换为 MusicXML 入库。')

        else:
            # 音频 → [MuScriptor 预留接口] → MIDI → MusicXML
            local_midi = workdir / f'{stem}.mid'
            result = muscriptor.transcribe(src, local_midi,
                                           contains_voice=piece.contains_voice)
            rel_midi = storage.store_midi(piece.id, local_midi, f'{stem}.mid')
            piece.midi_file = rel_midi
            local_xml = workdir / f'{stem}.musicxml'
            midi_to_musicxml(local_midi, local_xml)
            rel_xml = storage.store_musicxml(piece.id, local_xml, f'{stem}.musicxml')
            piece.musicxml_file = rel_xml
            piece.save(update_fields=['midi_file', 'musicxml_file'])
            summary.update({
                'num_instruments': result.num_instruments,
                'contains_voice': result.contains_voice,
                'warnings': result.warnings,
            })
            summary['notes'].append(
                '音频经 MuScriptor 转录为 MIDI（%s）入库，再转换为 MusicXML 入库。'
                % ('器乐+人声两遍管线' if piece.contains_voice else '纯器乐单遍管线'))
    finally:
        storage.cleanup_workdir(workdir)

    # 转换成功：清理临时源文件（音频 / 已入库的 MIDI、MusicXML 源）
    if storage.purge_source(piece):
        summary['notes'].append('临时源文件已清理，未做长期保存。')
    summary['storage'] = {
        'midi': piece.midi_file.name if piece.midi_file else '',
        'musicxml': piece.musicxml_file.name if piece.musicxml_file else '',
        'source_purged': piece.source_purged,
    }
    return summary
