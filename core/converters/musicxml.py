"""MIDI / MusicXML 转换：接入层的统一出口。

三条输入路径在 MusicXML 处汇合（计划书第三章总体设计）：
    audio   → [MuScriptor 预留接口] → MIDI → music21 → MusicXML
    midi    →                          music21 → MusicXML
    musicxml→                          直接使用
"""
from pathlib import Path

from django.conf import settings
from django.core.files.base import File

from . import muscriptor


def _rel_to_media(path: Path) -> str:
    """把 MEDIA_ROOT 下的绝对路径转成存储相对路径（FileField.name）。"""
    return str(Path(path).resolve().relative_to(Path(settings.MEDIA_ROOT).resolve())).replace('\\', '/')


def midi_to_musicxml(midi_path: str | Path, output_path: str | Path) -> Path:
    """MIDI → MusicXML（music21）。"""
    from music21 import converter
    score = converter.parse(str(midi_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write('musicxml', fp=str(output_path))
    return output_path


def convert_piece(piece) -> dict:
    """把 Piece 统一转换到 MusicXML，回写 piece.musicxml_file / piece.midi_file。

    返回摘要 dict（供 convert 步骤的 output 字段使用）。
    抛出的异常信息由 pipeline 捕获并写入 AnalysisResult.error。
    """
    summary = {'source_type': piece.source_type, 'notes': []}
    src = Path(piece.source_file.path)

    if piece.source_type == 'musicxml':
        # 直接使用上传的 MusicXML
        piece.musicxml_file = piece.source_file.name
        summary['notes'].append('直接使用上传的 MusicXML 文件。')
        piece.save()
        return summary

    if piece.source_type == 'midi':
        # MIDI → MusicXML
        piece.midi_file = piece.source_file.name
        out_xml = src.parent / (src.stem + '.musicxml')
        midi_to_musicxml(src, out_xml)
        piece.musicxml_file = _rel_to_media(out_xml)
        piece.save()
        summary['notes'].append('MIDI 已通过 music21 转换为 MusicXML。')
        return summary

    # 音频 → [MuScriptor 预留接口] → MIDI → MusicXML
    out_midi = src.parent / (src.stem + '.mid')
    result = muscriptor.transcribe(src, out_midi, contains_voice=piece.contains_voice)
    piece.midi_file = _rel_to_media(out_midi)
    out_xml = src.parent / (src.stem + '.musicxml')
    midi_to_musicxml(out_midi, out_xml)
    piece.musicxml_file = _rel_to_media(out_xml)
    piece.save()
    summary.update({
        'num_instruments': result.num_instruments,
        'contains_voice': result.contains_voice,
        'warnings': result.warnings,
    })
    summary['notes'].append(
        '音频经 MuScriptor 转录为 MIDI（%s），再转换为 MusicXML。'
        % ('器乐+人声两遍管线' if piece.contains_voice else '纯器乐单遍管线'))
    return summary
