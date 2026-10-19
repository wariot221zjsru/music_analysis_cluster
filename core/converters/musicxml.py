"""MIDI / MusicXML 转换：接入层的统一出口。

三条输入路径在 MusicXML 处汇合（计划书第三章总体设计）：
    audio   → [MuScriptor] → MIDI → music21 → MusicXML
    midi    →                 music21 → MusicXML
    musicxml→                 直接使用

存储策略（core/storage.py）：
  - 上传的原始音频是**临时文件**，转换成功后删除；
  - MIDI 与 MusicXML **永久保存且分目录**（media/midi/…、media/musicxml/…）；
  - 上传的 MIDI / MusicXML 源文件也复制进对应永久目录，随后清理临时源。

重跑转换（「重新转换」按钮）：
  - 若已入库的 MIDI 存在，则**复用该 MIDI、跳过昂贵的 MuScriptor 转录**，
    仅重新生成 MusicXML（转录一次约 6 分钟，没必要重复）；
  - 只有 MIDI 不存在时才真正调用 MuScriptor。
"""
from pathlib import Path

from .. import storage
from ..analysis.scoredata import detect_voice
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

    「是否含人声」的统一口径：
      - MIDI / MusicXML 上传：**只看文件里的声部**，覆盖上传页面的勾选；
      - 音频上传：以 MuScriptor 转录结果为准（谱面检出人声也算）。
      回写 ``piece.contains_voice``，全站（列表 / 详情 / 问答上下文）都用它。
    """
    summary = {'source_type': piece.source_type, 'notes': []}
    result = None          # MuScriptor 转录结果（仅音频路径有值）
    workdir = storage.new_workdir(prefix=f'convert_p{piece.id}_')

    has_source = bool(piece.source_file)
    src = Path(piece.source_file.path) if has_source else None

    def _on_disk(field) -> bool:
        """该 FileField 指向的文件在存储中确实存在。"""
        return bool(field) and field.storage.exists(field.name)

    def _restore(field, dest: Path) -> None:
        """把已入库的文件取回临时工作目录。"""
        with field.open('rb') as fh:
            dest.write_bytes(fh.read())

    try:
        # ============ ① MusicXML 直接入库 ============
        if piece.source_type == 'musicxml':
            if has_source:
                rel_xml = storage.store_musicxml(piece.id, src, src.name)
                piece.musicxml_file = rel_xml
                piece.save(update_fields=['musicxml_file'])
                summary['notes'].append('MusicXML 已入库永久保存（musicxml/）。')
            elif _on_disk(piece.musicxml_file):
                summary['notes'].append(
                    'MusicXML 已入库（musicxml/），无需重新转换。')
            else:
                raise RuntimeError(
                    '临时源文件已清理且无 MusicXML 产物，无法重新转换，请重新上传作品。')

        # ============ ② MIDI 入库 + 转 MusicXML ============
        elif piece.source_type == 'midi':
            if has_source:
                stem = src.stem
                piece.midi_file = storage.store_midi(piece.id, src, src.name)
                local_midi: Path = src
                summary['notes'].append('MIDI 已入库永久保存（midi/）。')
            elif _on_disk(piece.midi_file):
                stem = Path(piece.midi_file.name).stem
                local_midi = workdir / f'{stem}.mid'
                _restore(piece.midi_file, local_midi)
                summary['notes'].append('复用已入库的 MIDI（临时源已清理）。')
            else:
                raise RuntimeError(
                    '临时源文件已清理且无 MIDI 产物，无法重新转换，请重新上传作品。')

            local_xml = workdir / f'{stem}.musicxml'
            midi_to_musicxml(local_midi, local_xml)
            piece.musicxml_file = storage.store_musicxml(
                piece.id, local_xml, f'{stem}.musicxml')
            piece.save(update_fields=['midi_file', 'musicxml_file'])
            summary['notes'].append('经 music21 转换为 MusicXML 入库（musicxml/）。')

        # ============ ③ 音频 → MuScriptor → MIDI → MusicXML ============
        else:
            reused = False
            if _on_disk(piece.midi_file):
                # 快路径：已有转录产物，跳过昂贵的 MuScriptor 转录
                stem = Path(piece.midi_file.name).stem
                local_midi = workdir / f'{stem}.mid'
                _restore(piece.midi_file, local_midi)
                reused = True
                summary['notes'].append(
                    '复用已入库的 MIDI，跳过 MuScriptor 转录（已转录过，无需重复）。')
            elif has_source:
                stem = src.stem
                local_midi = workdir / f'{stem}.mid'
                result = muscriptor.transcribe(src, local_midi,
                                               contains_voice=piece.contains_voice)
                if not local_midi.exists() or local_midi.stat().st_size == 0:
                    raise RuntimeError(
                        f'MuScriptor 未生成有效的 MIDI：{local_midi}')
                piece.midi_file = storage.store_midi(
                    piece.id, local_midi, f'{stem}.mid')
                summary['notes'].append(
                    '音频经 MuScriptor 转录为 MIDI（%s）入库（midi/）。'
                    % ('器乐+人声两遍管线' if piece.contains_voice else '纯器乐单遍管线'))
            else:
                raise RuntimeError(
                    '临时源文件已清理且无可用 MIDI 产物，无法重新转换，请重新上传作品。')

            # MusicXML 始终从最终 MIDI 重新生成
            local_xml = workdir / f'{stem}.musicxml'
            midi_to_musicxml(local_midi, local_xml)
            piece.musicxml_file = storage.store_musicxml(
                piece.id, local_xml, f'{stem}.musicxml')
            piece.save(update_fields=['midi_file', 'musicxml_file'])
            summary['notes'].append('转换为 MusicXML 入库（musicxml/）。')

            if result is None:
                if not reused:
                    # 防御：转录接口未返回结果对象时，MIDI/MusicXML 已入库成功，
                    # 只丢失统计信息，不应让整条转换链失败。
                    summary['warnings'] = [
                        '转录接口未返回结果对象，乐器数等统计信息缺失。']
            else:
                summary.update({
                    'num_instruments': result.num_instruments,
                    'instruments': list(getattr(result, 'instruments', []) or []),
                    'contains_voice': result.contains_voice,
                    'warnings': list(result.warnings or []),
                })
    finally:
        storage.cleanup_workdir(workdir)

    # ============ ④ 统一「是否含人声」：以文件实际声部为准 ============
    user_choice = bool(piece.contains_voice)
    voice = None
    if piece.musicxml_file:
        try:
            voice = detect_voice(piece.musicxml_file.path)
        except Exception as exc:  # noqa: BLE001
            # 谱面读不出也不能让转换失败：退回用户勾选并记一条警告
            summary.setdefault('warnings', []).append(
                f'人声判定失败（{exc}），沿用上传时的勾选值。')
    if piece.source_type == 'audio':
        # 音频以 MuScriptor 的人声遍结果为准，谱面检出也算
        has_voice = bool(result is not None and result.contains_voice) \
            or bool(voice and voice['has_voice'])
        voice_source = 'transcribe'
    else:
        # MIDI / MusicXML：只认文件里的声部，忽略勾选
        has_voice = bool(voice and voice['has_voice'])
        voice_source = 'file'

    summary['contains_voice'] = has_voice
    summary['voice'] = {
        'source': voice_source,
        'user_choice': user_choice,
        'corrected': has_voice != user_choice,
        'voice_parts': (voice or {}).get('voice_parts', []),
        'part_count': (voice or {}).get('part_count'),
        'method': (voice or {}).get('method', ''),
    }
    label = '含人声' if has_voice else '纯器乐'
    old_label = '含人声' if user_choice else '纯器乐'
    if has_voice != user_choice:
        piece.contains_voice = has_voice
        piece.save(update_fields=['contains_voice', 'updated_at'])
        origin = {'file': '上传文件内的实际声部',
                  'transcribe': 'MuScriptor 转录结果'}[voice_source]
        summary['notes'].append(
            f'「是否含人声」按{origin}修正为「{label}」'
            f'（上传时勾选的是「{old_label}」）。')
    else:
        origin_short = ('文件内实际声部' if voice_source == 'file'
                        else '转录结果')
        summary['notes'].append(
            f'「是否含人声」=「{label}」（来源：{origin_short}，与上传勾选一致）。')

    # 转换成功：清理临时源文件（音频 / 已入库的 MIDI、MusicXML 源）
    if storage.purge_source(piece):
        summary['notes'].append('临时源文件已清理，未做长期保存。')
    summary['storage'] = {
        'midi': piece.midi_file.name if piece.midi_file else '',
        'musicxml': piece.musicxml_file.name if piece.musicxml_file else '',
        'source_purged': piece.source_purged,
    }
    return summary
