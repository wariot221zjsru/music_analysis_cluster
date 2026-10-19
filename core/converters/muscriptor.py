"""MuScriptor 音频转录接口。

==============================================================================
 本模块定义两个转录函数的**稳定接口**：
   - transcribe_instrumental()              纯器乐：单遍转录，转录即加节奏
   - transcribe_instrumental_with_voice()   器乐 + 人声：两遍转录后合并

 接入层（views / pipeline）只调用 transcribe()，不感知两种模式的差异。

 参考实现来源（两个实验脚本）：
   1. 纯器乐：     legacy/pyScriptor_base__func.py
   2. 器乐 + 人声： legacy/pyScriptor_rawaudioinstraments+voice__func.py
==============================================================================

【纯器乐（transcribe_instrumental）】

  单遍转录、直接加节奏。核心调用：

      result = model.transcribe_and_postprocess(
          audio,                   # 预处理后的音频张量
          instruments=None,        # 无条件引导
      )
      # 返回 (midi_bytes, beat_grid)：
      #   - transcribe 内部完成转录 + beat_this 节拍对齐（转录即加节奏）
      #   - midi_bytes 可直接写出 .mid 文件

【器乐 + 人声（transcribe_instrumental_with_voice）】

  两遍转录再合并。原因：MuScriptor 默认不解码人声
  （voice 组在无条件引导时几乎不输出），必须单独条件引导转一遍人声。

  第 1 遍（器乐遍）：
      用**整个音频**做默认转录，但**不走** transcribe_and_postprocess
      的"转录即加节奏"流程——节奏对齐延后统一处理。

  第 2 遍（人声遍）：
      instruments=["voice"] 条件引导，只允许 voice 组
      （GM 52/53/54 合唱 / 人声音色）单独转人声。

  节拍对齐与合并（两个关键坑）：
      a) 用 detect_beat_grid_for() 单独计算节拍网格，两遍共用；
         写 MIDI 时用 beat_grid.with_onset_delay() 对齐；
      b) **合并前必须给人声事件的 NoteStartEvent.index 重新编号**：
         两遍转录的 index 都从 0 起，若不重编号，
         events_to_midi_bytes 会按索引错配音符对。

【通用工程细节（两种模式通用）】

  - beat_this / MuScriptor 权重走本地加载（F:/music_models/...、F:/torch_home/...），
    可拦截 load_checkpoint 防止默认下载到 C 盘；
  - 事件流中的 ProgressEvent 可用于进度上报
    （本接口预留 progress_callback 参数）；
  - 长音频按模型要求的分块长度切分后逐块转录再拼接。

【重要：返回值约定】

  两个函数**必须**返回 TranscriptionResult，绝不能隐式返回 None。
  历史 bug：transcribe_instrumental() 调用成功后忘记 return，
  导致 core/converters/musicxml.py 里 result.num_instruments 抛
  AttributeError: 'NoneType' object has no attribute 'num_instruments'。
  统一用 _build_result() 构造返回值即可避免。
==============================================================================
"""
# ---------------- 导入 legacy（把项目根目录加入 sys.path） ----------------
import os
import sys
from pathlib import Path

# 本文件位于 <项目根>/core/converters/muscriptor.py，向上 3 级才是项目根目录
# （里面放着 legacy 包）。
_project_root = Path(os.path.abspath(__file__)).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from legacy.pyScriptor_base__func import (          # noqa: E402
    pyscriptor_transcribe_and_postprocess,          # 纯器乐转录（已实现）
)

from dataclasses import dataclass, field            # noqa: E402
from typing import Callable, List, Optional, Tuple  # noqa: E402

# ---------------------------------------------------------------------------
# 本地模型 / 权重路径（可用环境变量覆盖，默认沿用实验脚本的 F 盘布局）
# ---------------------------------------------------------------------------
MUSCRIPTOR_MODEL_PATH = os.environ.get(
    'MUSCRIPTOR_MODEL', r"F:\music_models\muscriptor-l\model.safetensors")

_TORCH_HOME = os.environ.get('TORCH_HOME', r"F:\torch_home")

_BEAT_CKPT_CANDIDATES = tuple(
    Path(_TORCH_HOME) / n for n in (
        'beat_this-final0.ckpt',
        'final0.ckpt',
        os.path.join('hub', 'checkpoints', 'beat_this-final0.ckpt'),
        os.path.join('hub', 'checkpoints', 'final0.ckpt'),
    )
)


@dataclass
class TranscriptionResult:
    """转录结果。"""
    midi_path: str                       # 输出 MIDI 文件路径
    num_instruments: int = 0             # 识别出的乐器（组）数
    instruments: List[str] = field(default_factory=list)   # 乐器名（去重）
    contains_voice: bool = False         # 人声遍是否检出人声
    warnings: List[str] = field(default_factory=list)


class MuScriptorNotImplemented(NotImplementedError):
    """提示实现者去哪里补代码的专用异常。"""


class MuScriptorTranscriptionError(RuntimeError):
    """转录过程本身失败（模型 / 权重 / 音频问题），区别于"接口未实现"。"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _patch_beat_this_checkpoint() -> None:
    """把 beat_this 的权重加载改到本地（F:\\torch_home），避免下载到 C 盘。"""
    try:
        import beat_this.inference as _bt
    except Exception:  # noqa: BLE001
        return
    if getattr(_bt.load_checkpoint, '_muscriptor_local_patched', False):
        return
    local = next((p for p in _BEAT_CKPT_CANDIDATES if p.exists()), None)
    orig = _bt.load_checkpoint

    def _load_local(checkpoint_path, device='cpu', _orig=orig, _local=local):
        if _local is not None:
            return _orig(str(_local), device)
        return _orig(checkpoint_path, device)

    _load_local._muscriptor_local_patched = True
    _bt.load_checkpoint = _load_local


def _require_midi(output_midi_path) -> Path:
    """校验转录产物 MIDI 确实写出来了，否则抛明确的运行时错误。"""
    path = Path(output_midi_path)
    if not path.exists() or path.stat().st_size == 0:
        raise MuScriptorTranscriptionError(
            f'MuScriptor 未生成有效的 MIDI 文件：{path}。'
            '请检查模型路径（MUSCRIPTOR_MODEL / F:\\music_models\\muscriptor-l）'
            '与音频文件是否可读。')
    return path


def _inspect_midi(midi_path) -> Tuple[int, List[str], List[str]]:
    """从已生成的 MIDI 统计乐器（声部）数量与名称。

    返回 ``(num_instruments, instrument_names, warnings)``。
    解析失败不抛异常，只记录 warning，避免把已成功的转录判为失败。
    """
    warnings: List[str] = []
    try:
        from music21 import converter
        score = converter.parse(str(midi_path))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f'MIDI 解析失败，乐器数按 0 计：{exc}')
        return 0, [], warnings

    names: List[str] = []
    for part in getattr(score, 'parts', []):
        name = ''
        try:
            inst = part.getInstrument()
            if inst is not None:
                name = (getattr(inst, 'instrumentName', '') or
                        getattr(inst, 'instrumentSound', '') or '')
        except Exception:  # noqa: BLE001
            name = ''
        if not name:
            name = getattr(part, 'partName', '') or getattr(part, 'id', '') or ''
        if name and name not in names:
            names.append(name)
    if not names:
        warnings.append('MIDI 中未解析到任何声部。')
    return len(names), names, warnings


def _build_result(output_midi_path,
                  contains_voice: bool = False,
                  extra_warnings: Optional[List[str]] = None) -> TranscriptionResult:
    """转录完成后统一构造 TranscriptionResult（保证不返回 None）。"""
    path = _require_midi(output_midi_path)
    num, names, warns = _inspect_midi(path)
    return TranscriptionResult(
        midi_path=str(path),
        num_instruments=num,
        instruments=names,
        contains_voice=bool(contains_voice),
        warnings=list(extra_warnings or []) + warns,
    )


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def transcribe_instrumental(
    audio_path: str | Path,
    output_midi_path: str | Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> TranscriptionResult:
    """纯器乐音频 → MIDI（单遍转录 + 直接加节奏）。

    参数：
        audio_path:         输入音频文件（mp3 / wav / flac …）
        output_midi_path:   输出 MIDI 文件路径
        progress_callback:  可选回调 callback(progress: 0~1, message: str)

    实现参见模块 docstring【纯器乐】一节。
    """
    rslt = pyscriptor_transcribe_and_postprocess(audio_path, output_midi_path)
    if rslt is not True:
        # legacy 内部 except 掉异常后返回 None（只打印 "[pyScriptor] 转录失败"），
        # 这里转成明确的运行时错误，避免上层拿到 None 再炸 AttributeError。
        detail = rslt if isinstance(rslt, str) else ''
        raise MuScriptorTranscriptionError(
            f'MuScriptor 纯器乐转录失败{("：" + detail) if detail else ""}。'
            '请查看服务端日志中 "[pyScriptor] 转录失败" 一行，'
            '并检查 legacy/pyScriptor_base__func.py 内的模型路径与音频文件。')

    if progress_callback:
        try:
            progress_callback(1.0, '转录完成')
        except Exception:  # noqa: BLE001
            pass
    return _build_result(output_midi_path)


def transcribe_instrumental_with_voice(
    audio_path: str | Path,
    output_midi_path: str | Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> TranscriptionResult:
    """器乐 + 人声音频 → MIDI（两遍转录再合并）。

    参数同 transcribe_instrumental。

    实现参见模块 docstring【器乐 + 人声】一节：
      第 1 遍：整音频默认转录（不加节奏）；
      第 2 遍：instruments=["voice"] 只转人声；
      合并：detect_beat_grid_for() 共用节拍网格，
            人声 NoteStartEvent.index 必须重新编号后合并。
    """
    _patch_beat_this_checkpoint()
    try:
        from muscriptor import TranscriptionModel
        from muscriptor.events import NoteStartEvent, ProgressEvent
    except Exception as exc:  # noqa: BLE001
        raise MuScriptorNotImplemented(
            f'未能导入 muscriptor 依赖（{exc}），无法执行"器乐+人声"两遍转录。'
            '请确认已安装 muscriptor 与 beat_this，或暂时改用纯器乐模式。') from exc

    def _report(p: float, msg: str) -> None:
        if progress_callback:
            try:
                progress_callback(p, msg)
            except Exception:  # noqa: BLE001
                pass

    _report(0.02, '加载 MuScriptor 模型…')
    model = TranscriptionModel.load_model(MUSCRIPTOR_MODEL_PATH)

    # 节拍网格两遍共用（beat_this 权重已切到本地）
    _report(0.08, '检测节拍网格…')
    beat_grid = model.detect_beat_grid_for(audio_path)

    def _collect(event_iter):
        """收集音符事件流，丢弃 ProgressEvent。"""
        return [ev for ev in event_iter if not isinstance(ev, ProgressEvent)]

    def _write_midi(events, fp) -> int:
        onsets = [ev.start_time for ev in events if isinstance(ev, NoteStartEvent)]
        grid = beat_grid.with_onset_delay(onsets) if beat_grid is not None else None
        Path(fp).write_bytes(model.events_to_midi_bytes(iter(events), beat_grid=grid))
        return len(onsets)

    _report(0.15, '第 1 遍：器乐转录…')
    inst_events = _collect(model.transcribe(audio_path))

    _report(0.60, '第 2 遍：人声转录（voice 条件引导）…')
    voice_events = _collect(model.transcribe(audio_path, instruments=['voice']))

    # 两遍的 NoteStartEvent.index 都从 0 起，合并前给人声事件重新编号，
    # 避免 events_to_midi_bytes 里 open_notes 按索引配对时错配。
    indices = [ev.index for ev in inst_events if isinstance(ev, NoteStartEvent)]
    offset = (max(indices) + 1) if indices else 0
    for ev in voice_events:
        if isinstance(ev, NoteStartEvent):
            ev.index += offset

    _report(0.90, '合并器乐 + 人声并写出 MIDI…')
    total_notes = _write_midi(inst_events + voice_events, output_midi_path)

    has_voice = any(
        isinstance(ev, NoteStartEvent) and getattr(ev, 'instrument', '') == 'voice'
        for ev in voice_events)
    warns: List[str] = []
    if not has_voice:
        warns.append('人声遍未检出 voice 事件，结果与纯器乐转录基本一致。')

    _report(1.0, f'完成（共 {total_notes} 个音符）')
    return _build_result(output_midi_path, contains_voice=has_voice,
                         extra_warnings=warns)


def transcribe(
    audio_path: str | Path,
    output_midi_path: str | Path,
    contains_voice: bool,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> TranscriptionResult:
    """统一入口：按上传时勾选的"是否含人声"分发到对应管线。

    接入层（views / pipeline）只调用本函数，不感知两种模式的差异。
    """
    if contains_voice:
        return transcribe_instrumental_with_voice(
            audio_path, output_midi_path, progress_callback)
    return transcribe_instrumental(
        audio_path, output_midi_path, progress_callback)
