"""MuScriptor 音频转录接口（预留 —— 由项目成员自行实现）。

==============================================================================
 本模块定义了两个转录函数的**稳定接口**。当前版本为占位实现，
 调用会抛出 NotImplementedError，并在错误信息中指向本说明。
 你只需要实现下面两个函数，替换 raise 部分，其余代码（上传、
 转换调度、MIDI→MusicXML、分析流水线）全部自动打通。

 参考实现来源（两个实验脚本）：
   1. 纯器乐：     pyScriptor_base.py
   2. 器乐 + 人声： pyScriptor_rawaudioinstraments+voice.py
==============================================================================

【实现要点：纯器乐（transcribe_instrumental）】

  单遍转录、直接加节奏。核心调用：

      result = model.transcribe_and_postprocess(
          audio,                   # 预处理后的音频张量
          instruments=None,        # 无条件引导
      )
      # 返回 (midi_bytes, beat_grid)：
      #   - transcribe 内部完成转录 + beat_this 节拍对齐（转录即加节奏）
      #   - midi_bytes 可直接写出 .mid 文件

  流程：
      加载音频 → 预处理（重采样/归一化，同实验脚本）
      → transcribe_and_postprocess() → 写出 midi_bytes

【实现要点：器乐 + 人声（transcribe_instrumental_with_voice）】

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

  - beat_this / MuScriptor 权重走本地加载（F:/music_models/...），
    可拦截 load_checkpoint 防止默认下载到 C 盘；
  - 事件流中的 ProgressEvent 可用于进度上报
    （本接口预留 progress_callback 参数）；
  - 长音频按模型要求的分块长度切分后逐块转录再拼接。

==============================================================================
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


@dataclass
class TranscriptionResult:
    """转录结果。"""
    midi_path: str                       # 输出 MIDI 文件路径
    num_instruments: int = 0             # 识别出的乐器（组）数
    contains_voice: bool = False         # 人声遍是否检出人声
    warnings: List[str] = field(default_factory=list)


class MuScriptorNotImplemented(NotImplementedError):
    """提示实现者去哪里补代码的专用异常。"""


def transcribe_instrumental(
    audio_path: str | Path,
    output_midi_path: str | Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> TranscriptionResult:
    """纯器乐音频 → MIDI（单遍转录 + 直接加节奏）。

    参数：
        audio_path:         输入音频文件（mp3 / wav / flac …）
        output_midi_path:   输出 MIDI 文件路径
        progress_callback:  可选回调 callback(progress: 0~1, message: str)，
                            可接 ProgressEvent 做进度上报

    实现参见模块 docstring【纯器乐】一节。
    """
    raise MuScriptorNotImplemented(
        'MuScriptor 纯器乐转录接口尚未实现。'
        '请在 core/converters/muscriptor.py 的 transcribe_instrumental() '
        '中按模块 docstring 的说明补全实现（参考 pyScriptor_base.py：'
        'model.transcribe_and_postprocess() 一步完成转录 + 节拍对齐）。'
    )


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
    raise MuScriptorNotImplemented(
        'MuScriptor 器乐+人声转录接口尚未实现。'
        '请在 core/converters/muscriptor.py 的 '
        'transcribe_instrumental_with_voice() 中按模块 docstring 的说明补全实现'
        '（参考 pyScriptor_rawaudioinstraments+voice.py：两遍转录 + index 重编号 + '
        '节拍网格合并）。'
    )


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
