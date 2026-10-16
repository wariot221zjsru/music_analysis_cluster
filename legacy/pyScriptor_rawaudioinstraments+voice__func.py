import os
from pathlib import Path

# =====================================================================
# 方案二：beat_this（节拍检测）权重改为本地读取，不再下载到 C 盘
# =====================================================================
# 1) 兜底：把 torch 的缓存/下载根目录也切到 F 盘。
#    这样即使下面没找到本地权重，回落去下载时也只会落在 F 盘。
os.environ.setdefault("TORCH_HOME", r"F:\torch_home")

import beat_this.inference as _bt

# 2) 可能的本地权重位置（取第一个真实存在的）。
#    手动下载一般叫 final0.ckpt；若由 torch.hub 自动下载则是 beat_this-final0.ckpt。
_BEAT_CKPT_CANDIDATES = (
    Path(r"F:\torch_home\beat_this-final0.ckpt"),
    Path(r"F:\torch_home\final0.ckpt"),
    Path(r"F:\torch_home\hub\checkpoints\beat_this-final0.ckpt"),
    Path(r"F:\torch_home\hub\checkpoints\final0.ckpt"),
)


def _find_local_beat_ckpt():
    """返回第一个存在的本地 beat_this 权重路径，找不到则返回 None。"""
    return next((p for p in _BEAT_CKPT_CANDIDATES if p.exists()), None)


# 3) 拦截 beat_this 的权重加载。
#    load_checkpoint 原本的逻辑就是「先按本地路径 torch.load，FileNotFoundError
#    才拼 URL 去下载」。这里把入参替换成本地文件路径，直接命中本地分支。
_orig_load_checkpoint = _bt.load_checkpoint


def _load_checkpoint_local(checkpoint_path, device="cpu"):
    local = _find_local_beat_ckpt()
    if local is not None:
        return _orig_load_checkpoint(str(local), device)
    return _orig_load_checkpoint(checkpoint_path, device)


_bt.load_checkpoint = _load_checkpoint_local

print(f"[pyScriptor] beat_this 本地权重: {_find_local_beat_ckpt() or '未找到，将回落到下载'}")

from muscriptor import TranscriptionModel
from muscriptor.events import NoteStartEvent, ProgressEvent
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES

# "large" resolves to hf://MuScriptor/muscriptor-large and downloads on first use.
model = TranscriptionModel.load_model(r"F:\music_models\muscriptor-l//model.safetensors")#I want local#large

music_audio_path = Path(r"D:\Muzik\旧电脑music\音频下载\陈蓉蓉\陈蓉蓉//11椰岛之歌.mp3")#D:\Muzik\歌曲下载//M500002sLc1b3MRv7r.mp3（印尼民歌联唱）M500002sLc1b3MRv7r
#D:\Muzik\旧电脑music\音频下载//美丽的凤尾竹-0900_c80b_a1dc_2cd707b1685b823ad6f87dd9b0d5da6a.mp3
#D:\Muzik\旧电脑music\音频下载\泛舟湄平河\muzmaze.org//Music from the North of Thailand - Long Mae Ping.mp3
#D:\Muzik\旧电脑music\音频下载\陈蓉蓉\陈蓉蓉//17划船歌 (1981年录音).mp3 #16爬山顶
#D:\Muzik\mp3//周华健 - 花心.wav
out_stem = "11椰岛之歌"
out_stem_2=out_stem+'//'+out_stem#先文件夹，后文件名处理#2026.9.18#因为一个文件夹下有3个这个歌曲的相关.mid文件
#Path("resultmidis").mkdir(exist_ok=True)
Path(f"resultmidis/{out_stem}").mkdir(exist_ok=True)#2026.09.19#没有歌名文件夹就建

# =====================================================================
# 两遍转写：muscriptor 默认不会解码人声（voice 组无条件引导时几乎不输出）。
#   第 1 遍：器乐 —— 传入除 voice 外的全部乐器组，硬性屏蔽人声 token；
#   第 2 遍：人声 —— instruments=["voice"]，只允许 voice 组（GM 52/53/54，
#            合唱/人声音色），屏蔽其它所有乐器 token。
# =====================================================================
'''
# 器乐组列表：全部组名去掉 voice（drums 保留，鼓组仍会被识别）
_instrumental_groups = [n for n in MT3_FULL_PLUS_GROUP_NAMES if n != "voice"]
print(f"[pyScriptor] 器乐遍: {len(_instrumental_groups)} 个乐器组（已排除 voice）")
'''
# 节拍网格两遍共用（beat_this 权重已在上方切到 F 盘本地）
beat_grid = model.detect_beat_grid_for(music_audio_path)


def _collect(events_iter):
    """收集 Note 事件流，丢弃 ProgressEvent。"""
    return [ev for ev in events_iter if not isinstance(ev, ProgressEvent)]


def _onsets(events):
    return [ev.start_time for ev in events if isinstance(ev, NoteStartEvent)]


def _write_midi(events, filename):
    grid = beat_grid.with_onset_delay(_onsets(events)) if beat_grid is not None else None
    Path(f"resultmidis/{filename}").write_bytes(
        model.events_to_midi_bytes(iter(events), beat_grid=grid)
    )
    n = len(_onsets(events))
    print(f"[pyScriptor] 已写出 resultmidis/{filename}（{n} 个音符）")


# ---- 第 1 遍：器乐（#这次没#屏蔽 voice）----
inst_events = _collect(model.transcribe(music_audio_path))#, instruments=_instrumental_groups
_write_midi(inst_events, f"{out_stem_2}_instr.mid")

# ---- 第 2 遍：人声（只允许 voice 组）----
voice_events = _collect(model.transcribe(music_audio_path, instruments=["voice"]))

# 两遍的 NoteStartEvent.index 都从 0 编号，合并前给人声事件重新编号，
# 避免 events_to_midi_bytes 里 open_notes 按索引配对时错配。
if inst_events:
    _offset = max(ev.index for ev in inst_events if isinstance(ev, NoteStartEvent)) + 1
    for ev in voice_events:
        if isinstance(ev, NoteStartEvent):
            ev.index += _offset

_write_midi(voice_events, f"{out_stem_2}_voice.mid")

# ---- 合并：器乐 + 人声 ----
_write_midi(inst_events + voice_events, f"{out_stem_2}_all.mid")
