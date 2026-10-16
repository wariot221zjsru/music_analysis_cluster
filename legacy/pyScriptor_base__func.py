import os
from pathlib import Path

# =====================================================================
# 方案二：beat_this（节拍检测）权重改为本地读取，不再下载到 C 盘
# =====================================================================
# 1) 兜底：把 torch 的缓存/下载根目录也切到 F 盘。
#    这样即使下面没找到本地权重，回落去下载时也只会落在 F 盘。
os.environ.setdefault("TORCH_HOME", r"F:\torch_home")

import beat_this.inference as _bt
'''
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
'''
local=r"F:\torch_home\final0.ckpt"

# 3) 拦截 beat_this 的权重加载。
#    load_checkpoint 原本的逻辑就是「先按本地路径 torch.load，FileNotFoundError
#    才拼 URL 去下载」。这里把入参替换成本地文件路径，直接命中本地分支。
_orig_load_checkpoint = _bt.load_checkpoint


def _load_checkpoint_local(checkpoint_path, device="cpu"):
    #local = _find_local_beat_ckpt()
    if local is not None:
        return _orig_load_checkpoint(str(local), device)
    return _orig_load_checkpoint(checkpoint_path, device)

def pyscriptor_transcribe_and_postprocess(music_audio_path,output_midi_path):
    
    try:
        _bt.load_checkpoint = _load_checkpoint_local
        #_find_local_beat_ckpt()
        print(f"[pyScriptor] beat_this 本地权重: {local or '未找到，将回落到下载'}")

        from muscriptor import TranscriptionModel

        # "large" resolves to hf://MuScriptor/muscriptor-large and downloads on first use.
        model = TranscriptionModel.load_model(r"F:\music_models\muscriptor-l//model.safetensors")#I want local#large

        #music_audio_path = Path(r"D:\Muzik\旧电脑music\音频下载//美丽的凤尾竹-0900_c80b_a1dc_2cd707b1685b823ad6f87dd9b0d5da6a.mp3")#D:\Muzik\歌曲下载//M500002sLc1b3MRv7r.mp3
        #D:\Muzik\旧电脑music\音频下载\泛舟湄平河\muzmaze.org//Music from the North of Thailand - Long Mae Ping.mp3
        #D:\Muzik\旧电脑music\音频下载\陈蓉蓉\陈蓉蓉//17划船歌 (1981年录音).mp3
        # Get a MIDI file directly (transcribe_and_postprocess returns (bytes, BeatGrid | None)):
        midi_bytes, _beat_grid = model.transcribe_and_postprocess(music_audio_path)
        Path(output_midi_path).write_bytes(midi_bytes)#保存midi文件#2026.10.15#
        
        # Or stream note events as they are transcribed:
        for event in model.transcribe(music_audio_path):
            print(event)  # NoteStartEvent / NoteEndEvent / ProgressEvent

        return True
    
    except Exception as e:
        print(f"[pyScriptor] 转录失败: {e}")#2026.10.15#
