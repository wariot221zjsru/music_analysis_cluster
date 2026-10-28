"""转调：大调 ⇄ 小调（自然小调 / 和声小调）。

## 玩法

原调由系统自动判断（调性布局分析的 ``main_key`` 优先，没跑过就用本地
KS / music21 调性判定兜底），用户在转调页用下拉框挑**目标调**：
大调源 → 12 个自然小调 + 12 个和声小调；小调源 → 12 个大调。

## 两步走的转换

1. **移调**：按「源主音 → 目标主音」的最近半音距离整体移调，
   拼写 / 和弦 / 谱号交给 music21（它比手写 XML 更懂记谱）。
   移完之后，谱子落在**目标主音的源调式**上（C 大调 → A 大调）。
2. **调式改写**：把落在源调式音级上（字母 + 音级都对得上）的音，
   改写到目标调式的同音级——
      大调 → 自然小调：降 3 / 6 / 7 级
      大调 → 和声小调：降 3 / 6 级（保留大调的导音）
      小调 → 大调：升 3 / 6 / 7 级
   调外变化音**保持变化音身份不动**（例如 C 大调里的 #4 到了 A 里还是 #4，
   不会被误当成某个音级去改），这符合"转调不改旋律性格"的直觉。

改完调号替换成目标调（和声小调用自然小调的调号，升 7 由临时记号给出），
然后写出新的 MusicXML 与 MIDI。
"""
from __future__ import annotations

import copy
import re

# ------------------------------------------------------------- 音名 / 音阶

#: 12 个主音的显示名与拼写（(音级, 字母, 变音记号)）
TONIC_SPELL = {
    0: ('C', 'C', 0),
    1: ('C♯', 'C', 1),
    2: ('D', 'D', 0),
    3: ('E♭', 'E', -1),
    4: ('E', 'E', 0),
    5: ('F', 'F', 0),
    6: ('F♯', 'F', 1),
    7: ('G', 'G', 0),
    8: ('A♭', 'A', -1),
    9: ('A', 'A', 0),
    10: ('B♭', 'B', -1),
    11: ('B', 'B', 0),
}

_LETTERS = 'CDEFGAB'
_LETTER_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

#: 三种调式的音级（相对主音的半音数）
MODE_STEPS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
}

#: 调式中文名
MODE_NAMES = {
    'major': '大调',
    'natural_minor': '自然小调',
    'harmonic_minor': '和声小调',
}

#: (源调式, 目标调式) → {级数: 半音改动}
#: 级数从 1 起算；只改**确实是源调式音级**的那些音
#: 键用模块内部调式名（'minor' 会先被归一成 natural_minor）
ALTER_RULES = {
    ('major', 'natural_minor'): {3: -1, 6: -1, 7: -1},
    ('major', 'harmonic_minor'): {3: -1, 6: -1},
    ('natural_minor', 'major'): {3: +1, 6: +1, 7: +1},
    ('natural_minor', 'natural_minor'): {},
    ('natural_minor', 'harmonic_minor'): {7: +1},
    ('major', 'major'): {},
}

TGT_MODES_BY_SRC = {
    'major': ('natural_minor', 'harmonic_minor'),
    'minor': ('major',),
}

_VALID_TARGETS = {'major', 'natural_minor', 'harmonic_minor'}


def tonic_name(pc: int) -> str:
    return TONIC_SPELL[pc % 12][0]


def _norm_mode(mode: str) -> str:
    """''minor'' 是判定结果的写法，本模块内部一律用 natural_minor。"""
    return 'natural_minor' if mode == 'minor' else mode


def key_name(tonic_pc: int, mode: str) -> str:
    """调名，如「C 大调」「A 自然小调」「A 和声小调」。"""
    return f'{tonic_name(tonic_pc)} {MODE_NAMES[_norm_mode(mode)]}'


def scale_degrees(tonic_pc: int, mode: str) -> list:
    """目标调式的七个音级 → [(音级 pc, 字母, 变音记号)]（含等音拼写）。

    按"字母逐个往下排"取拼写：C♯ 和声小调的第 7 级是 B♯ 而不是 C，
    记谱上才说得通。
    """
    tonic_pc = tonic_pc % 12
    _display, letter, _alter = TONIC_SPELL[tonic_pc]
    li = _LETTERS.index(letter)
    out = []
    for i, off in enumerate(MODE_STEPS[mode]):
        let = _LETTERS[(li + i) % 7]
        want = (tonic_pc + off) % 12
        alter = ((want - _LETTER_PC[let] + 6) % 12) - 6
        out.append((want, let, alter))
    return out


def parse_key_name(text) -> tuple | None:
    """调名字符串 → ``(主音音级, 'major'|'minor')``；解析不出返回 None。

    兼容 "C 大调" / "a小调" / "升C小调" / "G major" / "A minor"。
    """
    if not text:
        return None
    s = str(text)
    m = re.search(r'([升降♯♭#b]?)\s*([A-Ga-g])\s*([#♯b♭]?)', s)
    if not m:
        return None
    letter = m.group(2).upper()
    pc = _LETTER_PC.get(letter)
    if pc is None:
        return None
    acc = 0
    for ch in (m.group(1), m.group(3)):
        if ch in {'#', '♯', '升'}:
            acc += 1
        elif ch in {'b', '♭', '降'}:
            acc -= 1
    pc = (pc + acc) % 12
    if '大调' in s or 'major' in s.lower():
        mode = 'major'
    elif '小调' in s or 'minor' in s.lower():
        mode = 'minor'
    else:
        return None
    return pc, mode


def parse_target(value: str) -> tuple | None:
    """下拉框值 ``"9:natural_minor"`` → ``(主音音级, 调式)``。"""
    if not value or ':' not in value:
        return None
    pc_s, mode = value.split(':', 1)
    if mode not in _VALID_TARGETS:
        return None
    try:
        pc = int(pc_s) % 12
    except (TypeError, ValueError):
        return None
    return pc, mode


def target_options(src_tonic_pc: int, src_mode: str) -> tuple:
    """下拉框选项与默认值。返回 ``(分组列表, 默认值)``。

    组内是 ``(value, label)``；默认值取**关系大小调**（音级集合相同，
    转过去最自然）。
    """
    src_tonic_pc = src_tonic_pc % 12
    groups = []
    if src_mode == 'major':
        rel = (src_tonic_pc + 9) % 12          # 关系小调：往下小三度
        groups.append(('自然小调',
                       [(f'{t}:natural_minor', key_name(t, 'natural_minor'))
                        for t in range(12)]))
        groups.append(('和声小调',
                       [(f'{t}:harmonic_minor', key_name(t, 'harmonic_minor'))
                        for t in range(12)]))
        default = f'{rel}:natural_minor'
    else:
        rel = (src_tonic_pc + 3) % 12          # 关系大调：往上小三度
        groups.append(('大调',
                       [(f'{t}:major', key_name(t, 'major')) for t in range(12)]))
        default = f'{rel}:major'
    return groups, default


# ------------------------------------------------------------- 原调判定

def detect_key_from_tonality(output: dict) -> tuple | None:
    """调性布局分析的输出 → ``(主音, 调式)``（用其 main_key）。"""
    if not isinstance(output, dict):
        return None
    return parse_key_name(output.get('main_key'))


def detect_key_from_score(path: str) -> tuple | None:
    """本地兜底：music21 的调性判定（Krumhansl 系）。"""
    from music21 import converter
    try:
        score = converter.parse(str(path))
        k = score.analyze('key')
    except Exception:                       # noqa: BLE001 - 判定失败不影响其它功能
        return None
    if k is None:
        return None
    pc = int(k.tonic.pitchClass) % 12
    mode = 'major' if str(k.mode) == 'major' else 'minor'
    return pc, mode


# ------------------------------------------------------------- 转换本体

def _nearest_shift(src_pc: int, tgt_pc: int) -> int:
    """两个主音之间最近的半音距离（-6…+5）。"""
    s = (tgt_pc - src_pc) % 12
    return s - 12 if s > 6 else s


def _set_alter(p, alter: int):
    """写变音记号（music21 10 起 ``pitch.alter`` 只读，要改 accidental）。

    ``alter=0`` 会写出还原记号——转调里正是要它（G♯ 变回 G 得标 ♮）。
    """
    from music21 import pitch as m21pitch
    p.accidental = m21pitch.Accidental(int(alter))
    return p


def _spelled_pitch(letter: str, alter: int, octave: int):
    from music21 import pitch as m21pitch
    return _set_alter(m21pitch.Pitch(step=letter, octave=octave), alter)


def _target_key_signature(tonic_pc: int, mode: str):
    """目标调的调号（和声小调也用自然小调的调号，升 7 靠临时记号）。"""
    from music21 import key as m21key
    _name, letter, alter = TONIC_SPELL[tonic_pc % 12]
    tonic = letter + ('' if alter == 0 else ('#' if alter > 0 else '-'))
    return m21key.Key(tonic, 'major' if mode == 'major' else 'minor')


def _replace_key_signatures(score, tonic_pc: int, mode: str) -> int:
    """把所有调号换成目标调，返回替换个数；一个都没有就在开头插一个。"""
    from music21 import key as m21key, stream as m21stream

    replaced = 0
    for el in list(score.recurse().getElementsByClass(m21key.KeySignature)):
        site = el.activeSite
        if site is None:
            continue
        try:
            site.replace(el, _target_key_signature(tonic_pc, mode))
            replaced += 1
        except Exception:                    # noqa: BLE001 - 单个位置失败不致命
            continue
    if replaced == 0:
        parts = list(score.recurse().getElementsByClass(m21stream.Part))
        if parts:
            measures = parts[0].getElementsByClass(m21stream.Measure)
            if measures:
                measures[0].insert(0, _target_key_signature(tonic_pc, mode))
                replaced = 1
    return replaced


def _alter_notes(score, src_tonic_pc: int, src_mode: str,
                 tgt_tonic_pc: int, tgt_mode: str) -> int:
    """调式改写：源调式音级 → 目标调式同音级。返回改动音数。"""
    from music21 import chord as m21chord, note as m21note

    rule = ALTER_RULES.get((src_mode, tgt_mode)) or {}
    if not rule:
        return 0

    # 移调之后谱子落在"目标主音 + 源调式"上 → 用它认音级
    post = scale_degrees(tgt_tonic_pc,
                         'major' if src_mode == 'major' else 'natural_minor')
    target = scale_degrees(tgt_tonic_pc, tgt_mode)

    changed = 0
    for el in score.recurse().getElementsByClass((m21note.Note, m21chord.Chord)):
        pitches = list(el.pitches)
        new_pitches = []
        touched = False
        for p in pitches:
            np_ = p
            for deg, _delta in rule.items():
                pc, letter, _alter = post[deg - 1]
                if p.pitchClass != pc:
                    continue
                _tpc, tletter, talter = target[deg - 1]
                if p.step == letter or p.step == tletter:
                    # 同一个字母：改变音记号即可（E → E♭）
                    np_ = _set_alter(p, talter)
                else:
                    # 等音撞车（如记成 D♭ 的 3 级）：直接换成目标音级拼写
                    np_ = _spelled_pitch(tletter, talter, p.octave)
                    while np_.midi - p.midi > 6:
                        np_.octave -= 1
                    while p.midi - np_.midi > 6:
                        np_.octave += 1
                touched = True
                break
            new_pitches.append(np_)
        if not touched:
            continue
        if isinstance(el, m21chord.Chord):
            el.pitches = tuple(new_pitches)
        else:
            el.pitch = new_pitches[0]
        changed += 1
    return changed


def transpose_score(score, src_tonic_pc: int, src_mode: str,
                    tgt_tonic_pc: int, tgt_mode: str) -> tuple:
    """转调。返回 ``(新 score, 摘要 dict)``——原 score 对象不动。"""
    from music21 import interval as m21interval, pitch as m21pitch

    src_tonic_pc %= 12
    tgt_tonic_pc %= 12
    src_mode = _norm_mode(src_mode)
    tgt_mode = _norm_mode(tgt_mode)
    shift = _nearest_shift(src_tonic_pc, tgt_tonic_pc)

    if shift:
        _n, sletter, salter = TONIC_SPELL[src_tonic_pc]
        _t, tletter, talter = TONIC_SPELL[tgt_tonic_pc]
        p1 = _spelled_pitch(sletter, salter, 4)
        p2 = _spelled_pitch(tletter, talter, 4)
        while p2.midi - p1.midi != shift:
            p2.octave += 1 if p2.midi - p1.midi < shift else -1
        iv = m21interval.Interval(pitchStart=p1, pitchEnd=p2)
        score = score.transpose(iv)      # 返回新对象，原谱不动
    else:
        score = copy.deepcopy(score)     # 不移调也要写新文件，别改到原谱

    keys = _replace_key_signatures(score, tgt_tonic_pc, tgt_mode)
    notes = _alter_notes(score, src_tonic_pc, src_mode, tgt_tonic_pc, tgt_mode)
    info = {
        'shift': shift,
        'src_key': key_name(src_tonic_pc, src_mode),
        'tgt_key': key_name(tgt_tonic_pc, tgt_mode),
        'key_signatures': keys,
        'altered_notes': notes,
    }
    return score, info


def convert_file(src_path: str, out_musicxml: str, out_midi: str,
                 src_tonic_pc: int, src_mode: str,
                 tgt_tonic_pc: int, tgt_mode: str) -> dict:
    """读 MusicXML → 转调 → 写出 MusicXML / MIDI，返回摘要。"""
    from music21 import converter
    score = converter.parse(str(src_path))
    score, info = transpose_score(score, src_tonic_pc, src_mode,
                                  tgt_tonic_pc, tgt_mode)
    score.write('musicxml', fp=str(out_musicxml))
    try:
        score.write('midi', fp=str(out_midi))
        info['midi'] = str(out_midi)
    except Exception as exc:                # noqa: BLE001 - MIDI 失败不阻断 XML
        info['midi'] = ''
        info['midi_error'] = str(exc)
    info['musicxml'] = str(out_musicxml)
    return info
