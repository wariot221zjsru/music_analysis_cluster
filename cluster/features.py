"""三轴特征提取：调式 / 和声 / 旋律。

本模块只读 core 已有的分析结果（tonality / chords / melody_line）与乐谱，
**不回写、不修改 core 的任何模块**。特征算完交给上层存进 PieceVector。

## 调式轴：为什么"主音对齐"，而不是"循环移位取最大"

循环移位取最大（core 里 theme._pc_cos 的做法）回答的是
"忽略调高，这两个音级集合像不像"。它在主题识别里是对的（模进要容忍），
但在**调式分类**里会犯一个致命的错：

    多利亚就是自然大调的循环移位（D 多利亚与 C 自然大调的音级集合相同）

于是 C 大调、D 多利亚、E 弗里几亚……在"循环移位取最大"下全部互相满分，
被判成同一类 —— 多利亚和自然大调分不开。

所以调式轴改成：**先定主音 → 把音级分布旋转到主音为 0 → 再直接比余弦（不再移位）**

    C 大调 / D 大调 / E 大调    对齐后向量相同 → 一类（同调式、不同调高归一）
    C 大调 / D 多利亚 / F 立底亚  对齐后向量不同 → 能分开（真正的调式差异）

一次对齐同时满足"不同调高归一"与"不同调式要能区分"。主音取不到时才降级回
循环移位（mode_uncertain=True，页面要标出来）。

## 主音从哪来

1. `tonality.main_key`（LLM 判定，结合了和弦证据）—— 解析音名得音级；
2. 兜底：本地 KS（`ks_key_candidates`）在全曲音级分布上算出的候选 key；
3. 都没有 → uncertain，降级循环移位。

## 另两轴

和声：chords.progression 里每项的 function（罗马数字）→ 级数直方图 7 维，
chord 记号 → 性质 4 维，再加和声密度 1 维。级数天然相对主调，移调不变。

旋律：主旋律线（melody_line.segments）按 4 小节切乐句，每乐句用
core 的 theme.fingerprint 取指纹（音程 / 节奏 / 音级），另存一条 42 维检索
向量用于粗排；真正打分时用 theme.compare 在乐句对上取最大值（精排）。
"""
import math
import re

from core.analysis import melody_export as mex
from core.analysis import theme as theme_mod
from core.analysis.scoredata import (
    ks_key_candidates, midi_to_pc, parse_score,
)
from core.models import AnalysisResult, Piece

# ---- 维度 ----
MODE_DIM = 12                     # 调式：12 维音级分布
HARMONY_DIM = 12                  # 和声：级数 7 + 性质 4 + 密度 1
MELODY_VEC_DIM = 42               # 旋律乐句检索向量：音级 12 + 音程 25 + 节奏 5

# ---- 旋律乐句切分 ----
PHRASE_BARS = 4                   # 每几小节切一个乐句
MIN_PHRASE_NOTES = 3              # 少于几个音的残句不单独成乐句

# 音名 → 音级（0=C）
_PC_OF_LETTER = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_SHARP = {'#', '♯', '升'}
_FLAT = {'b', '♭', '降'}

# 罗马数字级数 → 1..7
_ROMAN_DEGREE = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7}


# ---------------------------------------------------------------- 小工具

def _result_output(piece: Piece, step: str):
    """取该作品某分析步骤的已完成输出；没跑 / 没完成返回 None。"""
    try:
        row = AnalysisResult.objects.get(piece=piece, step=step)
    except AnalysisResult.DoesNotExist:
        return None
    if row.status != AnalysisResult.Status.DONE:
        return None
    return row.output or None


def _norm(vec):
    """归一化成和为 1 的分布（全零则原样返回）。"""
    total = sum(vec)
    if total <= 0:
        return list(vec)
    return [x / total for x in vec]


def _hist(values, lo: int, hi: int):
    """整数序列 → [lo..hi] 区间的频率直方图（归一）。"""
    n = hi - lo + 1
    out = [0.0] * n
    for v in values:
        try:
            i = int(v) - lo
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            out[i] += 1.0
    return _norm(out)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def parse_tonic_pc(text) -> int | None:
    """调名字符串 → 主音音级（0=C…11=B），解析不出返回 None。

    兼容 "C 大调" / "C大调" / "a小调" / "升C小调" / "D# 大调" / "降E" / "G major"。
    """
    if not text:
        return None
    m = re.search(r'([升降♯♭#b]?)\s*([A-Ga-g])\s*([#♯b♭]?)', str(text))
    if not m:
        return None
    _pre, letter, _post = m.group(1), m.group(2).upper(), m.group(3)
    pc = _PC_OF_LETTER.get(letter)
    if pc is None:
        return None
    acc = 0
    for ch in (m.group(1), m.group(3)):
        if ch in _SHARP:
            acc += 1
        elif ch in _FLAT:
            acc -= 1
    return (pc + acc) % 12


def _degree_of_function(func) -> int | None:
    """功能标记（I / ii / V7 / vii°）→ 级数 1..7。"""
    if not func:
        return None
    m = re.match(r'\s*([IViv]+)', str(func))
    if not m:
        return None
    return _ROMAN_DEGREE.get(m.group(1).upper())


def _quality_of_chord(chord) -> str:
    """和弦记号 → maj / min / dom7 / other。"""
    c = str(chord or '')
    if not c:
        return 'other'
    low = c.lower()
    if 'dim' in low or '°' in c:
        return 'other'
    if '7' in c:
        return 'dom7'
    # m 但排除 maj / m7（m7 已在上面归 dom7）
    if 'm' in low and 'maj' not in low:
        return 'min'
    if 'maj' in low or re.match(r'^[A-G]', c):
        return 'maj'
    return 'other'


def _cosine(a, b) -> float:
    """余弦相似度（两向量等长；空向量返回 0）。"""
    if not a or not b:
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _pc_cos_max_shift(a, b) -> float:
    """音级分布余弦，**12 次循环移位取最大**（移调不变）—— 调式降级时用。"""
    if not a or not b:
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    best = 0.0
    for k in range(12):
        c = sum(a[i] * b[(i + k) % 12] for i in range(12))
        if c > best:
            best = c
    return best / (na * nb)


# ---------------------------------------------------------------- 调式轴

def full_pc_duration(data) -> list:
    """全曲音级时值分布（12 维），与 tonality 分段统计同口径。"""
    pc = [0.0] * 12
    for p in data.parts:
        for e in p.pitched_events:
            for pitch in e.midi_pitches:
                pc[midi_to_pc(pitch)] += e.duration
    return pc


def mode_vector(data, tonality_out):
    """调式向量：**按主音对齐**的 12 维音级分布。

    返回 ``(vec, tonic_pc, main_key, uncertain)``。
    """
    pc = full_pc_duration(data)
    if sum(pc) <= 0:
        return None, None, '', True

    tonality_out = tonality_out or {}
    main_key = tonality_out.get('main_key') or ''

    # 1) LLM 判的主调
    tonic = parse_tonic_pc(main_key)
    # 2) 兜底：本地 KS 在全曲分布上算
    if tonic is None:
        try:
            cands = ks_key_candidates(pc, top=1)
            if cands:
                tonic = parse_tonic_pc(cands[0].get('key'))
                if not main_key:
                    main_key = cands[0].get('key') or ''
        except Exception:
            tonic = None

    if tonic is None:
        # 主音定不了 → 降级：不做对齐，检索时改用循环移位比对
        return _norm(pc), None, main_key, True

    # 旋转到主音为 0：vec[i] = 相对主音第 i 个音级上的时值占比
    aligned = [pc[(tonic + i) % 12] for i in range(MODE_DIM)]
    return _norm(aligned), tonic, main_key, False


# ---------------------------------------------------------------- 和声轴

def harmony_vector(chords_out):
    """和声向量：级数 7 + 性质 4 + 密度 1 = 12 维。"""
    chords_out = chords_out or {}
    prog = chords_out.get('progression') or []
    prog = [r for r in prog if isinstance(r, dict)]
    if not prog:
        return None, 0

    deg = [0.0] * 7
    qual = {'maj': 0.0, 'min': 0.0, 'dom7': 0.0, 'other': 0.0}
    hit = 0
    for r in prog:
        d = _degree_of_function(r.get('function'))
        if d:
            deg[d - 1] += 1.0
            hit += 1
        q = _quality_of_chord(r.get('chord'))
        qual[q] += 1.0

    deg_hist = _norm(deg)
    qual_hist = _norm([qual['maj'], qual['min'], qual['dom7'], qual['other']])
    window_count = int(chords_out.get('window_count') or len(prog) or 1)
    density = _clamp(hit / max(window_count, 1), 0.0, 1.0)

    vec = list(deg_hist) + list(qual_hist) + [density]
    return vec, hit


# ---------------------------------------------------------------- 旋律轴（乐句级）

def _seg_notes(data, part_index, s, e):
    """段内音符 → [(offset, midi, duration)]（theme.fingerprint 的入参顺序），
    按时间排序，和弦取最高音（与 melody_line 同口径）。"""
    if not (0 <= part_index < len(data.parts)):
        return []
    evs = mex.events_in(data.parts[part_index], s, e)
    out = []
    for ev in sorted(evs, key=lambda x: x.offset):
        if not ev.midi_pitches:
            continue
        out.append((ev.offset, max(ev.midi_pitches), float(ev.duration)))
    return out


def _split_by_bars(notes, measure_offsets, bar_ql, start_m, end_m):
    """把一段音符按 PHRASE_BARS 小节切成乐句（丢掉过短的残句）。"""
    phrases = []
    m = start_m
    while m <= end_m:
        m_end = min(m + PHRASE_BARS - 1, end_m)
        s = measure_offsets.get(m)
        e = measure_offsets.get(m_end + 1)
        if s is None:
            m = m_end + 1
            continue
        if e is None:
            e = s + (m_end - m + 1) * bar_ql
        chunk = [n for n in notes if s <= n[0] < e]
        if len(chunk) >= MIN_PHRASE_NOTES:
            phrases.append(chunk)
        m = m_end + 1
    return phrases


def _fallback_part_index(data):
    """没有 melody_line 时的兜底：取发音数最多的声部当"主旋律"近似。"""
    best_i, best_n = None, 0
    for i, p in enumerate(data.parts):
        n = len(getattr(p, 'pitched_events', []) or [])
        if n > best_n:
            best_i, best_n = i, n
    return best_i


def phrase_vec(fp: dict):
    """乐句指纹 → 42 维检索向量（音级 12 + 音程 25 + 节奏 5，按 THEME_W 加权）。

    加权用 sqrt(w)：这样对拼接向量算余弦，近似等于三轴各自余弦的加权和。
    它只用于**粗排**，真正打分走 theme.compare 的乐句级精排。
    """
    w_iv, w_rh, w_pc = theme_mod.THEME_W
    pc = list(fp.get('pc') or [0.0] * 12)
    iv_hist = _hist([_clamp(v, -12, 12) for v in (fp.get('ivs') or [])], -12, 12)
    rh_hist = _hist([_clamp(v, -2, 2) for v in (fp.get('rhythm') or [])], -2, 2)
    vec = ([x * math.sqrt(w_pc) for x in pc]
           + [x * math.sqrt(w_iv) for x in iv_hist]
           + [x * math.sqrt(w_rh) for x in rh_hist])
    return vec


def melody_phrases(data, melody_line_out):
    """主旋律线 → 乐句指纹列表。返回 ``(phrases, note_count)``。

    phrases 每项 ``{ivs, rhythm, pc, vec, count, start_measure, end_measure}``。
    """
    if data is None:
        return [], 0

    from core.analysis.scoredata import build_measure_map
    measure_offsets, _measures, bar_ql = build_measure_map(data)

    ml_out = melody_line_out or {}
    segments = [s for s in (ml_out.get('segments') or []) if isinstance(s, dict)]

    # 没跑过 melody_line → 兜底用最活跃声部（不精确，但旋律轴不至于整个空掉）
    if not segments:
        idx = _fallback_part_index(data)
        if idx is None or not data.measure_count:
            return [], 0
        segments = [{'part_index': idx, 'start_measure': 1,
                     'end_measure': data.measure_count}]

    phrases = []
    note_count = 0
    for seg in segments:
        try:
            pi = int(seg.get('part_index'))
        except (TypeError, ValueError):
            continue
        if not (0 <= pi < len(data.parts)):
            continue
        sm = int(seg.get('start_measure') or 1)
        em = int(seg.get('end_measure') or sm)
        s, e = mex.seg_range(data, measure_offsets, bar_ql, sm, em)
        notes = _seg_notes(data, pi, s, e)
        if len(notes) < MIN_PHRASE_NOTES:
            continue
        chunks = _split_by_bars(notes, measure_offsets, bar_ql, sm, em)
        # 整段不足一个乐句长度时也别丢（短曲子常见）
        if not chunks and len(notes) >= MIN_PHRASE_NOTES:
            chunks = [notes]
        for chunk in chunks:
            fp = theme_mod.fingerprint(chunk)
            if not fp.get('count'):
                continue
            fp['vec'] = phrase_vec(fp)
            fp['start_measure'] = sm
            fp['end_measure'] = em
            phrases.append(fp)
            note_count += fp['count']
    return phrases, note_count


# ---------------------------------------------------------------- 主入口

def extract(piece: Piece):
    """算一首作品的三轴特征（不落库）。返回 dict。

    ``axes`` 是就绪的轴名列表；缺哪个轴由调用方按 ClusterSetting 决定
    是排除该作品还是只按可用轴比。
    """
    out = {
        'mode_vec': None, 'mode_key': '', 'mode_tonic_pc': None,
        'mode_uncertain': False,
        'harmony_vec': None, 'harmony_chord_count': 0,
        'melody_phrases': None, 'melody_phrase_count': 0,
        'melody_note_count': 0,
        'axes': [], 'error': '',
    }
    if not piece.musicxml_file:
        out['error'] = '没有 MusicXML 文件（未转换成功）'
        return out

    tonality_out = _result_output(piece, AnalysisResult.Step.TONALITY)
    chords_out = _result_output(piece, AnalysisResult.Step.CHORDS)
    ml_out = _result_output(piece, AnalysisResult.Step.MELODY_LINE)

    # 和声轴不需要谱面
    try:
        hv, cnt = harmony_vector(chords_out)
        out['harmony_vec'] = hv
        out['harmony_chord_count'] = cnt
    except Exception as exc:                      # 单轴失败不牵连别的轴
        out['error'] = f'和声轴：{exc}'

    # 调式 + 旋律都要谱面，解析一次复用
    try:
        data = parse_score(piece.musicxml_file.path)
    except Exception as exc:
        out['error'] = (out['error'] + '；' if out['error'] else '') + f'解析乐谱失败：{exc}'
        return out

    try:
        mv, tonic, mkey, uncertain = mode_vector(data, tonality_out)
        out['mode_vec'] = mv
        out['mode_tonic_pc'] = tonic
        out['mode_key'] = mkey or ''
        out['mode_uncertain'] = uncertain
    except Exception as exc:
        out['error'] = (out['error'] + '；' if out['error'] else '') + f'调式轴：{exc}'

    try:
        phrases, ncnt = melody_phrases(data, ml_out)
        out['melody_phrases'] = phrases or None
        out['melody_phrase_count'] = len(phrases)
        out['melody_note_count'] = ncnt
    except Exception as exc:
        out['error'] = (out['error'] + '；' if out['error'] else '') + f'旋律轴：{exc}'

    out['axes'] = [k for k in ('mode', 'harmony', 'melody') if out.get(f'{k}_vec')
                   or (k == 'melody' and out['melody_phrases'])]
    return out
