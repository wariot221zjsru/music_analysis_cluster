"""共享工具：MusicXML（music21）解析与音乐特征提取。

所有分析模块都消费这里产出的中间表示（parts / events / measure map），
避免各模块重复解析。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class NoteEvent:
    """一个发声事件（和弦按一个事件、多个音高处理）。"""
    part_index: int          # 所属 Part 序号（0 起）
    offset: float            # 相对全曲起始的四分音符位置
    duration: float          # 四分音符时值
    measure: int             # 小节号
    midi_pitches: List[int]  # 发声音高（MIDI 号；和弦多元素）
    is_chord: bool           # 是否同 Part 同时发声（≥2 音）


@dataclass
class PartInfo:
    """声部静态信息。"""
    index: int
    name: str                       # Part 名
    instrument: str = ''            # 乐器名（可能为空）
    is_voice: bool = False          # 是否人声类音色
    events: List[NoteEvent] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        return len(self.events)

    @property
    def total_duration(self) -> float:
        return sum(e.duration for e in self.events)

    @property
    def mean_pitch(self) -> float:
        allp = [p for e in self.events for p in e.midi_pitches]
        return sum(allp) / len(allp) if allp else 0.0


@dataclass
class ScoreData:
    """全曲中间表示（分析模块的统一输入）。"""
    parts: List[PartInfo]
    time_signatures: List[dict]     # [{measure, numerator, denominator}]
    tempos: List[dict]              # [{measure, bpm, text}]
    measure_offsets: dict           # {小节号: 全曲四分音符起点}
    measure_count: int
    total_ql: float                 # 总四分音符长度
    part_names: List[str] = field(default_factory=list)


def parse_score(musicxml_path: str) -> ScoreData:
    """解析 MusicXML / MIDI → ScoreData（music21）。

    统一走"小节结构"：若谱面无小节（部分 MIDI），先 makeNotation 补齐，
    这样 offset / measure 都有确定的全局值。
    """
    from music21 import converter, stream, meter, tempo as m21tempo

    score = converter.parse(str(musicxml_path))
    if not isinstance(score, stream.Score):
        score = score.stream()  # 单 Part 情况包装
    # MIDI 等来源可能没有小节结构 → 补齐（量化 + 分小节）
    if not score.recurse().getElementsByClass(stream.Measure):
        score = score.makeNotation(inPlace=False)

    # ---- 以第一个含小节的 Part 为基准建立 {小节号: 全局偏移} ----
    measure_offsets = {}
    time_signatures = []
    base_part = None
    for part in score.parts:
        measures = part.getElementsByClass(stream.Measure)
        if measures:
            base_part = part
            part_offset = 0.0
            try:
                part_offset = float(part.getOffsetInHierarchy(score))
            except Exception:
                part_offset = 0.0
            for m in measures:
                g = part_offset + float(m.offset)
                if m.number not in measure_offsets or g < measure_offsets[m.number]:
                    measure_offsets[m.number] = g
                ts = m.timeSignature
                if ts is not None:
                    entry = {'measure': m.number, 'numerator': ts.numerator,
                             'denominator': ts.denominator}
                    if entry not in time_signatures:
                        time_signatures.append(entry)
            break
    if base_part is None:  # 理论上不可达（上面已 makeNotation）
        raise ValueError('谱面无小节结构，且 makeNotation 失败。')

    # ---- 速度标记 ----
    tempos = []
    seen = set()
    for el in score.recurse().getElementsByClass(m21tempo.MetronomeMark):
        try:
            m_no = el.measureNumber or 1
        except Exception:
            m_no = 1
        key = (m_no, el.text, el.number)
        if key in seen:
            continue
        seen.add(key)
        tempos.append({'measure': m_no,
                       'bpm': float(el.number) if el.number else None,
                       'text': el.text or ''})
    tempos.sort(key=lambda d: d['measure'])

    # ---- 事件（按 Part × 小节 提取，全局偏移确定） ----
    parts: List[PartInfo] = []
    max_measure, max_ql = 0, 0.0
    for pi, part in enumerate(score.parts):
        info = PartInfo(index=pi, name=part.partName or f'Part {pi + 1}')
        try:
            inst = part.getInstrument(returnDefault=False)
            if inst is not None:
                info.instrument = (getattr(inst, 'instrumentName', None)
                                   or getattr(inst, 'partName', '') or '')
        except Exception:
            info.instrument = ''
        low = (info.instrument or '').lower()
        info.is_voice = any(k in low for k in
                            ('voice', 'vocal', 'choir', 'soprano', 'alto', 'tenor'))

        try:
            part_offset = float(part.getOffsetInHierarchy(score))
        except Exception:
            part_offset = 0.0
        for m in part.getElementsByClass(stream.Measure):
            for n in m.notes:
                if hasattr(n, 'pitches') and len(n.pitches) > 1:
                    pitches = [int(p.midi) for p in n.pitches]
                else:
                    pitches = [int(n.pitch.midi)]
                offset = part_offset + float(m.offset) + float(n.offset)
                info.events.append(NoteEvent(
                    part_index=pi, offset=offset,
                    duration=float(n.duration.quarterLength),
                    measure=int(m.number), midi_pitches=pitches,
                    is_chord=len(pitches) > 1))
                max_measure = max(max_measure, int(m.number))
                max_ql = max(max_ql, offset + float(n.duration.quarterLength))
        parts.append(info)

    if not time_signatures:
        time_signatures = [{'measure': 1, 'numerator': 4, 'denominator': 4}]

    return ScoreData(
        parts=parts,
        time_signatures=time_signatures,
        tempos=tempos,
        measure_offsets=measure_offsets,
        measure_count=max_measure,
        total_ql=max_ql,
        part_names=[p.name for p in parts],
    )


# ---------------- 纯 Python 音乐工具（本地确定性计算） ----------------

PC_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_pc(midi: int) -> int:
    return midi % 12


def midi_to_name(midi: int) -> str:
    return f'{PC_NAMES_SHARP[midi % 12]}{midi // 12 - 1}'


def build_measure_map(data: ScoreData):
    """返回 (measure_offsets, measures, bar_ql)。

    bar_ql 从第一个拍号推算（混合拍号时为首个小节长度，用于兜底）。
    """
    first = data.time_signatures[0] if data.time_signatures \
        else {'numerator': 4, 'denominator': 4}
    bar_ql = first['numerator'] * 4.0 / first['denominator']
    measures = sorted(data.measure_offsets.keys()) or list(range(1, data.measure_count + 1))
    if not data.measure_offsets:  # 兜底：均匀划分
        data.measure_offsets = {m: (m - 1) * bar_ql for m in range(1, data.measure_count + 1)}
        measures = list(range(1, data.measure_count + 1))
    return data.measure_offsets, measures, bar_ql


def windows_of_measures(data: ScoreData, window_bars: int):
    """把全曲切成若干时间窗（每窗 window_bars 个小节）。"""
    _, measures, bar_ql = build_measure_map(data)
    if not measures:
        return []
    out, w = [], 1
    for start in range(1, data.measure_count + 1, window_bars):
        end = min(start + window_bars - 1, data.measure_count)
        out.append({'window': w, 'start_measure': start, 'end_measure': end})
        w += 1
    return out


def sounding_pitches_at(part: PartInfo, t: float) -> List[int]:
    """某声部在四分音符位置 t 处正在发声的全部音高。"""
    return [p for e in part.events
            if e.offset <= t < e.offset + max(e.duration, 1e-6)
            for p in e.midi_pitches]


def interval_seq(pitches: List[int]) -> List[int]:
    """音高序列 → 有向音程序列（半音）。"""
    return [b - a for a, b in zip(pitches, pitches[1:])]


def lcs_ratio(a: List[int], b: List[int]) -> float:
    """两个音程序列的 LCS 相似度（0~1）。用于交接段的动机相似性。"""
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / max(len(a), len(b))


def ks_key_candidates(pc_duration: List[float], top: int = 2) -> List[dict]:
    """Krumhansl-Schmuckler 调性判定：24 个大小调候选按相关性排序。"""
    import math
    MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def pearson(x, y):
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
        vx = math.sqrt(sum((a - mx) ** 2 for a in x))
        vy = math.sqrt(sum((b - my) ** 2 for b in y))
        return cov / (vx * vy) if vx and vy else 0.0

    cands = []
    for tonic in range(12):
        for mode, profile in (('major', MAJOR), ('minor', MINOR)):
            rotated = pc_duration[tonic:] + pc_duration[:tonic]
            cands.append({
                'key': f'{PC_NAMES_SHARP[tonic]} {"大调" if mode == "major" else "小调"}',
                'mode': mode,
                'correlation': round(pearson(rotated, profile), 4),
            })
    cands.sort(key=lambda d: -d['correlation'])
    return cands[:top]
