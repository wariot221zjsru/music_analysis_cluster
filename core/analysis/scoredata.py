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
    midi_pitches: List[int]  # 发声音高（MIDI 号；和弦多元素；无音高打击乐为空）
    is_chord: bool           # 是否同 Part 同时发声（≥2 音）
    is_percussion: bool = False     # 打击乐事件（无确定音高 / 鼓组映射音高）


@dataclass
class PartInfo:
    """声部静态信息。"""
    index: int
    name: str                       # Part 名
    instrument: str = ''            # 乐器名（可能为空）
    is_voice: bool = False          # 是否人声类音色
    is_percussion: bool = False     # 是否打击乐声部
    events: List[NoteEvent] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        return len(self.events)

    @property
    def pitched_events(self) -> List[NoteEvent]:
        """有确定音高的事件（打击乐事件已排除）。

        打击乐的音高是鼓组映射号（36=底鼓、38=军鼓…），不是旋律音高，
        放进和弦 / 调性 / 旋律统计会严重污染结果，故统一在入口处滤掉。
        """
        return [e for e in self.events if not e.is_percussion and e.midi_pitches]

    @property
    def pitched_note_count(self) -> int:
        return len(self.pitched_events)

    @property
    def total_duration(self) -> float:
        return sum(e.duration for e in self.events)

    @property
    def mean_pitch(self) -> float:
        allp = [p for e in self.pitched_events for p in e.midi_pitches]
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


# 打击乐声部识别用的名称关键词（声部名 / 乐器名命中即判为打击乐）
_PERCUSSION_NAME_KEYS = ('drum', 'percussion', 'perc.', 'timpani', 'cymbal',
                         'triangle', 'tambourine', 'woodblock', '打击', '鼓')

# 人声判定：声部名 / 乐器名命中即判为人声
_VOICE_NAME_KEYS = ('voice', 'vocal', 'vocalist', 'choir', 'chorus', 'soprano',
                    'mezzo', 'alto', 'tenor', 'baritone', 'countertenor',
                    'cantus', 'bass voice', 'basso',
                    '人声', '声乐', '合唱', '独唱', '歌手', '歌者',
                    '女高音', '女中音', '女低音', '男高音', '男中音', '男低音')
# 但带这些词的都是**乐器**，不能因为含 alto/tenor/bass 就当人声
# （Alto Saxophone、Tenor Sax、Bass Clarinet、Bass Trombone…）
_NOT_VOICE_NAME_KEYS = ('saxophone', 'sax', 'clarinet', 'flute', 'recorder',
                        'trombone', 'tuba', 'horn', 'oboe', 'bassoon',
                        'guitar', 'violin', 'viola', 'cello', 'double bass',
                        'contrabass', 'drum', 'continuo')
# GM 音色里的人声类 program（52 Choir Aahs / 53 Voice Oohs / 54 Synth Voice）
_VOICE_MIDI_PROGRAMS = (52, 53, 54)

# 只在"同谱已检出明确人声"时才把这类声部名补判为人声（合唱谱的低音声部）
_CHOIR_BASS_NAMES = ('bass', 'basses', 'basso', 'bassi', 'bass voice')


def _is_percussion_part(part, info: 'PartInfo') -> bool:
    """判定一个声部是否为打击乐。

    三条判据命中任一即可（前两条在 MusicXML 有效，第三条兜住纯 MIDI）：
      ① 乐器对象是 music21 Percussion 家族；
      ② MIDI 通道 = 9（GM 标准打击乐通道）；
      ③ 声部名 / 乐器名含打击乐关键词。
    """
    from music21 import instrument as m21instrument

    inst = None
    try:
        inst = part.getInstrument(returnDefault=False)
    except Exception:  # noqa: BLE001
        inst = None
    if inst is not None and isinstance(inst, m21instrument.Percussion):
        return True
    if inst is not None and getattr(inst, 'midiChannel', None) == 9:
        return True
    blob = f'{info.name} {info.instrument}'.lower()
    return any(k in blob for k in _PERCUSSION_NAME_KEYS)


def _is_voice_part(part, info: 'PartInfo') -> bool:
    """判定一个声部是否人声（只看谱面 / 音色元数据，不看用户勾选）。

    四条判据，命中即人声：
      ① 乐器对象属于 music21 的 Vocalist 家族（Soprano / Alto / Tenor…）；
      ② MIDI 音色号是人声类（52 Choir Aahs / 53 Voice Oohs / 54 Synth Voice）；
      ③ 声部名 / 乐器名含人声关键词；
      ④ 上述 ③ 命中但整体含乐器关键词（Alto Saxophone 等）时判为**非**人声。
    """
    blob = f'{info.name} {info.instrument}'.lower()
    if any(k in blob for k in _NOT_VOICE_NAME_KEYS):
        return False
    try:
        from music21 import instrument as m21instrument
        inst = part.getInstrument(returnDefault=False)
        if inst is not None:
            if isinstance(inst, m21instrument.Vocalist):
                return True
            if getattr(inst, 'midiProgram', None) in _VOICE_MIDI_PROGRAMS:
                return True
    except Exception:  # noqa: BLE001
        pass
    return any(k in blob for k in _VOICE_NAME_KEYS)


def detect_voice(musicxml_path: str) -> dict:
    """从谱面判定整首曲子是否含人声。

    这是「是否含人声」的唯一事实来源：上传 MIDI / MusicXML 时一律以文件
    内实际声部为准，覆盖上传页面的勾选值。
    """
    data = parse_score(musicxml_path)
    voice_parts = [{'index': p.index, 'name': p.name,
                    'instrument': p.instrument or '（未标注）'}
                   for p in data.parts if p.is_voice]
    return {
        'has_voice': bool(voice_parts),
        'voice_parts': voice_parts,
        'part_count': len(data.parts),
        'method': '谱面声部元数据（声部名 / 乐器名 / 人声 MIDI 音色）',
    }


def _iter_notes(container, base: float = 0.0):
    """递归遍历容器（Part / Measure / Voice / 嵌套 Stream），
    yield ``(note, 相对 container 的四分音符偏移)``。

    不依赖 music21 的 activeSite（嵌套迭代时 activeSite 可能被破坏，
    getOffsetInHierarchy 会抛 SitesException 而静默丢音符），手工累加偏移。
    """
    for el in container:
        off = base + float(el.offset)
        if getattr(el, 'isStream', False):
            yield from _iter_notes(el, off)
        elif 'NotRest' in getattr(el, 'classes', ()):
            yield el, off


def parse_score(musicxml_path: str) -> ScoreData:
    """解析 MusicXML / MIDI → ScoreData（music21）。

    统一走"小节结构"：若谱面无小节（部分 MIDI），先 makeNotation 补齐，
    这样 offset / measure 都有确定的全局值。

    打击乐（music21 Unpitched / 鼓组通道）**保留事件但不给音高**：
    它参与节奏统计，不参与任何音高类分析（见 PartInfo.pitched_events）。
    """
    from music21 import converter, stream, meter, tempo as m21tempo
    from music21 import note as m21note

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

    # ---- 全局偏移 → 小节号 查表（基准与 measure_offsets 一致） ----
    meas_starts = sorted(((g, num) for num, g in measure_offsets.items()),
                         key=lambda t: t[0])

    def measure_of(offset: float) -> int:
        num = 0
        for g, n_ in meas_starts:
            if g <= offset + 1e-6:
                num = n_
            else:
                break
        return num or 1

    # ---- 事件（递归提取，全局偏移确定） ----
    # 注意：必须递归遍历。MuScriptor / music21 生成的 MusicXML 会把音符包在
    # <voice> 里，只用 Measure.notes 会漏掉绝大部分音符（实测 810 → 16）。
    # 另外打击乐有两种形态：Unpitched（单音）与 PercussionChord（多件鼓同击），
    # 后者 pitches 为空且没有 pitch 属性，最容易在取音高时被静默丢弃。
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

        info.is_percussion = _is_percussion_part(part, info)
        info.is_voice = _is_voice_part(part, info)

        try:
            part_offset = float(part.getOffsetInHierarchy(score))
        except Exception:
            part_offset = 0.0
        has_unpitched = False
        for n, rel_off in _iter_notes(part):
            n_cls = getattr(n, 'classes', ()) or ()
            # 无音高打击乐：Unpitched（单音）与 PercussionChord（同时敲多件鼓）。
            # 后者有 pitches 属性但为空、且没有 pitch，必须在此拦截，
            # 否则会被下面的 else 分支静默丢弃。
            if isinstance(n, m21note.Unpitched) or 'PercussionChord' in n_cls:
                has_unpitched = True
                pitches: List[int] = []
            elif hasattr(n, 'pitches') and len(n.pitches) > 1:
                pitches = [int(p.midi) for p in n.pitches]
            elif hasattr(n, 'pitch'):
                pitches = [int(n.pitch.midi)]
            else:
                # 未知的无音高对象：作为纯节奏事件保留，绝不静默丢弃
                pitches = []
            offset = part_offset + rel_off
            ql = float(n.duration.quarterLength)
            m_no = measure_of(offset)
            info.events.append(NoteEvent(
                part_index=pi, offset=offset, duration=ql,
                measure=m_no, midi_pitches=pitches,
                is_chord=len(pitches) > 1,
                is_percussion=bool(info.is_percussion or not pitches)))
            max_measure = max(max_measure, m_no)
            max_ql = max(max_ql, offset + ql)
        if has_unpitched:
            info.is_percussion = True
        parts.append(info)

    # 合唱谱兜底：SATB 里的 "Bass" 单看像贝斯，但同谱已检出明确人声时应算人声
    if any(p.is_voice for p in parts):
        for p in parts:
            if not p.is_voice and p.name.strip().lower() in _CHOIR_BASS_NAMES:
                p.is_voice = True

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


def pitched_parts(data: ScoreData) -> List[PartInfo]:
    """含有确定音高事件的声部（打击乐声部已排除）。"""
    return [p for p in data.parts if p.pitched_events]


def sounding_pitches_at(part: PartInfo, t: float) -> List[int]:
    """某声部在四分音符位置 t 处正在发声的全部音高（不含打击乐）。"""
    return [p for e in part.events
            if not e.is_percussion
            and e.offset <= t < e.offset + max(e.duration, 1e-6)
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
