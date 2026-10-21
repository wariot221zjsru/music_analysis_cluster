"""主旋律导出的共用工具：节奏 / 音色选项、小节区间换算、事件截取。

两张主旋律卡（⑨ 主旋律声部识别 / 选轨、⑩ 主旋律线追踪 / 逐段）都走这里，
保证导出行为一致、且都能被用户在页面上改：

  节奏  original = 按谱面 tempo 序列写入（变速也保留）
        changed  = 整条线统一 BPM（默认 120，可自定义）
  音色  original = 每段用它自己声部的 MIDI 音色号（多段接力时能听出换乐器）
        changed  = 整条线统一音色（默认钢琴 GM 0，可自定义任意 GM 音色）

以前导出是新建空 Part 只塞音符，既没有 tempo 也没有 instrument，
music21 写 MIDI 时就按默认 120 BPM + 默认钢琴落盘 —— 于是"听起来永远只有
一种乐器、速度也不对"。本模块就是为了修掉这个。
"""
import re

from .scoredata import ScoreData


# GM 标准音色号 → 中文名（供页面下拉与导出说明使用）
GM_PROGRAMS = [
    '大钢琴', '亮音钢琴', '电钢琴', '酒吧钢琴', '柔音电钢琴', '合唱电钢琴',
    '羽管键琴', '击弦古钢琴', '钢片琴', '钟琴', '八音盒', '颤音琴',
    '马林巴', '木琴', '管钟', '扬琴', '哈蒙德风琴', '打击风琴',
    '摇滚风琴', '教堂管风琴', '簧风琴', '手风琴', '口琴', '探戈手风琴',
    '尼龙弦吉他', '钢弦吉他', '爵士电吉他', '清音电吉他', '闷音电吉他',
    '过载电吉他', '失真电吉他', '吉他泛音', '原声贝斯', '指弹电贝斯',
    '拨片电贝斯', '无品贝斯', '击弦贝斯1', '击弦贝斯2', '合成贝斯1',
    '合成贝斯2', '小提琴', '中提琴', '大提琴', '低音提琴', '颤弓弦乐',
    '拨奏弦乐', '竖琴', '定音鼓', '弦乐合奏1', '弦乐合奏2', '合成弦乐1',
    '合成弦乐2', '合唱「啊」', '人声「哦」', '合成人声', '管弦乐重奏',
    '小号', '长号', '大号', '弱音小号', '圆号', '铜管合奏', '合成铜管1',
    '合成铜管2', '高音萨克斯', '中音萨克斯', '次中音萨克斯', '上低音萨克斯',
    '双簧管', '英国管', '大管', '单簧管', '短笛', '长笛', '竖笛', '排箫',
    '吹瓶', '尺八', '口哨', '陶笛', '方波主音', '锯齿波主音',
    '卡利欧佩主音', '吹管主音', '夏威夷主音', '合音主音', '合唱主音',
    '低音主音', '新时代垫音', '温暖垫音', '复音合成垫', '合唱垫音',
    '弓弦垫音', '金属垫音', '光环垫音', '扫弦垫音', '雨声', '原声带',
    '水晶', '大气', '明亮', '妖怪', '回声', '科幻', '西塔琴', '班卓琴',
    '三味线', '十三弦筝', '卡林巴', '风笛', '提琴', '沙纳依', '铃铛',
    '阿哥哥鼓', '钢鼓', '木鱼', '太鼓', '通鼓', '合成鼓', '反钹',
    '吉他品音', '呼吸声', '海浪声', '鸟鸣', '电话铃', '直升机', '掌声',
    '枪声',
]

DEFAULT_BPM = 120
DEFAULT_PROGRAM = 0            # 大钢琴


def gm_name(program: int) -> str:
    """GM 音色号 → 名称（越界时退回编号）。"""
    try:
        p = int(program)
    except (TypeError, ValueError):
        return '大钢琴'
    return GM_PROGRAMS[p] if 0 <= p < len(GM_PROGRAMS) else f'GM {p}'


# ---------------------------------------------------------------- 选项

def resolve_options(config=None, params: dict | None = None) -> dict:
    """把「系统配置默认值」与「页面提交的参数」合成一份导出选项。

    params 里的键（字符串）：tempo_mode / bpm / instrument_mode / program。
    """
    p = params or {}

    def cfg(name, default):
        return getattr(config, name, default) if config is not None else default

    tempo_mode = str(p.get('tempo_mode') or cfg('melody_export_tempo_mode', 'original'))
    if tempo_mode not in ('original', 'changed'):
        tempo_mode = 'original'
    instrument_mode = str(p.get('instrument_mode')
                          or cfg('melody_export_instrument_mode', 'original'))
    if instrument_mode not in ('original', 'changed'):
        instrument_mode = 'original'

    try:
        bpm = int(p.get('bpm') or cfg('melody_export_bpm', DEFAULT_BPM))
    except (TypeError, ValueError):
        bpm = int(cfg('melody_export_bpm', DEFAULT_BPM) or DEFAULT_BPM)
    bpm = max(20, min(300, bpm))

    try:
        program = int(p.get('program') if p.get('program') not in (None, '')
                      else cfg('melody_export_program', DEFAULT_PROGRAM))
    except (TypeError, ValueError):
        program = int(cfg('melody_export_program', DEFAULT_PROGRAM) or DEFAULT_PROGRAM)
    program = max(0, min(127, program))

    return {
        'tempo_mode': tempo_mode,
        'bpm': bpm,
        'instrument_mode': instrument_mode,
        'program': program,
    }


def options_label(options: dict) -> str:
    """选项的一句话说明（写进导出的 output，页面上回显）。"""
    tempo = ('原始节奏' if options['tempo_mode'] == 'original'
             else f"统一 {options['bpm']} BPM")
    inst = ('各段原始音色' if options['instrument_mode'] == 'original'
            else f"统一音色：{gm_name(options['program'])}（GM {options['program']}）")
    return f'{tempo} · {inst}'


def shape_stem_tag(shape_gate: bool, threshold=None) -> str:
    """形态资格层的文件名标记：开 → ``shape0.35``，关 → ``''``（与以前同名）。

    开与不开是两种不同口径的提取结果（可能归属到不同声部），写同一个文件名
    会互相覆盖 —— 开了形态层跑完再关掉重跑，前一份就没了，历史归档里那次
    运行的下载链接也就指向了别人的文件。带上标记后两者各存各的。
    """
    if not shape_gate:
        return ''
    try:
        return f'shape{float(threshold):g}'
    except (TypeError, ValueError):
        return 'shape'


def shape_label(shape_gate: bool, threshold=None) -> str:
    """下载文件名里用的可读形态标记：开 → ``形态0.35``，关 → ``''``。

    与 shape_stem_tag 的区别只是给人看的（一个中文、一个英文短标记）。
    """
    if not shape_gate:
        return ''
    try:
        return f'形态{float(threshold):g}'
    except (TypeError, ValueError):
        return '形态'


def run_stem_tag(seq) -> str:
    """运行序号标记：``r1`` / ``r2`` … 与「历史运行」里的「第 N 次」对得上。"""
    try:
        return f'r{max(int(seq), 1)}'
    except (TypeError, ValueError):
        return 'r1'


def time_stem_tag() -> str:
    """时间兜底标记（旧结果没有运行序号时用），保证每次落盘的文件名都不同。"""
    from django.utils import timezone
    return 't' + timezone.localtime().strftime('%m%d%H%M%S')


def next_run_seq(piece, step: str) -> int:
    """该步骤下一次运行会归档成的序号（1 起，与历史表里的 seq 一致）。

    只用来给导出文件编号，查不到（比如还没建表 / 非 ORM 场景）就退回 1。
    """
    try:
        from ..models import AnalysisRun
        if piece is None or getattr(piece, 'pk', None) is None:
            return 1
        last = (AnalysisRun.objects.filter(piece=piece, step=step)
                .order_by('-seq').first())
        return (last.seq + 1) if last else 1
    except Exception:  # noqa: BLE001 —— 编号失败不该拖垮导出
        return 1


def export_run_tag(output: dict | None, piece, step: str) -> str:
    """重新导出（写回结果）时用的运行标记。

    沿用本次结果里记下的运行序号：同一份结果反复改节奏 / 音色，换的是选项
    后缀（BPM / GM 号）而不是运行编号，于是"第 N 次"那一份始终只有一份，
    历史归档里那次运行的链接也一直指向它自己的文件。
    老结果里没记序号时用时间戳兜底，保证不与任何已有文件重名。
    """
    seq = (output or {}).get('export_run_seq') if isinstance(output, dict) else None
    if seq:
        return run_stem_tag(seq)
    return time_stem_tag()


def stem_with_options(stem: str, options: dict, shape_tag: str = '',
                      run_tag: str = '') -> str:
    """导出文件名（stem）带上形态层标记 + 节奏 / 音色选项 + 运行序号。

    以前所有选项都落同一个文件名（如 melody_line_absolute.mid），重新导出只是
    覆盖同名文件 —— 下载链接的 URL 一个字都没变，浏览器常常直接给缓存里的旧
    文件，于是"明明选了小提琴、下载的还是钢琴"。带上选项后 URL 随选项变化，
    既避开缓存，也方便同时留着几个版本对照。

    再加两段：
      shape_tag —— 开了「形态决定资格」的提取单独一份文件（口径不同、归属
                   可能不同，不能和没开的那份互相覆盖）；
      run_tag   —— 每次运行一份（``r1`` / ``r2``…），重跑不再覆盖上一次的
                   产物，历史归档里每一次运行都能下到它自己那份。
    """
    tempo = ('orig-tempo' if options['tempo_mode'] == 'original'
             else f'{int(options["bpm"])}bpm')
    inst = ('orig-inst' if options['instrument_mode'] == 'original'
            else f'gm{int(options["program"])}')
    parts = [stem]
    if shape_tag:
        parts.append(shape_tag)
    parts += [tempo, inst]
    if run_tag:
        parts.append(run_tag)
    return '__'.join(parts)


def download_filename(title: str, kind_label: str, options: dict,
                      ext: str = '.mid', shape_tag: str = '') -> str:
    """给"按当前设置直接下载"用的可读文件名。

    例：Potong Bebek Angsa_主旋律线_96BPM_小提琴.mid
    开了形态资格层时 kind_label 后面多一段「形态0.35」，下下来的两份一眼能分清。
    """
    tempo = ('原始节奏' if options['tempo_mode'] == 'original'
             else f'{int(options["bpm"])}BPM')
    inst = ('各段原始音色' if options['instrument_mode'] == 'original'
            else gm_name(options['program']))
    label = f'{kind_label}_{shape_tag}' if shape_tag else kind_label
    base = f'{title}_{label}_{tempo}_{inst}{ext}'
    # 去掉文件名里不安全的字符（Windows 不允许 \ / : * ? " < > |）
    return re.sub(r'[\\/:*?"<>|\r\n\t]', '_', base)[:120]


# ---------------------------------------------------------------- 区间

def seg_range(data: ScoreData, measure_offsets: dict, bar_ql: float,
              start_m: int, end_m: int):
    """段 [start_m, end_m] 小节的四分音符区间。

    与旧的 .get(m, 0.0) 兜底不同：小节号查不到时按 bar_ql 从**最近已知小节**
    推算，绝不退回 0.0（那等于把段起点当成全曲开头，导出会整段错位）。
    """
    known = sorted(measure_offsets.items(), key=lambda kv: kv[0])

    def lookup(m: int) -> float:
        if m in measure_offsets:
            return measure_offsets[m]
        prev = [kv for kv in known if kv[0] < m]
        if prev:
            n0, g0 = max(prev, key=lambda kv: kv[0])
            return g0 + (m - n0) * bar_ql
        nxt = [kv for kv in known if kv[0] > m]
        if nxt:
            n0, g0 = min(nxt, key=lambda kv: kv[0])
            return max(0.0, g0 - (n0 - m) * bar_ql)
        return (m - 1) * bar_ql

    s = lookup(start_m)
    if end_m + 1 in measure_offsets:
        e = measure_offsets[end_m + 1]
    elif end_m >= data.measure_count:
        e = data.total_ql + 1e-6
    else:
        e = lookup(end_m + 1)
    if e <= s:
        e = s + bar_ql
    return s, e


def events_in(part, s: float, e: float) -> list:
    """段内事件（按起点判定，同段内不会重复）。"""
    return [ev for ev in part.pitched_events if s <= ev.offset < e]


def clamped_duration(ev, e: float) -> float:
    """音符时值裁剪到段尾，避免拼接导出时上一段的长音压到下一段头上。"""
    dur = float(ev.duration)
    remain = e - float(ev.offset)
    if remain <= 0:
        return min(dur, 0.05)
    return min(dur, remain)


def make_note(ev, duration: float | None = None):
    """事件 → music21 音符（和弦取最高音，与显著性口径一致）。"""
    from music21 import note as m21note

    n = m21note.Note()
    n.pitch.midi = max(ev.midi_pitches)
    n.duration.quarterLength = duration if duration is not None else ev.duration
    return n


# ---------------------------------------------------------------- 节奏 / 音色

def program_of_part(part_info) -> int:
    """声部的 MIDI 音色号：优先谱面 program，其次乐器名匹配，最后钢琴。"""
    prog = getattr(part_info, 'midi_program', None)
    if prog is not None:
        try:
            return max(0, min(127, int(prog)))
        except (TypeError, ValueError):
            pass
    name = (getattr(part_info, 'instrument', '') or getattr(part_info, 'name', '') or '')
    key = name.strip().lower()
    if key:
        for i, gname in enumerate(GM_PROGRAMS):
            if gname == key or gname in key or key in gname:
                return i
    return DEFAULT_PROGRAM


def apply_tempo(part, data: ScoreData, measure_offsets: dict, bar_ql: float,
                options: dict) -> str:
    """给导出 Part 写入速度标记，返回说明。"""
    from music21 import tempo as m21tempo

    if options['tempo_mode'] == 'changed':
        part.insert(0.0, m21tempo.MetronomeMark(number=float(options['bpm'])))
        return f"统一 {options['bpm']} BPM"

    tempos = [t for t in (data.tempos or []) if t.get('bpm')]
    if not tempos:
        part.insert(0.0, m21tempo.MetronomeMark(number=float(DEFAULT_BPM)))
        return f'原始节奏（谱面未标速度，按 {DEFAULT_BPM} BPM）'
    for t in tempos:
        m = int(t.get('measure') or 1)
        off = measure_offsets.get(m, (m - 1) * bar_ql)
        part.insert(off, m21tempo.MetronomeMark(number=float(t['bpm'])))
    shown = '，'.join(f"{float(t['bpm']):.0f} BPM @ 第{int(t.get('measure') or 1)}小节"
                     for t in tempos[:3])
    more = f' 等 {len(tempos)} 处' if len(tempos) > 3 else ''
    return f'原始节奏（{shown}{more}）'


def apply_instruments(part, data: ScoreData, spans: list, options: dict) -> str:
    """给导出 Part 写入乐器（音色），返回说明。

    spans: [{'offset': 该段在导出里的起点, 'part_index': 该段所属声部}]
    """
    from music21 import instrument as m21inst

    if options['instrument_mode'] == 'changed':
        prog = options['program']
        part.insert(0.0, m21inst.instrumentFromMidiProgram(prog))
        return f"统一音色：{gm_name(prog)}（GM {prog}）"

    used = {}
    for seg in spans:
        idx = seg.get('part_index')
        if idx is None or idx >= len(data.parts):
            continue
        prog = program_of_part(data.parts[idx])
        used.setdefault(prog, data.parts[idx].name)
        part.insert(seg.get('offset', 0.0),
                    m21inst.instrumentFromMidiProgram(prog))
    if not used:
        part.insert(0.0, m21inst.instrumentFromMidiProgram(DEFAULT_PROGRAM))
        return f'各段原始音色（未读到音色号，按 {gm_name(DEFAULT_PROGRAM)}）'
    desc = '，'.join(f'{gm_name(p)}（{name}）' for p, name in list(used.items())[:4])
    more = f' 等 {len(used)} 种' if len(used) > 4 else ''
    return f'各段原始音色（{desc}{more}）'


def finalize_part(part):
    """导出前补齐小节结构与休止（留白段也要占满小节）。"""
    from music21 import stream

    try:
        has_measure = bool(part.recurse().getElementsByClass(stream.Measure))
    except Exception:  # noqa: BLE001
        has_measure = False
    if not has_measure:
        try:
            part = part.makeMeasures(inPlace=False)
            part.makeRests(fillGaps=True, inPlace=True)
        except Exception:  # noqa: BLE001
            pass
    return part
