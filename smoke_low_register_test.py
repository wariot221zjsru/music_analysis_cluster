"""低音区保护的回归脚本：低音声部该排的时候排，该让路的时候让路。

起因（用户提问）
----------------
「如果整首曲子全是低音，或者低音在听感上绝对占优的话，那就不应该排除所有
低音。考虑到了这种情况吗？」

低音区排除是一条**经验先验**（主旋律通常在更高音区），必须能进能退。本脚本
把四种情形钉死：

  A. 整首曲子都在低音区 → 相对差判据够不着，谁都不该被排除（这一条本来
     就成立，脚本把它固定住，免得以后改判据时改坏）；
  B. 低音主奏 + 上方只剩持续音铺底 → 低音必须翻盘（真主题在下面）；
  C. 常规织体（旋律在高音 / 稀疏长笛 + 走动贝斯 / 单音踏板贝斯）→ 不许翻盘，
     否则就是把昨天的 bug 放回来；
  D. 翻盘发生时，声部识别的矩阵归属、角色表、逐段追踪的分段三者要说同一件事。

用法：python smoke_low_register_test.py
跑完自动清理临时 DB / media 产物。
"""
import os
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402
from django.contrib.auth.models import User  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.core.management import call_command  # noqa: E402

from core.analysis import pipeline  # noqa: E402
from core.analysis import voice_id as vmod  # noqa: E402
from core.analysis.melody_line import _seg_pitch_seq  # noqa: E402
from core.analysis.scoredata import parse_score  # noqa: E402
from core.models import AnalysisResult, Piece, SystemConfig  # noqa: E402

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '_smoke_lowreg.sqlite3')
PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


# ------------------------------------------------------------ 造谱
BASS_TUNE = [36, 38, 40, 41, 40, 38, 36, 35]     # 一条真正在走的低音旋律线
WALK = [36, 38, 40, 41, 36, 38, 40, 41]          # 走动贝斯（伴奏画像）
FLUTE_LINE = [72, 74, 76, 79]                    # 稀疏高音旋律


def build_score(spec, bars=16):
    """spec: [(声部名, 乐器类, [(音高, 时值), ...] 或 [音高...], 每音时值[, 力度])]

    第 5 个元素可选：给该声部每个音都写上这个 velocity（1–127），用来造
    "谱面标了力度"的曲子。不写 = 谱面没标力度（velocity 落到 None）。
    """
    from music21 import instrument, note, stream

    sc = stream.Score()
    for item in spec:
        name, inst, pitches, ql = item[0], item[1], item[2], item[3]
        vel = item[4] if len(item) > 4 else None
        p = stream.Part()
        p.partName = name
        p.insert(0, inst)
        for m in range(bars):
            off = m * 4.0
            items = pitches if isinstance(pitches[0], tuple) else [
                (pc, ql) for pc in pitches]
            for i, (pc, d) in enumerate(items):
                n = note.Note(pc, quarterLength=d)
                if vel is not None:
                    n.volume.velocity = vel        # 音符自带力度
                p.insert(off + i * d, n)
        sc.append(p)
    return sc


def _dump(score):
    import tempfile
    path = tempfile.mktemp(suffix='.musicxml')
    score.write('musicxml', fp=path)
    with open(path, 'rb') as f:
        data = f.read()
    os.remove(path)
    return data


def prepare():
    for path in (DB_FILE,):
        if os.path.exists(path):
            os.remove(path)
    call_command('migrate', '--run-syncdb', verbosity=0)
    User.objects.filter(username='smoke').delete()
    User.objects.create_user('smoke', password='smoke123456')
    cfg = SystemConfig.get_config()
    cfg.melody_voice_priority = True
    cfg.melody_segment_min_bars = 2
    cfg.melody_confidence_high = 0.25
    cfg.melody_confidence_mid = 0.10
    cfg.analysis_window_bars = 4
    cfg.melody_shape_gate = False
    cfg.density_by_note_count = True
    cfg.melody_dynamics = True       # 力度参与"听感"判定（默认口径）
    cfg.save()
    print('[准备] SQLite 迁移 + 测试用户 + 配置复位 …… OK')


def make_piece(user, title, data: bytes):
    piece = Piece(owner=user, title=title, source_type='musicxml')
    piece.source_file.save('low.musicxml',
                           SimpleUploadedFile('low.musicxml', data), save=True)
    pipeline._run_convert(piece.id)
    piece.refresh_from_db()
    r = AnalysisResult.objects.get(piece=piece, step='convert')
    assert r.status == 'done', f'convert 失败：{r.error[:400]}'
    return piece


def seg_of(out):
    return [(s['start_measure'], s['end_measure'], s['part_name'])
            for s in out['segments']]


def run_pipeline(piece, score_data, ):
    pipeline._run_steps(piece.id, ['instrumentation', 'voice_id', 'melody_line'])
    vo = AnalysisResult.objects.get(piece=piece, step='voice_id').output
    ml = AnalysisResult.objects.get(piece=piece, step='melody_line').output
    return vo, ml


def matrix_owners(vo):
    """声部识别矩阵里每个窗的归属（排好序的小节起点）。"""
    return [(w['start_measure'],
             vo['salience']['part_names'][w['attribution']]
             if w['attribution'] is not None else '（静默）')
            for w in vo['salience']['matrix']]


def main():
    from music21 import instrument
    prepare()
    user = User.objects.get(username='smoke')
    pieces = []

    # ---------------------------------------------------------- A 全曲皆低音
    print('\n[A] 整首曲子都在低音区 → 一个声部都不该被排除')
    sc = build_score([
        ('Tuba', instrument.Tuba(), [38, 40, 41, 40], 1.0),
        ('Violoncello', instrument.Violoncello(), [50, 52, 53, 52], 1.0),
        ('Contrabass', instrument.Contrabass(), [33, 35, 31, 35], 1.0),
    ])
    piece = make_piece(user, 'all-low', _dump(sc))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    mp = vmod.part_mean_pitches(data)
    low = vmod.low_register_parts(data)
    check('三声部平均音高都在 C4 以下', max(mp.values()) < 60,
          str({k: round(v, 1) for k, v in mp.items()}))
    check('没有任何声部被判成低音区', low == set(), str(sorted(low)))
    vo, ml = run_pipeline(piece, data)
    check('覆盖率 100%', ml['coverage']['ratio'] == 1.0, str(ml['coverage']))
    names = {s[2] for s in seg_of(ml)}
    check('主旋律落在最高的那条线上（大提琴）', names == {'Violoncello'},
          str(seg_of(ml)))
    check('结果里不出现低音区保护字样',
          not ml.get('low_register_overridden'),
          str(ml.get('low_register_overridden')))

    # ---------------------------------------------------------- B 低音听感占优
    print('\n[B] 低音主奏 + 上方只剩持续音铺底 → 低音必须翻盘')
    sc = build_score([
        ('Lead Bass', instrument.ElectricBass(), BASS_TUNE, 0.5),
        ('Pad', instrument.Piano(), [(72, 2.0), (74, 2.0)], None),
    ])
    piece = make_piece(user, 'bass-led', _dump(sc))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    low = vmod.low_register_parts(data)
    check('低音声部本来会被判成低音区（说明排除规则确已生效）',
          len(low) == 1 and data.part_names[sorted(low)[0]] == 'Lead Bass',
          str(sorted(low)))
    check('上方 pad 只在两个音之间来回（够不上"一条线"）',
          not _looks_like(data, 'Pad', 0.0, 16.0), '铺垫：pad 不成线')
    check('低音自己是一条有起伏的线',
          _looks_like(data, 'Lead Bass', 0.0, 16.0))
    vo, ml = run_pipeline(piece, data)
    names = {s[2] for s in seg_of(ml)}
    check('分段把主旋律判给低音声部', names == {'Lead Bass'}, str(seg_of(ml)))
    check('声部识别矩阵的归属也是低音声部',
          {n for _, n in matrix_owners(vo)} == {'Lead Bass'},
          str(matrix_owners(vo)))
    check('两条路说的是同一件事', {s[2] for s in seg_of(ml)}
          == {n for _, n in matrix_owners(vo)})
    check('结果里写明了经历过低音区保护',
          ml.get('low_register_overridden') == ['Lead Bass'],
          str(ml.get('low_register_overridden')))
    check('方法说明交代了原因（上方没有成线的候选）',
          '低音区保护' in (ml.get('method') or '')
          and '踏板 / 持续音' in (ml.get('method') or ''),
          (ml.get('method') or '')[-80:])
    roles = {r['part_name']: r['role_candidate'] for r in vo['roles']}
    check('角色表把这人报成「主旋律（候选）」而不是「低音」',
          roles.get('Lead Bass') == '主旋律（候选）', str(roles))
    check('窗表里逐窗标出了翻盘',
          all(w.get('low_override') for w in ml['windows']),
          str([w.get('low_override') for w in ml['windows']]))
    check('翻盘不是白给置信度（给了形态分，不是 0）',
          all(w['margin'] > 0 for w in ml['windows']),
          str([w['margin'] for w in ml['windows']][:4]))

    # ---------------------------------------------------------- C 不该翻盘
    print('\n[C] 常规织体不许翻盘（把低音区排除的意义还回来）')

    # C1 旋律在高音 + 单音踏板贝斯
    print('\n  C1 高音旋律 + 单音反复的贝斯（贝斯只有 1 个音高）')
    sc = build_score([
        ('Lead', instrument.Violin(), [72, 74, 76, 74, 72, 71, 72, 76], 0.5),
        ('Pedal Bass', instrument.AcousticBass(), [36, 36, 36, 36], 1.0),
    ])
    piece = make_piece(user, 'no-flip-pedal', _dump(sc))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    vo, ml = run_pipeline(piece, data)
    names = {s[2] for s in seg_of(ml)}
    check('主旋律仍是高音声部', names == {'Lead'}, str(seg_of(ml)))
    check('没有触发低音区保护', not ml.get('low_register_overridden'),
          str(ml.get('low_register_overridden')))
    check('单音反复不成线 → 翻盘进不了门',
          not _looks_like(data, 'Pedal Bass', 0.0, 16.0))

    # C2 稀疏长笛旋律 + 走动贝斯
    print('\n  C2 稀疏的长笛旋律 + 走动贝斯（贝斯密度更高但它是伴奏）')
    sc = build_score([
        ('Flute', instrument.Flute(), FLUTE_LINE, 1.0),
        ('Contrabass', instrument.Contrabass(), WALK, 0.5),
    ])
    piece = make_piece(user, 'no-flip-walk', _dump(sc))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    vo, ml = run_pipeline(piece, data)
    names = {s[2] for s in seg_of(ml)}
    check('主旋律仍是长笛', names == {'Flute'}, str(seg_of(ml)))
    check('没有触发低音区保护', not ml.get('low_register_overridden'),
          str(ml.get('low_register_overridden')))

    # C3 常规三声部（旋律 + 人声 + 贝斯）
    print('\n  C3 常规三声部：低音区"长时值 + 高重复"画像不得翻盘')
    sc = build_score([
        ('Acoustic Bass', instrument.AcousticBass(), [36, 36, 36, 36], 1.0),
        ('Lead', instrument.Violin(), [72, 74, 76, 74, 72, 71, 72, 76], 0.5),
    ])
    piece = make_piece(user, 'no-flip-triad', _dump(sc))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    vo, ml = run_pipeline(piece, data)
    names = {s[2] for s in seg_of(ml)}
    check('贝斯没抢走任何一段', 'Acoustic Bass' not in names, str(seg_of(ml)))
    roles = {r['part_name']: r['role_candidate'] for r in vo['roles']}
    check('角色表仍把贝斯报成「低音」',
          roles.get('Acoustic Bass') == '低音', str(roles))

    # ---------------------------------------------------------- D 一致性
    print('\n[D] 翻盘 / 未翻盘两种情形下，覆盖率都不掉')
    for pc in pieces:
        out = AnalysisResult.objects.get(piece=pc, step='melody_line').output
        check(f'{pc.title} 覆盖率 100%', out['coverage']['ratio'] == 1.0,
              str(out['coverage']))

    # ---------------------------------------------------------- E 力度参与听感
    # 「听感上更像主旋律」离不开力度：谱面写了力度时，它就是最直接的证据 ——
    # 两条线别的方面分不出高下，耳朵跟着响的那条走。
    print('\n[E] 谱面标了力度时，"听感占优"要按力度说话')
    from core.analysis import melody_line as ml_mod
    from music21 import instrument

    PP, FF = 44, 108          # p / ff（与 music21 自己的响度标度一致）

    def _lead_word(ml):
        """第一窗的翻盘证据（没有翻盘则返回空 dict）。"""
        return ml['windows'][0].get('low_override_shape') or {}

    def bass_pad_score(bass_vel, pad_vel):
        """B 场景：低音主奏 + 上方只剩两个音来回的踏板；力度可分别指定。"""
        return build_score([
            ('Lead Bass', instrument.ElectricBass(), BASS_TUNE, 0.5, bass_vel),
            ('Pad', instrument.Piano(), [(72, 2.0), (74, 2.0)], None, pad_vel),
        ])

    # E1 低音主奏且写得比对手响 → 照旧翻盘（力度是加分项）
    print('\n  E1 低音主奏 + 写得比对手响 → 翻盘')
    piece = make_piece(user, 'dyn-bass-loud', _dump(bass_pad_score(FF, PP)))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    vo, ml = run_pipeline(piece, data)
    check('谱面把力度读进来了（两个声部都有力度数据）',
          ml['dynamics']['notated'] is True
          and len(ml['dynamics']['parts']) == 2,
          str(ml['dynamics']['parts']))
    check('更响的低音照旧当选', {s[2] for s in seg_of(ml)} == {'Lead Bass'},
          str(seg_of(ml)))
    check('窗证据里记下了正的力度领先',
          (_lead_word(ml).get('dyn_lead') or 0) > 0, str(_lead_word(ml)))
    check('方法说明交代了力度这条依据',
          '力度' in (ml.get('method') or ''), (ml.get('method') or '')[-70:])

    # E2 同一份织体，低音写成 pp、踏板写成 ff → 力度否决翻盘
    print('\n  E2 低音主奏但写得比对手弱 → 不许翻盘（力度否决）')
    piece = make_piece(user, 'dyn-bass-weak', _dump(bass_pad_score(PP, FF)))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    vo, ml = run_pipeline(piece, data)
    check('弱于对手的低音不当主旋律', {s[2] for s in seg_of(ml)} == {'Pad'},
          str(seg_of(ml)))
    check('没有触发低音区保护', not ml.get('low_register_overridden'),
          str(ml.get('low_register_overridden')))
    roles = {r['part_name']: r['role_candidate'] for r in vo['roles']}
    check('角色表仍把它报成「低音」', roles.get('Lead Bass') == '低音',
          str(roles))
    check('覆盖率没掉', ml['coverage']['ratio'] == 1.0, str(ml['coverage']))

    # E3 轮廓 / 实质都只差一点点、力度明显占优 → 放宽门槛后才够得着。
    # 这组参数是挑出来的（见下 check）：领先值落在"不放宽够不着、放宽一半够得着"
    # 的窄区间里 —— 这是唯一能验证"力度真把门槛放宽了"的情形。
    print('\n  E3 轮廓 / 实质领先不足 + 力度明显占优 → 放宽门槛后翻盘')
    BORDER_LOW = [33, 34, 37, 37]                       # 4 个音填满一小节
    BORDER_MEL = [71, 72, 75, 75, 75, 73, 75, 76,       # 16 个音填满一小节
                  68, 77, 74, 79, 77, 68, 75, 76]

    def border_score(bass_vel, other_vel):
        return build_score([
            ('Bass', instrument.ElectricBass(), BORDER_LOW, 1.0, bass_vel),
            ('Lead', instrument.Violin(), BORDER_MEL, 0.25, other_vel),
            ('Pad', instrument.Piano(), [(72, 4.0)], None, other_vel),
        ])

    piece = make_piece(user, 'dyn-relax-none', _dump(border_score(None, None)))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    seq_b = _seg_pitch_seq(data, data.part_names.index('Bass'), 0.0, 16.0)
    seq_l = _seg_pitch_seq(data, data.part_names.index('Lead'), 0.0, 16.0)
    lead_shape = round(ml_mod.contour_shape(seq_b)['score']
                       - ml_mod.contour_shape(seq_l)['score'], 4)
    check('样例落在设计好的区间里（0.05 ≤ 轮廓领先 < 0.10）',
          0.05 <= lead_shape < 0.10, f'实际轮廓领先 {lead_shape}')
    vo, ml = run_pipeline(piece, data)
    check('没标力度时这么点领先够不着门槛（纯形态口径）',
          {s[2] for s in seg_of(ml)} != {'Bass'}
          and not ml.get('low_register_overridden'), str(seg_of(ml)))

    piece = make_piece(user, 'dyn-relax-loud', _dump(border_score(FF, PP)))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    vo, ml = run_pipeline(piece, data)
    check('力度明显占优 → 门槛放宽，同一样例翻盘',
          {s[2] for s in seg_of(ml)} == {'Bass'}, str(seg_of(ml)))
    check('证据里标明了"门槛已放宽"',
          _lead_word(ml).get('dyn_relaxed') is True, str(_lead_word(ml)))

    piece = make_piece(user, 'dyn-relax-weak', _dump(border_score(PP, FF)))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    vo, ml = run_pipeline(piece, data)
    check('反过来写得比对手弱 → 连放宽都蹭不到',
          {s[2] for s in seg_of(ml)} != {'Bass'}
          and not ml.get('low_register_overridden'), str(seg_of(ml)))

    # E4 力度只"放宽 / 否决"，不代替形态：证据不足时再响也不许翻盘
    print('\n  E4 力度再响也不替它补上缺的那条：贝斯伴奏 + 稀疏长笛')
    sc = build_score([
        ('Flute', instrument.Flute(), FLUTE_LINE, 1.0, PP),
        ('Contrabass', instrument.Contrabass(), WALK, 0.5, FF),
    ])
    piece = make_piece(user, 'dyn-no-flip-walk', _dump(sc))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    vo, ml = run_pipeline(piece, data)
    check('主旋律仍是长笛（力度没把伴奏抬上来）',
          {s[2] for s in seg_of(ml)} == {'Flute'}, str(seg_of(ml)))

    # ---------------------------------------------------------- F 力度可开关
    # 力度是一条**证据**，不是真理：它只在轮廓 / 实质定不了时才说话。要不要
    # 让它说话，交给用户决定 —— 设置里默认开，关掉就退回纯形态口径。
    print('\n[F] 力度是可选口径（系统设置「主旋律判定考虑力度」，默认开）')
    cfg = SystemConfig.get_config()
    check('默认是开的', cfg.melody_dynamics is True, str(cfg.melody_dynamics))

    # F1 关掉 → E2 那份"低音偏弱"的谱回到早期口径（低音又翻盘了）
    p_weak = Piece.objects.get(title='dyn-bass-weak')
    data_weak = parse_score(p_weak.musicxml_file.path)
    cfg.melody_dynamics = False
    cfg.save()
    vo, ml = run_pipeline(p_weak, data_weak)
    check('关掉后力度的否决失效 → 回到纯形态口径（低音又当选）',
          {s[2] for s in seg_of(ml)} == {'Lead Bass'}, str(seg_of(ml)))
    check('结果里写明力度这条没启用',
          ml['dynamics']['enabled'] is False
          and ml['dynamics']['notated'] is False
          and ml['dynamics']['parts'] == [], str(ml['dynamics']))
    check('开关不一致时矩阵会被判为过期（自动重算）',
          vo['salience'].get('melody_dynamics') is False,
          str(vo['salience'].get('melody_dynamics')))

    # F2 没标力度的谱：开关两种取值必须逐段一致（加这条证据不能改动老结论）
    p_none = Piece.objects.get(title='dyn-relax-none')
    data_none = parse_score(p_none.musicxml_file.path)
    _, ml_off = run_pipeline(p_none, data_none)          # 此刻开关是关的
    cfg.melody_dynamics = True
    cfg.save()
    _, ml_on = run_pipeline(p_none, data_none)
    check('谱面没标力度 → 开关怎么拨都一样',
          seg_of(ml_off) == seg_of(ml_on),
          f'{seg_of(ml_off)} vs {seg_of(ml_on)}')

    # F3 设置页能存这个开关
    print('\n  F3 设置页前后的往返')
    from django.test import Client
    u = User.objects.get(username='smoke')
    if not u.is_staff:
        u.is_staff = True
        u.save()
    c = Client()
    c.force_login(u)
    base = {
        'polyphony_threshold': '0.30', 'exchange_min_switches': '2',
        'exchange_similarity_threshold': '0.60',
        'melody_voice_priority': 'on', 'melody_segment_min_bars': '2',
        'melody_confidence_high': '0.25', 'melody_confidence_mid': '0.10',
        'melody_export_tempo_mode': 'original', 'melody_export_bpm': '120',
        'melody_export_instrument_mode': 'original', 'melody_export_program': '0',
        'analysis_window_bars': '4', 'chord_window': 'measure',
        'density_by_note_count': 'on',      # 织体密度默认口径：按发音数
        'dsh_profile': 'headless', 'dsh_timeout': '600',
        'max_chord_windows_per_call': '24',
    }
    html = c.get('/settings/').content.decode('utf-8')
    check('设置页里有这个开关',
          'melody_dynamics' in html and '主旋律判定考虑力度' in html)
    r1 = c.post('/settings/', base)          # 不勾 → 复选框不提交
    check('POST 不勾 → 关闭', r1.status_code == 302
          and SystemConfig.get_config().melody_dynamics is False,
          str(SystemConfig.get_config().melody_dynamics))
    r2 = c.post('/settings/', dict(base, melody_dynamics='on'))
    check('POST 勾上 → 开启', r2.status_code == 302
          and SystemConfig.get_config().melody_dynamics is True,
          str(SystemConfig.get_config().melody_dynamics))
    cfg = SystemConfig.get_config()
    cfg.density_by_note_count = True
    cfg.melody_dynamics = True
    cfg.save()

    # ---------------------------------------------------------- 清理
    print('\n[清理] 临时 DB / media 产物')
    import shutil
    media = settings.MEDIA_ROOT
    for pc in pieces:
        for rel in (pc.musicxml_file.name, pc.midi_file.name,
                    pc.source_file.name):
            if not rel:
                continue
            p = os.path.join(media, rel)
            if os.path.exists(p):
                os.remove(p)
        pdir = os.path.join(media, 'melody')
        if os.path.isdir(pdir):
            for day in os.listdir(pdir):
                d = os.path.join(pdir, day, f'p{pc.id}')
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
    import time
    for _ in range(5):
        if not os.path.exists(DB_FILE):
            break
        try:
            os.remove(DB_FILE)
        except OSError:
            time.sleep(0.4)
    print('  ✓ 已删除')

    print('\n' + '=' * 56)
    print(f'通过 {len(PASS)} 项，失败 {len(FAIL)} 项')
    if FAIL:
        for n in FAIL:
            print('  ✗ ' + n)
        return 1
    print('全部通过 ✓')
    return 0


def _looks_like(data, name, s, e):
    """某个声部在 [s, e) 内够不够得上"一条有起伏的旋律线"。"""
    from core.analysis.melody_line import looks_like_line
    idx = data.part_names.index(name)
    return looks_like_line(_seg_pitch_seq(data, idx, s, e))


if __name__ == '__main__':
    sys.exit(main())
