"""主音判定的回归脚本：一串双音里哪个音是主旋律在唱的那个。

起因（用户提问）
----------------
「我反对用织体厚度判断。因为主旋律也有可能本身就是双音。有些乐曲就是用双音
演奏的。但是依然存在主旋律。」
「这就没办法了吗？我如何在一串双音中提出主旋律？可不可以根据和弦特征本身和
轮廓判断出来？」

于是有了两个东西：
  ① 系统设置「主旋律线保留双音」（默认开）—— 保留双音时原样导出和弦，
     压成单音时才做主音判定；
  ② core/analysis/principal.py —— 把每个事件按音高排位拆成几条候选层，
     用「候选层轮廓 / 和弦成员度 / 相对低音位置漂移 / 顶层先验」四条证据
     给每条线打分，选总分最高的那条；第一条第二名咬得太紧时交给 DeepSeek
     判定，并把它的判断与理由带回页面。

本脚本把这些钉死：

  A. 平行三度双音旋律 + 柱式和弦伴奏 → 主旋律归给双音那条；单音模式选出的是
     **上声部**（人跟着唱的那条），保留双音模式导出的是和弦；
  B. voicing 中途重排（三度变六度）→ 不该把整条线带走样，判定仍能落在流动
     的那一层上；
  C. 纯单音旋律（一个多音事件都没有）→ 直接放行，不去做层判定；
  D. 多音事件太少 → **照样判**：换少样本口径（轮廓降权、连接平滑度升权、
     顶层先验压到最低），不许退回"取最高音"。整段只有一两处双音时，挑出来
     的可以是下面那条；
  E. DeepSeek 兜底：正常采纳 / 返回不合法的音不采纳 / 调用失败退回本地，
     三种都不许把整段导出拖垮；
  F. 设置页往返：勾上与不勾分别写进配置；
  G. 导出口径：同一份结果，保留双音导出和弦、压成单音导出单音，文件名各带
     doubles / single；
  H. 双音口径**不改窗归属** —— 两态下的分段必须逐段一致。

用法：python smoke_principal_test.py
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
from core.analysis import principal as pmod  # noqa: E402
from core.analysis import melody_export as mex  # noqa: E402
from core.analysis.melody_line import build_export_parts  # noqa: E402
from core.analysis.scoredata import parse_score, build_measure_map  # noqa: E402
from core.models import AnalysisResult, Piece, SystemConfig  # noqa: E402

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '_smoke_principal.sqlite3')
PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


# ------------------------------------------------------------ 造谱
# 平行三度：上声部走一条有起伏的线，下声部低三度跟着
THIRDS_TOP = [72, 74, 76, 79, 76, 74, 72, 69, 71, 74, 76, 72]
# 中途 voicing 重排：后一半改成低六度（同和弦的转位），最高音不再是同一条 melody对象的那条
THIRDS_RE = [72, 74, 76, 79, 76, 74, 72, 65, 67, 70, 72, 68]
PAD_CHORDS = [[50, 57, 62], [48, 55, 60], [53, 60, 65], [45, 52, 57]]


def build_score(spec, bars=8):
    """spec: [(声部名, 乐器, chords, ql)]

    ``chords`` 是**和弦列表**，每项是 [音高…]（单元素就是单音）；每个和弦占
    ql 拍，按小节循环铺满 bars 小节。
    """
    from music21 import chord, instrument, note, stream

    sc = stream.Score()
    for name, inst, chords, ql in spec:
        p = stream.Part()
        p.partName = name
        p.insert(0, inst)
        per_bar = max(int(4.0 / ql), 1)
        for m in range(bars):
            off = m * 4.0
            for i in range(per_bar):
                picks = chords[(m * per_bar + i) % len(chords)]
                if len(picks) >= 2:
                    n = chord.Chord(sorted(picks), quarterLength=ql)
                else:
                    n = note.Note(picks[0], quarterLength=ql)
                p.insert(off + i * ql, n)
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
    u = User.objects.create_user('smoke', password='smoke123456')
    # /settings/ 是 staff_only（views.py 里的 user_passes_test），不是管理员看不了
    u.is_staff = True
    u.save()
    cfg = SystemConfig.get_config()
    cfg.melody_voice_priority = True
    cfg.melody_segment_min_bars = 2
    cfg.analysis_window_bars = 4
    cfg.melody_shape_gate = False
    cfg.density_by_note_count = True
    cfg.melody_dynamics = True
    cfg.melody_keep_doubles = True          # 新开关，默认保留双音
    cfg.melody_llm_judge = False            # 默认关：回归要可复现，不能依赖网络
    cfg.save()
    print('[准备] SQLite 迁移 + 测试用户 + 配置复位 …… OK')


def make_piece(user, title, data: bytes):
    piece = Piece(owner=user, title=title, source_type='musicxml')
    piece.source_file.save('p.musicxml',
                           SimpleUploadedFile('p.musicxml', data), save=True)
    pipeline._run_convert(piece.id)
    piece.refresh_from_db()
    r = AnalysisResult.objects.get(piece=piece, step='convert')
    assert r.status == 'done', f'convert 失败：{r.error[:400]}'
    return piece


def seg_of(out):
    return [(s['start_measure'], s['end_measure'], s['part_name'])
            for s in out['segments']]


def run_pipeline(piece, keep_doubles):
    cfg = SystemConfig.get_config()
    cfg.melody_keep_doubles = keep_doubles
    cfg.save()
    pipeline._run_steps(piece.id, ['instrumentation', 'voice_id', 'melody_line'])
    ml = AnalysisResult.objects.get(piece=piece, step='melody_line').output
    return ml


def notes_of(part):
    """导出 Part 里的元素按 offset 排好 → [(offset, (音高…))]"""
    from music21 import chord as m21chord, note as m21note
    out = []
    for el in part.recurse().notes:
        pitches = ([p.midi for p in el.pitches] if isinstance(el, m21chord.Chord)
                   else [el.pitch.midi])
        out.append((round(float(el.offset), 3), tuple(sorted(pitches))))
    out.sort(key=lambda t: t[0])
    return out


BASE_POST = {
    'polyphony_threshold': '0.30', 'exchange_min_switches': '2',
    'exchange_similarity_threshold': '0.60',
    'melody_voice_priority': 'on', 'melody_segment_min_bars': '2',
    'melody_confidence_high': '0.25', 'melody_confidence_mid': '0.10',
    'melody_export_tempo_mode': 'original', 'melody_export_bpm': '120',
    'melody_export_instrument_mode': 'original', 'melody_export_program': '0',
    'analysis_window_bars': '4', 'chord_window': 'measure',
    'density_by_note_count': 'on',
    'dsh_profile': 'headless', 'dsh_timeout': '600',
    'max_chord_windows_per_call': '24',
}


def post_settings(client, **extra):
    return client.post('/settings/', dict(BASE_POST, **extra))


def main():
    from music21 import instrument
    prepare()
    user = User.objects.get(username='smoke')
    cfg = SystemConfig.get_config()

    # ------------------------------------------------ A 平行三度双音旋律
    print('\n[A] 平行三度双音旋律 + 柱式和弦伴奏')
    sc = build_score([
        ('Lead', instrument.Violin(),
         [[t, t - 3] for t in THIRDS_TOP], 1.0),
        ('Pad', instrument.Piano(), PAD_CHORDS, 1.0),
    ])
    piece = make_piece(user, 'thirds', _dump(sc))
    data = parse_score(piece.musicxml_file.path)
    lead = next(p.index for p in data.parts if p.name == 'Lead')
    measure_offsets, _, bar_ql = build_measure_map(data)

    pr = pmod.resolve_principal(data, lead, 0.0, 32.0, measure_offsets, bar_ql,
                                cfg)
    check('段里认出 32 个多音事件（全都是双音）',
          pr['chord_event_count'] == 32, str(pr['chord_event_count']))
    check('主音判定层真的启用了（不是因为样本太少被跳过）',
          pr['resolved_by'] in ('local', 'deepseek'), pr['resolved_by'])
    check('选中的是第 0 层（最高的那条外轮廓）',
          pr['chosen_rank'] == 0, str(pr['chosen_rank']))
    tops = [row['principal'] for row in pr['per_event']]
    # 32 个事件是把 12 音的那条循环铺满 8 小节 → 期望值就是它循环取整番
    want_tops = (THIRDS_TOP * 3)[:len(tops)]
    check('主音逐个落成上声部（人跟着唱的那条）',
          tops == want_tops,
          f'{tops[:12]} … 共 {len(tops)} 个')
    check('四条证据都算出来了（轮廓 / 外音率 / 位置漂移 / 顶层先验）',
          all(set(l.keys()) >= {'shape', 'nonmember', 'bass_spread', 'top_prior'}
              for l in pr['layers']), str(pr['layers'][:1]))
    check('顶层先验把第 0 层排在前面',
          pr['layers'][0]['top_prior'] >= 0.99,
          str([round(l['top_prior'], 3) for l in pr['layers']]))

    # ------------------------------------------------ B voicing 中途重排
    print('\n[B] 中途重排（三度变六度）后仍能判到流动的那层')
    sc = build_score([
        ('Lead', instrument.Violin(),
         [[t, t - 3] if i < 6 else [t, t - 9]
          for i, t in enumerate(THIRDS_RE)], 1.0),
        ('Pad', instrument.Piano(), PAD_CHORDS, 1.0),
    ])
    piece2 = make_piece(user, 'revoice', _dump(sc))
    data2 = parse_score(piece2.musicxml_file.path)
    lead2 = next(p.index for p in data2.parts if p.name == 'Lead')
    mo2, _, bq2 = build_measure_map(data2)
    pr2 = pmod.resolve_principal(data2, lead2, 0.0, 32.0, mo2, bq2, cfg)
    check('重排后选中层仍是第 0 层（轮廓在上声部）',
          pr2['chosen_rank'] == 0, str(pr2['chosen_rank']))
    check('每个事件的主音都落在它自己的候选项里',
          all(row['principal'] in row['pitches']
              for row in pr2['per_event']),
          str(pr2['per_event'][:2]))

    # ------------------------------------------------ C 纯单音
    print('\n[C] 纯单音旋律 → 不做层判定，原样放行')
    sc = build_score([
        ('Solo', instrument.Flute(), [[p] for p in [72, 74, 76, 74]], 1.0),
    ])
    piece3 = make_piece(user, 'solo', _dump(sc))
    data3 = parse_score(piece3.musicxml_file.path)
    mo3, _, bq3 = build_measure_map(data3)
    pr3 = pmod.resolve_principal(data3, 0, 0.0, 32.0, mo3, bq3, cfg)
    check('认出这是纯单音（resolved_by=single-notes）',
          pr3['resolved_by'] == 'single-notes', pr3['resolved_by'])
    check('主音就是它自己那一个音',
          [r['principal'] for r in pr3['per_event']][:8]
          == [72, 74, 76, 74] * 2,
          str([r['principal'] for r in pr3['per_event']][:8]))

    # ------------------------------------------------ D 多音事件太少也要判
    print('\n[D] 多音事件不足 → 少样本口径照样判，不许退回最高音')
    mix = [[72], [74], [76], [72, 76], [74], [72], [76], [74]]
    # 只铺 2 小节 → 16 个事件里只有 2 个是双音，够不上常规统计口径
    sc = build_score([('Few', instrument.Violin(), mix, 1.0)], bars=2)
    piece4 = make_piece(user, 'few', _dump(sc))
    data4 = parse_score(piece4.musicxml_file.path)
    mo4, _, bq4 = build_measure_map(data4)
    pr4 = pmod.resolve_principal(data4, 0, 0.0, 32.0, mo4, bq4, cfg)
    check('D1 少样本不再短路（没有 too-few-chords 这种退路了）',
          pr4['resolved_by'] in ('local', 'deepseek'), pr4['resolved_by'])
    check('D1b 段被标成少样本口径',
          pr4['small_sample'] is True, str(pr4['small_sample']))
    check('D1c 候选层照拆照打分，且带第五条证据（连接平滑度）',
          len(pr4['layers']) >= 2
          and all('link' in l for l in pr4['layers']),
          str(pr4['layers'][:1]))
    check('D1d 每个事件的主音都在它自己的候选项里',
          all(r['principal'] in r['pitches'] for r in pr4['per_event']),
          str(pr4['per_event'][:4]))
    check('D1e 少样本权重里顶层先验被压到最低（不再默认"高的那个在唱"）',
          pmod.PRINCIPAL_WEIGHTS_SMALL[3] < pmod.PRINCIPAL_WEIGHTS[3]
          and pmod.PRINCIPAL_WEIGHTS_SMALL[4] > pmod.PRINCIPAL_WEIGHTS[4],
          f"{pmod.PRINCIPAL_WEIGHTS_SMALL} vs {pmod.PRINCIPAL_WEIGHTS}")

    # D2 关键反例：整段只有两处双音，叠上去的高音是硬跳、下面那条才是顺着
    # 级进接上来的 —— 少样本口径必须挑下面那条，不许取最高音。
    print('\n[D2] 少样本下叠上来的高音是硬跳 → 挑顺着接上来的低音那条')
    mix2 = [[72], [74], [76], [84, 77], [79], [81], [79], [76]]
    sc = build_score([('FewLow', instrument.Violin(), mix2, 1.0)], bars=2)
    piece5 = make_piece(user, 'few_low', _dump(sc))
    data5 = parse_score(piece5.musicxml_file.path)
    mo5, _, bq5 = build_measure_map(data5)
    pr5 = pmod.resolve_principal(data5, 0, 0.0, 32.0, mo5, bq5, cfg)
    check('D2a 选中的是第 1 层（低的那条），不是最高音那条',
          pr5['chosen_rank'] == 1, str(pr5['chosen_rank']))
    row5 = [r for r in pr5['per_event'] if len(r['pitches']) >= 2]
    check('D2b 双音处的主音是顺着级进接上来的 77，不是硬跳上去的 84',
          all(r['principal'] == 77 for r in row5),
          str([(r['pitches'], r['principal']) for r in row5]))
    check('D2c 依据确实是连接平滑度（低那条衔接分更高）',
          pr5['layers'][0]['link'] > pr5['layers'][1]['link'],
          str([(l['rank'], l['link']) for l in pr5['layers']]))
    check('D2d 少样本口径标记在结果里',
          pr5['small_sample'] is True, str(pr5['small_sample']))

    # D3 极端：整段只有一处双音 → 也要有判定过程，不许"懒得判就取最高音"
    print('\n[D3] 只有一处双音 → 照样走判定')
    mix3 = [[72], [74], [76], [84, 77]]
    sc = build_score([('One', instrument.Violin(), mix3, 1.0)], bars=1)
    piece6 = make_piece(user, 'one_chord', _dump(sc))
    data6 = parse_score(piece6.musicxml_file.path)
    mo6, _, bq6 = build_measure_map(data6)
    pr6 = pmod.resolve_principal(data6, 0, 0.0, 32.0, mo6, bq6, cfg)
    check('D3a 只有一个多音事件也做了层判定',
          pr6['chord_event_count'] == 1 and pr6['resolved_by'] == 'local',
          f"{pr6['chord_event_count']}/{pr6['resolved_by']}")
    check('D3b 主音按证据挑（这里是顺着接上来的 77）',
          [r['principal'] for r in pr6['per_event'] if len(r['pitches']) >= 2]
          == [77],
          str(pr6['per_event'][-1]))

    # ------------------------------------------------ E DeepSeek 兜底
    print('\n[E] DeepSeek 兜底三种情形（选音层）')
    saved = pmod._judge_by_llm
    cfg.melody_llm_judge = True             # 只有开关开着才会真的去问
    cfg.save()
    mo, _, bq = build_measure_map(data)
    # E1 调用失败 → 退回本地，不许抛
    pmod._judge_by_llm = lambda task, c: {'_error': 'dsh 不可用'}
    pmod.PRINCIPAL_LLM_MARGIN = 1.0          # 强制判为"咬得太紧"
    pr_e1 = pmod.resolve_principal(data, lead, 0.0, 32.0, mo, bq, cfg)
    check('E1 LLM 失败也不抛异常，仍是本地结果',
          pr_e1['resolved_by'] == 'local', pr_e1['resolved_by'])
    check('E1 失败原因被记下来供页面展示',
          (pr_e1['llm'] or {}).get('raw_error') == 'dsh 不可用',
          str((pr_e1['llm'] or {}).get('raw_error')))
    # E2 返回一个不在候选里的音 → 不采纳
    pmod._judge_by_llm = lambda task, c: {
        'decisions': [{'offset': 0.0, 'principal': 'C9', 'reason': '乱说'}],
        'summary': '按高处外轮廓'}
    pr_e2 = pmod.resolve_principal(data, lead, 0.0, 32.0, mo, bq, cfg)
    check('E2 模型给了一个不在候选里的音 → 不采纳',
          (pr_e2['llm'] or {}).get('rejected') == 1,
          str((pr_e2['llm'] or {}).get('rejected')))
    # E3 合法判定 → 采纳，source 标成 deepseek
    first = pr['per_event'][0]
    want, why = first['pitches'][1], '低音那层在这里有经过音'
    from core.analysis.scoredata import midi_to_name
    pmod._judge_by_llm = lambda task, c: {
        'decisions': [{'offset': first['offset'],
                       'principal': midi_to_name(want), 'reason': why}],
        'summary': '这里下面那层更连贯'}
    pr_e3 = pmod.resolve_principal(data, lead, 0.0, 32.0, mo, bq, cfg)
    check('E3 合法判定被采纳（source=deepseek）',
          pr_e3['resolved_by'] == 'deepseek'
          and pr_e3['per_event'][0]['source'] == 'deepseek',
          f"{pr_e3['resolved_by']}/{pr_e3['per_event'][0]['source']}")
    check('E3 主音换成了模型选的那个音',
          pr_e3['per_event'][0]['principal'] == want,
          str(pr_e3['per_event'][0]))
    check('E3 理由与 summary 都带回来了（页面右侧面板要用）',
          pr_e3['per_event'][0]['reason'] == why
          and (pr_e3['llm'] or {}).get('summary') == '这里下面那层更连贯',
          str(pr_e3['per_event'][0]['reason']))
    # E3b 少样本段统计不可靠 → 界线放宽，也会被问到
    pmod._judge_by_llm = lambda task, c: {'decisions': [],
                                          'summary': '少样本，按前后衔接判'}
    pmod.PRINCIPAL_LLM_MARGIN_SMALL = 1.0
    pr_e5 = pmod.resolve_principal(data5, 0, 0.0, 32.0, mo5, bq5, cfg)
    check('E3b 少样本段也会被问（统计不可靠 → 更愿意交给模型）',
          (pr_e5['llm'] or {}).get('called') is True,
          str((pr_e5['llm'] or {}).get('called')))
    check('E3b2 问的是少样本那段（small_sample 标记带过去了）',
          pr_e5['small_sample'] is True, str(pr_e5['small_sample']))
    pmod.PRINCIPAL_LLM_MARGIN_SMALL = 0.12
    pmod.PRINCIPAL_LLM_MARGIN = 0.05
    pmod._judge_by_llm = saved
    # E4 开关关着 → 一次都不许问（离线可用性）
    cfg.melody_llm_judge = False
    cfg.save()
    pmod._judge_by_llm = lambda task, c: (_ for _ in ()).throw(
        AssertionError('开关关着竟然还去调 LLM 了'))
    pmod.PRINCIPAL_LLM_MARGIN = 1.0
    pr_e4 = pmod.resolve_principal(data, lead, 0.0, 32.0, mo, bq, cfg)
    check('E4 开关关着 → 不发问，仍是本地结果',
          pr_e4['resolved_by'] == 'local'
          and (pr_e4['llm'] or {}).get('called') is False,
          f"{pr_e4['resolved_by']}/{pr_e4['llm']}")
    pmod.PRINCIPAL_LLM_MARGIN = 0.05
    pmod._judge_by_llm = saved

    # ------------------------------------------------ F 设置页往返
    print('\n[F] 系统设置「主旋律线保留双音」往返')
    from django.test import Client
    client = Client()
    client.force_login(user)
    post_settings(client, melody_dynamics='on', melody_keep_doubles='on',
                  melody_llm_judge='on')
    cfg.refresh_from_db()
    check('F1 勾上 → 存 True（默认口径）',
          cfg.melody_keep_doubles is True, str(cfg.melody_keep_doubles))
    check('F3 DeepSeek 兜底勾上 → 存 True',
          cfg.melody_llm_judge is True, str(cfg.melody_llm_judge))
    post_settings(client, melody_dynamics='on')
    cfg.refresh_from_db()
    check('F2 不勾 → 存 False（POST 里没有这个键）',
          cfg.melody_keep_doubles is False, str(cfg.melody_keep_doubles))
    check('F4 DeepSeek 兜底不勾 → 存 False',
          cfg.melody_llm_judge is False, str(cfg.melody_llm_judge))
    cfg.melody_keep_doubles = True
    cfg.melody_llm_judge = False            # 后面 G/H 的粗管道不许联网
    cfg.save()

    # ------------------------------------------------ G 导出口径
    print('\n[G] 保留双音导出和弦 / 压成单音导出单音')
    ml_keep = run_pipeline(piece, True)
    ml_one = run_pipeline(piece, False)
    check('G1 两种口径记在结果里',
          ml_keep['double']['keep'] is True
          and ml_one['double']['keep'] is False,
          f"{ml_keep['double']['keep']}/{ml_one['double']['keep']}")
    parts_keep = build_export_parts(data, ml_keep['segments'],
                                    mex.resolve_options(cfg), double_tag='doubles')
    built_one = build_export_parts(data, ml_one['segments'],
                                   mex.resolve_options(cfg), double_tag='single')
    nk = notes_of(parts_keep[0]['part'])
    no = notes_of(built_one[0]['part'])
    check('G2 保留双音 → 产物里是和弦',
          all(len(p) == 2 for _, p in nk), str(nk[:3]))
    check('G3 压成单音 → 产物里全是单音',
          all(len(p) == 1 for _, p in no), str(no[:3]))
    check('G4 压成单音的那条就是上声部轮廓',
          [p[0] for _, p in no][:12] == THIRDS_TOP,
          str([p[0] for _, p in no][:12]))
    check('G5 文件名分别带 doubles / single',
          'doubles' in parts_keep[0]['stem'] and 'single' in built_one[0]['stem'],
          f"{parts_keep[0]['stem']} | {built_one[0]['stem']}")
    check('G6 两态的产物事件数一致（只是每个事件的音数不同）',
          len(nk) == len(no), f'{len(nk)} vs {len(no)}')

    # ------------------------------------------------ H 归属不变
    print('\n[H] 双音口径不改窗归属')
    check('H1 两态分段逐段一致',
          seg_of(ml_keep) == seg_of(ml_one),
          f'{seg_of(ml_keep)} vs {seg_of(ml_one)}')
    check('H2 主旋律仍归给双音那条（不是柱式伴奏）',
          {s[2] for s in seg_of(ml_one)} == {'Lead'}, str(seg_of(ml_one)))

    # ------------------------------------------------ I DeepSeek 兜底裁定（选轨）
    print('\n[I] DeepSeek 兜底裁定：本地咬得太紧的窗交给模型')
    from core.analysis import llm as llm_mod
    from core.analysis import melody_judge as jmod
    from core.analysis.melody_line import _window_choices
    from core.analysis.voice_id import window_scores

    ws = window_scores(data, 0.0, 32.0, by_note_count=True)
    row = {'scores': {str(k): v for k, v in ws['scores'].items()}}
    choices = _window_choices(row, data)
    win = {'window': 1, 'start_measure': 1, 'end_measure': 8}
    pad = next(p.index for p in data.parts if p.name == 'Pad')
    check('I1 候选够两个、领先极小 → 判为难判',
          jmod.is_hard(choices, 0.01) is True, str(len(choices)))
    check('I1b 领先明显 → 不难判（本地说了算）',
          jmod.is_hard(choices, 0.5) is False)
    check('I1c 只有一个候选 → 谈不上难判',
          jmod.is_hard(choices[:1], 0.0) is False)

    saved_rllm = llm_mod.run_llm_json_soft
    cfg.melody_llm_judge = False
    cfg.save()
    r_off = jmod.judge_window(data, cfg, win, choices, ws['per_part'], 0.0,
                              32.0, lead, 0.01, [jmod.JUDGE_MAX_CALLS])
    check('I2 开关关着 → 一次都不问',
          r_off['called'] is False and r_off['applied'] is False,
          str(r_off['reason']))

    cfg.melody_llm_judge = True
    cfg.save()
    llm_mod.run_llm_json_soft = lambda *a, **k: {
        'pick': pad, 'reason': '上面那条是柱式和弦伴奏', 'confidence': 0.8}
    r_ok = jmod.judge_window(data, cfg, win, choices, ws['per_part'], 0.0,
                             32.0, lead, 0.01, [jmod.JUDGE_MAX_CALLS])
    check('I3 模型选了一个合法声部 → 采纳并改判',
          r_ok['called'] and r_ok['applied'] and r_ok['pick'] == pad,
          str(r_ok))
    check('I3b 改判把理由一起带回来了（页面右侧面板要用）',
          r_ok['reason'] == '上面那条是柱式和弦伴奏' and r_ok['confidence'] == 0.8,
          str(r_ok['reason']))

    llm_mod.run_llm_json_soft = lambda *a, **k: {'pick': 99, 'reason': '乱指'}
    r_bad = jmod.judge_window(data, cfg, win, choices, ws['per_part'], 0.0,
                              32.0, lead, 0.01, [jmod.JUDGE_MAX_CALLS])
    check('I4 模型选了不存在的声部 → 不采纳',
          r_bad['called'] and not r_bad['applied'] and '不在候选' in r_bad['reason'],
          str(r_bad['reason']))

    llm_mod.run_llm_json_soft = lambda *a, **k: {'_error': 'dsh 不可用'}
    r_err = jmod.judge_window(data, cfg, win, choices, ws['per_part'], 0.0,
                              32.0, lead, 0.01, [jmod.JUDGE_MAX_CALLS])
    check('I5 调用失败 → 静默退回本地，不抛异常',
          r_err['called'] and not r_err['applied'], str(r_err['reason']))

    def _boom(*a, **k):
        raise RuntimeError('网络炸了')

    llm_mod.run_llm_json_soft = _boom
    r_exc = jmod.judge_window(data, cfg, win, choices, ws['per_part'], 0.0,
                              32.0, lead, 0.01, [jmod.JUDGE_MAX_CALLS])
    check('I6 调用抛异常 → 兜住，退回本地',
          not r_exc['applied'] and 'RuntimeError' in r_exc['reason'],
          str(r_exc['reason']))

    r_budget = jmod.judge_window(data, cfg, win, choices, ws['per_part'], 0.0,
                                 32.0, lead, 0.01, [0])
    check('I7 预算用完 → 不再发问（大曲子不会拖死）',
          r_budget['called'] is False and '预算' in r_budget['reason'],
          str(r_budget['reason']))

    # I8 端到端：真的跑一次管道，让模型把窗改判给另一个声部
    llm_mod.run_llm_json_soft = lambda *a, **k: {
        'pick': pad, 'reason': '伴奏更连贯', 'confidence': 0.7}
    saved_margin, saved_calls = jmod.JUDGE_MARGIN, jmod.JUDGE_MAX_CALLS
    jmod.JUDGE_MARGIN, jmod.JUDGE_MAX_CALLS = 1.0, 2   # 强制每窗都判为难判
    cfg.melody_llm_judge = True
    cfg.melody_keep_doubles = True
    cfg.save()
    ml_judge = run_pipeline(piece, True)
    j_out = ml_judge['llm_judge']
    check('I8 记录在结果里：问过几个窗 / 采纳几个',
          j_out['called'] >= 1 and j_out['applied'] >= 1,
          f"called={j_out['called']} applied={j_out['applied']}")
    rows = ml_judge.get('windows') or []
    check('I8b 改判真的落到窗归属上了',
          any(r['part_index'] == pad for r in rows),
          str([(r['start_measure'], r['part_name']) for r in rows]))
    check('I8c 每个裁定的窗都带着本地原判与模型的理由（页面要用）',
          all((w['reason'] or '') and w['local_part_name'] and w['part_name']
              for w in j_out['windows'] if w['called']),
          str(j_out['windows'][:1]))
    jmod.JUDGE_MARGIN, jmod.JUDGE_MAX_CALLS = saved_margin, saved_calls
    llm_mod.run_llm_json_soft = saved_rllm
    cfg.melody_llm_judge = False
    cfg.save()

    print('\n' + '=' * 62)
    print(f'通过 {len(PASS)} 项，失败 {len(FAIL)} 项')
    if FAIL:
        for f in FAIL:
            print('  ✗ ' + f)
        sys.exit(1)
    print('全部通过。')


if __name__ == '__main__':
    main()
