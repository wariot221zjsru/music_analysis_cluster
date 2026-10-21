"""有音高打击乐（颤音琴 / 钟琴 / 马林巴 …）当主旋律时的回归脚本。

起因（线上真实事故：naik.xml）
------------------------------
一首 66 小节的曲子，主题旋律从头到尾都在，但**前半段的主旋律怎么都提不出来**：

    颤音琴  m1–32  ┐ 同一条主题
    钟琴    m1–32  ┘（逐音相同的齐奏加倍）
    合唱    m33–64 ┐ 同一条主题，换成人声
    人声    m33–64 ┘
    钢琴    m2–66  柱式和弦伴奏
    贝斯    m2–66  低音

后半段（合唱 / 人声）一直提得出来，前半段提不出来。排查下来是**解析入口**
就把颤音琴和钟琴判成了"打击乐"：

  music21 把 Vibraphone / Glockenspiel / Marimba / Xylophone / Timpani / Kalimba
  都归在 ``instrument.Percussion`` 家族下（``PitchedPercussion`` 子类），和军鼓、
  底鼓（``UnpitchedPercussion``）同族。旧代码只问"是不是 Percussion"，
  于是这两个声部的音高在 ``PartInfo.pitched_events`` 里被**整体滤掉** ——
  声部识别、主旋律线根本看不见它们，覆盖率照旧报 100%，
  但 m1–32 报的是"钢琴伴奏"，整段主题一个音都没进结果。

本脚本把三件事钉死：
  A. 有音高打击乐**不当**无音高打击乐，音高必须留在 pitched_events 里；
  B. 柱式和弦伴奏不能因为"只数事件个数"反而显得比单声部旋律更"稀疏"；
  C. 齐奏加倍（逐音相同的两条声部）算**同一条**线索，不互相竞争，
     不把置信度压成 0。

用法：python smoke_mallet_melody_test.py
跑完自动清理临时 DB / media 产物。
"""
import os
import shutil
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402
from django.contrib.auth.models import User  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.test import Client  # noqa: E402

from core.analysis import pipeline  # noqa: E402
from core.analysis import scoredata as sd  # noqa: E402
from core.analysis import voice_id as vmod  # noqa: E402
from core.analysis.scoredata import parse_score  # noqa: E402
from core.models import AnalysisResult, Piece, SystemConfig  # noqa: E402

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '_smoke_mallet.sqlite3')
PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


# ------------------------------------------------------------ 造谱
# 复刻 naik 的声部配置：前半段主题在颤音琴 + 钟琴（逐音相同的齐奏），
# 后半段同样的主题交给合唱；钢琴弹柱式和弦伴奏，贝斯走低音，另加一条鼓轨
# 当作"真的无音高打击乐"的对照组。

MEL = [72, 74, 76, 74]      # 级进为主（真旋律）
MEL_HI = [79, 77, 76, 74]   # 后半段人声主题


def _bar(m, bar_ql=4.0):
    return (m - 1) * bar_ql


def build_score(total_bars=16, bar_ql=4.0):
    from music21 import chord, instrument, note, stream

    sc = stream.Score()

    def mk(name, inst):
        p = stream.Part()
        p.partName = name
        p.insert(0, inst)
        return p

    vib = mk('Vibraphone', instrument.Vibraphone())
    glo = mk('Glockenspiel', instrument.Glockenspiel())
    pno = mk('Piano', instrument.Piano())
    cho = mk('Choir', instrument.Vocalist())
    bas = mk('Acoustic Bass', instrument.AcousticBass())
    drm = mk('Drums', instrument.Percussion())

    for m in range(1, total_bars + 1):
        off = _bar(m, bar_ql)
        # 主题：四个四分音符（前半段颤音琴 + 钟琴同奏，后半段交给合唱）
        for i, pc in enumerate(MEL):
            at = off + i * 1.0
            vib.insert(at, note.Note(pc, quarterLength=1.0))
            glo.insert(at, note.Note(pc, quarterLength=1.0))
        if m > total_bars / 2:
            for i, pc in enumerate(MEL_HI):
                cho.insert(off + i * 1.0, note.Note(pc, quarterLength=1.0))
        # 钢琴：柱式和弦伴奏（每小节两下三和弦）
        for i in range(2):
            pno.insert(off + i * 2.0,
                       chord.Chord([60, 64, 67], quarterLength=1.0))
        # 贝斯：两个二分音符
        for i in range(2):
            bas.insert(off + i * 2.0, note.Note(36, quarterLength=2.0))
        # 鼓：无音高打击乐（对照组，音高应被滤掉）
        for i in range(4):
            drm.insert(off + i * 1.0, note.Note(60, quarterLength=1.0))

    for p in (vib, glo, pno, cho, bas, drm):
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
    cfg.density_by_note_count = True      # 织体密度默认口径：按发音数
    cfg.melody_dynamics = True            # 力度参与"听感"判定（默认口径）
    cfg.save()
    print('[准备] SQLite 迁移 + 测试用户 + 配置复位 …… OK')


def make_piece(user, title, data: bytes):
    piece = Piece(owner=user, title=title, source_type='musicxml')
    piece.source_file.save('mallet.musicxml',
                           SimpleUploadedFile('mallet.musicxml', data), save=True)
    pipeline._run_convert(piece.id)
    piece.refresh_from_db()
    r = AnalysisResult.objects.get(piece=piece, step='convert')
    assert r.status == 'done', f'convert 失败：{r.error[:400]}'
    return piece


def seg_of(out):
    return [(s['start_measure'], s['end_measure'], s['part_name'])
            for s in out['segments']]


def _post_settings(user, extra=None):
    """以管理员身份提交一次设置页，返回保存后的 density_by_note_count。"""
    u = User.objects.get(pk=user.pk)
    if not u.is_staff:
        u.is_staff = True
        u.save()
    base = {
        'polyphony_threshold': '0.30', 'exchange_min_switches': '2',
        'exchange_similarity_threshold': '0.60',
        'melody_voice_priority': 'on', 'melody_segment_min_bars': '2',
        'melody_confidence_high': '0.25', 'melody_confidence_mid': '0.10',
        'melody_export_tempo_mode': 'original', 'melody_export_bpm': '120',
        'melody_export_instrument_mode': 'original', 'melody_export_program': '0',
        'analysis_window_bars': '4', 'chord_window': 'measure',
        'melody_dynamics': 'on',            # 力度参与"听感"判定（默认口径）
        'dsh_profile': 'headless', 'dsh_timeout': '600',
        'max_chord_windows_per_call': '24',
    }
    base.update(extra or {})
    c = Client()
    c.force_login(u)
    c.post('/settings/', base)
    return SystemConfig.get_config().density_by_note_count


def main():
    prepare()
    user = User.objects.get(username='smoke')
    pieces = []

    print('\n[造谱] 颤音琴 + 钟琴（齐奏主题）/ 钢琴柱式和弦 / 合唱 / 贝斯 / 鼓')
    piece = make_piece(user, 'mallet', _dump(build_score()))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    names = data.part_names
    print('  声部：', [(p.index, p.name, p.instrument, p.is_percussion,
                        p.is_pitched_percussion, len(p.pitched_events))
                       for p in data.parts])

    # ---------------------------------------------------------- A 有音高打击乐
    print('\n[A] 有音高打击乐不当无音高打击乐（naik 前半段主题消失的根因）')
    for nm in ('Vibraphone', 'Glockenspiel'):
        i = names.index(nm)
        p = data.parts[i]
        check(f'{nm} 不被判成无音高打击乐', not p.is_percussion)
        check(f'{nm} 被标记为有音高打击乐', p.is_pitched_percussion)
        check(f'{nm} 的音高留在 pitched_events 里',
              len(p.pitched_events) > 0, str(len(p.pitched_events)))
        check(f'{nm} 平均音高算得出来', p.mean_pitch > 0, f'{p.mean_pitch:.2f}')
    i = names.index('Drums')
    check('真·无音高打击乐（鼓组）仍是打击乐', data.parts[i].is_percussion)
    check('鼓组的音高不进音高类分析',
          len(data.parts[i].pitched_events) == 0,
          str(len(data.parts[i].pitched_events)))

    pipeline._run_steps(piece.id, ['instrumentation'])
    inst_out = AnalysisResult.objects.get(piece=piece, step='instrumentation').output
    fam = {p['name']: p['family'] for p in inst_out['parts']}
    check('配器表把颤音琴标成「打击乐（有音高）」',
          fam.get('Vibraphone') == '打击乐（有音高）', str(fam.get('Vibraphone')))
    check('配器表把鼓组标成「打击乐」', fam.get('Drums') == '打击乐',
          str(fam.get('Drums')))

    # ---------------------------------------------------------- B 密度按发音数
    print('\n[B] 柱式和弦伴奏不该显得比单声部旋律更"稀疏"')
    vib_i, pno_i = names.index('Vibraphone'), names.index('Piano')
    wsc = vmod.window_scores(data, 0.0, 16.0)
    d = {x['part_index']: x for x in wsc['per_part'] if not x.get('empty')}
    check('柱式和弦的发音数比单声部旋律多',
          d[pno_i]['note_count'] > d[vib_i]['note_count'],
          f"钢琴 {d[pno_i]['note_count']} vs 颤音琴 {d[vib_i]['note_count']}")
    check('所以柱式和弦的稀疏度更低（更密）',
          d[pno_i]['sparse'] < d[vib_i]['sparse'],
          f"钢琴 {d[pno_i]['sparse']:.3f} vs 颤音琴 {d[vib_i]['sparse']:.3f}")

    # ---------------------------------------------------------- C 前半段归属
    print('\n[C] 前半段主旋律归颤音琴，不是钢琴伴奏')
    pipeline._run_steps(piece.id, ['instrumentation', 'voice_id', 'melody_line'])
    out = AnalysisResult.objects.get(piece=piece, step='melody_line').output
    segs = seg_of(out)
    print('  分段：', segs)
    check('覆盖率 100%', out['coverage']['ratio'] == 1.0, str(out['coverage']))
    check('钢琴一段都没被选成主旋律',
          'Piano' not in {s[2] for s in segs}, str(segs))
    check('前半段归有音高打击乐（颤音琴 / 钟琴）',
          any(s[0] <= 1 and s[1] >= 8 and s[2] in ('Vibraphone', 'Glockenspiel')
              for s in segs), str(segs))
    check('后半段归人声（合唱）',
          any(s[2] == 'Choir' for s in segs), str(segs))
    check('贝斯没被选成主旋律',
          'Acoustic Bass' not in {s[2] for s in segs}, str(segs))

    # ---------------------------------------------------------- D 齐奏加倍
    print('\n[D] 齐奏加倍算同一条线索，不把置信度压成 0')
    first = out['segments'][0]
    check('第一段 margin > 0（加倍声部不算竞争对手）',
          first['margin'] > 0, str(first['margin']))
    check('第一段置信度不是 0', first['confidence'] > 0, str(first['confidence']))
    doubled = [w for w in out['windows'] if w.get('doubled_with')]
    check('窗表里记下了齐奏加倍关系', len(doubled) > 0, str(len(doubled)))
    if doubled:
        peers = [names[j] for j in doubled[0]['doubled_with']]
        check('加倍对象是另一件同奏主题乐器',
              all(p in ('Vibraphone', 'Glockenspiel') for p in peers), str(peers))
    check('方法说明里讲清了齐奏加倍', '齐奏加倍' in out['method'],
          out['method'][-80:])

    # ---------------------------------------------------------- E 反证
    print('\n[E-反证] 按旧规则（Percussion 一律当鼓组）→ 前半段主题必然丢失')
    orig = sd._is_percussion_part

    def _old_is_percussion(part, info):
        from music21 import instrument as m21instrument
        inst = part.getInstrument(returnDefault=False)
        if inst is not None and isinstance(inst, m21instrument.Percussion):
            return True
        if inst is not None and getattr(inst, 'midiChannel', None) == 9:
            return True
        blob = f'{info.name} {info.instrument}'.lower()
        return any(k in blob for k in ('drum', 'percussion', 'perc.', 'timpani',
                                       'cymbal', 'triangle', 'tambourine',
                                       'woodblock', '打击', '鼓'))

    sd._is_percussion_part = _old_is_percussion
    try:
        # 注意：必须连同声部识别一起按旧口径重跑，否则显著性矩阵里还留着
        # 颤音琴的分数，反证就证不出来（melody_line 只认矩阵里的声部）。
        old_data = parse_score(piece.musicxml_file.path)
        old_vo = vmod.voice_id(old_data, SystemConfig.get_config(),
                               {'parts': []})
    finally:
        sd._is_percussion_part = orig
    vi = names.index('Vibraphone')
    check('旧规则下颤音琴被判成打击乐', old_data.parts[vi].is_percussion)
    check('旧规则下颤音琴的音高被整体滤掉',
          len(old_data.parts[vi].pitched_events) == 0,
          str(len(old_data.parts[vi].pitched_events)))
    check('旧规则下显著性矩阵里没有颤音琴',
          all(str(vi) not in (row.get('scores') or {})
              for row in old_vo['salience']['matrix']))
    from core.analysis import melody_line as ml_mod
    old_res = ml_mod.melody_line(old_data, SystemConfig.get_config(), None,
                                 old_vo, run_seq=99)
    old_names = {s[2] for s in seg_of(old_res)}
    print('  旧规则分段：', seg_of(old_res))
    check('旧规则下前半段主题声部提不出来',
          not (old_names & {'Vibraphone', 'Glockenspiel'}), str(old_names))
    check('旧规则下前半段退而报成钢琴伴奏', 'Piano' in old_names, str(old_names))

    # ---------------------------------------------------------- F 过期依赖
    print('\n[F] 矩阵里"少了颤音琴"也能被判为过期（窗数一个不少）')
    vo_now = AnalysisResult.objects.get(piece=piece, step='voice_id').output
    check('当前矩阵不算过期',
          pipeline._stale_deps(data, SystemConfig.get_config(),
                               {'voice_id': vo_now}) == set())
    # 模拟"声部识别是按旧解析口径跑的"：把颤音琴 / 钟琴从每个窗的分数里抹掉。
    # 窗数一个不少，只看窗数是查不出来的 —— 这正是这次改解析规则后的情形。
    import copy
    stale_vo = copy.deepcopy(vo_now)
    gone = {str(names.index('Vibraphone')), str(names.index('Glockenspiel'))}
    for row in stale_vo['salience']['matrix']:
        row['scores'] = {k: v for k, v in row['scores'].items() if k not in gone}
    check('抹掉后窗数不变',
          len(stale_vo['salience']['matrix']) == len(vo_now['salience']['matrix']),
          str(len(stale_vo['salience']['matrix'])))
    check('矩阵缺声部 → voice_id 判为过期',
          pipeline._stale_deps(data, SystemConfig.get_config(),
                               {'voice_id': stale_vo}) == {'voice_id'})

    # ---------------------------------------------------------- 页面
    print('\n[页面] 详情页能打开')
    c = Client()
    c.force_login(user)
    r = c.get(f'/pieces/{piece.id}/')
    check('详情页 200', r.status_code == 200, str(r.status_code))

    # ---------------------------------------------------------- G 密度口径开关
    print('\n[G] 织体密度口径可切换（默认按发音数，可改为按事件数）')
    cfg = SystemConfig.get_config()
    check('默认口径 = 按发音数', cfg.density_by_note_count is True,
          str(cfg.density_by_note_count))
    # 取 2 小节窗：钢琴 4 个三和弦（4 事件 / 12 音），颤音琴 8 个单音
    w_note = vmod.window_scores(data, 0.0, 8.0, by_note_count=True)
    w_ev = vmod.window_scores(data, 0.0, 8.0, by_note_count=False)
    dn = {x['part_index']: x for x in w_note['per_part'] if not x.get('empty')}
    de = {x['part_index']: x for x in w_ev['per_part'] if not x.get('empty')}
    check('按发音数：钢琴密度 = 和弦内部音数',
          dn[pno_i]['density'] == dn[pno_i]['note_count'] == 12,
          f"{dn[pno_i]['density']} / {dn[pno_i]['note_count']}")
    check('按事件数：钢琴密度 = 发声事件数',
          de[pno_i]['density'] == de[pno_i]['onsets'] == 4,
          f"{de[pno_i]['density']} / {de[pno_i]['onsets']}")
    check('note_count 两种口径下都报真实音数（不受开关影响）',
          de[pno_i]['note_count'] == 12, str(de[pno_i]['note_count']))
    check('按发音数：伴奏比单声部旋律更密（不冤枉它当旋律）',
          dn[pno_i]['sparse'] < dn[vib_i]['sparse'],
          f"钢琴 {dn[pno_i]['sparse']:.3f} vs 颤音琴 {dn[vib_i]['sparse']:.3f}")
    check('切回按事件数：同一段伴奏反而显得更稀疏（旧口径的毛病）',
          de[pno_i]['sparse'] > de[vib_i]['sparse'],
          f"钢琴 {de[pno_i]['sparse']:.3f} vs 颤音琴 {de[vib_i]['sparse']:.3f}")

    vo_out = AnalysisResult.objects.get(piece=piece, step='voice_id').output
    check('声部识别把所用口径记进了结果（供过期判断）',
          vo_out['salience'].get('density_by_note_count') is True,
          str(vo_out['salience'].get('density_by_note_count')))
    check('口径没变 → 矩阵不算过期',
          pipeline._stale_deps(data, cfg, {'voice_id': vo_out}) == set())

    cfg.density_by_note_count = False
    cfg.save()
    check('改成按事件数 → 矩阵被判为过期（会自动补跑）',
          pipeline._stale_deps(data, cfg, {'voice_id': vo_out}) == {'voice_id'})
    # 单步重跑主旋律线时也要识别口径变了：不能用旧口径的矩阵算新口径的归属
    ml_res = ml_mod.melody_line(data, cfg, None, vo_out, run_seq=98)
    check('主旋律线单步重跑时按新口径整表重算',
          len(ml_res.get('matrix_filled') or []) == len(ml_res['windows']),
          f"重算 {len(ml_res.get('matrix_filled') or [])} / {len(ml_res['windows'])} 窗")
    check('重算后仍覆盖全曲', ml_res['coverage']['ratio'] == 1.0,
          str(ml_res['coverage']))

    # 设置页：勾上 / 不勾都能存
    check('设置页 POST 不勾 → 关闭（checkbox 不提交）',
          _post_settings(user, {}) is False)
    check('设置页 POST 勾上 → 开启',
          _post_settings(user, {'density_by_note_count': 'on'}) is True)
    check('测试结束前口径已复位为按发音数',
          SystemConfig.get_config().density_by_note_count is True)

    # ---------------------------------------------------------- 清理
    print('\n[清理] 临时 DB / media 产物')
    media = settings.MEDIA_ROOT
    for pc in pieces:
        for rel in (pc.musicxml_file.name, pc.midi_file.name, pc.source_file.name):
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
        sys.exit(1)
    print('全部通过 ✓')


if __name__ == '__main__':
    main()
