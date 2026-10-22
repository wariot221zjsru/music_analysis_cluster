"""主旋律线提取「覆盖率 + 低音区误判」的回归脚本。

起因（线上真实事故）
--------------------
一首 66 小节的曲子，主旋律线只追到前 32 小节：

1. **谱面解析改过之后，声部识别的结果没跟着重跑。** 早先 `parse_score` 只取
   第一个含小节的声部当小节基准（那是一条 32 小节的鼓轨），于是全曲被当成
   32 小节，显著性矩阵只有 8 个窗。后来解析改成"所有声部小节并集" → 真实
   66 小节、17 个窗，但库里那份 voice_id 还是 8 窗的旧值。
2. **melody_line 单步重跑会直接复用它**（依赖已完成就不重跑），矩阵里查不到
   的窗**一律按静默处理** → 34 小节凭空消失，覆盖率 48%。真正的主题（第 33
   小节起的人声）一个音都没进结果。
3. 顺带查出的第二个毛病：**贝斯被当成主旋律**。显著性四项在窗内 min-max
   归一，贝斯靠"长时值 + 高重复 + 稀疏"三项，0.625 压过人声的 0.598，
   于是前奏段交替出现"钢琴 / 贝斯"的怪结果。

本脚本把这三件事都钉死：
  A. 低音区声部不参与主旋律竞选；
  B. 显著性矩阵缺窗时 melody_line **就地补算**（同一套算法），覆盖率不掉；
  C. 流水线能识别"过期依赖"并自动补跑声部识别。

用法：python smoke_melody_coverage_test.py
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

from core.analysis import melody_export as mex  # noqa: E402
from core.analysis import melody_line as ml_mod  # noqa: E402
from core.analysis import pipeline  # noqa: E402
from core.analysis import voice_id as vmod  # noqa: E402
from core.analysis.scoredata import parse_score  # noqa: E402
from core.models import AnalysisResult, Piece, SystemConfig  # noqa: E402

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '_smoke_melody_coverage.sqlite3')
PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


# ------------------------------------------------------------ 造谱
# 三声部：低音（长时值 + 高重复 + 稀疏，正是"显著性容易赢"的画像）/
#         高音固定动机（真旋律）/ 人声（只在后半段进来）。

LEAD_PATTERN = [72, 74, 76, 74, 72, 71, 72, 76]   # 级进为主，动机不重复


def _bar(m, bar_ql=4.0):
    return (m - 1) * bar_ql


def build_score(total_bars=16, bar_ql=4.0, bass_name='Acoustic Bass'):
    from music21 import instrument, note, stream

    sc = stream.Score()
    bass = stream.Part()
    bass.partName = bass_name
    bass.insert(0, instrument.AcousticBass())
    lead = stream.Part()
    lead.partName = 'Lead'
    lead.insert(0, instrument.Violin())
    voice = stream.Part()
    voice.partName = 'Voice'
    voice.insert(0, instrument.Vocalist())

    for m in range(1, total_bars + 1):
        off = _bar(m, bar_ql)
        # 低音：每小节两个二分音符、音高不变 → 时值总量大、动机重复度≈1、稀疏
        for i in range(2):
            bass.insert(off + i * (bar_ql / 2), note.Note(36, quarterLength=bar_ql / 2))
        # 高音：八个八分音符的级进动机（真旋律的画像：音高高、音符密）
        step = bar_ql / len(LEAD_PATTERN)
        for i, pc in enumerate(LEAD_PATTERN):
            lead.insert(off + i * step, note.Note(pc, quarterLength=step))
        # 人声：只在后半段进来
        if m > total_bars / 2:
            vstep = bar_ql / 4
            for i, pc in enumerate([79, 77, 76, 74]):
                voice.insert(off + i * vstep, note.Note(pc, quarterLength=vstep))

    sc.append(bass)
    sc.append(lead)
    sc.append(voice)
    return sc


def _dump(score, suffix='.musicxml'):
    import tempfile
    path = tempfile.mktemp(suffix=suffix)
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
    cfg.save()
    print('[准备] SQLite 迁移 + 测试用户 + 配置复位 …… OK')


def make_piece(user, title, data: bytes):
    piece = Piece(owner=user, title=title, source_type='musicxml')
    piece.source_file.save('cov.musicxml',
                           SimpleUploadedFile('cov.musicxml', data), save=True)
    pipeline._run_convert(piece.id)
    piece.refresh_from_db()
    r = AnalysisResult.objects.get(piece=piece, step='convert')
    assert r.status == 'done', f'convert 失败：{r.error[:400]}'
    return piece


def seg_of(out):
    return [(s['start_measure'], s['end_measure'], s['part_name'])
            for s in out['segments']]


def main():
    prepare()
    # 回归可复现：本机装着 dsh，兜底开关默认开着会让管道真去调模型
    _cfg = SystemConfig.get_config()
    _cfg.melody_llm_judge = False
    _cfg.save()
    user = User.objects.get(username='smoke')
    pieces = []

    print('\n[造谱] 低音（长音 + 重复）+ 高音动机 + 后半段人声')
    piece = make_piece(user, 'coverage', _dump(build_score()))
    pieces.append(piece)
    data = parse_score(piece.musicxml_file.path)
    check('谱面 16 小节', data.measure_count == 16, str(data.measure_count))

    # ---------------------------------------------------------- A 低音区
    print('\n[A] 低音区声部不参与主旋律竞选')
    mp = vmod.part_mean_pitches(data)
    low = vmod.low_register_parts(data)
    bass_idx = data.part_names.index('Acoustic Bass')
    check('贝斯平均音高最低且低于 C3', mp[bass_idx] < 48,
          f'{mp[bass_idx]:.1f}')
    check('贝斯被判为低音区声部', bass_idx in low, str(sorted(low)))
    lead_idx = data.part_names.index('Lead')
    check('高音声部不被误判成低音区', lead_idx not in low)

    pipeline._run_steps(piece.id, ['instrumentation', 'voice_id', 'melody_line'])
    out = AnalysisResult.objects.get(piece=piece, step='melody_line').output
    check('覆盖率 100%', out['coverage']['ratio'] == 1.0, str(out['coverage']))
    check('结果里记下了低音区声部',
          out.get('low_register_names') == ['Acoustic Bass'],
          str(out.get('low_register_names')))
    names = {s[2] for s in seg_of(out)}
    check('贝斯一段都没被选成主旋律', 'Acoustic Bass' not in names, str(names))
    check('前段归高音声部、后段归人声',
          {'Lead', 'Voice'} <= names, str(seg_of(out)))
    roles = {r['part_name']: r['role_candidate']
             for r in AnalysisResult.objects.get(
                 piece=piece, step='voice_id').output['roles']}
    check('角色表把贝斯报成「低音」', roles.get('Acoustic Bass') == '低音',
          str(roles.get('Acoustic Bass')))
    check('角色表仍点得出「主旋律（候选）」',
          any(v == '主旋律（候选）' for v in roles.values()), str(roles))

    # 反证：把低音区排除关掉，贝斯就会抢走主旋律（这就是改之前的行为）
    print('\n[A-反证] 关掉低音区排除 → 贝斯会抢走主旋律')
    orig = vmod.low_register_parts
    vmod.low_register_parts = lambda d: set()
    try:
        AnalysisResult.objects.filter(piece=piece, step='melody_line').delete()
        pipeline._run_steps(piece.id, ['melody_line'])
        bad = AnalysisResult.objects.get(piece=piece, step='melody_line').output
    finally:
        vmod.low_register_parts = orig
    bad_names = {s[2] for s in seg_of(bad)}
    check('不做低音区排除时贝斯确实会入选（说明这一步不能省）',
          'Acoustic Bass' in bad_names, str(seg_of(bad)))

    # ---------------------------------------------------------- B 矩阵缺窗自愈
    print('\n[B] 显著性矩阵缺窗 → 本步就地补算，覆盖率不掉')
    vo = AnalysisResult.objects.get(piece=piece, step='voice_id')
    full = len(vo.output['salience']['matrix'])
    check('完整矩阵 4 个窗', full == 4, str(full))
    # 模拟"声部识别是按旧谱解析跑的"：只留前 2 个窗
    vo.output['salience']['matrix'] = vo.output['salience']['matrix'][:2]
    vo.save()
    # 直接调 melody_line（绕过流水线）—— 流水线那层会自动补跑 voice_id，
    # 这里要验的是"就算 voice_id 没跟着重跑，本步也不会丢掉半首曲子"。
    AnalysisResult.objects.filter(piece=piece, step='melody_line').delete()
    out2 = ml_mod.melody_line(data, SystemConfig.get_config(), piece, vo.output)
    check('矩阵被截短后覆盖率仍是 100%',
          out2['coverage']['ratio'] == 1.0, str(out2['coverage']))
    filled = out2.get('matrix_filled') or []
    check('记下了就地补算的窗', len(filled) == 4, str(filled))
    check('提示文案不再写"按静默处理"',
          '静默' not in (out2.get('window_warning') or ''),
          out2.get('window_warning'))
    names2 = {s[2] for s in seg_of(out2)}
    check('补算后贝斯仍不入选', 'Acoustic Bass' not in names2, str(names2))
    check('补算后的分段与完整矩阵时一致', seg_of(out2) == seg_of(out),
          f'{seg_of(out2)} vs {seg_of(out)}')

    # ---------------------------------------------------------- C 过期依赖自动补跑
    print('\n[C] 流水线识别过期依赖并自动补跑声部识别')
    stale = pipeline._stale_deps(data, SystemConfig.get_config(),
                                 {'voice_id': vo.output})
    check('矩阵被截短 → voice_id 判为过期', stale == {'voice_id'}, str(sorted(stale)))
    # 重跑声部识别把矩阵补齐，再验一次"不该报过期"
    pipeline._run_steps(piece.id, ['voice_id'])
    ok_out = AnalysisResult.objects.get(piece=piece, step='voice_id').output
    check('矩阵补齐后不再判为过期',
          pipeline._stale_deps(data, SystemConfig.get_config(),
                               {'voice_id': ok_out}) == set())
    check('补齐后矩阵又是 4 个窗',
          len(ok_out['salience']['matrix']) == 4,
          str(len(ok_out['salience']['matrix'])))

    # ---------------------------------------------------------- D 3/4 拍导出
    print('\n[D] 3/4 的曲子导出后小节仍是 3/4')
    from music21 import meter, note, stream as m21stream
    p34 = m21stream.Part()
    for i in range(6):
        p34.insert(i * 1.0, note.Note(72, quarterLength=1.0))
    ts = mex.source_time_sig(data)
    check('本谱拍号是 4/4', ts == {'numerator': 4, 'denominator': 4}, str(ts))
    fin = mex.finalize_part(p34, {'numerator': 3, 'denominator': 4})
    ms = fin.getElementsByClass(m21stream.Measure)
    check('按 3/4 分小节 → 2 小节', len(ms) == 2, str(len(ms)))
    check('每小节 3 拍',
          all(abs(m.duration.quarterLength - 3.0) < 1e-6 for m in ms),
          str([float(m.duration.quarterLength) for m in ms]))
    check('拍号写进导出谱',
          ms[0].timeSignature is not None
          and ms[0].timeSignature.ratioString == '3/4',
          str(ms[0].timeSignature))

    # ---------------------------------------------------------- 页面
    print('\n[页面] 详情页能打开且含新字段')
    c = Client()
    c.force_login(user)
    r = c.get(f'/pieces/{piece.id}/')
    check('详情页 200', r.status_code == 200, str(r.status_code))
    body = r.content.decode('utf-8')
    check('含「主旋律线追踪（逐段）」', '主旋律线追踪（逐段）' in body)

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
        return 1
    print('全部通过 ✓')
    return 0


if __name__ == '__main__':
    sys.exit(main())
