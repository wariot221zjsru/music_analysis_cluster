"""「形态（旋律轮廓）决定资格」回归脚本 —— 主旋律线追踪（逐段）的可选增强。

需求（用户要求）：
  1. 只在「主旋律线追踪（逐段）」这一步上加，别的步骤一律不动；
  2. 开关放在**账户总体设置** /settings/ 里，作品页不提供单独开关；
  3. 默认关闭，行为与以前完全一致。

设计口径：**形态决定「有没有资格」，显著性决定「谁更强」** —— 形态只做否决，
不参与显著性加分、不改 margin / 置信度。

本脚本造一张"分解和弦赢了显著性"的谱，验证：
  关闭 → 归属是分解和弦（证明漏洞真实存在）；
  开启 → 分解和弦因级进比≈0 被否决，改由形态合格的人声旋律承担。

用法：python smoke_shape_gate_test.py
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

from core.analysis import melody_line as ml_mod  # noqa: E402
from core.analysis import pipeline  # noqa: E402
from core.analysis.scoredata import parse_score  # noqa: E402
from core.models import AnalysisResult, Piece, SystemConfig  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


# ------------------------------------------------------------ 造谱

def _write(score):
    import tempfile
    path = tempfile.mktemp(suffix='.musicxml')
    score.write('musicxml', fp=path)
    with open(path, 'rb') as f:
        data = f.read()
    os.remove(path)
    return data


def _add(part, measure, pitches, ql):
    """在第 measure 小节内均匀铺音符。"""
    from music21 import note
    off = (measure - 1) * 4.0
    step = 4.0 / max(len(pitches), 1)
    for i, pc in enumerate(pitches):
        part.insert(off + i * step, note.Note(pc, quarterLength=ql))


def arp_score():
    """分解和弦（跳进、周期重复、音区高、密度大）+ 人声旋律（级进）+ 低音。

    四项显著性里分解和弦赢三项（音高 / 时值 / 动机重复），只输稀疏度 ——
    这正是"周期音型被误判成主旋律"的典型场景。
    """
    from music21 import instrument, meter, stream

    sc = stream.Score()
    sc.insert(0, meter.TimeSignature('4/4'))

    arp = stream.Part()
    arp.partName = 'Arpeggio'
    arp.insert(0, instrument.Piano())
    voice = stream.Part()
    voice.partName = 'Voice'
    voice.insert(0, instrument.Vocalist())
    bass = stream.Part()
    bass.partName = 'Bass'
    bass.insert(0, instrument.AcousticBass())

    arp_bar = [72, 76, 79, 84, 79, 76, 72, 67]     # 上下行分解和弦：全是跳进
    voice_bars = [[60, 62, 64, 67], [67, 64, 62, 60],
                  [62, 64, 65, 67], [67, 65, 64, 60]]   # 级进为主，逐小节有变化
    bass_bars = [[36, 43], [38, 45], [40, 47], [36, 43]]  # 根音跳动：跳进

    for m in range(1, 13):
        _add(arp, m, arp_bar, 0.5)                  # 8 个八分音符 / 小节
        _add(voice, m, voice_bars[(m - 1) % 4], 0.75)
        _add(bass, m, bass_bars[(m - 1) % 4], 1.0)
    sc.append(arp)
    sc.append(voice)
    sc.append(bass)
    return sc


def prepare():
    call_command('migrate', '--run-syncdb', verbosity=0)
    User.objects.filter(username='smoke').delete()
    User.objects.create_user('smoke', password='smoke123456')
    User.objects.filter(username='boss').delete()
    User.objects.create_superuser('boss', 'boss@x.com', 'boss123456')
    # 所有 smoke 共用同一个 DB，配置会串味 → 先把本脚本依赖的字段复位
    cfg = SystemConfig.get_config()
    cfg.melody_voice_priority = False     # 关掉人声掩膜，让显著性说话
    cfg.melody_segment_min_bars = 2
    cfg.melody_confidence_high = 0.25
    cfg.melody_confidence_mid = 0.10
    cfg.analysis_window_bars = 4
    cfg.melody_shape_gate = False         # 默认关
    cfg.melody_shape_min_score = 0.35
    cfg.save()
    print('[准备] 迁移 + 用户 + 配置复位 …… OK')


def make_piece(user, title, data):
    piece = Piece(owner=user, title=title, source_type='musicxml')
    piece.source_file.save('arp.musicxml',
                           SimpleUploadedFile('arp.musicxml', data), save=True)
    pipeline._run_convert(piece.id)
    piece.refresh_from_db()
    return piece


def run_line(piece):
    pipeline._run_steps(piece.id, ['instrumentation', 'voice_id', 'melody_line'])
    return AnalysisResult.objects.get(piece=piece, step='melody_line')


def cleanup(piece):
    day_dir = os.path.join(str(settings.MEDIA_ROOT), 'melody')
    target = os.path.join(day_dir, f'p{piece.id}')
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    for d in os.listdir(day_dir) if os.path.isdir(day_dir) else []:
        p = os.path.join(day_dir, d, f'p{piece.id}')
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


# ------------------------------------------------------------ 用例

def main():
    prepare()
    user = User.objects.get(username='smoke')
    cfg = SystemConfig.get_config()

    print('\n[配置] 形态资格层默认值与开关位置')
    check('默认关闭（行为与以前一致）', cfg.melody_shape_gate is False,
          str(cfg.melody_shape_gate))
    check('形态资格下限默认 0.35', cfg.melody_shape_min_score == 0.35,
          str(cfg.melody_shape_min_score))
    check('形态层只作用于 melody_line（voice_id 不受影响）',
          not hasattr(cfg, 'salience_shape_gate'))

    print('\n[造谱] 分解和弦 vs 人声旋律（12 小节，3 个声部）')
    piece = make_piece(user, '琶音对照曲', _write(arp_score()))
    data = parse_score(piece.musicxml_file.path)
    check('解析出 3 个声部', len(data.parts) == 3, str(data.part_names))
    names = data.part_names
    i_arp, i_voice = names.index('Arpeggio'), names.index('Voice')

    # ---------- ① 关闭：分解和弦赢（漏洞确认） ----------
    print('\n[关闭形态层] 单看显著性 → 分解和弦被判成主旋律')
    cfg.melody_shape_gate = False
    cfg.melody_voice_priority = False
    cfg.save()
    r0 = run_line(piece)
    segs0 = (r0.output or {}).get('segments') or []
    check('有分段结果', len(segs0) >= 1, str(len(segs0)))
    owner0 = segs0[0]['part_index'] if segs0 else None
    check('关闭时归属是分解和弦（漏洞真实存在）', owner0 == i_arp,
          f'owner={names[owner0] if owner0 is not None else None}')
    check('关闭时不带形态字段', segs0[0].get('shape') is None,
          str(segs0[0].get('shape')))
    check('关闭时没有替换记录',
          (r0.output or {}).get('shape_replacements') == 0,
          str((r0.output or {}).get('shape_replacements')))
    check('result 里 shape_gate=False',
          (r0.output or {}).get('shape_gate') is False)

    # ---------- ② 开启：形态否决分解和弦 ----------
    print('\n[开启形态层] 形态定资格 → 改由人声旋律承担')
    cfg.melody_shape_gate = True
    cfg.melody_shape_min_score = 0.35
    cfg.save()
    r1 = run_line(piece)
    out1 = r1.output or {}
    segs1 = out1.get('segments') or []
    check('result 里 shape_gate=True', out1.get('shape_gate') is True)
    check('阈值写进了 result', out1.get('shape_gate_threshold') == 0.35,
          str(out1.get('shape_gate_threshold')))
    owner1 = segs1[0]['part_index'] if segs1 else None
    check('开启后归属改成人声旋律', owner1 == i_voice,
          f'owner={names[owner1] if owner1 is not None else None}')
    check('记下了被顶替的声部',
          segs1[0].get('shape_replaced_from') == i_arp,
          str(segs1[0].get('shape_replaced_from')))
    check('替换原因写明"形态分…低于下限"',
          '低于下限' in (segs1[0].get('shape_reason') or ''),
          str(segs1[0].get('shape_reason'))[:120])
    check('全局替换计数 = 1', out1.get('shape_replacements') == 1,
          str(out1.get('shape_replacements')))
    check('替换明细含 from/to',
          (out1.get('shape_replacement_notes') or [{}])[0].get('from') == 'Arpeggio'
          and (out1.get('shape_replacement_notes') or [{}])[0].get('to') == 'Voice',
          str(out1.get('shape_replacement_notes'))[:160])
    check('模式字段体现形态层', '形态资格' in (out1.get('mode') or ''),
          str(out1.get('mode')))
    check('method 说明了形态层', '形态资格筛选' in (out1.get('method') or ''))

    sh = segs1[0].get('shape') or {}
    check('段里带形态三项指标',
          all(k in sh for k in ('step_ratio', 'arch', 'contour_smooth')),
          str(sh))
    check('人声形态分明显高于下限', (sh.get('score') or 0) >= 0.6, str(sh.get('score')))

    # ---------- ③ 分解和弦的形态分确实不合格 ----------
    print('\n[形态分] 分解和弦 vs 人声旋律')
    arp_seq = ml_mod._seg_pitch_seq(data, i_arp, 0.0, 48.0)
    voice_seq = ml_mod._seg_pitch_seq(data, i_voice, 0.0, 48.0)
    sh_arp = ml_mod.contour_shape(arp_seq)
    sh_voice = ml_mod.contour_shape(voice_seq)
    # 注意别写 `x or 1` 兜底：step_ratio 合法值就是 0.0，会被 or 换成 1
    arp_step = sh_arp.get('step_ratio')
    check('分解和弦级进比接近 0',
          arp_step is not None and arp_step < 0.15, str(arp_step))
    check('人声旋律级进比 > 0.5', (sh_voice.get('step_ratio') or 0) > 0.5,
          str(sh_voice.get('step_ratio')))
    check('分解和弦形态分 < 下限 0.35', sh_arp['score'] < 0.35, str(sh_arp['score']))
    check('人声旋律形态分 ≥ 0.6', sh_voice['score'] >= 0.6, str(sh_voice['score']))
    check('形态合格性判定正确',
          ml_mod.shape_eligible(sh_arp, 0.35) is False
          and ml_mod.shape_eligible(sh_voice, 0.35) is True)

    # ---------- ④ 音符太少 → 不作判定，一律放行 ----------
    print('\n[安全设计] 证据不足放行 / 全不合格保留原归属')
    check('3 个音时不判定（score=None）',
          ml_mod.contour_shape([60, 62, 64]).get('score') is None)
    check('不足 4 音仍判为合格（放行）',
          ml_mod.shape_eligible(ml_mod.contour_shape([60, 62, 64]), 0.9) is True)
    check('全音程 0（同音反复）不崩', ml_mod.contour_shape([60, 60, 60, 60]).get('score')
          is not None)

    cfg.melody_shape_min_score = 0.99      # 卡到没有任何声部合格
    cfg.save()
    r2 = run_line(piece)
    out2 = r2.output or {}
    segs2 = out2.get('segments') or []
    check('全部不合格时保留原归属（不丢段）',
          segs2[0]['part_index'] == i_arp,
          f'owner={names[segs2[0]["part_index"]]}')
    check('全部不合格时不计替换', out2.get('shape_replacements') == 0,
          str(out2.get('shape_replacements')))
    check('原因写明"没有形态合格的替代声部"',
          '没有形态合格的替代声部' in (segs2[0].get('shape_reason') or ''),
          str(segs2[0].get('shape_reason'))[:140])
    cfg.melody_shape_min_score = 0.35
    cfg.save()

    # ---------- ⑤ 开启人声优先时形态层不该乱改 ----------
    print('\n[与人声优先共存] 人声本来就赢 → 形态层不干预')
    cfg.melody_voice_priority = True
    cfg.melody_shape_gate = True
    cfg.save()
    r3 = run_line(piece)
    out3 = r3.output or {}
    segs3 = out3.get('segments') or []
    check('人声掩膜生效，归属是人声', segs3[0]['part_index'] == i_voice,
          f'owner={names[segs3[0]["part_index"]]}')
    check('人声形态合格 → 形态层不换人',
          out3.get('shape_replacements') == 0
          and segs3[0].get('shape_replaced_from') is None,
          str(out3.get('shape_replacements')))

    # ---------- ⑥ 设置页（账户总体设置）保存开关 ----------
    print('\n[设置页] /settings/ 保存形态资格层开关')
    c = Client()
    check('未登录不能进设置页', c.get('/settings/').status_code in (302, 403),
          str(c.get('/settings/').status_code))
    c.login(username='boss', password='boss123456')
    base = {
        'polyphony_threshold': '0.30', 'exchange_min_switches': '2',
        'exchange_similarity_threshold': '0.60',
        'melody_voice_priority': 'on', 'melody_segment_min_bars': '2',
        'melody_confidence_high': '0.25', 'melody_confidence_mid': '0.10',
        'melody_export_tempo_mode': 'original', 'melody_export_bpm': '120',
        'melody_export_instrument_mode': 'original', 'melody_export_program': '0',
        'analysis_window_bars': '4', 'chord_window': 'measure',
        'density_by_note_count': 'on',      # 织体密度默认口径：按发音数
        'melody_dynamics': 'on',            # 力度参与"听感"判定（默认口径）
        'dsh_profile': 'headless', 'dsh_timeout': '600',
        'max_chord_windows_per_call': '24',
    }
    r4 = c.post('/settings/', dict(base, melody_shape_gate='on',
                                   melody_shape_min_score='0.45'))
    check('POST 带开关返回 302（保存成功）', r4.status_code == 302,
          str(r4.status_code))
    cfg.refresh_from_db()
    check('设置页能打开形态资格层', cfg.melody_shape_gate is True,
          str(cfg.melody_shape_gate))
    check('设置页能改形态资格下限', cfg.melody_shape_min_score == 0.45,
          str(cfg.melody_shape_min_score))
    r5 = c.post('/settings/', base)      # 不勾 → 复选框不提交
    check('POST 不带开关即关闭', r5.status_code == 302
          and SystemConfig.get_config().melody_shape_gate is False,
          str(SystemConfig.get_config().melody_shape_gate))
    check('未传阈值时用默认 0.35（不报错）',
          SystemConfig.get_config().melody_shape_min_score == 0.35,
          str(SystemConfig.get_config().melody_shape_min_score))
    html = c.get('/settings/').content.decode('utf-8')
    check('设置页有「形态（旋律轮廓）决定资格」开关',
          'melody_shape_gate' in html and '形态（旋律轮廓）决定资格' in html)
    check('设置页有形态资格下限输入', 'melody_shape_min_score' in html)

    # ---------- ⑦ 其它步骤不受影响 ----------
    print('\n[边界] 只动 melody_line，其它步骤不受影响')
    check('melody_line 依赖仍只有 voice_id',
          pipeline.STEP_DEPS['melody_line'] == ['voice_id'],
          str(pipeline.STEP_DEPS['melody_line']))
    check('melody_extraction 依赖不变',
          pipeline.STEP_DEPS['melody_extraction']
          == ['voice_id', 'polyphony', 'exchange'],
          str(pipeline.STEP_DEPS['melody_extraction']))

    cleanup(piece)
    cfg.melody_voice_priority = True
    cfg.melody_shape_gate = False
    cfg.melody_shape_min_score = 0.35
    cfg.save()

    print('\n' + '=' * 56)
    print(f'通过 {len(PASS)} 项，失败 {len(FAIL)} 项')
    if FAIL:
        for f in FAIL:
            print('  ✗ ' + f)
        sys.exit(1)
    print('全部通过 ✓')


if __name__ == '__main__':
    main()
