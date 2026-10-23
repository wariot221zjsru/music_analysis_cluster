"""「平线否决」回归脚本 —— 柱式和弦 / 踏板音不许抢主旋律。

起因（真实事故）
----------------
que sera sera 的主旋律线第 4–12 小节提出来的是 PIZZICATO 的柱式和弦，
不是手风琴的旋律。逐位对过：

  * 原谱 MELODY 第 4 小节只有 3 个音（rest + C4 + D4 + rest + E4），音区 C4–G4；
  * 原谱 PIZZICATO 同小节是**三连音柱式和弦**，一个事件同时含 G4+C5+E5；
  * 导出的正是那个和弦 —— single 口径下每个事件只留主音，而主音判定每次都
    取最高音 E5，于是导出成一条 E5 同音平线。

为什么柱式和弦能赢：显著性四项里有两项站在它一边，而这两项都是**假证据** ——

  1. 平均音高：和弦一律"取最高音"（voice_id.window_scores 与 _seg_pitch_seq
     都是 max()），G4-C5-E5 被记成 E5，比真旋律高一个八度；
  2. 动机重复：取最高音后序列是 E5,E5,E5…，音程全 0，3-gram 全部重复 → 满分。

更糟的是**形态闸门会反过来帮忙**：平线的音程全是 0 → 级进比 1.0、拱形度 1.0、
平滑度 1.0 → **形态分 1.0，比真旋律还高**。于是形态层不但不拦截伴奏，还会主动
把主旋律让给它。

本脚本验证的修复
----------------
把"这是不是一条线"（:func:`melody_line.is_flat_line`）从**低音区翻盘那条支路**
提到**主选路径**上：

  * 窗级候选先过闸，平线失去当主旋律的资格（人声豁免 —— 人在唱就是主旋律）；
  * 形态闸门（``shape_eligible``）接同一道闸，堵住"形态 1.0 反超"的漏洞；
  * 这道闸**恒生效**，不是可选项，关掉形态层它照样在；
  * 证据不足（音太少）一律放行 —— 只做否决，不做好处。

用法：python smoke_flat_line_test.py
跑完自动清理临时 DB / media 产物（与其它 smoke 一样共用 _smoke_db，须串行跑）。
"""
import os
import shutil
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django  # noqa: E402

django.setup()

from django.contrib.auth.models import User  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.core.management import call_command  # noqa: E402

from core.analysis import melody_line as ml_mod  # noqa: E402
from core.analysis import pipeline  # noqa: E402
from core.analysis.scoredata import parse_score  # noqa: E402
from core.models import AnalysisResult, Piece, SystemConfig  # noqa: E402

PASS, FAIL = [], []

# 真实谱（用户研究目录里的原谱）。存在就额外跑一遍"真实谱复核"，不存在则跳过。
REAL_SCORE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir,
    '音乐分析和处理智能体-结果下载', '研究', 'que sera sera研究',
    'Que-Sera-Sera-1.xml')


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


def _bar(m, beats=4.0):
    return (m - 1) * beats


def _chord(part, offset, pitches, ql=0.5):
    """一个柱式和弦事件（同一 offset 上多个音）。"""
    from music21 import chord
    part.insert(offset, chord.Chord(pitches, quarterLength=ql))


def _note(part, offset, pitch, ql=1.0):
    from music21 import note
    part.insert(offset, note.Note(pitch, quarterLength=ql))


def pizz_score():
    """复刻 que sera sera 的织体：三连音柱式和弦伴奏 + 稀疏的真旋律 + 低音。

    PIZZICATO 每个和弦事件都是 [G4, C5, E5]，**取最高音恒为 E5** —— 这正是
    真实曲子里出事的那个形态：音高虚高一个八度、动机重复满分、密度垫底。
    """
    from music21 import instrument, meter, stream

    sc = stream.Score()
    sc.insert(0, meter.TimeSignature('4/4'))

    mel = stream.Part()
    mel.partName = 'MELODY'
    mel.insert(0, instrument.Accordion())
    pizz = stream.Part()
    pizz.partName = 'PIZZICATO'
    pizz.insert(0, instrument.Violoncello())
    bass = stream.Part()
    bass.partName = 'BASS'
    bass.insert(0, instrument.AcousticBass())

    # 真旋律：每小节 3 个音，级进为主，音区 C4–G4（真实谱就是这样）
    mel_bars = [[60, 62, 64], [65, 64, 62], [60, 62, 64], [67, 65, 64],
                [62, 64, 65], [67, 65, 64], [62, 60, 62], [64, 62, 60]]
    # 柱式和弦：每小节 6 个三连音式事件，最高音恒为 E5（76）
    ch_bars = [[67, 72, 76], [69, 72, 76], [64, 67, 76], [67, 71, 76],
               [67, 72, 76], [69, 72, 76], [64, 67, 76], [67, 71, 76]]
    bass_bars = [[36, 43], [38, 45], [40, 47], [36, 43],
                 [38, 45], [40, 47], [36, 43], [38, 45]]

    for i in range(8):
        m = i + 1
        for j, pc in enumerate(mel_bars[i]):
            _note(mel, _bar(m) + j * 1.0, pc, 0.75)
        for j in range(6):
            _chord(pizz, _bar(m) + j * (4.0 / 6), ch_bars[i], 4.0 / 6)
        for j, pc in enumerate(bass_bars[i]):
            _note(bass, _bar(m) + j * 2.0, pc, 2.0)
    for p in (mel, pizz, bass):
        sc.append(p)
    return sc


def chant_score():
    """人声豁免：人声吟诵（同音反复 = 平线），伴奏是柱式和弦。

    人在唱就是主旋律，这是掩膜级的强先验 —— 平线闸不许把吟诵让给伴奏。
    """
    from music21 import instrument, meter, stream

    sc = stream.Score()
    sc.insert(0, meter.TimeSignature('4/4'))
    voice = stream.Part()
    voice.partName = 'VOICE'
    voice.insert(0, instrument.Vocalist())
    pizz = stream.Part()
    pizz.partName = 'PIZZICATO'
    pizz.insert(0, instrument.Violoncello())

    for i in range(8):
        m = i + 1
        for j in range(4):                     # 吟诵：整小节同音反复
            _note(voice, _bar(m) + j * 1.0, 60, 1.0)
        for j in range(6):
            _chord(pizz, _bar(m) + j * (4.0 / 6), [67, 72, 76], 4.0 / 6)
    for p in (voice, pizz):
        sc.append(p)
    return sc


def gate_score():
    """形态闸门不许把主旋律让给平线。

    LEAD 是条大跳旋律（级进比 0，形态分低于阈值）但显著性最高 → owner 是它；
    PAD 是柱式和弦平线（形态分 1.0）。修复前形态层会把 LEAD 换成 PAD ——
    形态层本该拦伴奏，实际成了"把主旋律推给伴奏"的通道。
    """
    from music21 import instrument, meter, stream

    sc = stream.Score()
    sc.insert(0, meter.TimeSignature('4/4'))
    lead = stream.Part()
    lead.partName = 'LEAD'
    lead.insert(0, instrument.Violin())
    pad = stream.Part()
    pad.partName = 'PAD'
    pad.insert(0, instrument.Piano())

    lead_bars = [[60, 72, 62], [71, 61, 72], [60, 70, 61], [72, 62, 71],
                 [61, 72, 60], [70, 62, 71], [60, 71, 62], [72, 60, 71]]
    for i in range(8):
        m = i + 1
        for j, pc in enumerate(lead_bars[i]):
            _note(lead, _bar(m) + j * 1.0, pc, 1.0)
        for j in range(4):
            _chord(pad, _bar(m) + j * 1.0, [55, 60, 64], 1.0)
    for p in (lead, pad):
        sc.append(p)
    return sc


def prepare():
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
    cfg.melody_shape_min_score = 0.35
    cfg.melody_llm_judge = False       # 回归可复现：不许真去调模型
    cfg.melody_keep_doubles = True
    cfg.save()
    print('[准备] SQLite 迁移 + 测试用户 + 配置复位 …… OK')


def make_piece(user, title, data: bytes, filename='song.musicxml'):
    piece = Piece(owner=user, title=title, source_type='musicxml')
    piece.source_file.save(filename, SimpleUploadedFile(filename, data), save=True)
    pipeline._run_convert(piece.id)
    piece.refresh_from_db()
    r = AnalysisResult.objects.get(piece=piece, step='convert')
    assert r.status == 'done', f'convert 失败：{(r.error or "")[:400]}'
    return piece


def run_ml(piece, data, cfg, steps=('instrumentation', 'voice_id')):
    """跑前置步骤 + melody_line 主入口，返回 melody_line 的输出。"""
    pipeline._run_steps(piece.id, list(steps))
    res = AnalysisResult.objects.get(piece=piece, step='voice_id')
    assert res.status == 'done', (res.error or '')[:300]
    return ml_mod.melody_line(data, cfg, piece, res.output)


def seg_desc(segs):
    return ' | '.join(f"{s['start_measure']}-{s['end_measure']}:{s['part_name']}"
                      for s in segs)


# ------------------------------------------------------------ 用例

def main():
    prepare()
    cfg = SystemConfig.get_config()
    user = User.objects.get(username='smoke')

    print('\n[A] is_flat_line：只做否决，证据不足一律放行')
    check('平线（同音反复 8 次）判为平线',
          ml_mod.is_flat_line([76] * 8) is True)
    check('真旋律（级进）不是平线',
          ml_mod.is_flat_line([60, 62, 64, 65]) is False)
    check('音太少（2 个音）→ 证据不足，放行',
          ml_mod.is_flat_line([60, 62]) is False)
    check('空序列 → 放行', ml_mod.is_flat_line([]) is False)
    # 跨度不看：四个音的小音型跨度常常只有 3 半音，按跨度否掉会误杀真旋律
    # （smoke_low_register 用例 A 的大提琴 50,52,53,52 就是这么被杀掉的）。
    check('不同音高够、跨度虽只有 3 半音 → 放行（否决不看跨度）',
          ml_mod.is_flat_line([60, 61, 62, 63]) is False)
    check('四个音的小音型（三个不同音高）不是平线',
          ml_mod.is_flat_line([50, 52, 53, 52]) is False)
    check('柱式和弦取最高音后恒为 E5 → 平线',
          ml_mod.is_flat_line([76, 76, 76, 76, 76, 76]) is True)

    print('\n[B] 形态分 1.0 是假满分 vs 真表扬（漏洞确实存在）')
    # 柱式和弦口径：一个起音同时响好几个音（ratio=1.0）
    flat_sh = ml_mod.contour_shape([76] * 8, 1.0)
    # 级进比 1.0（音程全 0）、平滑度 1.0（没有极值点）、拱形度因"最高点取中位
    # 位置"而接近满分 → 总分 0.99，仍高于绝大多数真旋律。
    check('平线的形态分接近满分（级进比 / 平滑度齐满分，拱形度近满分）',
          flat_sh['score'] >= 0.95, str(flat_sh['score']))
    check('平线的级进比与平滑度确实是满分',
          flat_sh['step_ratio'] == 1.0 and flat_sh['contour_smooth'] == 1.0,
          f"{flat_sh['step_ratio']}/{flat_sh['contour_smooth']}")
    check('contour_shape 把它标为 flat', flat_sh.get('flat') is True)
    check('标的是**柱式和弦**那种平线',
          flat_sh.get('flat_kind') == 'chord', str(flat_sh.get('flat_kind')))
    check('形态闸门不再放行柱式和弦平线（堵住 1.0 反超）',
          ml_mod.shape_eligible(flat_sh, 0.35) is False)
    # 同一条序列、但走"单音吟诵"口径（ratio=0.0）：形态满分这次是**表扬**，
    # 不是假满分 —— 它说的是"一直在同一个音上"，证据指向"人在唱歌"。
    chant_sh = ml_mod.contour_shape([76] * 8, 0.0)
    check('同音反复标的是 chant',
          chant_sh.get('flat_kind') == 'chant', str(chant_sh.get('flat_kind')))
    check('形态闸门不否决同音反复（吟诵的满分不许当成嫌疑）',
          ml_mod.shape_eligible(chant_sh, 0.35) is True)
    check('音太少时形态闸门照旧放行（不作否决）',
          ml_mod.shape_eligible(ml_mod.contour_shape([60, 62, 64]), 0.9) is True)
    check('真旋律形态闸门放行',
          ml_mod.shape_eligible(ml_mod.contour_shape([60, 62, 64, 65]), 0.35)
          is True)

    print('\n[C] _pick：所有候选都是平线时不许把整窗丢掉')
    from core.analysis.melody_line import _pick
    flat_only = [
        {'part_index': 0, 'part_name': 'Pad', 'salience': 0.9, 'is_voice': False,
         'low_register': False, 'group': 0, 'flat_line': True},
        {'part_index': 1, 'part_name': 'Sparse', 'salience': 0.2,
         'is_voice': False, 'low_register': False, 'group': 1,
         'flat_line': True},
    ]
    best, _margin = _pick(flat_only, False, None)
    check('只剩平线时仍选显著性最高的那个（绝不记为静默）',
          best is not None and best['part_index'] == 0,
          best['part_name'] if best else 'None')
    mixed = [
        {'part_index': 0, 'part_name': 'Pad', 'salience': 0.9, 'is_voice': False,
         'low_register': False, 'group': 0, 'flat_line': True},
        {'part_index': 1, 'part_name': 'Melody', 'salience': 0.3,
         'is_voice': False, 'low_register': False, 'group': 1,
         'flat_line': False},
    ]
    best2, _m2 = _pick(mixed, False, None)
    check('有非平线候选时平线出局', best2['part_index'] == 1, best2['part_name'])
    voice_flat = [
        {'part_index': 0, 'part_name': 'Pad', 'salience': 0.9, 'is_voice': False,
         'low_register': False, 'group': 0, 'flat_line': True},
        {'part_index': 1, 'part_name': 'Voice', 'salience': 0.3,
         'is_voice': True, 'low_register': False, 'group': 1,
         'flat_line': True},
    ]
    best3, _m3 = _pick(voice_flat, True, None)
    check('人声豁免：吟诵（平线）的人声仍是主旋律',
          best3['part_index'] == 1, best3['part_name'])

    print('\n[D] 柱式和弦伴奏不抢主旋律（复刻 que sera sera 织体）')
    piece = make_piece(user, '柱式和弦伴奏测试曲', _write(pizz_score()))
    data = parse_score(piece.musicxml_file.path)
    check('解析出 3 个声部', len(data.parts) == 3, str(data.part_names))
    i_mel = data.part_names.index('MELODY')
    i_pizz = data.part_names.index('PIZZICATO')
    seq_mel = ml_mod._seg_pitch_seq(data, i_mel, 0.0, 16.0)
    seq_pizz = ml_mod._seg_pitch_seq(data, i_pizz, 0.0, 16.0)
    check('MELODY 的窗内序列不是平线',
          ml_mod.is_flat_line(seq_mel) is False, str(sorted(set(seq_mel))))
    check('PIZZICATO 取最高音后是一条平线',
          ml_mod.is_flat_line(seq_pizz) is True, str(sorted(set(seq_pizz))))
    check('PIZZICATO 的音高确实压过 MELODY（漏洞的前提成立）',
          max(seq_pizz) > max(seq_mel), f'{max(seq_pizz)} vs {max(seq_mel)}')

    out = run_ml(piece, data, cfg)
    segs = out.get('segments') or []
    names = {s['part_name'] for s in segs}
    check('主旋律线归 MELODY，柱式和弦没抢走',
          names == {'MELODY'}, seg_desc(segs))
    check('PIZZICATO 被记进 flat_excluded（页面上看得见）',
          'PIZZICATO' in (out.get('flat_excluded_parts') or []),
          str(out.get('flat_excluded_parts')))
    check('窗级 flat_excluded 里能查到 PIZZICATO',
          any('PIZZICATO' in (r.get('flat_excluded') or [])
              for r in out.get('windows') or []))
    check('method 里交代了平线否决',
          '平线否决' in (out.get('method') or ''))
    check('覆盖率没有被这道闸打掉（段没丢，只是换了归属）',
          out['coverage']['covered_bars'] >= 8,
          f"{out['coverage']['covered_bars']}/{out['coverage']['total_bars']}")

    print('\n[E] 人声豁免：吟诵（平线）不让位给柱式和弦')
    cfg.melody_voice_priority = True
    cfg.save()
    piece2 = make_piece(user, '吟诵测试曲', _write(chant_score()))
    data2 = parse_score(piece2.musicxml_file.path)
    out2 = run_ml(piece2, data2, cfg)
    segs2 = out2.get('segments') or []
    check('归人声，没有被伴奏顶掉',
          {s['part_name'] for s in segs2} == {'VOICE'}, seg_desc(segs2))
    check('人声自己不进 flat_excluded（豁免）',
          'VOICE' not in (out2.get('flat_excluded_parts') or []),
          str(out2.get('flat_excluded_parts')))
    check('柱式和弦仍被记进 flat_excluded',
          'PIZZICATO' in (out2.get('flat_excluded_parts') or []))

    print('\n[F] 形态闸门不许把主旋律让给平线（形态 1.0 反超）')
    cfg.melody_shape_gate = True
    cfg.melody_shape_min_score = 0.35
    cfg.save()
    piece3 = make_piece(user, '形态闸门反超测试曲', _write(gate_score()))
    data3 = parse_score(piece3.musicxml_file.path)
    i_lead = data3.part_names.index('LEAD')
    i_pad = data3.part_names.index('PAD')
    sh_lead = ml_mod.contour_shape(
        ml_mod._seg_pitch_seq(data3, i_lead, 0.0, 16.0))
    sh_pad = ml_mod.contour_shape(
        ml_mod._seg_pitch_seq(data3, i_pad, 0.0, 16.0))
    check('LEAD 形态分低于阈值（不合格）',
          sh_lead['score'] is not None and sh_lead['score'] < 0.35,
          str(sh_lead['score']))
    check('PAD 是平线且形态分接近满分（修复前它就是靠这个顶上来）',
          sh_pad['flat'] is True and sh_pad['score'] >= 0.95,
          f"flat={sh_pad['flat']} score={sh_pad['score']}")
    check('PAD 形态分确实压过 LEAD（反超的前提成立）',
          sh_pad['score'] > sh_lead['score'],
          f"{sh_pad['score']} vs {sh_lead['score']}")
    out3 = run_ml(piece3, data3, cfg)
    segs3 = out3.get('segments') or []
    check('形态层没有把 LEAD 让给 PAD',
          {s['part_name'] for s in segs3} == {'LEAD'}, seg_desc(segs3))
    reason = segs3[0].get('shape_reason') or '' if segs3 else ''
    check('改判理由写明"没有形态合格的替代声部"',
          '没有形态合格的替代' in reason, reason[:120])
    check('形态层仍在工作（这一段确实被判过）',
          (out3.get('shape_gate') is True))
    cfg.melody_shape_gate = False
    cfg.save()

    print('\n[G] 真实谱复核：que sera sera')
    if not os.path.exists(REAL_SCORE):
        print(f'  – 跳过（找不到 {os.path.basename(REAL_SCORE)}）')
    else:
        with open(REAL_SCORE, 'rb') as f:
            raw = f.read()
        piece4 = make_piece(user, 'que sera sera', raw,
                            filename='Que-Sera-Sera-1.xml')
        data4 = parse_score(piece4.musicxml_file.path)
        out4 = run_ml(piece4, data4, cfg)
        segs4 = out4.get('segments') or []
        # 出事的是第 4–12 小节（被 PIZZICATO 占着）
        bad = [s for s in segs4
               if not (s['end_measure'] < 4 or s['start_measure'] > 12)
               and s['part_name'] == 'PIZZICATO']
        check('第 4–12 小节不再归柱式和弦声部',
              not bad, seg_desc(segs4))
        check('平线否决在真实谱上真的触发了',
              bool(out4.get('flat_excluded_parts')),
              str(out4.get('flat_excluded_parts')))

    print(f'\n[汇总] 通过 {len(PASS)} 项，失败 {len(FAIL)} 项')
    if FAIL:
        for name in FAIL:
            print('  ✗ ' + name)
        sys.exit(1)
    print('全部通过')


if __name__ == '__main__':
    try:
        main()
    finally:
        for path in ('_smoke_media',):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
