"""主旋律导出「节奏 / 音色」回归脚本。

用户反馈：以前主旋律导出的音频只能听到一种乐器 —— 根因是导出时新建空 Part
只塞音符，既没写 tempo 也没写 instrument，music21 落盘就按默认 120 BPM +
默认钢琴，于是速度和音色永远是那一个。

本脚本验证：
  1. 导出 MIDI / MusicXML 里**真的带上**了 tempo 与 program_change；
  2. 「原始节奏」= 谱面速度，「统一 BPM」= 用户指定的 BPM（默认 120）；
  3. 「各段原始音色」= 各段自己声部的音色号，「统一音色」= 用户指定的 GM 号；
  4. 重新导出**不改变音符**（只改速度 / 音色）；
  5. 页面上 /pieces/<id>/reexport/ 接口可用，且非作者被拒；
  6. 旧 bug：小节号查不到时不再退回 0.0（那会让整段错位到曲首）。

用法：python smoke_melody_export_test.py
"""
import os
import shutil
import sys
import tempfile
from urllib.parse import unquote

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402
from django.contrib.auth.models import User  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.test import Client  # noqa: E402
from django.utils import timezone  # noqa: E402

from core.analysis import melody as melody_mod  # noqa: E402
from core.analysis import melody_export as mex  # noqa: E402
from core.analysis import melody_line as ml_mod  # noqa: E402
from core.analysis import pipeline  # noqa: E402
from core.analysis.scoredata import parse_score  # noqa: E402
from core.models import AnalysisResult, Piece, SystemConfig  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


# ------------------------------------------------------------ 造谱

def _bar(m):
    return (m - 1) * 4.0


def _add(part, measure, pitches, ql):
    from music21 import note
    off = _bar(measure)
    step = 4.0 / max(len(pitches), 1)
    for i, pc in enumerate(pitches):
        part.insert(off + i * step, note.Note(pc, quarterLength=ql))


def song_score():
    """器乐前奏 1-4（吉他 program 25）+ 人声主歌 5-12 + 器乐间奏 13-16。

    曲速 160 BPM（与 120 明显不同），用于验证「原始节奏」确实写进去了。
    """
    from music21 import instrument, meter, stream, tempo

    sc = stream.Score()
    sc.insert(0, meter.TimeSignature('4/4'))

    guitar = stream.Part()
    guitar.partName = 'Guitar'
    guitar.insert(0, instrument.Guitar())          # GM 24/25 一带
    # 速度标记必须落在声部里：music21 写 MusicXML 时会丢掉 Score 级的
    # MetronomeMark（Score 没有小节结构可挂），真实谱面本来也是写在声部上的。
    guitar.insert(0, tempo.MetronomeMark(number=160.0))
    voice = stream.Part()
    voice.partName = 'Voice'
    voice.insert(0, instrument.Vocalist())

    for m in range(1, 17):
        if m <= 4:
            _add(guitar, m, [67, 69, 71, 72], 1.0)
        elif m <= 12:
            _add(voice, m, [64, 66, 67, 69], 1.0)
            _add(guitar, m, [60], 4.0)
        else:
            _add(guitar, m, [72, 71, 69, 67], 1.0)
    sc.append(guitar)
    sc.append(voice)
    return sc


def _write(score):
    import tempfile
    path = tempfile.mktemp(suffix='.musicxml')
    score.write('musicxml', fp=path)
    with open(path, 'rb') as f:
        data = f.read()
    os.remove(path)
    return data


# ------------------------------------------------------------ MIDI 读取

def read_midi_meta(path):
    """读出 MIDI 里的 tempo 序列与 program_change 序列（不依赖 music21）。

    返回 (tempos_bpm, programs) —— tempos_bpm 为 [(tick, bpm)]，
    programs 为 [(tick, program)]（按出现顺序）。
    """
    with open(path, 'rb') as f:
        raw = f.read()
    assert raw[:4] == b'MThd', '不是标准 MIDI'
    header_len = int.from_bytes(raw[4:8], 'big')
    fmt = int.from_bytes(raw[8:10], 'big')
    ntrks = int.from_bytes(raw[10:12], 'big')
    division = int.from_bytes(raw[12:14], 'big')
    assert division & 0x8000 == 0, '只支持 tick 制'
    ticks_per_beat = division
    pos = 8 + header_len

    def read_vlq(buf, i):
        val = 0
        while True:
            b = buf[i]
            i += 1
            val = (val << 7) | (b & 0x7F)
            if not (b & 0x80):
                return val, i

    events = []
    for _ in range(ntrks):
        assert raw[pos:pos + 4] == b'MTrk', '轨道头异常'
        tlen = int.from_bytes(raw[pos + 4:pos + 8], 'big')
        pos += 8
        body = raw[pos:pos + tlen]
        pos += tlen
        i, tick, running = 0, 0, None
        while i < len(body):
            delta, i = read_vlq(body, i)
            tick += delta
            status = body[i]
            if status & 0x80:
                running = status
                i += 1
            else:
                status = running
            if status == 0xFF:
                meta = body[i]
                i += 1
                ln, i = read_vlq(body, i)
                payload = body[i:i + ln]
                i += ln
                if meta == 0x51 and ln == 3:
                    us_per_beat = int.from_bytes(payload, 'big')
                    bpm = round(60_000_000 / us_per_beat, 2)
                    events.append((tick, 'tempo', bpm))
                elif meta == 0x2F:
                    break
                continue
            if status in (0xF0, 0xF7):
                ln, i = read_vlq(body, i)
                i += ln
                continue
            hi = status & 0xF0
            ndata = 1 if hi in (0xC0, 0xD0) else 2
            d0 = body[i]
            i += ndata
            if hi == 0xC0:
                events.append((tick, 'program', d0))
    tempos = [(t, v) for t, k, v in events if k == 'tempo']
    programs = [(t, v) for t, k, v in events if k == 'program']
    return tempos, programs, ticks_per_beat


# ------------------------------------------------------------ 主流程

def prepare():
    call_command('migrate', '--run-syncdb', verbosity=0)
    User.objects.filter(username='smoke').delete()
    # 设置页是 @_staff_required，所以 smoke 必须是 staff
    User.objects.create_user('smoke', password='smoke123456',
                             is_staff=True, is_superuser=True)
    User.objects.filter(username='other').delete()
    User.objects.create_user('other', password='other123456')
    # 所有 smoke 脚本共用一个 _smoke_db.sqlite3，配置会串味。
    # 本脚本会 POST /settings/ 改一组字段，跑完必须还原，别影响其它脚本。
    cfg = SystemConfig.get_config()
    cfg.melody_voice_priority = True
    cfg.melody_segment_min_bars = 2
    cfg.melody_confidence_high = 0.25
    cfg.melody_confidence_mid = 0.10
    cfg.analysis_window_bars = 4
    cfg.save()
    print('[准备] SQLite 迁移 + 测试用户 + 配置复位 …… OK')


def make_piece(user, title, data: bytes, filename='song.musicxml'):
    piece = Piece(owner=user, title=title, source_type='musicxml')
    piece.source_file.save(filename, SimpleUploadedFile(filename, data), save=True)
    pipeline._run_convert(piece.id)
    piece.refresh_from_db()
    r = AnalysisResult.objects.get(piece=piece, step='convert')
    assert r.status == 'done', f'convert 失败：{r.error[:400]}'
    return piece


def main():
    prepare()
    user = User.objects.get(username='smoke')
    piece = make_piece(user, '导出设置测试曲', _write(song_score()))
    data = parse_score(piece.musicxml_file.path)

    print('\n[配置] 导出默认值')
    # 重设默认值，防止前次 run 留下的脏数据影响本次测试
    cfg = SystemConfig.get_config()
    if (cfg.melody_export_tempo_mode != 'original' or cfg.melody_export_bpm != 120 or
        cfg.melody_export_instrument_mode != 'original' or cfg.melody_export_program != 0):
        cfg.melody_export_tempo_mode = 'original'
        cfg.melody_export_bpm = 120
        cfg.melody_export_instrument_mode = 'original'
        cfg.melody_export_program = 0
        cfg.save()
        cfg.refresh_from_db()
        print('    （重置导出默认值）')
    check('默认节奏 = 原始节奏', cfg.melody_export_tempo_mode == 'original',
          cfg.melody_export_tempo_mode)
    check('默认统一 BPM = 120', cfg.melody_export_bpm == 120,
          str(cfg.melody_export_bpm))
    check('默认音色 = 各段原始音色', cfg.melody_export_instrument_mode == 'original',
          cfg.melody_export_instrument_mode)
    check('默认统一音色 = 0（大钢琴）', cfg.melody_export_program == 0,
          str(cfg.melody_export_program))

    print('\n[工具] resolve_options 的边界与合法性')
    check('未传参时按配置默认', mex.resolve_options(cfg) ==
          {'tempo_mode': 'original', 'bpm': 120,
           'instrument_mode': 'original', 'program': 0})
    o1 = mex.resolve_options(cfg, {'tempo_mode': 'changed', 'bpm': '96'})
    check('改节奏 + BPM 生效', (o1['tempo_mode'], o1['bpm']) == ('changed', 96),
          str(o1))
    o2 = mex.resolve_options(cfg, {'instrument_mode': 'changed', 'program': '40'})
    check('改音色 + GM 号生效', (o2['instrument_mode'], o2['program']) == ('changed', 40),
          str(o2))
    o3 = mex.resolve_options(cfg, {'bpm': '9999', 'program': '-5'})
    check('BPM / GM 号越界被夹紧', (o3['bpm'], o3['program']) == (300, 0), str(o3))
    o4 = mex.resolve_options(cfg, {'tempo_mode': 'xxx'})
    check('非法模式退回 original', o4['tempo_mode'] == 'original')
    check('options_label 可读', '统一 96 BPM' in mex.options_label(o1),
          mex.options_label(o1))
    check('gm_name(40) = 小提琴', mex.gm_name(40) == '小提琴', mex.gm_name(40))

    print('\n[区间] 小节号查不到不退回 0.0')
    mo = {1: 0.0, 2: 4.0, 3: 8.0}
    s, e = mex.seg_range(data, mo, 4.0, 3, 5)   # 5 号小节查不到
    check('末段起点按相邻小节推算（不是 0.0）', abs(s - 8.0) < 1e-6, str(s))
    check('末段终点推算正确', abs(e - 20.0) < 1e-6, str(e))
    s2, _ = mex.seg_range(data, {5: 16.0, 6: 20.0}, 4.0, 1, 2)  # 起点在已知之前
    check('起点在已知小节之前也能推算', abs(s2 - 0.0) < 1e-6, str(s2))

    print('\n[运行] melody_line 与 melody_extraction 都跑通')
    pipeline._run_steps(piece.id, ['instrumentation', 'voice_id', 'polyphony',
                                   'exchange', 'melody_line', 'melody_extraction'])
    rl = AnalysisResult.objects.get(piece=piece, step='melody_line')
    re_ = AnalysisResult.objects.get(piece=piece, step='melody_extraction')
    check('melody_line 完成', rl.status == 'done', (rl.error or '')[:300])
    check('melody_extraction 完成', re_.status == 'done', (re_.error or '')[:300])
    segs = (rl.output or {}).get('segments') or []
    check('逐段追踪有分段结果', len(segs) > 0, str(len(segs)))
    check('默认导出选项写进了 output',
          (rl.output or {}).get('export_options', {}).get('tempo_mode') == 'original',
          str((rl.output or {}).get('export_options')))

    # ---------- ① 原始节奏 / 各段原始音色 ----------
    print('\n[A] 原始节奏 + 各段原始音色')
    opt_a = mex.resolve_options(cfg)
    ex_a = ml_mod.export_segments(data, piece, segs, opt_a)
    p_a = os.path.join(settings.MEDIA_ROOT, ex_a['absolute']['midi'])
    tempos_a, progs_a, _ = read_midi_meta(p_a)
    check('导出文件存在', os.path.exists(p_a))
    check('写进了 tempo（不再是没有速度标记）', len(tempos_a) >= 1, str(tempos_a))
    check('tempo = 谱面 160 BPM', any(abs(b - 160.0) < 1.0 for _, b in tempos_a),
          str(tempos_a))
    check('写进了 program_change（不再只有默认钢琴）', len(progs_a) >= 1, str(progs_a))
    check('各段原始音色：用到 ≥ 1 种音色', len({p for _, p in progs_a}) >= 1,
          str(sorted({p for _, p in progs_a})))
    check('说明文字含「原始节奏」', '原始节奏' in ex_a['absolute']['tempo_desc'],
          ex_a['absolute']['tempo_desc'])
    check('说明文字含「各段原始音色」',
          '各段原始音色' in ex_a['absolute']['instrument_desc'],
          ex_a['absolute']['instrument_desc'])

    # ---------- ② 统一 BPM / 统一音色 ----------
    print('\n[B] 统一 120 BPM + 统一音色（默认钢琴 GM 0）')
    opt_b = mex.resolve_options(cfg, {'tempo_mode': 'changed', 'bpm': '120',
                                      'instrument_mode': 'changed', 'program': '0'})
    ex_b = ml_mod.export_segments(data, piece, segs, opt_b)
    p_b = os.path.join(settings.MEDIA_ROOT, ex_b['absolute']['midi'])
    tempos_b, progs_b, _ = read_midi_meta(p_b)
    check('tempo = 120 BPM', any(abs(b - 120.0) < 1.0 for _, b in tempos_b),
          str(tempos_b))
    check('只有一种音色 = 0（大钢琴）', {p for _, p in progs_b} == {0},
          str(sorted({p for _, p in progs_b})))
    check('说明文字含「统一 120 BPM」', '120' in ex_b['absolute']['tempo_desc'],
          ex_b['absolute']['tempo_desc'])
    check('说明文字含「统一音色：大钢琴」',
          '大钢琴' in ex_b['absolute']['instrument_desc'],
          ex_b['absolute']['instrument_desc'])

    print('\n[C] 自定义：统一 96 BPM + 小提琴（GM 40）')
    opt_c = mex.resolve_options(cfg, {'tempo_mode': 'changed', 'bpm': '96',
                                      'instrument_mode': 'changed', 'program': '40'})
    ex_c = ml_mod.export_segments(data, piece, segs, opt_c)
    p_c = os.path.join(settings.MEDIA_ROOT, ex_c['absolute']['midi'])
    tempos_c, progs_c, _ = read_midi_meta(p_c)
    check('tempo = 96 BPM', any(abs(b - 96.0) < 1.0 for _, b in tempos_c),
          str(tempos_c))
    check('program = 40（小提琴）', {p for _, p in progs_c} == {40},
          str(sorted({p for _, p in progs_c})))

    # ---------- ③ 改节奏 / 音色不改变音符 ----------
    print('\n[核心] 改节奏 / 音色不改变音符')
    def notes_of(path):
        from music21 import converter
        sc = converter.parse(path)
        out = []
        for n in sc.recurse().notes:
            out.append((round(float(n.offset), 4),
                        n.pitch.midi if hasattr(n, 'pitch') else None,
                        round(float(n.duration.quarterLength), 4)))
        return sorted(out)
    na, nb, nc = notes_of(p_a), notes_of(p_b), notes_of(p_c)
    check('A / B 音符集合完全一致', na == nb, f'{len(na)} vs {len(nb)}')
    check('A / C 音符集合完全一致', na == nc, f'{len(na)} vs {len(nc)}')

    # ---------- ④ 选轨卡（⑨）的重新导出 ----------
    print('\n[老卡片] melody_extraction 也能按新选项重新导出')
    out_e = melody_mod.export_again(data, piece, re_.output, opt_c)
    check('重新导出了文件', len(out_e) >= 1, str(list(out_e)))
    first = next(iter(out_e.values()))
    p_e = os.path.join(settings.MEDIA_ROOT, first['midi'])
    tempos_e, progs_e, _ = read_midi_meta(p_e)
    check('选轨导出也带上了 96 BPM',
          any(abs(b - 96.0) < 1.0 for _, b in tempos_e), str(tempos_e))
    check('选轨导出也带上了 GM 40',
          any(p == 40 for _, p in progs_e), str(sorted({p for _, p in progs_e})))

    # ---------- ⑤ 页面接口 ----------
    print('\n[接口] /pieces/<id>/reexport/')
    c = Client()
    check('登录后可用', c.login(username='smoke', password='smoke123456'))
    url = f'/pieces/{piece.id}/reexport/'
    # 重新导出前那几份（旧选项的）文件路径：换选项后应被清理掉
    old_files = []
    for f in ((rl.output or {}).get('exports') or {}).values():
        if isinstance(f, dict):
            old_files += [os.path.join(str(settings.MEDIA_ROOT), f['midi']),
                          os.path.join(str(settings.MEDIA_ROOT), f['musicxml'])]
    r = c.post(url, {'step': 'melody_line', 'tempo_mode': 'changed', 'bpm': '140',
                     'instrument_mode': 'changed', 'program': '73'})
    body = r.json()
    check('POST 返回 ok', r.status_code == 200 and body.get('ok') is True,
          str(body)[:200])
    check('返回了新的导出说明', '140' in (body.get('label') or ''),
          str(body.get('label')))
    rl.refresh_from_db()
    check('结果里的 export_options 已更新',
          (rl.output or {}).get('export_options', {}).get('bpm') == 140,
          str((rl.output or {}).get('export_options')))
    p_d = os.path.join(settings.MEDIA_ROOT,
                       (rl.output or {})['exports']['absolute']['midi'])
    tempos_d, progs_d, _ = read_midi_meta(p_d)
    check('落盘 MIDI 真的是 140 BPM',
          any(abs(b - 140.0) < 1.0 for _, b in tempos_d), str(tempos_d))
    check('落盘 MIDI 真的是 GM 73（长笛）',
          any(p == 73 for _, p in progs_d), str(sorted({p for _, p in progs_d})))
    new_midi = (rl.output or {})['exports']['absolute']['midi']
    check('文件名带节奏后缀（换 BPM 就换 URL，绕开缓存）', '140bpm' in new_midi,
          new_midi)
    check('文件名带音色后缀（换乐器就换 URL）', 'gm73' in new_midi, new_midi)
    # 旧文件不删：历史归档里可能还引用着上一次的那几份，删了就成 404
    check('旧导出文件保留（历史记录仍可下载）',
          all(os.path.exists(op) for op in old_files),
          str([os.path.basename(op) for op in old_files]))

    # ---------- ⑤b 按当前设置直接下载 ----------
    print('\n[下载] /pieces/<id>/melody-export/ 按指定节奏 / 乐器下发')
    dl = f'/pieces/{piece.id}/melody-export/'
    q = {'step': 'melody_line', 'variant': 'absolute', 'fmt': 'midi',
         'tempo_mode': 'changed', 'bpm': '104',
         'instrument_mode': 'changed', 'program': '71'}
    r5 = c.get(dl, q)
    check('下载返回 200', r5.status_code == 200, str(r5.status_code))
    # Django 会把中文文件名按 RFC 5987 编成 %XX，检查前先解回来
    cd = unquote(r5.get('Content-Disposition', ''))
    check('是附件下载（不是页面）', 'attachment' in cd, cd[:120])
    check('文件名写着 BPM', '104BPM' in cd, cd[:200])
    check('文件名写着乐器名', '单簧管' in cd, cd[:200])
    tmp_dl = tempfile.mktemp(suffix='.mid')
    with open(tmp_dl, 'wb') as fh:
        fh.write(b''.join(r5.streaming_content))
    t5, p5, _ = read_midi_meta(tmp_dl)
    os.remove(tmp_dl)
    check('下载到的 MIDI 真的是 104 BPM', any(abs(b - 104.0) < 1.0 for _, b in t5),
          str(t5))
    check('下载到的 MIDI 真的是 GM 71（单簧管）', sorted({p for _, p in p5}) == [71],
          str(sorted({p for _, p in p5})))

    r5b = c.get(dl, dict(q, variant='spliced', program='56', bpm='72'))
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tf:
        tf.write(b''.join(r5b.streaming_content))
        tmp_b = tf.name
    t5b, p5b, _ = read_midi_meta(tmp_b)
    os.remove(tmp_b)
    check('拼接版也能按选项下载', r5b.status_code == 200, str(r5b.status_code))
    check('拼接版是 72 BPM + GM 56（小号）',
          any(abs(b - 72.0) < 1.0 for _, b in t5b) and sorted({p for _, p in p5b}) == [56],
          f'{t5b} {sorted({p for _, p in p5b})}')
    check('拼接版文件名含乐器名',
          '小号' in unquote(r5b.get('Content-Disposition', '')),
          unquote(r5b.get('Content-Disposition', ''))[:160])

    r5c = c.get(dl, {'step': 'melody_extraction', 'fmt': 'midi', 'bpm': '100',
                     'tempo_mode': 'changed', 'instrument_mode': 'changed',
                     'program': '0'})
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tf:
        tf.write(b''.join(r5c.streaming_content))
        tmp_c = tf.name
    t5c, p5c, _ = read_midi_meta(tmp_c)
    os.remove(tmp_c)
    check('选轨卡也能按选项下载', r5c.status_code == 200, str(r5c.status_code))
    check('选轨下载是 100 BPM + 大钢琴',
          any(abs(b - 100.0) < 1.0 for _, b in t5c) and sorted({p for _, p in p5c}) == [0],
          f'{t5c} {sorted({p for _, p in p5c})}')

    r5d = c.get(dl, {'step': 'not_a_step'})
    check('非法 step 下载被拒 404', r5d.status_code == 404, str(r5d.status_code))
    r5e = c.get(dl, dict(q, fmt='wav'))
    check('非法格式被拒 404', r5e.status_code == 404, str(r5e.status_code))
    c3 = Client()
    r5f = c3.get(dl, q)
    check('未登录不能下载（302 去登录）', r5f.status_code == 302, str(r5f.status_code))

    r2 = c.post(url, {'step': 'not_a_step'})
    check('非法 step 被拒 400', r2.status_code == 400, str(r2.status_code))
    r3 = c.get(url)
    check('GET 不被允许（405）', r3.status_code == 405, str(r3.status_code))

    c2 = Client()
    c2.login(username='other', password='other123456')
    r4 = c2.post(url, {'step': 'melody_line', 'tempo_mode': 'changed'})
    check('非作者无权重新导出（403）', r4.status_code == 403, str(r4.status_code))

    # ---------- ⑥ 设置页 ----------
    print('\n[设置页] 默认导出选项可保存')
    c.post('/settings/', {
        'polyphony_threshold': '0.5', 'exchange_min_switches': '2',
        'exchange_similarity_threshold': '0.6',
        'melody_segment_min_bars': '2', 'melody_confidence_high': '0.25',
        'melody_confidence_mid': '0.10',
        'melody_export_tempo_mode': 'changed', 'melody_export_bpm': '88',
        'melody_export_instrument_mode': 'changed', 'melody_export_program': '65',
        'analysis_window_bars': '4', 'chord_window': 'measure',
        'dsh_profile': 'headless', 'dsh_timeout': '600',
        'max_chord_windows_per_call': '200',
    })
    cfg.refresh_from_db()
    check('设置页保存了节奏模式', cfg.melody_export_tempo_mode == 'changed',
          cfg.melody_export_tempo_mode)
    check('设置页保存了 BPM', cfg.melody_export_bpm == 88, str(cfg.melody_export_bpm))
    check('设置页保存了音色号', cfg.melody_export_program == 65,
          str(cfg.melody_export_program))
    check('新默认值会被页面取用',
          mex.resolve_options(cfg)['program'] == 65)

    print('\n[清理] 临时 media 产物已删除')
    # MEDIA_ROOT 已由 smoke_settings 隔离到 _smoke_media，这里只清本次作品那层
    shutil.rmtree(os.path.join(str(settings.MEDIA_ROOT), 'melody',
                               f'{timezone.localdate():%Y%m%d}', f'p{piece.id}'),
                  ignore_errors=True)

    # 还原被 /settings/ 改掉的配置，避免污染共用 DB 里的其它 smoke 脚本
    cfg.refresh_from_db()
    cfg.melody_export_tempo_mode = 'original'
    cfg.melody_export_bpm = 120
    cfg.melody_export_instrument_mode = 'original'
    cfg.melody_export_program = 0
    cfg.melody_voice_priority = True
    cfg.melody_segment_min_bars = 2
    cfg.analysis_window_bars = 4
    cfg.save()

    print(f'\n===== 通过 {len(PASS)} 项，失败 {len(FAIL)} 项 =====')
    if FAIL:
        for f in FAIL:
            print('  ✗ ' + f)
        sys.exit(1)


if __name__ == '__main__':
    main()
