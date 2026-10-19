"""「是否含人声」以文件实际声部为准的回归脚本。

规则（用户要求）：
  - 上传 MIDI / MusicXML 时，不管上传页勾选与否，一律按**文件里自带的声部**判定；
  - 整首曲子的「是否含人声」记录要统一（列表 / 详情 / 问答上下文 / 配器），
    都按文件实际情况，不按用户勾选；
  - 音频上传仍按 MuScriptor 转录结果。

用法：python smoke_voice_test.py
跑完自动清理临时 DB / media 产物。
"""
import os
import sys
import time

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django  # noqa: E402

django.setup()

from django.contrib.auth.models import User  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.test import Client  # noqa: E402

from core.analysis import pipeline, qa as qa_mod  # noqa: E402
from core.analysis.scoredata import detect_voice, parse_score  # noqa: E402
from core.models import AnalysisResult, Piece  # noqa: E402

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_smoke_voice.sqlite3')
PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


# ---------------- 造谱 ----------------

def _write(score, suffix):
    import tempfile
    path = tempfile.mktemp(suffix=suffix)
    fmt = 'musicxml' if suffix == '.musicxml' else 'midi'
    score.write(fmt, fp=path)
    with open(path, 'rb') as f:
        data = f.read()
    os.remove(path)
    return data


def choir_score():
    """SATB 合唱（Soprano / Alto / Tenor / Bass）——真的有人声。"""
    from music21 import corpus, stream
    src = corpus.parse('bach/bwv66.6')
    out = stream.Score()
    for part in src.parts:
        p = stream.Part()
        p.partName = part.partName
        for m in part.getElementsByClass(stream.Measure)[:8]:
            p.append(m)
        out.append(p)
    return out


def band_score(specs):
    """任意乐器组合：specs = [(声部名, music21 乐器对象), ...]"""
    from music21 import instrument, note, stream
    out = stream.Score()
    for name, inst in specs:
        p = stream.Part()
        p.partName = name
        if inst is not None:
            p.insert(0, inst)
        for i in range(8):
            p.append(note.Note(60 + i % 5, quarterLength=1.0))
        out.append(p)
    return out


def prepare():
    for path in (DB_FILE,):
        if os.path.exists(path):
            os.remove(path)
    call_command('migrate', '--run-syncdb', verbosity=0)
    User.objects.filter(username='smoke').delete()
    User.objects.create_user('smoke', password='smoke123456')
    print('[准备] SQLite 迁移 + 测试用户 …… OK')


def make_piece(user, title, data: bytes, filename: str, source_type: str,
               contains_voice: bool):
    """上传 → 同步转换，返回 (piece, convert output)。"""
    piece = Piece(owner=user, title=title, source_type=source_type,
                  contains_voice=contains_voice)
    piece.source_file.save(filename, SimpleUploadedFile(filename, data), save=True)
    pipeline._run_convert(piece.id)
    piece.refresh_from_db()
    r = AnalysisResult.objects.get(piece=piece, step='convert')
    assert r.status == 'done', f'convert 失败：{r.error[:400]}'
    return piece, (r.output or {})


def main():
    prepare()
    user = User.objects.get(username='smoke')

    choir_xml = _write(choir_score(), '.musicxml')
    from music21 import instrument
    piano_flute_xml = _write(
        band_score([('Piano', instrument.Piano()),
                    ('Flute', instrument.Flute())]), '.musicxml')
    piano_flute_mid = _write(
        band_score([('Piano', instrument.Piano()),
                    ('Flute', instrument.Flute())]), '.mid')
    vocal_mid = _write(band_score([('Voice', instrument.Vocalist())]), '.mid')
    altosax_xml = _write(
        band_score([('Alto Saxophone', instrument.AltoSaxophone())]), '.musicxml')

    print('\n[场景 1] MusicXML 合唱谱，上传时勾「不含人声」→ 应改判为含人声')
    p1, out1 = make_piece(user, '合唱 SATB', choir_xml, 'choir.musicxml',
                          'musicxml', False)
    check('piece.contains_voice = True', p1.contains_voice is True, str(p1.contains_voice))
    check('voice.source = file', (out1.get('voice') or {}).get('source') == 'file',
          str((out1.get('voice') or {}).get('source')))
    check('voice.corrected = True', (out1.get('voice') or {}).get('corrected') is True)
    check('voice.user_choice 保留原勾选',
          (out1.get('voice') or {}).get('user_choice') is False)
    names = [p['name'] for p in (out1.get('voice') or {}).get('voice_parts', [])]
    check('人声声部含 Soprano/Alto/Tenor/Bass',
          all(n in names for n in ('Soprano', 'Alto', 'Tenor', 'Bass')), str(names))
    check('summary.contains_voice = True', out1.get('contains_voice') is True)
    check('notes 里写明了修正', any('修正为' in n for n in out1.get('notes', [])),
          str(out1.get('notes', [])[-1:]))

    print('\n[场景 2] MusicXML 纯器乐（钢琴+长笛），勾「含人声」→ 应改判为纯器乐')
    p2, out2 = make_piece(user, '钢琴与长笛', piano_flute_xml, 'pf.musicxml',
                          'musicxml', True)
    check('piece.contains_voice = False', p2.contains_voice is False, str(p2.contains_voice))
    check('voice.source = file', (out2.get('voice') or {}).get('source') == 'file')
    check('voice.corrected = True', (out2.get('voice') or {}).get('corrected') is True)
    check('voice_parts 为空', (out2.get('voice') or {}).get('voice_parts') == [])

    print('\n[场景 3] MIDI 纯器乐，勾「含人声」→ 应改判为纯器乐')
    p3, out3 = make_piece(user, 'MIDI 钢琴长笛', piano_flute_mid, 'pf.mid',
                          'midi', True)
    check('piece.contains_voice = False', p3.contains_voice is False, str(p3.contains_voice))
    check('voice.source = file', (out3.get('voice') or {}).get('source') == 'file')

    print('\n[场景 4] MIDI 人声音色（program 52），勾「不含人声」→ 应改判为含人声')
    p4, out4 = make_piece(user, 'MIDI 人声', vocal_mid, 'vocal.mid', 'midi', False)
    check('piece.contains_voice = True', p4.contains_voice is True, str(p4.contains_voice))
    check('voice_parts 非空', bool((out4.get('voice') or {}).get('voice_parts')))

    print('\n[场景 5] 中音萨克斯（Alto Saxophone）不误判为人声')
    p5, out5 = make_piece(user, '中音萨克斯', altosax_xml, 'sax.musicxml',
                          'musicxml', True)
    check('piece.contains_voice = False', p5.contains_voice is False, str(p5.contains_voice))
    check('Alto Saxophone 未进人声清单',
          (out5.get('voice') or {}).get('voice_parts') == [])

    print('\n[一致] 配器 / 问答上下文与 piece 完全一致')
    for piece, expect in ((p1, True), (p2, False), (p3, False), (p4, True), (p5, False)):
        pipeline._run_steps(piece.id, ['instrumentation'])
        res = AnalysisResult.objects.get(piece=piece, step='instrumentation')
        ok = res.status == 'done' and res.output.get('has_voice') is expect
        check(f'{piece.title}：配器 has_voice = {expect}', ok,
              str((res.output or {}).get('has_voice')))
        data = parse_score(piece.musicxml_file.path)
        ctx = qa_mod.build_context(piece, data, {})
        check(f'{piece.title}：问答上下文含人声 = {expect}',
              ctx['作品']['含人声'] is expect, str(ctx['作品']['含人声']))

    print('\n[页面] 详情页标签带判定来源说明')
    c = Client()
    check('登录', c.login(username='smoke', password='smoke123456'))
    resp = c.get(f'/pieces/{p1.id}/')
    body = resp.content.decode('utf-8', 'replace')
    check('详情页 200', resp.status_code == 200)
    check('标签显示「含人声」', '>含人声<' in body)
    check('title 说明按文件实际声部判定',
          '按上传文件内的实际声部判定' in body, )

    print('\n[命令] refresh_voice 重算')
    call_command('refresh_voice', verbosity=0)
    p1.refresh_from_db()
    p2.refresh_from_db()
    check('合唱作品仍为 True', p1.contains_voice is True)
    check('器乐作品仍为 False', p2.contains_voice is False)

    print('\n[单元] detect_voice 直接调用')
    info = detect_voice(p1.musicxml_file.path)
    check('has_voice', info['has_voice'] is True)
    check('part_count = 4', info['part_count'] == 4, str(info['part_count']))
    check('method 非空', bool(info['method']))

    cleanup([p1, p2, p3, p4, p5])
    print(f'\n===== 通过 {len(PASS)} 项，失败 {len(FAIL)} 项 =====')
    if FAIL:
        print('失败项：' + '、'.join(FAIL))
    return 0 if not FAIL else 1


def cleanup(pieces):
    from django.conf import settings
    from django.db import connection
    connection.close()
    media = settings.MEDIA_ROOT
    for piece in pieces:
        for rel in (piece.musicxml_file.name, piece.midi_file.name,
                    piece.source_file.name):
            if not rel:
                continue
            p = os.path.join(media, rel)
            if os.path.exists(p):
                os.remove(p)
    for _ in range(5):
        if not os.path.exists(DB_FILE):
            break
        try:
            os.remove(DB_FILE)
        except OSError:
            time.sleep(0.4)
    print('[清理] 临时 DB / media 产物已删除')


if __name__ == '__main__':
    sys.exit(main())
