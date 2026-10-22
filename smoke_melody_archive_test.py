"""主旋律提取结果的**存档**回归脚本：每次运行都留一份，文件名互不相同。

需求（用户要求）：
  1. 开了「形态（旋律轮廓）决定资格」的那次提取，MIDI 也要保存，
     文件名要和没开的那次不一样（两种口径结果可能不同，不能互相覆盖）；
  2. 「历史运行」子窗口要显示该次运行有没有按旋律形态判断资格；
  3. **每一次**主旋律提取的导出文件都要保存下来，每次的文件名都不同
     （重跑不覆盖上一次的产物，历史归档里的下载链接始终指向自己那份）；
  4. 历史子窗口要能看该次运行的消息与结果。

用法：python smoke_melody_archive_test.py
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
from core.analysis import pipeline  # noqa: E402
from core.models import AnalysisResult, AnalysisRun, Piece, SystemConfig  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


def exists(rel):
    return bool(rel) and os.path.exists(os.path.join(str(settings.MEDIA_ROOT), rel))


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
    from music21 import note
    off = (measure - 1) * 4.0
    step = 4.0 / max(len(pitches), 1)
    for i, pc in enumerate(pitches):
        part.insert(off + i * step, note.Note(pc, quarterLength=ql))


def demo_score():
    """人声旋律 + 分解和弦（伴奏）+ 低音，8 小节。"""
    from music21 import instrument, meter, stream

    sc = stream.Score()
    sc.insert(0, meter.TimeSignature('4/4'))
    voice = stream.Part()
    voice.partName = 'Voice'
    voice.insert(0, instrument.Vocalist())
    arp = stream.Part()
    arp.partName = 'Arpeggio'
    arp.insert(0, instrument.Piano())
    bass = stream.Part()
    bass.partName = 'Bass'
    bass.insert(0, instrument.AcousticBass())

    voice_bars = [[60, 62, 64, 67], [67, 64, 62, 60],
                  [62, 64, 65, 67], [67, 65, 64, 60]]
    arp_bar = [72, 76, 79, 84, 79, 76, 72, 67]
    bass_bars = [[36, 43], [38, 45], [40, 47], [36, 43]]
    for m in range(1, 9):
        _add(voice, m, voice_bars[(m - 1) % 4], 0.75)
        _add(arp, m, arp_bar, 0.5)
        _add(bass, m, bass_bars[(m - 1) % 4], 1.0)
    sc.append(voice)
    sc.append(arp)
    sc.append(bass)
    return sc


def prepare():
    call_command('migrate', '--run-syncdb', verbosity=0)
    User.objects.filter(username='smoke').delete()
    User.objects.create_user('smoke', password='smoke123456')
    cfg = SystemConfig.get_config()
    cfg.melody_voice_priority = False     # 让显著性说话，形态层才有活干
    cfg.melody_segment_min_bars = 2
    cfg.melody_confidence_high = 0.25
    cfg.melody_confidence_mid = 0.10
    cfg.analysis_window_bars = 4
    cfg.melody_shape_gate = False
    cfg.melody_shape_min_score = 0.35
    cfg.melody_export_tempo_mode = 'original'
    cfg.melody_export_bpm = 120
    cfg.melody_export_instrument_mode = 'original'
    cfg.melody_export_program = 0
    cfg.save()
    print('[准备] 迁移 + 用户 + 配置复位 …… OK')


def make_piece(user, title, data):
    piece = Piece(owner=user, title=title, source_type='musicxml')
    piece.source_file.save('demo.musicxml',
                           SimpleUploadedFile('demo.musicxml', data), save=True)
    pipeline._run_convert(piece.id)
    piece.refresh_from_db()
    return piece


def run_line(piece):
    pipeline._run_steps(piece.id, ['instrumentation', 'voice_id', 'melody_line'])
    return AnalysisResult.objects.get(piece=piece, step='melody_line')


def run_extraction(piece):
    pipeline._run_steps(piece.id, ['instrumentation', 'voice_id', 'polyphony',
                                   'exchange', 'melody_extraction'])
    return AnalysisResult.objects.get(piece=piece, step='melody_extraction')


def cleanup(piece):
    day_dir = os.path.join(str(settings.MEDIA_ROOT), 'melody')
    for d in os.listdir(day_dir) if os.path.isdir(day_dir) else []:
        p = os.path.join(day_dir, d, f'p{piece.id}')
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


# ------------------------------------------------------------ 用例

def main():
    prepare()
    # 回归可复现：本机装着 dsh，兜底开关默认开着会让管道真去调模型
    _cfg = SystemConfig.get_config()
    _cfg.melody_llm_judge = False
    _cfg.save()
    user = User.objects.get(username='smoke')
    cfg = SystemConfig.get_config()

    print('\n[造谱] 人声旋律 + 分解和弦伴奏 + 低音（8 小节）')
    piece = make_piece(user, '存档对照曲', _write(demo_score()))

    # ---------- ① 关闭形态层跑两次：两次的文件都要留着 ----------
    print('\n[① 每次运行单独一份] 关闭形态层连跑两次')
    r1 = run_line(piece)
    f1 = (r1.output or {})['exports']['absolute']
    check('第 1 次文件名带运行序号 r1', 'r1' in f1['midi'], f1['midi'])
    check('第 1 次文件名不带形态标记', 'shape' not in f1['midi'], f1['midi'])
    check('第 1 次 MIDI 已落盘', exists(f1['midi']), f1['midi'])
    check('第 1 次 MusicXML 已落盘', exists(f1['musicxml']))

    r2 = run_line(piece)
    f2 = (r2.output or {})['exports']['absolute']
    check('第 2 次文件名带运行序号 r2', 'r2' in f2['midi'], f2['midi'])
    check('两次文件名不同（重跑不覆盖）', f1['midi'] != f2['midi'],
          f'{f1["midi"]} vs {f2["midi"]}')
    check('第 1 次那份还在磁盘上（没被覆盖）', exists(f1['midi']))
    check('第 2 次那份也在磁盘上', exists(f2['midi']))

    # ---------- ② 开启形态层：文件名要和不开启的不一样 ----------
    print('\n[② 形态资格层] 开启后单独存一份，名字要不一样')
    cfg.melody_shape_gate = True
    cfg.melody_shape_min_score = 0.35
    cfg.save()
    r3 = run_line(piece)
    out3 = r3.output or {}
    f3 = out3['exports']['absolute']
    check('开启时 shape_gate=True 写进结果', out3.get('shape_gate') is True)
    check('形态标记写进文件名', 'shape0.35' in f3['midi'], f3['midi'])
    check('与没开形态层的文件名不同', f3['midi'] != f2['midi'] and f3['midi'] != f1['midi'],
          f3['midi'])
    check('开了形态层这份也保存了', exists(f3['midi']) and exists(f3['musicxml']))
    check('无缝拼接版同样带形态标记和运行序号',
          'shape0.35' in out3['exports']['spliced']['midi']
          and 'r3' in out3['exports']['spliced']['midi'],
          out3['exports']['spliced']['midi'])
    check('前三份都在磁盘上（没有互相覆盖）',
          all(exists(x['midi']) for x in (f1, f2, f3)))
    check('记下了本次运行序号', out3.get('export_run_seq') == 3,
          str(out3.get('export_run_seq')))

    # ---------- ③ 关掉再跑：又是另一份 ----------
    print('\n[③ 口径切换] 关掉形态层再跑一次')
    cfg.melody_shape_gate = False
    cfg.save()
    r4 = run_line(piece)
    f4 = (r4.output or {})['exports']['absolute']
    check('第 4 次文件名带 r4 且无形态标记',
          'r4' in f4['midi'] and 'shape' not in f4['midi'], f4['midi'])
    check('四份文件名两两不同',
          len({x['midi'] for x in (f1, f2, f3, f4)}) == 4)
    check('四份都在磁盘上', all(exists(x['midi']) for x in (f1, f2, f3, f4)))

    # ---------- ④ 历史运行：显示形态资格开关 ----------
    print('\n[④ 历史运行子窗口] 每次都标明有没有按形态判断资格')
    c = Client()
    c.login(username='smoke', password='smoke123456')
    res = c.get(f'/pieces/{piece.id}/history/?step=melody_line')
    data = res.json()
    check('历史接口返回 ok', data.get('ok') is True)
    check('归档了 4 次运行', data.get('total') == 4, str(data.get('total')))
    items = data.get('items') or []
    check('列表按最近在前', [it['seq'] for it in items] == [4, 3, 2, 1],
          str([it['seq'] for it in items]))
    gate_by_seq = {it['seq']: it.get('shape_gate') for it in items}
    check('每次都带 shape_gate 字段（true / false，不是缺失）',
          gate_by_seq == {4: False, 3: True, 2: False, 1: False},
          str(gate_by_seq))
    check('第 3 次还带形态下限', items[1].get('shape_gate_threshold') == 0.35,
          str(items[1].get('shape_gate_threshold')))
    check('第 3 次带形态层替换段数',
          items[1].get('shape_replacements') is not None,
          str(items[1].get('shape_replacements')))
    check('每次都带运行序号（与文件名里的 rN 一致）',
          {it['seq']: it.get('export_run_seq') for it in items}
          == {4: 4, 3: 3, 2: 2, 1: 1},
          str({it['seq']: it.get('export_run_seq') for it in items}))

    # 单条详情：历史结果 + 该次自己的导出文件
    run3 = AnalysisRun.objects.get(piece=piece, step='melody_line', seq=3)
    det = c.get(f'/pieces/{piece.id}/history/?step=melody_line&run={run3.id}').json()
    check('单条详情返回该次 output', (det.get('item') or {}).get('output') is not None)
    out_hist = (det.get('item') or {}).get('output') or {}
    check('详情里的导出文件就是第 3 次那份',
          out_hist.get('exports', {}).get('absolute', {}).get('midi') == f3['midi'],
          str(out_hist.get('exports', {}).get('absolute', {}).get('midi')))
    check('第 3 次那份文件仍然可下载（没被后续运行覆盖）',
          exists(out_hist['exports']['absolute']['midi']))
    check('详情里带形态资格标记', (det.get('item') or {}).get('shape_gate') is True)
    check('详情里带该次的导出设置说明',
          '原始节奏' in ((det.get('item') or {}).get('export_options_label') or ''),
          str((det.get('item') or {}).get('export_options_label')))

    # 历史里每一步的文件都还在（这是"每次都要保存"的硬指标）
    all_files = []
    for run in AnalysisRun.objects.filter(piece=piece, step='melody_line'):
        ex = ((run.output or {}).get('exports') or {}).get('absolute') or {}
        if ex.get('midi'):
            all_files.append((run.seq, ex['midi']))
    check('历史 4 次各有自己的文件（4 个不同路径）',
          len({p for _, p in all_files}) == 4, str(all_files))
    check('历史 4 次的文件全都还在磁盘上',
          all(exists(p) for _, p in all_files),
          str([p for _, p in all_files if not exists(p)]))

    # 别的步骤没有形态层 → 不显示这个标
    res_v = c.get(f'/pieces/{piece.id}/history/?step=voice_id').json()
    check('没有形态层的步骤 shape_gate 为 null（前端不显示该标）',
          all(it.get('shape_gate') is None for it in (res_v.get('items') or [])),
          str([it.get('shape_gate') for it in (res_v.get('items') or [])]))

    # ---------- ⑤ 重新导出（改节奏 / 音色）沿用同一运行编号 ----------
    print('\n[⑤ 重新导出] 换选项换后缀，运行编号不变')
    rep = c.post(f'/pieces/{piece.id}/reexport/', {
        'step': 'melody_line', 'tempo_mode': 'changed', 'bpm': '140',
        'instrument_mode': 'changed', 'program': '40',
    })
    d5 = rep.json()
    check('重新导出成功', d5.get('ok') is True, str(d5)[:160])
    f5 = (d5.get('exports') or {}).get('absolute') or {}
    check('换选项后文件名带 140bpm + gm40',
          '140bpm' in (f5.get('midi') or '') and 'gm40' in (f5.get('midi') or ''),
          str(f5.get('midi')))
    check('仍是第 4 次运行那份（编号 r4 不变）', 'r4' in (f5.get('midi') or ''),
          str(f5.get('midi')))
    check('第 4 次的原始节奏那份也还在', exists(f4['midi']))
    check('新导出这份已落盘', exists(f5.get('midi')))

    # ---------- ⑥ 下载文件名也区分形态口径 ----------
    print('\n[⑥ 现算下载] 开了形态层的文件名里写明「形态0.35」')
    from django.utils.http import urlencode
    q = urlencode({'step': 'melody_line', 'fmt': 'midi', 'variant': 'absolute',
                   'tempo_mode': 'changed', 'bpm': '104',
                   'instrument_mode': 'changed', 'program': '71'})
    resp = c.get(f'/pieces/{piece.id}/melody-export/?{q}')
    cd = resp.headers.get('Content-Disposition', '')
    from urllib.parse import unquote
    cd = unquote(cd)
    check('下载带 BPM 与乐器名', '104BPM' in cd and '单簧管' in cd, cd[:200])
    check('下载不写库（现算），文件名不带形态标记（当前这次没开）',
          '形态' not in cd, cd[:200])
    cfg.melody_shape_gate = True
    cfg.save()
    r6 = run_line(piece)
    resp2 = c.get(f'/pieces/{piece.id}/melody-export/?{q}')
    cd2 = unquote(resp2.headers.get('Content-Disposition', ''))
    check('开启形态层后下载文件名写明「形态0.35」', '形态0.35' in cd2, cd2[:200])
    check('开启形态层那次的文件也存了一份',
          exists((r6.output or {})['exports']['absolute']['midi']),
          str((r6.output or {}).get('exports', {}).get('absolute', {}).get('midi')))
    cfg.melody_shape_gate = False
    cfg.save()

    # ---------- ⑦ 选轨卡（⑨）同样是每次一份 ----------
    print('\n[⑦ 选轨卡] 主旋律声部识别也每次单独保存')
    def first_file(g):
        """选轨的 exports 是扁平的 {midi, musicxml,…}；兼容嵌套写法。"""
        if not isinstance(g, dict):
            return None
        if g.get('midi'):
            return g
        for v in g.values():
            if isinstance(v, dict) and v.get('midi'):
                return v
        return None

    e1 = run_extraction(piece)
    first = first_file((e1.output or {}).get('exports'))
    check('选轨导出落盘', bool(first) and exists(first.get('midi')), str(first))
    e2 = run_extraction(piece)
    second = first_file((e2.output or {}).get('exports'))
    check('选轨两次文件名不同',
          bool(first) and bool(second) and first['midi'] != second['midi'],
          f'{first} vs {second}')
    check('选轨第一次那份还在', exists(first['midi']))
    check('选轨结果记了运行序号',
          (e1.output or {}).get('export_run_seq') == 1
          and (e2.output or {}).get('export_run_seq') == 2,
          f'{(e1.output or {}).get("export_run_seq")} / '
          f'{(e2.output or {}).get("export_run_seq")}')

    # ---------- ⑧ 前端：历史子窗口能看形态开关 / 该次文件 / 消息 ----------
    print('\n[⑧ 前端] 历史子窗口的渲染逻辑')
    html = c.get(f'/pieces/{piece.id}/').content.decode('utf-8')
    check('历史行渲染形态资格标（shapeChip）', 'shapeChip' in html)
    check('形态标有「开 / 关」两种文案',
          '形态资格 开' in html and '形态资格 关' in html)
    check('旧运行（没记开关）显示「未记录」而不是装作开或关',
          '形态资格 未记录' in html)
    check('历史里列出该次自己的导出文件（histExports）',
          'histExports' in html)
    check('历史里能看该次的消息（histMessages）', 'histMessages' in html)
    check('历史模式不再摆"按当前设置下载"（renderOutput 带 opts.hist）',
          'opts.hist' in html or 'opts && opts.hist' in html)
    css = ''
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'static', 'css', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, encoding='utf-8') as fh:
            css = fh.read()
    check('「形态资格 关」有对应样式', '.tag-shape-off' in css)

    # ---------- ⑨ 文件名工具本身 ----------
    print('\n[⑨ 文件名工具] 标记拼接')
    opt = {'tempo_mode': 'original', 'bpm': 120,
           'instrument_mode': 'original', 'program': 0}
    s_off = mex.stem_with_options('melody_line_absolute', opt, '', 'r1')
    s_on = mex.stem_with_options('melody_line_absolute', opt,
                                 mex.shape_stem_tag(True, 0.35), 'r1')
    check('没开形态层 → 文件名里没有 shape 段', 'shape' not in s_off, s_off)
    check('开了形态层 → 文件名多一段 shape0.35',
          s_on == 'melody_line_absolute__shape0.35__orig-tempo__orig-inst__r1', s_on)
    check('两个名字确实不同', s_off != s_on)
    check('运行序号标记 r1 / r2 / r7',
          [mex.run_stem_tag(i) for i in (1, 2, 7)] == ['r1', 'r2', 'r7'])
    check('形态标记可读版：形态0.35 / 空',
          mex.shape_label(True, 0.35) == '形态0.35'
          and mex.shape_label(False, 0.35) == '')

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
