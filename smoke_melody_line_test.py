"""「主旋律声部识别（选轨）」与「主旋律线追踪（逐段）」两张卡的回归脚本。

需求（用户要求）：
  1. 老卡片改名「主旋律声部识别（选轨）」，**内部方法与阈值一律不动**
     （仍先按阈值判定是否交接，再决定选轨形态）；
  2. 老卡片之后新增「主旋律线追踪（逐段）」卡片，用**新办法**：
     人声掩膜 + 显著性逐段定归属，**不受 exchange 门控**，
     器乐演奏的前奏 / 间奏同样要提取到；
  3. 两张卡各自独立运行、独立重跑。

用法：python smoke_melody_line_test.py
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
from django.db import connection  # noqa: E402
from django.test import Client  # noqa: E402

from core.analysis import melody_line as ml_mod  # noqa: E402
from core.analysis import pipeline  # noqa: E402
from core.analysis.scoredata import parse_score  # noqa: E402
from core.models import (  # noqa: E402
    AnalysisResult, AnalysisRun, Piece, SystemConfig,
)

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '_smoke_melody_line.sqlite3')
PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


# ------------------------------------------------------------ 造谱

def _write(score, suffix='.musicxml'):
    import tempfile
    path = tempfile.mktemp(suffix=suffix)
    fmt = 'musicxml' if suffix == '.musicxml' else 'midi'
    score.write(fmt, fp=path)
    with open(path, 'rb') as f:
        data = f.read()
    os.remove(path)
    return data


def _bar(m):
    return (m - 1) * 4.0


def _add(part, measure, pitches, ql):
    """在第 measure 小节内按拍铺音符。"""
    from music21 import note
    off = _bar(measure)
    step = 4.0 / max(len(pitches), 1)
    for i, pc in enumerate(pitches):
        n = note.Note(pc, quarterLength=ql)
        part.insert(off + i * step, n)


def song_score(gap_bars=0):
    """器乐前奏（1-4）+ 人声主歌（5-12）+ 器乐间奏（13-16）。

    gap_bars > 0 时在末尾追加一段"全体静默"，用于验证覆盖率统计。
    """
    from music21 import instrument, meter, stream

    sc = stream.Score()
    sc.insert(0, meter.TimeSignature('4/4'))

    guitar = stream.Part()
    guitar.partName = 'Guitar'
    guitar.insert(0, instrument.Guitar())
    piano = stream.Part()
    piano.partName = 'Piano'
    piano.insert(0, instrument.Piano())
    voice = stream.Part()
    voice.partName = 'Voice'
    voice.insert(0, instrument.Vocalist())

    riff = [67, 69, 71, 72]
    solo = [72, 71, 69, 67]
    for m in range(1, 17):
        # Piano：全曲和弦伴奏（中低音区、密度中等）
        _add(piano, m, [48, 55], 2.0)
        if m <= 4:                       # 前奏：吉他 riff（动机重复、音区高）
            _add(guitar, m, riff, 1.0)
        elif m <= 12:                    # 主歌：吉他让位，只铺长音
            _add(guitar, m, [60], 4.0)
            _add(voice, m, [64, 66, 67, 69], 1.0)
        else:                            # 间奏：吉他 solo
            _add(guitar, m, solo, 1.0)

    sc.append(guitar)
    sc.append(piano)
    sc.append(voice)
    if gap_bars:
        # 尾部追加一段**纯打击乐**（无确定音高）：小节数被撑到 20，
        # 但该窗没有 pitched event → 归属为空 → 计入"未覆盖"。
        from music21 import note as m21note
        drums = stream.Part()
        drums.partName = 'Drums'
        drums.insert(0, instrument.Percussion())
        for m in range(17, 17 + gap_bars):
            for i in range(4):
                drums.insert(_bar(m) + i, m21note.Unpitched(quarterLength=1.0))
        sc.append(drums)
    return sc


def prepare():
    for path in (DB_FILE,):
        if os.path.exists(path):
            os.remove(path)
    call_command('migrate', '--run-syncdb', verbosity=0)
    User.objects.filter(username='smoke').delete()
    User.objects.create_user('smoke', password='smoke123456')
    # 所有 smoke 脚本共用 _smoke_db.sqlite3，配置会串味（例如别的脚本 POST
    # /settings/ 时漏传 melody_voice_priority，就会被置成 False）。
    # 这里先把本脚本依赖的字段复位，保证每次跑都从默认值出发。
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


def run(piece, steps):
    pipeline._run_steps(piece.id, list(steps))
    return {s: AnalysisResult.objects.get(piece=piece, step=s) for s in steps}


# ------------------------------------------------------------ 用例

def main():
    prepare()
    # 回归可复现：本机装着 dsh，兜底开关默认开着会让管道真去调模型
    _cfg = SystemConfig.get_config()
    _cfg.melody_llm_judge = False
    _cfg.save()
    user = User.objects.get(username='smoke')

    print('\n[静态] 步骤常量与依赖')
    labels = dict(AnalysisResult.Step.choices)
    check('melody_extraction 改名为「主旋律声部识别（选轨）」',
          labels['melody_extraction'] == '主旋律声部识别（选轨）',
          labels['melody_extraction'])
    check('melody_line 名为「主旋律线追踪（逐段）」',
          labels['melody_line'] == '主旋律线追踪（逐段）', labels['melody_line'])
    check('melody_line 依赖只有 voice_id（不看 polyphony / exchange）',
          pipeline.STEP_DEPS['melody_line'] == ['voice_id'],
          str(pipeline.STEP_DEPS['melody_line']))
    check('melody_extraction 依赖保持不变',
          pipeline.STEP_DEPS['melody_extraction']
          == ['voice_id', 'polyphony', 'exchange'],
          str(pipeline.STEP_DEPS['melody_extraction']))
    order = pipeline.CANONICAL_ORDER
    check('melody_line 紧跟在 melody_extraction 之后',
          order.index('melody_line') == order.index('melody_extraction') + 1)
    check('GROUPS 新增 line 组', pipeline.GROUPS['line'] == ['melody_line'],
          str(pipeline.GROUPS['line']))
    check('extraction 组不受影响',
          pipeline.GROUPS['extraction'] == ['melody_extraction'])

    print('\n[配置] 新增参数默认值')
    cfg = SystemConfig.get_config()
    check('人声优先默认开', cfg.melody_voice_priority is True)
    check('分段最小长度默认 2 小节', cfg.melody_segment_min_bars == 2,
          str(cfg.melody_segment_min_bars))
    check('置信度高/中默认 0.25 / 0.10',
          (cfg.melody_confidence_high, cfg.melody_confidence_mid) == (0.25, 0.10),
          f'{cfg.melody_confidence_high}/{cfg.melody_confidence_mid}')
    check('交接两个阈值未被改动',
          (cfg.exchange_min_switches, cfg.exchange_similarity_threshold) == (2, 0.60),
          f'{cfg.exchange_min_switches}/{cfg.exchange_similarity_threshold}')

    print('\n[造谱] 器乐前奏 1-4 / 人声主歌 5-12 / 器乐间奏 13-16')
    piece = make_piece(user, '前奏间奏测试曲', _write(song_score()))
    data = parse_score(piece.musicxml_file.path)
    check('解析出 3 个声部', len(data.parts) == 3, str(data.part_names))
    check('Voice 被判为人声', data.parts[2].is_voice is True)
    check('Guitar / Piano 非人声',
          (data.parts[0].is_voice, data.parts[1].is_voice) == (False, False))
    check('作品判定为含人声', piece.contains_voice is True)

    print('\n[运行] 两张卡各自独立跑通')
    res = run(piece, ['instrumentation', 'voice_id', 'melody_line'])
    check('voice_id 完成', res['voice_id'].status == 'done',
          (res['voice_id'].error or '')[:200])
    check('melody_line 完成', res['melody_line'].status == 'done',
          (res['melody_line'].error or '')[:300])
    out = res['melody_line'].output or {}

    print('\n[核心] 前奏 / 间奏必须被提出来')
    segs = out.get('segments') or []
    desc = ' | '.join(f"{s['start_measure']}-{s['end_measure']}:{s['part_name']}"
                      f"{'(人声)' if s['is_voice'] else ''}" for s in segs)
    check('分出 3 段', len(segs) == 3, desc)
    if len(segs) == 3:
        s1, s2, s3 = segs
        check('第 1 段 = 器乐前奏（1-4 小节，非人声）',
              (s1['start_measure'], s1['end_measure'], s1['is_voice'])
              == (1, 4, False), desc)
        check('前奏段音符数 > 0（旧口径会在这里丢掉）',
              s1['note_count'] > 0, str(s1['note_count']))
        check('前奏段来自 Guitar', s1['part_name'] == 'Guitar', s1['part_name'])
        check('第 2 段 = 人声主歌（5-12 小节，人声）',
              (s2['start_measure'], s2['end_measure'], s2['is_voice'])
              == (5, 12, True), desc)
        check('第 3 段 = 器乐间奏（13-16 小节，非人声）',
              (s3['start_measure'], s3['end_measure'], s3['is_voice'])
              == (13, 16, False), desc)
        check('间奏段音符数 > 0', s3['note_count'] > 0, str(s3['note_count']))
    cov = out.get('coverage') or {}
    check('覆盖 16/16 小节', (cov.get('covered_bars'), cov.get('total_bars'))
          == (16, 16), str(cov))
    check('覆盖率 = 1.0', cov.get('ratio') == 1.0, str(cov.get('ratio')))

    print('\n[置信度] margin 与分级')
    check('每段都有 margin / confidence / level',
          all(('margin' in s and 'confidence' in s and s['level'] in ('高', '中', '低'))
              for s in segs))
    check('人声段的 confidence ≥ margin（人声先验加权）',
          all(s['confidence'] >= s['margin'] for s in segs if s['is_voice']),
          str([(s['margin'], s['confidence']) for s in segs]))
    check('窗级明细齐全（4 个窗）', len(out.get('windows') or []) == 4,
          str(len(out.get('windows') or [])))
    check('窗 1 人声休止、窗 2 人声在唱',
          (out['windows'][0]['voice_sounding'] is False
           and out['windows'][1]['voice_sounding'] is True),
          str([w['voice_sounding'] for w in out['windows']]))

    print('\n[导出] 原位 + 无缝拼接两套文件')
    ex = out.get('exports') or {}
    for kind in ('absolute', 'spliced'):
        spec = ex.get(kind) or {}
        ok = bool(spec.get('midi')) and bool(spec.get('musicxml'))
        check(f'{kind} 生成 midi + musicxml', ok, str(spec))
        if ok:
            p = os.path.join(settings.MEDIA_ROOT, spec['midi'])
            check(f'{kind} MIDI 文件真实存在', os.path.exists(p), p)
    check('导出未报错', 'export_error' not in out, str(out.get('export_error', '')))

    print('\n[开关] 人声优先关闭后仍全程按显著性选轨')
    cfg.melody_voice_priority = False
    cfg.save()
    out2 = ml_mod.melody_line(data, cfg, piece, res['voice_id'].output)
    check('voice_priority 字段 = False', out2['voice_priority'] is False)
    check('关闭后仍有分段', len(out2.get('segments') or []) > 0,
          str(len(out2.get('segments') or [])))
    check('关闭后覆盖率仍为 1.0',
          (out2.get('coverage') or {}).get('ratio') == 1.0)
    cfg.melody_voice_priority = True
    cfg.save()

    print('\n[碎片合并] melody_segment_min_bars 生效')
    cfg.melody_segment_min_bars = 8
    out3 = ml_mod.melody_line(data, cfg, piece, res['voice_id'].output)
    check('最小段长 8 小节 → 4 小节的碎片被并入相邻段',
          len(out3.get('segments') or []) == 1,
          str(len(out3.get('segments') or [])))
    if out3.get('segments'):
        check('被吸收的声部记录在 absorbed_from',
              bool(out3['segments'][0].get('absorbed_from')),
              str(out3['segments'][0].get('absorbed_from')))
    cfg.melody_segment_min_bars = 2
    cfg.save()

    print('\n[间隙] 整窗静默计入未覆盖')
    piece2 = make_piece(user, '带静默段的曲子', _write(song_score(gap_bars=4)))
    res2 = run(piece2, ['instrumentation', 'voice_id', 'melody_line'])
    out4 = res2['melody_line'].output or {}
    cov2 = out4.get('coverage') or {}
    check('覆盖 16 / 总 20 小节', (cov2.get('covered_bars'), cov2.get('total_bars'))
          == (16, 20), str(cov2))
    check('gap_bars = 4', cov2.get('gap_bars') == 4, str(cov2.get('gap_bars')))
    check('覆盖率 0.8', abs((cov2.get('ratio') or 0) - 0.8) < 1e-6,
          str(cov2.get('ratio')))

    print('\n[老卡片] melody_extraction 逻辑与阈值不动')
    res3 = run(piece, ['polyphony', 'exchange', 'melody_extraction'])
    check('polyphony 完成', res3['polyphony'].status == 'done')
    check('exchange 完成', res3['exchange'].status == 'done')
    check('melody_extraction 完成', res3['melody_extraction'].status == 'done',
          (res3['melody_extraction'].error or '')[:300])
    ex_out = res3['exchange'].output
    check('exchange 仍按 2 次 / 0.60 判定',
          (ex_out['min_switches'], ex_out['similarity_threshold']) == (2, 0.60))
    old = res3['melody_extraction'].output or {}
    check('老卡片仍有 candidates', bool(old.get('candidates')))
    check('老卡片仍有 mode 字段', bool(old.get('mode')), str(old.get('mode')))
    check('老卡片产出与逐段追踪相互独立（keys 不同）',
          'segments' not in old and 'candidates' not in out)

    print('\n[历史] 逐段追踪重跑会归档（seq 递增）')
    run(piece, ['melody_line'])
    seqs = list(AnalysisRun.objects.filter(piece=piece, step='melody_line')
                .order_by('seq').values_list('seq', flat=True))
    check('melody_line 归档了 2 次', seqs == [1, 2], str(seqs))
    check('最新一次结果仍在 AnalysisResult 里',
          AnalysisResult.objects.get(piece=piece, step='melody_line').status == 'done')

    print('\n[问答] 上下文带上逐段追踪')
    from core.analysis import qa as qa_mod
    results = {r.step: r.output for r in
               AnalysisResult.objects.filter(piece=piece, status='done')}
    ctx = qa_mod.build_context(piece, data, results)
    check('上下文含「主旋律线追踪（逐段）」',
          '主旋律线追踪（逐段）' in ctx['已完成分析'],
          str(list(ctx['已完成分析'].keys())))

    print('\n[页面] 主页两张卡都在')
    c = Client()
    check('登录', c.login(username='smoke', password='smoke123456'))
    resp = c.get(f'/pieces/{piece.id}/')
    body = resp.content.decode('utf-8', 'replace')
    check('详情页 200', resp.status_code == 200)
    check('含「主旋律声部识别（选轨）」', '主旋律声部识别（选轨）' in body)
    check('含「主旋律线追踪（逐段）」', '主旋律线追踪（逐段）' in body)
    check('含「主旋律声部识别（选轨）」按钮',
          'data-group="extraction"' in body and '主旋律声部识别（选轨）' in body)
    check('含「主旋律线追踪（逐段）」按钮',
          'data-group="line"' in body)
    check('老卡片仍未写「主旋律提取」', '主旋律提取' not in body)

    print('\n[管道] start_steps_async(line) 只排 melody_line 及其依赖')
    started = pipeline.start_steps_async(piece, 'line')
    check('返回的步骤含 melody_line', 'melody_line' in started, str(started))
    check('不含 exchange', 'exchange' not in started, str(started))
    import time
    time.sleep(1.5)   # 等后台线程跑完，避免清理时文件被占用

    cleanup([piece, piece2])
    print(f'\n===== 通过 {len(PASS)} 项，失败 {len(FAIL)} 项 =====')
    if FAIL:
        print('失败项：' + '、'.join(FAIL))
    return 0 if not FAIL else 1


def cleanup(pieces):
    connection.close()
    media = str(settings.MEDIA_ROOT)
    # 只删**本次测试自己造出来的**作品产物。
    # 以前这里 walk 整个 media/melody/ 删文件 —— 而 smoke_settings 的 MEDIA_ROOT
    # 就是真实项目的 media，结果把用户作品的导出产物一起删光，页面上点下载就 404。
    # 现在 MEDIA_ROOT 已隔离到 _smoke_media，这里再按 piece id 精确清理，双保险。
    for piece in pieces:
        for rel in (piece.musicxml_file.name, piece.midi_file.name,
                    piece.source_file.name):
            if not rel:
                continue
            p = os.path.join(media, rel)
            if os.path.exists(p):
                os.remove(p)
        # 旋律导出产物：media/melody/<日期>/p<id>/
        pdir = os.path.join(media, 'melody')
        if os.path.isdir(pdir):
            for day in os.listdir(pdir):
                d = os.path.join(pdir, day, f'p{piece.id}')
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
    print('[清理] 临时 DB / media 产物已删除')


if __name__ == '__main__':
    sys.exit(main())
