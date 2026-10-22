"""结构 / 织体 / 风格三步骤 + 自定义问答页的回归脚本（与 smoke_test.py 并列）。

与 smoke_test.py 的区别：LLM 用假回答替换（不再真的调用 dsh），
因此几十秒就能跑完整条链路，用来验证本地算法、落库、页面与权限。

用法：python smoke_sts_test.py
跑完自动清理临时 DB / MIDI / media 产物。
"""
import ast
import json
import os
import re
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

from core.analysis import llm as llm_mod  # noqa: E402
from core.analysis import pipeline, qa as qa_mod  # noqa: E402
from core.models import AnalysisResult, Piece, QAQuestion, SystemConfig  # noqa: E402

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_smoke_db.sqlite3')
MIDI_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_smoke_sts.mid')
PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


# ---------------- 假 LLM ----------------

def _longest_list(text):
    best = None
    for m in re.finditer(r'(\[\{.*?\}\])', text, re.S):
        try:
            obj = ast.literal_eval(m.group(1))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, list) and (best is None or len(obj) > len(best)):
            best = obj
    return best


def fake_run_llm(task, profile='headless', timeout=240):
    if '专属音乐分析助手' in task:
        return '这是测试回答第一段（结论）。\n\n第二段：依据来自第 1-8 小节的终止式。'
    payload = _longest_list(task)
    if payload and isinstance(payload[0], dict) and 'pitches' in payload[0]:
        return json.dumps([{'label': w['label'], 'chord': 'C', 'quality': 'major',
                            'inversion': None, 'function': 'I', 'confidence': 0.8}
                           for w in payload], ensure_ascii=False)
    for key, body in (
        ('节奏分析师', {'groove': '稳定行进', 'character': '均匀',
                    'syncopation': '少量切分', 'main_patterns': '四分音符为主',
                    'description': '节奏平稳'}),
        ('调性分析师', {'main_key': 'F 大调',
                    'regions': [{'start_measure': 1, 'end_measure': 8, 'key': 'F 大调',
                                 'type': '大调', 'evidence': '终止式'}],
                    'modulations': [], 'confidence': 0.8, 'description': '单一调性'}),
        ('旋律分析师', {'contour_summary': '波浪型', 'phrase_analysis': '4 小节一句',
                    'motive_analysis': '级进为主', 'highlights': '首句动机鲜明'}),
        ('曲式分析师', {'form_name': '单三部曲式（ABA）',
                    'section_plan': [{'label': 'A', 'measures': '1-4',
                                      'function': '呈示', 'comment': '主题呈示'}],
                    'phrase_comment': '方整型乐句', 'highlights': '再现完整',
                    'confidence': 0.7}),
        ('织体分析师', {'main_texture': '主调织体 · 分解和弦伴奏',
                    'accompaniment_pattern': '分解和弦',
                    'timeline_comment': '织体统一', 'density_comment': '四声部',
                    'highlights': '声部清晰', 'confidence': 0.75}),
        ('风格分析师', {'genre': '古典主义', 'period': '18 世纪', 'region': '德奥',
                    'evidence': ['功能和声', '方整乐句', '四声部织体'],
                    'harmonic_language': '三度叠置功能和声',
                    'rhythm_feel': '平稳', 'instrumentation_comment': '混声四部',
                    'description': '典型古典主义合唱织体。', 'confidence': 0.7}),
    ):
        if key in task:
            return json.dumps(body, ensure_ascii=False)
    return json.dumps({'ok': True}, ensure_ascii=False)


# ---------------- 准备 ----------------

def prepare():
    for path in (DB_FILE, MIDI_OUT):
        if os.path.exists(path):
            os.remove(path)
    call_command('migrate', '--run-syncdb', verbosity=0)
    User.objects.filter(username='smoke').delete()
    User.objects.create_user('smoke', password='smoke123456')
    print('[准备] SQLite 迁移 + 测试用户 …… OK')


def make_piece():
    from music21 import corpus, stream
    score = corpus.parse('bach/bwv66.6')
    short = stream.Score()
    for part in score.parts:
        p2 = stream.Part()
        p2.partName = part.partName
        for m in part.getElementsByClass(stream.Measure)[:16]:
            p2.append(m)
        short.append(p2)
    short.write('midi', fp=MIDI_OUT)

    user = User.objects.get(username='smoke')
    with open(MIDI_OUT, 'rb') as f:
        uploaded = SimpleUploadedFile('bwv66.mid', f.read())
    piece = Piece(owner=user, title='冒烟 · 结构织体风格', source_type='midi')
    piece.source_file.save('bwv66.mid', uploaded, save=True)
    pipeline._run_convert(piece.id)
    conv = AnalysisResult.objects.get(piece=piece, step='convert')
    assert conv.status == 'done', f'convert 失败: {conv.error[:400]}'
    piece.refresh_from_db()
    print(f'[准备] 作品 #{piece.id} 转换完成 …… OK（{piece.musicxml_file.name}）')
    return piece


def run_all(piece):
    """同步跑完全部依赖（rhythm + melody 组 + 三项）。"""
    for group in ('rhythm', 'melody'):
        done = set(AnalysisResult.objects.filter(piece=piece, status='done')
                   .values_list('step', flat=True))
        steps = pipeline.expand_group(group, done)
        if steps:
            pipeline._run_steps(piece.id, steps)
    # 三项各自单独跑（依赖已由上面两组跑完，ctx 会复用其输出）
    for step in ('structure', 'texture', 'style'):
        pipeline._run_steps(piece.id, [step])
    print('[运行] 全部步骤执行完毕')


def main():
    # 装假 LLM：llm 模块（其余步骤经 run_llm_json_soft 间接命中）
    # 与 qa 模块（它直接 import 了 run_llm，必须单独替换）
    llm_mod.run_llm = fake_run_llm
    qa_mod.run_llm = fake_run_llm
    prepare()
    piece = make_piece()
    run_all(piece)

    print('\n[结果] 步骤状态：')
    for step in ('rhythm_grid', 'rhythm_pattern', 'instrumentation', 'voice_id',
                 'polyphony', 'chords', 'tonality', 'melody_shape',
                 'structure', 'texture', 'style'):
        r = AnalysisResult.objects.filter(piece=piece, step=step).first()
        ok = bool(r and r.status == 'done' and r.output)
        check(f'{step} 完成', ok, ('' if ok else (r.error[:200] if r else '无记录')))

    print('\n[结果] 输出字段：')
    for step, fields in (
        ('structure', ('measure_count', 'phrase_count', 'phrases', 'sections', 'form')),
        ('texture', ('windows', 'timeline', 'ratios', 'main_texture', 'density')),
        ('style', ('fingerprint', 'local_hints', 'genre', 'evidence')),
    ):
        out = AnalysisResult.objects.get(piece=piece, step=step).output or {}
        missing = [f for f in fields if f not in out]
        check(f'{step} 字段齐全', not missing, f'缺 {missing}' if missing else '')
        if step == 'structure':
            f = out.get('form') or {}
            print(f"    曲式 {f.get('name')} / 骨架 {f.get('scheme')} / "
                  f"乐段 {len(out.get('sections') or [])} 个 / "
                  f"乐句 {len(out.get('phrases') or [])} 个 / "
                  f"终止式 {len(out.get('cadences') or [])} 处")
            check('structure 曲式名非空', bool(f.get('name')))
            check('structure 乐段非空', bool(out.get('sections')))
        if step == 'texture':
            print(f"    主织体 {out.get('main_texture')} / 时间线 "
                  f"{len(out.get('timeline') or [])} 段 / 比例 {out.get('ratios')}")
            check('texture 主织体非空', bool(out.get('main_texture')))
            check('texture 窗口非空', bool(out.get('windows')))
        if step == 'style':
            fp = out.get('fingerprint') or {}
            print(f"    流派 {out.get('genre')} / {out.get('period')} / {out.get('region')}"
                  f" / 证据 {len(out.get('evidence') or [])} 条")
            check('style 流派非空', bool(out.get('genre')))
            check('style 指纹有和声词汇', 'chord_count' in fp)

    print('\n[页面] HTTP：')
    c = Client()
    check('登录', c.login(username='smoke', password='smoke123456'))
    for url, name in ((f'/pieces/{piece.id}/sts/', '结构/织体/风格页'),
                      (f'/pieces/{piece.id}/ask/', '问答页'),
                      (f'/pieces/{piece.id}/status/', '步骤状态接口')):
        resp = c.get(url)
        body = resp.content.decode('utf-8', 'replace')
        check(f'{name} 200', resp.status_code == 200, f'status={resp.status_code}')
        if url.endswith('/sts/'):
            check('sts 页三个卡片', body.count('class="card sts-card"') == 3)
            check('sts 页含问答链接', '/ask/' in body)
        if url.endswith('/ask/'):
            check('qa 页含结构页链接', '/sts/' in body)

    # 「音乐结构」窗口上的「分析曲式」开关：关 → 存盘 + 标需重跑；再开回来
    print('\n[曲式开关] HTTP 往返：')
    resp = c.get(f'/pieces/{piece.id}/status/')
    jd = resp.json() if resp.status_code == 200 else {}
    check('状态接口带曲式开关', 'structure_form_enabled' in (jd.get('config') or {}),
          str(jd.get('config')))
    resp = c.post(f'/pieces/{piece.id}/structure-form/', {'enabled': '0'})
    jd = resp.json() if resp.status_code == 200 else {}
    check('关闭曲式 200', jd.get('ok') is True, str(jd))
    check('关闭后写进系统配置',
          SystemConfig.get_config().structure_form_enabled is False)
    check('关闭后已有结果标「需重跑」', jd.get('stale') is True, str(jd))
    resp = c.post(f'/pieces/{piece.id}/structure-form/', {'enabled': '1'})
    jd = resp.json() if resp.status_code == 200 else {}
    check('再开回来 200 且配置为 True',
          jd.get('ok') is True
          and SystemConfig.get_config().structure_form_enabled is True, str(jd))
    check('开关不改动已有结构结果本身',
          bool((AnalysisResult.objects.get(piece=piece, step='structure').output or {})
               .get('phrases')))

    print('\n[问答] 提交与回答：')
    resp = c.post(f'/pieces/{piece.id}/ask/submit/', {'question': '这首曲子的织体如何？'})
    data = resp.json() if resp.status_code == 200 else {}
    check('提交提问 200', resp.status_code == 200, str(data))
    qid = data.get('id')
    for _ in range(20):
        time.sleep(0.3)
        item = QAQuestion.objects.filter(pk=qid).first()
        if item and item.status in ('done', 'error'):
            break
    check('问答落库完成', bool(item and item.status == 'done'),
          f'status={item.status if item else None}')
    if item and item.status == 'done':
        check('答案非空', bool(item.answer.strip()), item.answer[:60])
        check('上下文含已完成方面',
              bool((item.details or {}).get('已完成分析')),
              str(list((item.details or {}).get('已完成分析', {}).keys())))
    resp = c.get(f'/pieces/{piece.id}/ask/status/')
    check('问答状态接口 200', resp.status_code == 200)
    if resp.status_code == 200:
        jd = resp.json()
        check('状态接口含 1 条记录', len(jd.get('items', [])) == 1)
        check('状态接口带 based_on', bool(jd['items'][0].get('based_on')),
              str(jd['items'][0].get('based_on')))

    # 权限：陌生用户不能对他人私有作品提问
    User.objects.filter(username='other').delete()
    User.objects.create_user('other', password='other123456')
    c2 = Client()
    c2.login(username='other', password='other123456')
    check('私有作品他人不可见 404',
          c2.get(f'/pieces/{piece.id}/sts/').status_code == 404)
    check('私有作品他人不可提问 403',
          c2.post(f'/pieces/{piece.id}/ask/submit/',
                  {'question': 'x'}).status_code == 403)

    cleanup(piece)
    print(f'\n===== 通过 {len(PASS)} 项，失败 {len(FAIL)} 项 =====')
    if FAIL:
        print('失败项：' + '、'.join(FAIL))
    return 0 if not FAIL else 1


def cleanup(piece):
    from django.conf import settings
    from django.db import connection
    connection.close()          # 释放 SQLite 文件句柄
    media = settings.MEDIA_ROOT
    for rel in (piece.musicxml_file.name, piece.midi_file.name,
                piece.source_file.name):
        if not rel:
            continue
        p = os.path.join(media, rel)
        if os.path.exists(p):
            os.remove(p)
    for path in (DB_FILE, MIDI_OUT):
        for _ in range(5):
            if not os.path.exists(path):
                break
            try:
                os.remove(path)
            except OSError:
                time.sleep(0.4)     # 等后台线程释放句柄
    print('[清理] 临时 DB / MIDI / media 产物已删除')


if __name__ == '__main__':
    sys.exit(main())
