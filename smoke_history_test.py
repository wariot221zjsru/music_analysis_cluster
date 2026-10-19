"""运行历史归档 + 问答历史保留的回归脚本（与 smoke_sts_test.py 并列）。

覆盖：
  1. 每次运行（done / error）都往 AnalysisRun 归档一份，重跑不丢旧结果；
  2. "未执行：前置步骤失败"这类占位不进历史；
  3. /pieces/<id>/history/ 摘要列表与单条详情接口（含 404 / 403）；
  4. /pieces/<id>/status/ 带 run_counts；
  5. 问答历史完整保留（多次提问都在），首屏 json_script 直出；
  6. 前端历史入口（sts 页按钮 / 主页 histBtn）存在。

用法：python smoke_history_test.py
跑完自动清理临时 DB / MIDI / media 产物。
"""
import json
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

from core.analysis import pipeline  # noqa: E402
from core.models import AnalysisResult, AnalysisRun, Piece, QAQuestion  # noqa: E402
from core.views import _trigger_label  # noqa: E402

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_smoke_hist.sqlite3')
MIDI_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_smoke_hist.mid')
PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + (f' — {extra}' if extra else ''))


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
    piece = Piece(owner=user, title='冒烟 · 历史归档', source_type='midi')
    piece.source_file.save('bwv66.mid', uploaded, save=True)
    pipeline._run_convert(piece.id)
    conv = AnalysisResult.objects.get(piece=piece, step='convert')
    assert conv.status == 'done', f'convert 失败: {conv.error[:400]}'
    piece.refresh_from_db()
    print(f'[准备] 作品 #{piece.id} 转换完成 …… OK')
    return piece


def main():
    prepare()
    piece = make_piece()

    # ---------- 1. 归档：同一步骤跑两次 ----------
    print('\n[归档] 同一步骤重复运行：')
    pipeline._run_steps(piece.id, ['rhythm_grid'], trigger='group:rhythm')
    check('第一次运行成功',
          AnalysisResult.objects.get(piece=piece, step='rhythm_grid').status == 'done')
    check('归档 1 条', AnalysisRun.objects.filter(piece=piece, step='rhythm_grid').count() == 1,
          f"实际 {AnalysisRun.objects.filter(piece=piece, step='rhythm_grid').count()} 条")

    first_out = AnalysisResult.objects.get(piece=piece, step='rhythm_grid').output
    pipeline._run_steps(piece.id, ['rhythm_grid'], trigger='rerun:rhythm_grid')
    runs = list(AnalysisRun.objects.filter(piece=piece, step='rhythm_grid').order_by('-seq'))
    check('归档 2 条', len(runs) == 2, f'实际 {len(runs)} 条')
    check('seq 递增（2,1）', [r.seq for r in runs] == [2, 1],
          str([r.seq for r in runs]))
    check('旧结果仍保留', bool(runs[1].output), str(type(runs[1].output)))
    check('两次输出都有内容', bool(runs[0].output) and bool(runs[1].output))
    check('最新结果留在 AnalysisResult',
          AnalysisResult.objects.filter(piece=piece, step='rhythm_grid').count() == 1)
    check('首次输出未被覆盖丢失',
          runs[1].output.get('measure_count') == (first_out or {}).get('measure_count'),
          f"{runs[1].output.get('measure_count')} vs {(first_out or {}).get('measure_count')}")
    check('trigger 被记录', runs[0].trigger == 'rerun:rhythm_grid', runs[0].trigger)
    check('第一次 trigger 为按钮', runs[1].trigger == 'group:rhythm', runs[1].trigger)

    # ---------- 2. 失败也归档；占位不归档 ----------
    print('\n[归档] 失败与占位：')
    pipeline._run_steps(piece.id, ['melody_shape'], trigger='group:melody')
    err = AnalysisResult.objects.get(piece=piece, step='melody_shape')
    check('依赖缺失的步骤被标记失败', err.status == 'error')
    check('失败也归档 1 条',
          AnalysisRun.objects.filter(piece=piece, step='melody_shape').count() == 1)
    before = AnalysisRun.objects.filter(piece=piece, step='tonality').count()
    pipeline._set_result(piece, 'tonality', 'error', archive=False, trigger='group:melody',
                         error='未执行：前置步骤失败，请重跑本组。')
    check('"未执行"占位不进历史',
          AnalysisRun.objects.filter(piece=piece, step='tonality').count() == before)

    # ---------- 3. trigger 文案 ----------
    print('\n[文案] 触发方式：')
    check('convert', _trigger_label('convert') == '上传后自动转换', _trigger_label('convert'))
    check('group:melody', _trigger_label('group:melody') == '「分析旋律」按钮',
          _trigger_label('group:melody'))
    check('rerun:chords', '单模块重跑' in _trigger_label('rerun:chords'),
          _trigger_label('rerun:chords'))
    check('空值', _trigger_label('') == '—')

    # ---------- 4. 历史接口 ----------
    print('\n[接口] 历史：')
    c = Client()
    check('登录', c.login(username='smoke', password='smoke123456'))
    resp = c.get(f'/pieces/{piece.id}/history/', {'step': 'rhythm_grid'})
    check('摘要接口 200', resp.status_code == 200, str(resp.status_code))
    jd = json.loads(resp.content)
    check('摘要 ok', jd.get('ok') is True)
    check('摘要 2 条', len(jd.get('items', [])) == 2, str(len(jd.get('items', []))))
    check('摘要不含 output（避免大体量）', 'output' not in (jd['items'][0] if jd.get('items') else {}))
    check('摘要带 trigger_label', bool(jd['items'][0].get('trigger_label')),
          jd['items'][0].get('trigger_label'))
    check('摘要带 finished_at', bool(jd['items'][0].get('finished_at')))
    check('total 正确', jd.get('total') == 2, str(jd.get('total')))
    check('倒序（最近在前）', [i['seq'] for i in jd['items']] == [2, 1],
          str([i['seq'] for i in jd['items']]))

    run_id = jd['items'][0]['id']
    resp = c.get(f'/pieces/{piece.id}/history/', {'step': 'rhythm_grid', 'run': run_id})
    jd2 = json.loads(resp.content)
    check('单条详情 200', resp.status_code == 200)
    check('单条含 output', bool((jd2.get('item') or {}).get('output')))
    check('单条 output 有 measure_count',
          'measure_count' in ((jd2.get('item') or {}).get('output') or {}))
    resp = c.get(f'/pieces/{piece.id}/history/', {'step': 'rhythm_grid', 'run': 999999})
    check('不存在的记录 404', resp.status_code == 404, str(resp.status_code))
    resp = c.get(f'/pieces/{piece.id}/history/', {'step': 'nope'})
    check('未知步骤返回空列表',
          resp.status_code == 200 and json.loads(resp.content)['items'] == [])

    # ---------- 5. 状态接口带 run_counts ----------
    resp = c.get(f'/pieces/{piece.id}/status/')
    jd3 = json.loads(resp.content)
    check('status 带 run_counts', 'run_counts' in jd3)
    check('rhythm_grid 次数 = 2', (jd3.get('run_counts') or {}).get('rhythm_grid') == 2,
          str((jd3.get('run_counts') or {}).get('rhythm_grid')))
    check('melody_shape 次数 = 1', (jd3.get('run_counts') or {}).get('melody_shape') == 1)

    # ---------- 6. 问答历史 ----------
    print('\n[问答] 历史保留：')
    for q in ('这首曲子的织体如何？', '主旋律在哪个声部？'):
        item = QAQuestion.objects.create(piece=piece, user=piece.owner,
                                         question=q, status='running')
        pipeline._run_question(item.id)
    items = list(QAQuestion.objects.filter(piece=piece).order_by('id'))
    check('两次提问都落库', len(items) == 2, f'实际 {len(items)}')
    check('两条都有回答', all(i.answer.strip() for i in items),
          str([bool(i.answer.strip()) for i in items]))
    resp = c.get(f'/pieces/{piece.id}/ask/status/')
    jq = json.loads(resp.content)
    check('问答状态 200', resp.status_code == 200)
    check('返回 2 条', len(jq.get('items', [])) == 2, str(len(jq.get('items', []))))
    check('total 字段', jq.get('total') == 2, str(jq.get('total')))
    check('带 finished_at', all(i.get('finished_at') for i in jq['items']))
    check('带 based_on', any(i.get('based_on') for i in jq['items']),
          str(jq['items'][0].get('based_on')))

    resp = c.get(f'/pieces/{piece.id}/ask/')
    body = resp.content.decode('utf-8', 'replace')
    check('问答页 200', resp.status_code == 200)
    check('首屏直出历史（json_script）', 'id="qa-initial"' in body)
    check('页面带总数', 'id="qa-total"' in body and '>2<' in body, )
    resp = c.get(f'/pieces/{piece.id}/sts/')
    body = resp.content.decode('utf-8', 'replace')
    check('sts 页 200', resp.status_code == 200)
    check('sts 三卡都有历史按钮', body.count('data-histbtn=') == 3,
          str(body.count('data-histbtn=')))
    check('sts 三卡都有历史容器', body.count('class="hist-slot"') == 3,
          str(body.count('class="hist-slot"')))

    # ---------- 7. 权限：他人私有作品看不到历史 ----------
    print('\n[权限] 他人访问：')
    User.objects.filter(username='other').delete()
    User.objects.create_user('other', password='other123456')
    c2 = Client()
    c2.login(username='other', password='other123456')
    check('他人看历史 404',
          c2.get(f'/pieces/{piece.id}/history/', {'step': 'rhythm_grid'}).status_code == 404,
          str(c2.get(f'/pieces/{piece.id}/history/', {'step': 'rhythm_grid'}).status_code))
    check('他人看问答页 404', c2.get(f'/pieces/{piece.id}/ask/').status_code == 404)

    # 公开作品：他人可看历史，但不能提问
    piece.is_public = True
    piece.save(update_fields=['is_public'])
    resp = c2.get(f'/pieces/{piece.id}/history/', {'step': 'rhythm_grid'})
    check('公开作品他人可看历史 200', resp.status_code == 200, str(resp.status_code))
    check('公开作品他人不能提问 403',
          c2.post(f'/pieces/{piece.id}/ask/submit/', {'question': 'x'}).status_code == 403)
    check('公开作品他人看不到别人的提问',
          len(json.loads(c2.get(f'/pieces/{piece.id}/ask/status/').content)['items']) == 0)

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
