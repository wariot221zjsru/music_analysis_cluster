"""端到端冒烟测试：SQLite + music21 自带巴赫众赞歌 → 全链路跑到旋律形态分析。

用法（不依赖 PostgreSQL，用独立 SQLite）：
    python smoke_test.py

验证内容：
  1. 建库迁移、建用户
  2. music21 语料生成 MIDI → 上传件
  3. convert（MIDI→MusicXML）
  4. 三组分析（rhythm / melody / extraction）全链路，含 dsh LLM 调用
  5. 每步输出落库检查
"""
import os
import sys
import django

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

django.setup()

from django.core.management import call_command  # noqa: E402
from django.contrib.auth.models import User       # noqa: E402
from core.models import Piece, AnalysisResult     # noqa: E402
from core.converters import musicxml as conv_mod  # noqa: E402
from core.analysis import pipeline                # noqa: E402

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_smoke_db.sqlite3')
MIDI_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_smoke_bwv66.mid')


def prepare():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    call_command('migrate', '--run-syncdb', verbosity=0)
    User.objects.filter(username='smoke').delete()
    User.objects.create_user('smoke', password='smoke123456')
    print('[1/6] SQLite 迁移 + 测试用户 …… OK')


def make_test_midi():
    """music21 核心语料：巴赫众赞歌 bwv66.6（4 声部，真实和声）→ 截前 12 小节 → MIDI。"""
    from music21 import corpus
    score = corpus.parse('bach/bwv66.6')
    # 截短以控制 LLM 调用时长（保留 12 小节）
    from music21 import stream
    short = stream.Score()
    for part in score.parts:
        p2 = stream.Part()
        p2.partName = part.partName
        for m in part.getElementsByClass(stream.Measure)[:12]:
            p2.append(m)
        short.append(p2)
    short.write('midi', fp=MIDI_OUT)
    print(f'[2/6] 测试 MIDI 生成 …… OK（{MIDI_OUT}）')


def main():
    prepare()
    make_test_midi()

    user = User.objects.get(username='smoke')
    from django.core.files.uploadedfile import SimpleUploadedFile
    with open(MIDI_OUT, 'rb') as f:
        uploaded = SimpleUploadedFile('bwv66.mid', f.read())
    piece = Piece(owner=user, title='冒烟测试 · 巴赫众赞歌 bwv66.6（节选）',
                  source_type='midi', contains_voice=False)
    piece.source_file.save('bwv66.mid', uploaded, save=True)
    print(f'[3/6] 上传件入库 …… OK（piece #{piece.id}）')

    # convert（同步调用线程主体）
    pipeline._run_convert(piece.id)
    conv = AnalysisResult.objects.get(piece=piece, step='convert')
    assert conv.status == 'done', f'convert 失败: {conv.error[:500]}'
    piece.refresh_from_db()
    print(f'[4/6] MIDI → MusicXML 转换 …… OK（{piece.musicxml_file.name}）')

    # 三组分析（直接同步跑线程主体；顺序 = 前端按钮可点击的顺序）
    for group in ('rhythm', 'melody', 'extraction'):
        done = set(AnalysisResult.objects.filter(piece=piece, status='done')
                   .values_list('step', flat=True))
        steps = pipeline.expand_group(group, done)
        print(f'[5/6] 组「{group}」→ 步骤: {steps}')
        if not steps:
            print('      （已全部完成，跳过）')
            continue
        pipeline._run_steps(piece.id, steps)

    # 校验
    print('[6/6] 结果检查：')
    ok, bad = 0, []
    for r in AnalysisResult.objects.filter(piece=piece).order_by('id'):
        line = f'      {r.step:<20} {r.status}'
        if r.status == 'done' and r.output:
            ok += 1
            print(line)
        else:
            bad.append(r.step)
            print(line, '⚠', (r.error or '')[:200])
    print(f'\n===== 完成 {ok} 步，失败 {len(bad)} 步: {bad} =====')

    # 关键输出摘要
    for step in ('polyphony', 'exchange', 'tonality', 'melody_shape', 'melody_extraction'):
        r = AnalysisResult.objects.filter(piece=piece, step=step).first()
        if r and r.status == 'done':
            out = r.output or {}
            if step == 'polyphony':
                print(f"复调判定: {out.get('branch')}（ratio={out.get('ratio')}, thr={out.get('threshold')}）")
            elif step == 'exchange':
                print(f"交接判定: {out.get('branch')}（switches={out.get('switch_count')}）")
            elif step == 'tonality':
                print(f"主调: {out.get('main_key')}，区域 {len(out.get('regions') or [])} 个，转调 {len(out.get('modulations') or [])} 处")
            elif step == 'melody_shape':
                print(f"旋律形态分支: {out.get('branch')}，分析段落 {len(out.get('analyses') or [])} 份")
            elif step == 'melody_extraction':
                print(f"主旋律模式: {out.get('mode')}，候选 {len(out.get('candidates') or [])}，导出: {out.get('exports')}")

    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
