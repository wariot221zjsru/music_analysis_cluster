"""转调（大调 ⇄ 自然小调 / 和声小调）的回归脚本。

覆盖：
  1. 引擎层：C 大调 → A 自然小调 / A 和声小调 / C 自然小调；A 小调 → C 大调；
     调外变化音保持变化音身份；和弦一起被改写；
  2. 调号被换成目标调（和声小调用自然小调调号）；
  3. 转调页 GET：原调自动判断（调性分析优先 / music21 兜底）、下拉框分组与默认值；
  4. 转调页 POST：另存为新作品（默认私有）、原作品不动、新谱内容正确；
  5. 权限：看不到别人的私有作品（404），公开作品的转调页可进。

用法：python smoke_transpose_test.py
"""
import os
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_cluster_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django  # noqa: E402

django.setup()

from django.contrib.auth.models import User  # noqa: E402
from django.core.files.base import ContentFile  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.test import Client  # noqa: E402
from music21 import converter, key as m21key, note as m21note  # noqa: E402
from music21 import chord as m21chord  # noqa: E402

from core.analysis import transpose as tp  # noqa: E402
from core.models import AnalysisResult, Piece  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


def make_xml(notes, key_sig=None, chords=False) -> bytes:
    """用 music21 造一段 MusicXML，返回字节。"""
    from music21 import stream
    s = stream.Score()
    p = stream.Part()
    if key_sig:
        p.append(key_sig)
    for n in notes:
        if chords:
            p.append(m21chord.Chord([n, n], quarterLength=1))
        else:
            p.append(m21note.Note(n, quarterLength=1))
    s.append(p)
    path = '_tmp_smoke_transpose.musicxml'
    s.write('musicxml', fp=path)
    with open(path, 'rb') as fh:
        data = fh.read()
    os.remove(path)
    return data


def notes_of(xml_bytes):
    """谱面音名序列（含八度）。"""
    path = '_tmp_smoke_read.musicxml'
    with open(path, 'wb') as fh:
        fh.write(xml_bytes)
    sc = converter.parse(path)
    os.remove(path)
    return [n.nameWithOctave for n in sc.recurse().getElementsByClass(
        (m21note.Note, m21chord.Chord))]


def key_of(xml_bytes):
    path = '_tmp_smoke_read2.musicxml'
    with open(path, 'wb') as fh:
        fh.write(xml_bytes)
    sc = converter.parse(path)
    os.remove(path)
    k = sc.recurse().getElementsByClass(m21key.KeySignature).first()
    return str(k) if k is not None else ''


call_command('migrate', verbosity=0)

# ---------------------------------------------------------------- 1) 引擎层
print('1) 引擎层：转换结果')


def conv(notes, src, tgt):
    data = make_xml(notes)
    out = '_tmp_smoke_out.musicxml'
    with open('_tmp_smoke_in.musicxml', 'wb') as fh:
        fh.write(data)
    info = tp.convert_file('_tmp_smoke_in.musicxml', out, '_tmp_smoke_out.mid',
                           *src, *tgt)
    with open(out, 'rb') as fh:
        xml = fh.read()
    os.remove('_tmp_smoke_in.musicxml')
    os.remove(out)
    return notes_of(xml), key_of(xml), info


C_MAJ = 'C4 D4 E4 F4 G4 A4 B4 C5'.split()
names, k, info = conv(C_MAJ, (0, 'major'), (9, 'natural_minor'))
check(names == ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4'],
      f'C 大调 → A 自然小调：{names}')
check('a minor' in k, f'调号换成 a minor（实际 {k!r}）')

names, k, info = conv(C_MAJ, (0, 'major'), (9, 'harmonic_minor'))
check(names == ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G#4', 'A4'],
      f'C 大调 → A 和声小调：{names}')
check('a minor' in k, f'和声小调用自然小调调号（实际 {k!r}）')

names, k, info = conv(C_MAJ, (0, 'major'), (0, 'natural_minor'))
check(names == ['C4', 'D4', 'E-4', 'F4', 'G4', 'A-4', 'B-4', 'C5'],
      f'C 大调 → C 自然小调：{names}')

names, k, info = conv('A4 B4 C5 D5 E5 F5 G5 A5'.split(), (9, 'minor'), (0, 'major'))
check(names == ['C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6'],
      f'A 自然小调 → C 大调：{names}')
check('C major' in k, f'调号换成 C major（实际 {k!r}）')

names, k, info = conv('A4 B4 C5 D5 E5 F5 G#5 A5'.split(), (9, 'minor'), (0, 'major'))
check(names == ['C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6'],
      f'A 和声小调 → C 大调（升过的 7 级原样保留）：{names}')

# 调外变化音：C 大调的 #4 到 A 里还是 #4
names, _k, _i = conv('C4 D4 F#4 G4 A4 B4 C5'.split(), (0, 'major'), (9, 'natural_minor'))
check(names == ['A3', 'B3', 'D#4', 'E4', 'F4', 'G4', 'A4'],
      f'调外 #4 保持变化音身份：{names}')

# 和弦
data = make_xml(C_MAJ)
from music21 import stream  # noqa: E402

s = stream.Score()
p = stream.Part()
p.append(m21key.Key('C'))
p.append(m21chord.Chord(['C4', 'E4', 'G4']))
s.append(p)
s.write('musicxml', fp='_tmp_smoke_chord.musicxml')
sc = tp.transpose_score(converter.parse('_tmp_smoke_chord.musicxml'),
                        0, 'major', 9, 'natural_minor')[0]
os.remove('_tmp_smoke_chord.musicxml')
ch = sc.recurse().getElementsByClass(m21chord.Chord).first()
check(sorted(x.name for x in ch.pitches) == ['A', 'C', 'E'],
      f'C 大三和弦 → A 小三和弦（实际 {sorted(x.name for x in ch.pitches)}）')

# ---------------------------------------------------------------- 2) 页面
print('2) 转调页：原调自动判断 + 下拉框 + 转调落库')

admin = User.objects.create_user('t_admin', password='x', is_staff=True)
other = User.objects.create_user('t_other', password='x')

piece = Piece.objects.create(owner=admin, title='测试曲', source_type='musicxml')
piece.musicxml_file.save('t.musicxml', ContentFile(make_xml(C_MAJ)), save=True)
# 造一条已完成的调性分析结果（模拟"系统已判断过"）
AnalysisResult.objects.create(
    piece=piece, step=AnalysisResult.Step.TONALITY,
    status=AnalysisResult.Status.DONE, output={'main_key': 'C 大调'})

c = Client()
c.force_login(admin)
resp = c.get(f'/pieces/{piece.id}/transpose/')
check(resp.status_code == 200, f'转调页 200（实际 {resp.status_code}）')
body = resp.content.decode()
check('C 大调' in body, '页面显示自动判断的原调「C 大调」')
check('调性布局分析' in body, '页面标明判断依据')
check('自然小调' in body and '和声小调' in body, '大调源给出自然/和声小调两组选项')
check('value="9:natural_minor" selected' in body, '默认选中关系小调 A 自然小调')

print('3) 转调页 POST：另存为新作品')
resp = c.post(f'/pieces/{piece.id}/transpose/', {'target': '9:harmonic_minor'})
check(resp.status_code == 302, f'POST 后跳转（实际 {resp.status_code}）')
new_piece = Piece.objects.exclude(pk=piece.pk).first()
check(new_piece is not None, '生成了新作品')
check(new_piece and '和声小调' in new_piece.title,
      f'新作品标题带目标调（{new_piece.title if new_piece else "无"}）')
check(new_piece and new_piece.is_public is False, '新作品默认私有')
check(new_piece and bool(new_piece.musicxml_file), '新作品带 MusicXML')
if new_piece and new_piece.musicxml_file:
    with new_piece.musicxml_file.open('rb') as fh:
        xml = fh.read()
    got = notes_of(xml)
    check(got == ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G#4', 'A4'],
          f'新作品谱面是 A 和声小调：{got}')

# 原作品没被动过
piece.refresh_from_db()
with piece.musicxml_file.open('rb') as fh:
    orig = fh.read()
check(notes_of(orig) == ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'],
      '原作品谱面未被修改')

print('4) 兜底判定（没有调性分析结果时用 music21）')
p2 = Piece.objects.create(owner=admin, title='兜底曲', source_type='musicxml')
p2.musicxml_file.save('t2.musicxml',
                      ContentFile(make_xml('A4 B4 C5 D5 E5 F5 G5 A5'.split())),
                      save=True)
body = c.get(f'/pieces/{p2.id}/transpose/').content.decode()
check('A 小调' in body or 'A 自然小调' in body, 'music21 兜底判出 A 小调')
check('大调' in body, '小调源给出大调选项')

print('5) 权限')
co = Client()
co.force_login(other)
check(co.get(f'/pieces/{piece.id}/transpose/').status_code == 404,
      '别人的私有作品：转调页 404')
piece.is_public = True
piece.save(update_fields=['is_public'])
check(co.get(f'/pieces/{piece.id}/transpose/').status_code == 200,
      '公开作品的转调页可进')

# 清理
for p in Piece.objects.all():
    if p.musicxml_file:
        p.musicxml_file.delete(save=False)
    if p.midi_file:
        p.midi_file.delete(save=False)
Piece.objects.all().delete()
User.objects.all().delete()

print('\n结果：' + ('全部通过' if not FAILS else f'{len(FAILS)} 项失败'))
sys.exit(1 if FAILS else 0)
