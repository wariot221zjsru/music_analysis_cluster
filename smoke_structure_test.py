"""音乐结构的回归脚本：乐句层 / 主题层 / 曲式层三层分离。

起因（用户提问）
----------------
「曲式'主题'和乐曲结构是不一样的吗？分开分析好还是一起分析好？」
「曲式功能为何最好和前两层做在一起？请分析」
「音乐结构分析部分，乐句层和主题层，按您说的改。在结构分析的页面上，
  直接加一个 是否分析曲式 的开关。这次直接在页面的 音乐结构窗口 上加。」

于是结构分析拆成三层单向依赖：

  乐句层 → 边界在哪（终止式 / 长音 / 稀疏），**留全部候选**，不只留一种切法
  主题层 → 这段材料是谁（动机 + 节奏型 + 音级分布），**留全部两两证据**，不急着定死
  曲式层 → 这段在干什么（主题序列 + 调性 + 终止式 + 长度均衡），全局打分挑模板

并且做一轮反馈：曲式假设若含"再现"，把"相似度不够但调性回归"的段提升为同一主题。

本脚本把这些钉死：

  A. 主题指纹：移调 / 倒影 / 逆行 / 节奏轴 / 碎片 / 无关材料；
  B. 聚类：强阈值 / 弱阈值 / 碎片开关；
  C. 曲式打分：ABA 带再现 / ABACA 回旋 / 一部 / 二部，以及"调性没回来"要扣分；
  D. 反馈回路：调性回归才提升，"假再现"（材料像、调性没回）不许提升；
  E. 端到端：同一份谱，曲式开关开 / 关两种输出；关闭时不该调大模型。

用法：python smoke_structure_test.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django  # noqa: E402

django.setup()

from core.analysis import structure as smod  # noqa: E402
from core.analysis import theme as tmod  # noqa: E402
from core.analysis.scoredata import parse_score  # noqa: E402

PASS = 0
FAIL = 0
FAILS = []


def check(name, cond, extra=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  [PASS] {name}')
    else:
        FAIL += 1
        FAILS.append(name)
        print(f'  [FAIL] {name}  {extra}')


def notes_of(pitches, durs=None):
    """[(offset, midi, duration)]，按给定顺序铺开。"""
    durs = durs or [1.0] * len(pitches)
    out, t = [], 0.0
    for p, d in zip(pitches, durs):
        out.append((t, p, d))
        t += d
    return out


# ---------------- A 主题指纹 ----------------

def test_fingerprint():
    print('\n[A] 主题指纹：三条轴各自干自己的活')

    base = [72, 74, 76, 77, 79]
    a = tmod.fingerprint(notes_of(base))

    # A1 移调（模进）：音高轴与音级轴都该满，节奏轴在等时值句里退成中性 0.5
    up = tmod.fingerprint(notes_of([p + 7 for p in base]))
    c1 = tmod.compare(a, up)
    check('A1 移调后仍是同一主题（音高轴满分、音级轴移调不变）',
          c1['motif'] > 0.95 and c1['pc'] > 0.95
          and c1['score'] > tmod.THEME_STRONG and c1['kind'] == '原型',
          f"score={c1['score']} motif={c1['motif']} pc={c1['pc']}")
    check('A1b 等时值句的节奏轴退成中性值（没信息就不该给满分）',
          c1['rhythm'] == 0.5, str(c1['rhythm']))

    # A2 倒影：音程取反、顺序不变
    inv = tmod.fingerprint(notes_of([72, 70, 68, 67, 65]))
    c2 = tmod.compare(a, inv)
    check('A2 倒影被认出来（不是"新主题"）',
          c2['kind'] == '倒影' and c2['motif'] > 0.95,
          f"kind={c2['kind']} motif={c2['motif']}")

    # A3 逆行：音高序列反转（音程取反且反序）
    retro = tmod.fingerprint(notes_of(list(reversed(base))))
    c3 = tmod.compare(a, retro)
    check('A3 逆行被认出来（与倒影区分开）',
          c3['kind'] == '逆行' and c3['motif'] > 0.95,
          f"kind={c3['kind']} motif={c3['motif']}")

    # A4 节奏轴：同音高、节奏完全不同 → 明显掉分（老实现这里会给满分）
    b = tmod.fingerprint(notes_of(base, [1.0, 4.0, 0.5, 2.0, 1.0]))
    c4 = tmod.compare(a, b)
    check('A4 节奏完全不同 → 节奏轴掉到 0，总分够不上强阈值',
          c4['rhythm'] < 0.2 and c4['score'] < tmod.THEME_STRONG,
          f"rhythm={c4['rhythm']} score={c4['score']} thr={tmod.THEME_STRONG}")
    check('A4b 但音高轴仍是满的（说明掉分确实来自节奏）',
          c4['motif'] > 0.95, f"motif={c4['motif']}")

    # A5 时值整体缩放（放慢一倍）→ 节奏型不变
    # 用非等时值的句子测，否则节奏轴本来就退化成中性值，测不出缩放不变性
    uneven = [1.0, 2.0, 0.5, 2.0, 1.0]
    x1 = tmod.fingerprint(notes_of(base, uneven))
    x2 = tmod.fingerprint(notes_of(base, [d * 2 for d in uneven]))
    c5 = tmod.compare(x1, x2)
    check('A5 整体放慢一倍不算变形（比值不变）',
          c5['rhythm'] > 0.95 and c5['score'] > 0.95,
          f"rhythm={c5['rhythm']} score={c5['score']}")

    # A6 碎片：长句里取开头 40%
    long_notes = notes_of([60, 62, 64, 65, 67, 69, 71, 72, 74, 76])
    frag = tmod.fingerprint(long_notes[:4])
    c6 = tmod.compare(tmod.fingerprint(long_notes), frag)
    check('A6 短句几乎整条落进长句 → 标碎片',
          c6['fragment'] is True and c6['cover'] > 0.9,
          f"fragment={c6['fragment']} cover={c6['cover']} ratio={c6['ratio']}")

    # A7 无关材料
    far = tmod.fingerprint(notes_of([60, 71, 61, 70, 62]))
    c7 = tmod.compare(a, far)
    check('A7 无关材料分数低（不会被并成同一主题）',
          c7['score'] < 0.55, f"score={c7['score']}")
    return a, b, frag


def test_labels(a, b):
    print('\n[B] 聚类：证据留着，阈值可以换')

    fps = [a, a, b]
    got = tmod.build_labels(tmod.links_of(fps), 3, tmod.THEME_STRONG)
    # 标签带「第几次出现」的撇号（A / A' / A''），比较主题身份要 rstrip
    check('B1 前两句同一材料、第三句不同 → A A B',
          [x.rstrip("'") for x in got['labels']] == ['A', 'A', 'B'],
          str(got['labels']))
    check('B1b 主题清单记下出现位置',
          got['themes'][0]['occurrences'] == [1, 2],
          str(got['themes'][0]))

    # 同一批证据换弱阈值：节奏不同那两句会被并起来
    weak = tmod.build_labels(tmod.links_of(fps), 3, tmod.THEME_WEAK)
    check('B2 换成弱阈值 → 第三句被并进 A（证据没变，只是判得更宽）',
          [x.rstrip("'") for x in weak['labels']] == ['A', 'A', 'A'],
          str(weak['labels']))

    # 碎片开关
    long_notes = notes_of([60, 62, 64, 65, 67, 69, 71, 72])
    fps2 = [tmod.fingerprint(long_notes), tmod.fingerprint(long_notes[:-4])]
    on = tmod.build_labels(tmod.links_of(fps2), 2, tmod.THEME_STRONG,
                           allow_fragment=True)
    off = tmod.build_labels(tmod.links_of(fps2), 2, tmod.THEME_STRONG,
                            allow_fragment=False)
    check('B3 认碎片时并入（展开部的动机碎片不是新主题）',
          [x.rstrip("'") for x in on['labels']] == ['A', 'A'], str(on['labels']))
    check('B3b 不认碎片时分开（严格口径可关掉这条）',
          off['labels'] == ['A', 'B'], str(off['labels']))


# ---------------- C 曲式打分 ----------------

def _sec(bars, cad, key):
    return {'bars': bars, 'cadence': cad, 'key': key, 'theme': 'A',
            'start_measure': 1, 'end_measure': bars}


def test_form():
    print('\n[C] 曲式：主题序列 + 调性 + 终止式 + 长度均衡，四轴全局打分')

    T = '完全正格终止（V-I）'
    H = '半终止（停在 V）'

    # C1 标准 ABA：末段回主调
    secs = [_sec(8, T, 'C 大调'), _sec(8, H, 'G 大调'), _sec(8, T, 'C 大调')]
    f1 = smod._infer_form(['A', 'B', 'A'], secs)
    check('C1 ABA 且末段回主调 → 单三部曲式',
          '单三部' in f1['name'] and f1['has_return'] is True, f1['name'])
    check('C1b 调性轴拿满分（材料回归 + 调性回归）',
          f1['candidates'][0]['parts']['key'] == 1.0,
          str(f1['candidates'][0]['parts']))

    # C2 假再现：材料回来了、调性没回来 → 调性轴必须扣分
    secs2 = [_sec(8, T, 'C 大调'), _sec(8, H, 'G 大调'), _sec(8, T, 'a 小调')]
    check('C2 末段调性没回主调 → 调性轴只剩 0.25（假再现扣分）',
          smod._key_fit(True, secs2) == 0.25,
          str(smod._key_fit(True, secs2)))
    f2 = smod._infer_form(['A', 'B', 'A'], secs2)
    aba2 = next(c for c in f2['candidates'] if '单三部' in c['name'])
    aba1 = next(c for c in f1['candidates'] if '单三部' in c['name'])
    check('C2b 同一个模板，假再现的总分低于真再现',
          aba2['score'] < aba1['score'],
          f"{aba2['score']} vs {aba1['score']}")
    check('C2c 但段数对得上，ABA 仍不该输给两段的 AB',
          '单三部' in f2['name'], f2['name'])

    # C3 回旋
    secs3 = [_sec(8, T, 'C'), _sec(8, H, 'G'), _sec(8, T, 'C'),
             _sec(8, H, 'F'), _sec(8, T, 'C')]
    f3 = smod._infer_form(['A', 'B', 'A', 'C', 'A'], secs3)
    check('C3 ABACA → 回旋曲式', '回旋' in f3['name'], f3['name'])

    # C4 单一主题
    f4 = smod._infer_form(['A'], [_sec(16, T, 'C')])
    check('C4 只有一个主题 → 一部曲式', '一部' in f4['name'], f4['name'])

    # C5 两个主题
    f5 = smod._infer_form(['A', 'B'], [_sec(8, T, 'C'), _sec(8, T, 'G')])
    check('C5 AB → 单二部曲式', '单二部' in f5['name'], f5['name'])

    # C6 候选是可解释的：每个模板都带四项分数
    check('C6 候选模板全部带四轴分数（不是黑箱）',
          len(f5['candidates']) == len(smod._TEMPLATES)
          and all(set(c['parts']) == {'material', 'key', 'cadence', 'balance'}
                  for c in f5['candidates']),
          str(f5['candidates'][:1]))
    check('C6b 编辑距离：完全一致为 0、少一段为 1',
          smod._edit_distance(['A', 'B', 'A'], ['A', 'B', 'A']) == 0
          and smod._edit_distance(['A', 'B'], ['A', 'B', 'A']) == 1)


# ---------------- D 反馈回路 ----------------

def test_feedback():
    print('\n[D] 反馈：材料不够、但调性回来了才算再现')

    base = [72, 74, 76, 77, 79]
    a = tmod.fingerprint(notes_of(base))
    # 变形稍大：音高轮廓保留、节奏改掉一部分 → 强阈值够不上、弱阈值能过
    var = tmod.fingerprint(notes_of(base, [1.0, 2.0, 0.5, 2.0, 1.0]))
    other = tmod.fingerprint(notes_of([60, 71, 61, 70, 62]))
    fps = [a, other, var]
    links = tmod.links_of(fps)
    got = tmod.build_labels(links, 3, tmod.THEME_STRONG)
    leaders = {}
    for g in got['groups']:
        for m in g['members']:
            leaders[m] = g['leader']
    check('D0 前提：变形句在强阈值下是独立主题',
          got['labels'] == ['A', 'B', 'C'], str(got['labels']))

    # D1 调性回归 → 提升
    new, promoted = smod._apply_feedback(
        links, 3, got['labels'], ['C 大调', 'G 大调', 'C 大调'], leaders)
    check('D1 材料相似度不足但调性回归 → 提升为同一主题',
          new == ['A', 'B', 'A'] and len(promoted) == 1,
          f"{new} / {promoted}")
    check('D1b 改判理由带回来了',
          promoted and '调性' in promoted[0]['reason'], str(promoted))

    # D2 调性没回来（假再现）→ 不许提升
    new2, promoted2 = smod._apply_feedback(
        links, 3, got['labels'], ['C 大调', 'G 大调', 'a 小调'], leaders)
    check('D2 材料像但调性没回来 → 不提升（假再现不算再现）',
          new2 == ['A', 'B', 'C'] and not promoted2,
          f"{new2} / {promoted2}")


# ---------------- E 端到端 ----------------

def _build_score(rows, bars):
    """每小节一个音高列表，四分音符铺满。"""
    from music21 import chord as m21chord
    from music21 import instrument, meter, note, stream

    sc = stream.Score()
    p = stream.Part()
    p.append(instrument.Violin())
    p.append(meter.TimeSignature('4/4'))
    for m in range(bars):
        meas = stream.Measure(number=m + 1)
        for x in rows[m]:
            if isinstance(x, (list, tuple)):
                meas.append(m21chord.Chord(list(x), quarterLength=1.0))
            else:
                meas.append(note.Note(x, quarterLength=1.0))
        meas.makeRests(fillGaps=True, inPlace=True)
        p.append(meas)
    sc.append(p)
    return sc


def test_end_to_end():
    print('\n[E] 端到端：同一份谱，曲式开关开 / 关')

    ph_a = [72, 74, 76, 77]
    # B 段刻意用不同的音程结构（+1 +2 +1）：如果只用 A 的下移八度，
    # 从"音高动机 + 音级集合"看它本来就是同一材料，测不出分段能力。
    ph_b = [60, 61, 63, 64]
    rows = [ph_a, ph_a,      # m1-2  句1  主题 A
            ph_a, ph_a,      # m3-4  句2  主题 A（重复）
            ph_b, ph_b,      # m5-6  句3  主题 B
            ph_a, ph_a]      # m7-8  句4  再现 A
    sc = _build_score(rows, len(rows))

    tmp = tempfile.mkdtemp(prefix='smoke_struct_')
    path = os.path.join(tmp, 'in.musicxml')
    sc.write('musicxml', fp=path)
    data = parse_score(path)

    # 终止式落在偶数小节（奇数小节 V、偶数小节 I）→ 每 2 小节一句
    chords_out = {'progression': [
        {'label': f'第 {m} 小节', 'chord': 'C',
         'function': 'I' if m % 2 == 0 else 'V'}
        for m in range(1, len(rows) + 1)]}
    tonality_out = {'main_key': 'C 大调', 'regions': [
        {'start_measure': 1, 'end_measure': 8, 'key': 'C 大调'}]}
    rhythm_out = {'measure_count': len(rows), 'tempos': [120],
                  'time_signatures': ['4/4']}

    # 不调模型
    smod._llm_structure = lambda payload, cfg: {
        'form_name': '单三部曲式（ABA）', 'confidence': 0.7}

    cfg_on = SimpleNamespace(structure_form_enabled=True,
                             dsh_profile='headless', dsh_timeout=30)
    out = smod.structure(data, cfg_on, rhythm_out, chords_out, tonality_out)

    check('E1 三层都产出了：乐句 / 主题 / 乐段',
          out['phrases'] and out['themes'] and out['sections'],
          f"{len(out['phrases'])}/{len(out['themes'])}/{len(out['sections'])}")
    check('E2 主题清单给出了"谁在哪出现"',
          out['themes'][0]['occurrences']
          and out['themes'][0]['count'] >= 2,
          str(out['themes'][0]))
    check('E3 乐句带主题标签',
          all('theme' in p for p in out['phrases']),
          str(out['phrases'][:1]))
    check('E4 乐段按主题合并（连续同主题 → 一段）',
          all(s['theme'] for s in out['sections']),
          str([(s['label'], s['theme'], s['start_measure'], s['end_measure'])
               for s in out['sections']][:4]))
    check('E5 曲式开关开着 → 有结论、有候选分数',
          out['form']['enabled'] is True
          and out['form']['candidates']
          and out['form']['name'],
          str(out['form']['name']))
    check('E6 乐句边界候选也留着（切法不止一种）',
          len(out['phrase_boundaries']) >= len(out['phrases']),
          str(len(out['phrase_boundaries'])))
    check('E7 主题参数写进输出（可复现 / 可对照）',
          out['theme_params']['threshold'] and 'weights' in out['theme_params'],
          str(out['theme_params']))

    cfg_off = SimpleNamespace(structure_form_enabled=False,
                              dsh_profile='headless', dsh_timeout=30)
    out2 = smod.structure(data, cfg_off, rhythm_out, chords_out, tonality_out)
    check('E8 开关关掉 → form.enabled=False，名字说明未分析',
          out2['form']['enabled'] is False
          and '未分析' in out2['form']['name'],
          str(out2['form']['name']))
    check('E9 关掉后不调大模型（llm_analysis 为空）',
          out2['llm_analysis'] is None, str(out2['llm_analysis']))
    check('E10 关掉后乐句 / 主题照样有（只是不推断曲式）',
          out2['phrases'] and out2['themes'] and out2['sections'],
          f"{len(out2['phrases'])}/{len(out2['themes'])}")
    check('E11 关掉后没有候选模板分数（那一层没算）',
          out2['form']['candidates'] == [], str(out2['form']['candidates']))

    # 曲式层不动切分与指纹：关掉只是少了"曲式"这一层结论
    check('E12 关掉时不做反馈改判（feedback.applied=False）',
          out2['form']['feedback']['applied'] is False,
          str(out2['form']['feedback']))
    check('E13 曲式开关不改乐句切分（分层互不污染）',
          [(p['start_measure'], p['end_measure']) for p in out['phrases']]
          == [(p['start_measure'], p['end_measure']) for p in out2['phrases']],
          str([(p['start_measure'], p['end_measure']) for p in out['phrases']]))
    check('E14 主题数量在两种口径下一致（反馈只会合并、不会新增）',
          len(out['themes']) <= len(out2['themes']),
          f"{len(out['themes'])} vs {len(out2['themes'])}")
    check('E15 四个乐句、三个乐段（A 两句合并 + B + 再现 A）',
          out2['phrase_count'] == 4 and len(out2['sections']) == 3,
          f"{out2['phrase_count']} 句 / {len(out2['sections'])} 段")
    # 主题清单必须与乐句标签对得上（反馈改判过 labels 时最容易脱节）
    labs = [p['theme_base'] for p in out['phrases']]
    occ = []
    for t in out['themes']:
        occ += t['occurrences']
    check('E16 主题清单与乐句标签一致（不会出现"清单里没有 A"）',
          sorted(occ) == list(range(1, len(labs) + 1))
          and {t['id'] for t in out['themes']} == set(labs),
          f"清单 {[t['id'] for t in out['themes']]} vs 标签 {labs}")


def main():
    print('=== 音乐结构：三层分离 / 曲式开关 ===')
    a, b, _frag = test_fingerprint()
    test_labels(a, b)
    test_form()
    test_feedback()
    test_end_to_end()
    print(f'\n=== 完成：{PASS} 项通过，{FAIL} 项失败 ===')
    if FAIL:
        for x in FAILS:
            print('  失败：', x)
        sys.exit(1)


if __name__ == '__main__':
    main()
