"""临时诊断：chant_gate_score 的窗级归属到底谁赢了、为什么。"""
import os
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import django  # noqa: E402

django.setup()

from core.analysis import melody_line as ml_mod  # noqa: E402
from core.analysis import voice_id as vmod  # noqa: E402
from core.analysis.scoredata import parse_score  # noqa: E402
from smoke_chant_line_test import banuwa_score, chant_gate_score, _write, make_piece, prepare  # noqa: E402
from core.models import SystemConfig  # noqa: E402
from django.contrib.auth.models import User  # noqa: E402

prepare()
cfg = SystemConfig.get_config()
user = User.objects.get(username='smoke')


def diag(score, tag):
    piece = make_piece(user, tag, _write(score()))
    data = parse_score(piece.musicxml_file.path)
    low = vmod.low_register_parts(data)
    print(f'===== {tag}  low={sorted(low)}  names={data.part_names}')
    for start, end in ((1, 4), (5, 8)):
        mo, _, bq = ml_mod.build_measure_map(data)
        s = mo.get(start, 0.0)
        e = mo.get(end + 1, s + 4 * bq)
        ws = vmod.window_scores(data, s, e)['per_part']
        print(f'-- 窗 {start}-{end}')
        for d in sorted(ws, key=lambda x: -x.get('salience', 0)):
            if d.get('empty'):
                continue
            idx = d['part_index']
            seq = ml_mod._seg_pitch_seq(data, idx, s, e)
            cr = ml_mod.chord_ratio(data, idx, s, e)
            print(f"   {data.part_names[idx]:12s} sal={d.get('salience')}"
                  f" pitch_n={round(d.get('pitch_n', 0),3)} dur_n={round(d.get('dur_n',0),3)}"
                  f" motif_n={round(d.get('motif_n',0),3)} sparse_n={round(d.get('sparse_n',0),3)}"
                  f" | flat_kind={ml_mod.flat_kind(seq, cr)} ratio={cr}"
                  f" ull chord_ratio nolike={not ml_mod.looks_like_line(seq)}")


diag(chant_gate_score, 'diag-gate')
diag(banuwa_score, 'diag-banuwa')
