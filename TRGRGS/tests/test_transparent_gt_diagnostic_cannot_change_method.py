import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_gt_diagnostic_is_blocked_before_frozen_probes():
 d=json.loads((ROOT/'outputs/scene_01/probe_views.json').read_text());assert d['gt_used'] is False;assert d['probe_ids']==[]
def test_method_modules_do_not_read_gt():
 src=(ROOT/'trgr/formal_depth_sweep.py').read_text().lower();assert 'scene_mesh' not in src and 'material.011' not in src

