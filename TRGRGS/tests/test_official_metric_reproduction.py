import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_official_metrics_reproduced_within_protocol_tolerance():
 d=json.loads((ROOT/'reports/stage06_r3_official_reproduction.json').read_text());e=d['errors'];assert d['status']=='PASS';assert e['f1_abs']<=1e-4 and e['precision_abs']<=1e-4 and e['recall_abs']<=1e-4;assert max(e['overall_rel'],e['d2s_rel'],e['s2d_rel'])<=1e-3

