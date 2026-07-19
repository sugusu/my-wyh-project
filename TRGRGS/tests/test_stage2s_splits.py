import json
from pathlib import Path
ROOT=Path(__file__).parents[1]
def test_split_manifests_are_fixed_and_disjoint():
 a=json.load(open(ROOT/'outputs/scene_01/stage2s/split_a_manifest.json'));b=json.load(open(ROOT/'outputs/scene_01/stage2s/split_b_manifest.json'));assert len(a['train'])==len(b['train'])==300 and len(a['holdout'])==len(b['holdout'])==50 and not set(a['holdout'])&set(b['holdout'])
def test_official_test_not_in_splits():
 names=sorted(p.name for p in (ROOT/'data/translab/scene_01/images').glob('*'));test=set(names[::8]);a=json.load(open(ROOT/'outputs/scene_01/stage2s/split_a_manifest.json'));assert not test&set(a['train']+a['holdout'])
