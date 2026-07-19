import inspect
import trgr.hypothesis_consensus as c
def test_consensus_has_no_gt_dependency():
 s=inspect.getsource(c).lower();assert 'mesh' not in s and 'intersection' not in s
