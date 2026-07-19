import inspect
from trgr.view_selection_r2 import select_probe_views_r2
def test_no_gt_and_no_largest_gate():
 s=inspect.getsource(select_probe_views_r2).lower();assert 'mesh' not in s and "largest_component_ratio']<=" not in s

