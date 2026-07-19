import inspect
from trgr.view_selection import select_probe_views
def test_probe_selector_has_no_gt():
 assert 'mesh' not in inspect.signature(select_probe_views).parameters
 assert 'mesh' not in inspect.getsource(select_probe_views).lower()
