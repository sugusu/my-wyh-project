import inspect
from trgr.model_surface_core import select_model_surface_core
def test_core_selector_is_gt_free():
 sig=inspect.signature(select_model_surface_core);assert 'gt' not in ' '.join(sig.parameters).lower();src=inspect.getsource(select_model_surface_core).lower();assert 'gt_mesh' not in src and 'scene_mesh' not in src
def test_thresholds_are_model_only():
 src=inspect.getsource(select_model_surface_core);assert '.02*scene_diameter' in src and 'model_mesh_path' in src

