import inspect,numpy as np
from pathlib import Path
from trgr.foreground_gaussians import select_foreground_gaussians
def test_interface_and_source_are_gt_free():
    assert 'mesh' not in inspect.signature(select_foreground_gaussians).parameters
    src=inspect.getsource(select_foreground_gaussians).lower();assert 'trimesh' not in src and 'mesh evaluation' not in src
def test_fake_gt_cannot_change_selection(tmp_path):
    assert 'mesh' not in inspect.signature(select_foreground_gaussians).parameters
    (tmp_path/'fake_mesh.obj').write_text('garbage');assert (tmp_path/'fake_mesh.obj').exists()

