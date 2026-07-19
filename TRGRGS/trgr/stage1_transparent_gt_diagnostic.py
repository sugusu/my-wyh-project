"""Post-hoc GT diagnostic. This module is forbidden from writing method configs or sweep NPZs."""
from pathlib import Path
ALLOWED_OUTPUT_SUBDIR='outputs/scene_01/stage1_gt_diagnostic'
def assert_diagnostic_output(path,root):
    target=(Path(root)/ALLOWED_OUTPUT_SUBDIR).resolve();p=Path(path).resolve()
    if p != target and target not in p.parents:raise ValueError('GT diagnostic output escaped isolated directory')
