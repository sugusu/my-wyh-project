from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_training_command_has_no_gt_mesh_reference():
    text=(ROOT/'scripts/01_baseline.sh').read_text()
    train=text.split('TRAIN=(',1)[1].split('RENDER=(',1)[0]
    forbidden=('scene_mesh','gt_mesh','meshes/','coordinate_audit','ICP','icp')
    assert not any(x in train for x in forbidden)
def test_no_future_stage_modules_implemented():
    forbidden=('depth_sweep.py','counterfactual_score.py','relocate.py','optimizer_reset.py','refine.py')
    assert not any((ROOT/'trgr'/x).exists() for x in forbidden)

