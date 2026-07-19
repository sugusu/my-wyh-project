from pathlib import Path
def assert_isolated(path,root):
 target=(Path(root)/'outputs/scene_01/stage2a_gt_diagnostic').resolve();p=Path(path).resolve()
 if p!=target and target not in p.parents:raise ValueError('GT diagnostic escaped isolated directory')
