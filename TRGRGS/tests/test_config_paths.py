from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def test_paths_and_links_exist():
    cfg=yaml.safe_load((ROOT/'configs/scene01_dev.yaml').read_text())
    assert Path(cfg['paths']['project_root']) == ROOT
    for key in ('tsgs_root','data_root','scene_path','gt_mesh'):
        assert Path(cfg['paths'][key]).exists(), key
    assert cfg['baseline']['normal_folder'] == 'normals'
    assert cfg['experiment']['allowed_gpus'] == [2,3]

