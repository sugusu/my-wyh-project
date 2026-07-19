#!/usr/bin/env python3
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; model=ROOT/'checkpoints/baseline/scene_01'
train=(ROOT/'logs/scene01_baseline_train.log').read_text(errors='replace') if (ROOT/'logs/scene01_baseline_train.log').exists() else ''
mesh=model/'mesh/tsdf_fusion_post_30000.ply'; ckpt=model/'point_cloud/iteration_30000/point_cloud.ply'
nan=bool(re.search(r'(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])',train,re.I))
metric=re.search(r'\[ITER 30000\] Evaluating test: L1 ([0-9.e+-]+) PSNR ([0-9.e+-]+) SSIM ([0-9.e+-]+) LPIPS ([0-9.e+-]+)',train)
geometry=json.loads((model/'mesh/results_30000.json').read_text())
header=ckpt.read_bytes()[:4096].decode('ascii',errors='ignore')
count=int(re.search(r'element vertex (\d+)',header).group(1))
train_renders=len(list((model/'train/ours_30000/renders').glob('*')))
test_renders=len(list((model/'test/ours_30000/renders').glob('*')))
report={"stage":0,"status":"PASS" if ckpt.stat().st_size>0 and mesh.stat().st_size>0 and not nan and metric and train_renders==350 and test_renders==50 else "FAIL",
        "iteration_30000_checkpoint":str(ckpt),"checkpoint_bytes":ckpt.stat().st_size,
        "mesh":str(mesh),"mesh_bytes":mesh.stat().st_size,"nan_or_inf_in_train_log":nan,
        "gaussian_count":count,"train_render_count":train_renders,"test_render_count":test_renders,
        "test_metrics":{"l1":float(metric.group(1)),"psnr":float(metric.group(2)),"ssim":float(metric.group(3)),"lpips":float(metric.group(4))},
        "geometry_metrics":geometry,
        "command_file":str(model/'command.txt'),"normal_folder":"normals",
        "config_sha256":hashlib.sha256((ROOT/'configs/scene01_dev.yaml').read_bytes()).hexdigest()}
(ROOT/'reports/stage0_baseline_parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
