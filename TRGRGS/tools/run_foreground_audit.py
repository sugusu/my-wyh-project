#!/usr/bin/env python3
import json,sys,hashlib,subprocess,platform,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trgr.camera_geometry import load_colmap_cameras
from trgr.foreground_gaussians import *
from trgr.foreground_coordinate_audit import audit_foreground
scene=ROOT/'data/translab/scene_01'; ply=ROOT/'checkpoints/baseline/scene_01/point_cloud/iteration_30000/point_cloud.ply';out=ROOT/'outputs/scene_01/foreground_audit';out.mkdir(parents=True,exist_ok=True)
xyz=load_gaussian_xyz(ply); cams=load_colmap_cameras(scene,ROOT/'third_party/TSGS'); cams=[c for i,c in enumerate(cams) if i%8!=0]
idx,support=select_foreground_gaussians(xyz,cams,scene/'transparent_masks')
np.save(out/'foreground_indices.npy',idx);np.save(out/'foreground_support_count.npy',support);write_xyz_ply(out/'foreground_gaussians.ply',xyz[idx])
stats={'gaussian_count':len(xyz),'camera_count':len(cams),'foreground_count':len(idx),'min_support':3,'chunk_size':8192,'selection_uses_gt':False}
(out/'selection_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
# GT is read only after immutable selection artifacts above have been saved.
report=audit_foreground(xyz,idx,scene/'meshes/scene_mesh.obj');report.update(stats)
report['config_sha256']=hashlib.sha256((ROOT/'configs/scene01_dev.yaml').read_bytes()).hexdigest()
def git_commit(path):
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None
report['environment']={'python':sys.version,'pytorch':torch.__version__,'cuda':torch.version.cuda,'gpu':'physical GPU 2',
                       'tsgs_git_commit':git_commit((ROOT/'third_party/TSGS').resolve()),'trgrgs_git_commit':git_commit(ROOT)}
(ROOT/'reports/stage06_foreground_coordinate_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
raise SystemExit(0 if report['status']=='PASS' else 2)
