#!/usr/bin/env python3
import sys,json,hashlib,subprocess,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trgr.camera_geometry import load_colmap_cameras
from trgr.foreground_gaussians import select_mask_cone_candidates,write_xyz_ply
from trgr.model_surface_core import select_model_surface_core
from trgr.foreground_coordinate_audit_r2 import audit_core_r2
out=ROOT/'outputs/scene_01/foreground_audit_r2';out.mkdir(parents=True,exist_ok=True);scene=ROOT/'data/translab/scene_01'
a=np.load(out/'activated_gaussians_from_tsgs.npz');xyz,scales,opacity,rotation=[a[k] for k in ('xyz','scales','opacity','rotation')]
cams=load_colmap_cameras(scene,ROOT/'third_party/TSGS');cams=[c for i,c in enumerate(cams) if i%8!=0]
idx,sup,eroded=select_mask_cone_candidates(xyz,cams,scene/'transparent_masks')
np.save(out/'mask_cone_candidate_indices.npy',idx);np.save(out/'mask_cone_support_count.npy',sup);np.save(out/'mask_cone_eroded_support_count.npy',eroded);write_xyz_ply(out/'mask_cone_candidates.ply',xyz[idx])
ms={'status':'MASK_CONE_SELECTION_NOT_SURFACE_SPECIFIC','mask_cone_candidate_count':len(idx),'center_offset_ratio_historical':.22045,'raw_bbox_diagonal_ratio_historical':2.51773,'selection_uses_gt':False,'erosion_pixels':3};(out/'mask_cone_stats.json').write_text(json.dumps(ms,indent=2)+'\n')
metrics,stats=select_model_surface_core(xyz,scales,opacity,rotation,idx,sup,eroded,ROOT/'checkpoints/baseline/scene_01/mesh/tsdf_fusion_post_30000.ply')
np.save(out/'model_surface_core_indices.npy',metrics['indices']);np.savez_compressed(out/'model_surface_core_metrics.npz',**metrics);write_xyz_ply(out/'model_surface_core_gaussians.ply',xyz[metrics['indices']]);(out/'model_surface_core_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
# Only now, after core indices are persisted, may the GT mesh be opened.
report=audit_core_r2(xyz,metrics['indices'],opacity,sup,scene/'meshes/scene_mesh.obj');report.update({'mask_cone_stats':ms,'model_surface_core_stats':stats,'config_sha256':hashlib.sha256((ROOT/'configs/scene01_dev.yaml').read_bytes()).hexdigest()})
def git_info(path):
 try:
  commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip();top=subprocess.check_output(['git','-C',str(path),'rev-parse','--show-toplevel'],text=True).strip();return {'commit':commit,'git_status':'ok','toplevel':top}
 except subprocess.CalledProcessError:return {'commit':None,'git_status':'not_a_git_repository','toplevel':None}
report['git']={'tsgs':git_info(ROOT/'third_party/TSGS'),'trgrgs':git_info(ROOT)}
(ROOT/'reports/stage06_r2_model_surface_coordinate_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
raise SystemExit(0 if report['status']=='PASS' else 2)

