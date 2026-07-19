import json,numpy as np
from pathlib import Path
from trgr.foreground_coordinate_audit_r2 import audit_core_r2
ROOT=Path(__file__).resolve().parents[1]
def test_artificial_shift_is_detected():
 p=ROOT/'outputs/scene_01/foreground_audit_r2';a=np.load(p/'activated_gaussians_from_tsgs.npz');idx=np.load(p/'model_surface_core_indices.npy');m=np.load(p/'model_surface_core_metrics.npz')
 base=audit_core_r2(a['xyz'],idx,a['opacity'],np.bincount(idx,weights=m['mask_support'],minlength=len(a['xyz'])),ROOT/'data/translab/scene_01/meshes/scene_mesh.obj')
 shifted=a['xyz'].copy();shifted[idx,0]+=.10*.5481526628412927
 bad=audit_core_r2(shifted,idx,a['opacity'],np.bincount(idx,weights=m['mask_support'],minlength=len(a['xyz'])),ROOT/'data/translab/scene_01/meshes/scene_mesh.obj')
 assert bad['gt_surface_distance']['p90']>base['gt_surface_distance']['p90'];assert bad['failed_checks']
