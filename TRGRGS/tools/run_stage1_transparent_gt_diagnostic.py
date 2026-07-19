#!/usr/bin/env python3
"""Post-hoc GT diagnostic. It never changes probes, sweep arrays, or method state."""
import json,sys
from pathlib import Path
import numpy as np, trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trgr.camera_geometry import load_colmap_cameras
from trgr.transparent_ray_intersections import list_positive_intersections
from trgr.stage1_transparent_gt_diagnostic import assert_diagnostic_output
gtfree=json.load(open(ROOT/'reports/stage1_gt_free_depth_sweep_r2.json'))
if gtfree.get('status')!='PASS_GT_FREE_DEPTH_SWEEP':raise SystemExit('BLOCKED: GT-free depth sweep R2 did not pass')
probe=json.load(open(ROOT/'outputs/scene_01/probe_views_r2.json'))
scene=ROOT/'data/translab/scene_01';out=ROOT/'outputs/scene_01/stage1_gt_diagnostic';assert_diagnostic_output(out,ROOT);out.mkdir(parents=True,exist_ok=True)
mesh_path=scene/'meshes/scene_mesh.obj';mesh=trimesh.load(mesh_path,force='mesh',process=False)
# Audited official inverse Blender-to-COLMAP convention from Stage 0.5: (x,y,z)->(x,-z,y).
v=np.asarray(mesh.vertices,np.float32);v=np.stack([v[:,0],-v[:,2],v[:,1]],1);f=np.asarray(mesh.faces,np.int32)
cams={c.name:c for c in load_colmap_cameras(scene,ROOT/'third_party/TSGS')};rng=np.random.default_rng(42);rows=[]
for name in probe['probe_ids']:
 z=np.load(ROOT/'outputs/scene_01/depth_sweep'/f'{Path(name).stem}.npz');mask=z['transparent_mask'].astype(bool);ys,xs=np.nonzero(mask)
 take=np.arange(len(xs)) if len(xs)<=20000 else np.sort(rng.choice(len(xs),20000,replace=False));xs=xs[take];ys=ys[take]
 c=cams[name];K=c.K.copy();K[:2]/=2.;cam_dirs=np.stack([(xs-K[0,2])/K[0,0],(ys-K[1,2])/K[1,1],np.ones(len(xs))],1)
 R=c.world_to_camera[:3,:3];center=-R.T@c.world_to_camera[:3,3];orig=np.repeat(center[None],len(xs),0);dirs=cam_dirs@R
 vals,offs=list_positive_intersections(v,f,orig,dirs,c.world_to_camera,tolerance=1e-5)
 pred=z['depths'][:,ys,xs].astype(np.float32);base=z['baseline_default_depth'][ys,xs].astype(np.float32);has=np.diff(offs)>0
 best=np.full(len(xs),np.nan,np.float32);berr=np.full(len(xs),np.nan,np.float32);nint=np.diff(offs)
 for i in np.flatnonzero(has):
  q=vals[offs[i]:offs[i+1]];best[i]=np.min(np.abs(pred[:,i,None]-q[None,:]));berr[i]=np.min(np.abs(base[i]-q))
 valid=has&np.isfinite(best)&np.isfinite(berr);rows.append({'view':name,'sampled_mask_rays':int(len(xs)),'rays_with_gt_intersection':int(has.sum()),'intersection_coverage':float(has.mean()),'multi_intersection_ratio':float((nint>1).mean()),'sweep_best_abs_error_mean':float(np.mean(best[valid])),'baseline_abs_error_mean':float(np.mean(berr[valid])),'sweep_improves_baseline_ratio':float(np.mean(best[valid]<berr[valid]))})
 (out/f'{Path(name).stem}.json').write_text(json.dumps(rows[-1],indent=2)+'\n')
report={'status':'PASS_TRANSPARENT_GT_DIAGNOSTIC_COMPLETED','diagnostic_only':True,'gt_read_after_gt_free_pass':True,'method_outputs_modified':False,'probe_ids_modified':False,'gt_mesh':str(mesh_path.resolve()),'coordinate_transform':'inverse Blender-to-COLMAP: (x,y,z)->(x,-z,y)','ray_sampling':{'seed':42,'maximum_per_view':20000},'views':rows,'aggregate':{'intersection_coverage_mean':float(np.mean([r['intersection_coverage'] for r in rows])),'multi_intersection_ratio_mean':float(np.mean([r['multi_intersection_ratio'] for r in rows])),'sweep_improves_baseline_ratio_mean':float(np.mean([r['sweep_improves_baseline_ratio'] for r in rows]))}}
(ROOT/'reports/stage1_transparent_gt_diagnostic_r2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
