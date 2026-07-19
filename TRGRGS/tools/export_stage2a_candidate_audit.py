#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trgr.camera_geometry import load_colmap_cameras
from trgr.hypothesis_reprojection import pixels_depth_to_world,world_to_pixels,local_set_residual
scene=ROOT/'data/translab/scene_01';names=json.load(open(ROOT/'outputs/scene_01/probe_views_r2.json'))['probe_ids'];cams={c.name:c for c in load_colmap_cameras(scene,ROOT/'third_party/TSGS')};D={}
for n in names:
 z=np.load(ROOT/'outputs/scene_01/depth_sweep'/f'{Path(n).stem}.npz');K=cams[n].K.copy();K[:2]/=2.;D[n]={'depths':z['depths'],'mask':z['transparent_mask'].astype(bool),'K':K}
rng=np.random.default_rng(42);records=[]
for si,n in enumerate(names):
 dep=D[n]['depths'];tau,y,x=np.nonzero(np.isfinite(dep)&(dep>0)&D[n]['mask'][None]);order=rng.permutation(len(x));made=0
 for ii in order:
  target=names[(si+1+(made%7))%8]
  if target==n:continue
  world=pixels_depth_to_world(np.array([x[ii]]),np.array([y[ii]]),np.array([dep[tau[ii],y[ii],x[ii]]]),D[n]['K'],cams[n].world_to_camera);uv,z=world_to_pixels(world,D[target]['K'],cams[target].world_to_camera);rr,bp,bt,bd=local_set_residual(uv,z,D[target]['depths'],D[target]['mask'])
  if not np.isfinite(rr[0]):continue
  tx,ty=bp[0];records.append({'source_view':n,'source_pixel':[int(x[ii]),int(y[ii])],'source_tau_index':int(tau[ii]),'source_depth':float(dep[tau[ii],y[ii],x[ii]]),'world_point':world[0].tolist(),'target_view':target,'target_projected_pixel':uv[0].tolist(),'target_projected_depth':float(z[0]),'target_local_hypotheses':D[target]['depths'][:,ty,tx].astype(float).tolist(),'matched_target_pixel':[int(tx),int(ty)],'matched_target_tau_index':int(bt[0]),'matched_target_depth':float(bd[0]),'residual':float(rr[0])});made+=1
  if made==125:break
assert len(records)==1000
(ROOT/'outputs/scene_01/stage2a_consensus/summary/audit_candidates_1000.json').write_text(json.dumps(records,indent=2)+'\n');print(len(records))
