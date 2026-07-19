#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
ROOT=Path(__file__).resolve().parents[1];rd=ROOT/'outputs/scene_01/stage2s/regions';pref=np.load(rd/'dispersion_prefilter.npz');xyz=pref['xyz'];cand=pref['candidate_indices'];prior=pref['dispersion_prior'];full_coarse=np.load(rd/'full_model_assignments.npz')['region_id'];selected=list(range(16));fine=[]
for cr in selected:
 ids=cand[full_coarse[cand]==cr];parts=[ids]
 while len(parts)<4:
  i=max(range(len(parts)),key=lambda j:len(parts[j]));q=parts.pop(i);axis=int(np.argmax(np.ptp(xyz[q],axis=0)));order=q[np.argsort(xyz[q,axis],kind='stable')];parts.extend([order[:len(order)//2],order[len(order)//2:]])
 for q in parts:fine.append({'fine_region_id':len(fine),'coarse_region_id':cr,'min_bound':xyz[q].min(0).tolist(),'max_bound':xyz[q].max(0).tolist(),'candidate_count':len(q),'dispersion_prior_mean':float(prior[q].mean())})
(rd/'fine_regions.json').write_text(json.dumps({'selected_coarse_regions':selected,'tie_break':'coarse region ID ascending because all symmetric scores are zero','fine_region_count':64,'regions':fine},indent=2)+'\n')
def loadxyz(p):
 v=PlyData.read(p)['vertex'].data;return np.stack([v['x'],v['y'],v['z']],1)
for name,p in {'full':ROOT/'checkpoints/baseline/scene_01/point_cloud/iteration_30000/point_cloud.ply','split_a':ROOT/'checkpoints/split_a/scene_01/point_cloud/iteration_30000/point_cloud.ply','split_b':ROOT/'checkpoints/split_b/scene_01/point_cloud/iteration_30000/point_cloud.ply'}.items():
 x=loadxyz(p);a=np.full(len(x),-1,np.int16)
 for r in fine:
  lo=np.array(r['min_bound'])-1e-7;hi=np.array(r['max_bound'])+1e-7;hit=np.all((x>=lo)&(x<=hi),1)&(a<0);a[hit]=r['fine_region_id']
 counts=np.bincount(a[a>=0],minlength=64);np.savez_compressed(rd/f'{name}_fine_assignments.npz',region_id=a,counts=counts,insufficient=counts<32)
print(json.dumps({'fine_regions':64,'selected_coarse':selected},indent=2))
