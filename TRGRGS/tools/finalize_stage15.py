#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np,trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trgr.camera_geometry import load_colmap_cameras
from trgr.transparent_ray_intersections import list_positive_intersections
from trgr.stage15_finalize_metrics import stats,reliability
scene=ROOT/'data/translab/scene_01';probe=json.load(open(ROOT/'outputs/scene_01/probe_views_r2.json'))['probe_ids'];cams={c.name:c for c in load_colmap_cameras(scene,ROOT/'third_party/TSGS')}
mesh=trimesh.load(scene/'meshes/scene_mesh.obj',force='mesh',process=False);v=np.asarray(mesh.vertices,np.float32);v=np.stack([v[:,0],-v[:,2],v[:,1]],1);f=np.asarray(mesh.faces,np.int32);rawdir=ROOT/'outputs/scene_01/stage1_gt_diagnostic/raw_intersections';rawdir.mkdir(parents=True,exist_ok=True)
all_single=[];all_oracle=[];all_disp=[];views=[]
for name in probe:
 z=np.load(ROOT/'outputs/scene_01/depth_sweep'/f'{Path(name).stem}.npz');m=z['transparent_mask'].astype(bool)&(z['num_valid_hypotheses']>=3);ys,xs=np.nonzero(m);c=cams[name];K=c.K.copy();K[:2]/=2.;dirs_cam=np.stack([(xs-K[0,2])/K[0,0],(ys-K[1,2])/K[1,1],np.ones(len(xs))],1);R=c.world_to_camera[:3,:3];center=-R.T@c.world_to_camera[:3,3];vals,offs=list_positive_intersections(v,f,np.repeat(center[None],len(xs),0),dirs_cam@R,c.world_to_camera,1e-5);np.savez_compressed(rawdir/f'{Path(name).stem}.npz',x=xs,y=ys,values=vals,offsets=offs)
 pred=z['depths'][:,ys,xs].astype(np.float32);base=z['baseline_default_depth'][ys,xs].astype(np.float32);disp=z['threshold_conditioned_depth_dispersion'][ys,xs];single=[];oracle=[];dd=[]
 for i in np.flatnonzero(np.diff(offs)>0):
  q=vals[offs[i]:offs[i+1]];single.append(np.min(np.abs(base[i]-q)));oracle.append(np.min(np.abs(pred[:,i,None]-q[None,:])));dd.append(disp[i])
 single=np.asarray(single);oracle=np.asarray(oracle);dd=np.asarray(dd);all_single.extend(single);all_oracle.extend(oracle);all_disp.extend(dd);views.append({'view':name,'baseline':stats(single),'oracle':stats(oracle),'mean_improvement_ratio':float(1-oracle.mean()/single.mean()),'median_improvement_ratio':float(1-np.median(oracle)/np.median(single)),'dispersion':reliability(dd,single)})
S=np.asarray(all_single);O=np.asarray(all_oracle);D=np.asarray(all_disp);bs,os=stats(S),stats(O);utility={'status':'PASS_HYPOTHESIS_UTILITY' if (1-os['median']/bs['median']>=.1 or 1-os['mean']/bs['mean']>=.1 or os['within_0.005']-bs['within_0.005']>=.05) else 'FAIL_HYPOTHESIS_UTILITY','baseline':bs,'all_hypothesis_oracle':os,'mean_improvement_ratio':float(1-O.mean()/S.mean()),'median_improvement_ratio':float(1-np.median(O)/np.median(S)),'unweighted_8view_mean':{'baseline_error':float(np.mean([x['baseline']['mean'] for x in views])),'oracle_error':float(np.mean([x['oracle']['mean'] for x in views]))},'views':views}
rel=reliability(D,S);passed=rel['spearman']>=.2 or rel['dispersion_auprc']>=1.15*rel['random_auprc'] or rel['top_10pct_error_rate']>=rel['global_error_rate']+.1;rel.update({'status':'PASS_DISPERSION_RELIABILITY' if passed else 'FAIL_DISPERSION_RELIABILITY','views':[{ 'view':x['view'],**x['dispersion']} for x in views]})
(ROOT/'reports/stage15_hypothesis_utility.json').write_text(json.dumps(utility,indent=2)+'\n');(ROOT/'reports/stage15_dispersion_reliability.json').write_text(json.dumps(rel,indent=2)+'\n');status='FAIL_HYPOTHESIS_UTILITY' if utility['status'].startswith('FAIL') else ('PASS_HYPOTHESIS_AND_DISPERSION' if passed else 'PASS_HYPOTHESIS_ONLY');decision={'status':status,'stage2a_authorized':not status.startswith('FAIL'),'dispersion_allowed_in_consensus':passed,'gt_used_only_for_diagnostic_metrics':True};(ROOT/'reports/stage15_final_decision.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision,indent=2))
