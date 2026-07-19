#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trgr.camera_geometry import load_colmap_cameras,world_to_camera,project
assert json.load(open(ROOT/'reports/stage2s_split_model_parity.json'))['status']=='PASS_SPLIT_MODEL_PARITY'
def load(p):
 v=PlyData.read(p)['vertex'].data;names=v.dtype.names;arr=lambda n:np.asarray(v[n],np.float32);xyz=np.stack([arr('x'),arr('y'),arr('z')],1);op=1/(1+np.exp(-arr('opacity')));sc=np.exp(np.stack([arr('scale_0'),arr('scale_1'),arr('scale_2')],1));return xyz,op,sc
models={'full':ROOT/'checkpoints/baseline/scene_01/point_cloud/iteration_30000/point_cloud.ply','split_a':ROOT/'checkpoints/split_a/scene_01/point_cloud/iteration_30000/point_cloud.ply','split_b':ROOT/'checkpoints/split_b/scene_01/point_cloud/iteration_30000/point_cloud.ply'};xyz,op,sc=load(models['full']);probe=json.load(open(ROOT/'outputs/scene_01/probe_views_r2.json'))['probe_ids'];cams={c.name:c for c in load_colmap_cameras(ROOT/'data/translab/scene_01',ROOT/'third_party/TSGS')};num=np.zeros(len(xyz),np.int16);sw=np.zeros(len(xyz));sd=np.zeros(len(xyz));mx=np.zeros(len(xyz));means=np.zeros(len(xyz));disp_all=[]
for n in probe:
 z=np.load(ROOT/'outputs/scene_01/depth_sweep'/f'{Path(n).stem}.npz');disp=z['threshold_conditioned_depth_dispersion'];mask=z['transparent_mask'].astype(bool);c=cams[n];K=c.K.copy();K[:2]/=2.;pc=world_to_camera(xyz,c.world_to_camera);uv,depth=project(pc,K);x=np.rint(uv[:,0]).astype(int);y=np.rint(uv[:,1]).astype(int);inside=(depth>0)&(x>=0)&(x<800)&(y>=0)&(y<800);idx=np.flatnonzero(inside);idx=idx[mask[y[idx],x[idx]]];radius=np.max(sc[idx],1)*K[0,0]/depth[idx];p95=np.quantile(radius,.95);w=op[idx]*np.clip(radius,0,p95);d=disp[y[idx],x[idx]];valid=np.isfinite(d);idx=idx[valid];w=w[valid];d=d[valid];num[idx]+=1;sw[idx]+=w;sd[idx]+=w*d;means[idx]+=d;mx[idx]=np.maximum(mx[idx],d);disp_all.extend(d)
prior=sd/(sw+1e-8);mean=means/np.maximum(num,1);eligible=np.flatnonzero(num>=3);cut=np.median(prior[eligible]);cand=eligible[prior[eligible]>=cut];low=eligible[np.argsort(prior[eligible],kind='stable')[:len(cand)]];od=ROOT/'outputs/scene_01/stage2s/regions';od.mkdir(parents=True,exist_ok=True);np.savez_compressed(od/'dispersion_prefilter.npz',support_count=num,dispersion_prior=prior,mean_dispersion=mean,max_dispersion=mx,opacity=op,scale=sc,xyz=xyz,candidate_indices=cand,low_control_indices=low)
# Balanced KD-tree leaves with root bounds covering all full-model Gaussians.
leaves=[(cand,xyz.min(0)-1e-6,xyz.max(0)+1e-6,[])]
while len(leaves)<64:
 i=max(range(len(leaves)),key=lambda q:len(leaves[q][0]));ids,lo,hi,path=leaves.pop(i);axis=int(np.argmax(np.ptp(xyz[ids],axis=0)));order=ids[np.argsort(xyz[ids,axis],kind='stable')];mid=len(order)//2;cut=float((xyz[order[mid-1],axis]+xyz[order[mid],axis])/2);h1=hi.copy();h1[axis]=cut;l2=lo.copy();l2[axis]=cut;leaves.extend([(order[:mid],lo,h1,path+[[axis,cut,0]]),(order[mid:],l2,hi,path+[[axis,cut,1]])])
regions=[]
for i,(ids,lo,hi,path) in enumerate(leaves):regions.append({'region_id':i,'min_bound':lo.tolist(),'max_bound':hi.tolist(),'candidate_count':len(ids),'path':path,'dispersion_prior_mean':float(prior[ids].mean())})
(od/'coarse_regions.json').write_text(json.dumps({'region_count':64,'construction':'balanced KD-tree on frozen full-baseline candidates','regions':regions},indent=2)+'\n')
for name,p in models.items():
 q,_,_=load(p);a=np.full(len(q),-1,np.int16)
 for r in regions:
  lo=np.array(r['min_bound']);hi=np.array(r['max_bound']);hit=np.all((q>=lo)&(q<=hi),1)&(a<0);a[hit]=r['region_id']
 counts=np.bincount(a[a>=0],minlength=64);label='full_model' if name=='full' else name;np.savez_compressed(od/f'{label}_assignments.npz',region_id=a,counts=counts,insufficient=counts<32)
print(json.dumps({'full_gaussians':len(xyz),'eligible':len(eligible),'candidates':len(cand),'low_controls':len(low),'regions':64},indent=2))
