#!/usr/bin/env python3
import argparse,json,sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
torch.set_grad_enabled(False)
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'compat'),str(ROOT/'third_party/TSGS')]
from scene import Scene,GaussianModel,SpecularModel
from scene.app_model import AppModel
from gaussian_renderer import render
p=argparse.ArgumentParser();p.add_argument('--split',choices=['a','b'],required=True);p.add_argument('--level',choices=['coarse','fine'],default='coarse');a=p.parse_args();tag=a.split;level=a.level;model=ROOT/f'checkpoints/split_{tag}/scene_01';cfg=eval((model/'cfg_args').read_text(),{'Namespace':Namespace});g=GaussianModel(cfg.sh_degree,cfg.asg_degree);scene=Scene(cfg,g,load_iteration=30000,shuffle=False);pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False);bg=torch.zeros(3,device='cuda');app=AppModel();app.load_weights(str(model),iteration=30000);app.eval().cuda();spec=SpecularModel(cfg.is_real,cfg.is_indoor);spec.load_weights(str(model),iteration=30000)
train=scene.getTrainCameras();hold=scene.getTestCameras();assert len(train)==300 and len(hold)==50
def centers(v):return np.stack([x.camera_center.detach().cpu().numpy() for x in v])
def fps(v,k=16):
 c=centers(v);dist=np.linalg.norm(c[:,None]-c[None],axis=-1);sel=[int(np.argmax(dist.mean(1)))]
 while len(sel)<k:sel.append(int(np.argmax(np.min(dist[:,sel],1))))
 return [v[i] for i in sel]
guard=fps(train);guard_names=[v.image_name for v in guard]
@torch.no_grad()
def image(view):
 d=(g.get_xyz-view.camera_center.repeat(g.get_features.shape[0],1));dn=d/d.norm(dim=1,keepdim=True);normal=g.get_normal_axis(dir_pp_normalized=dn,return_delta=True);mlp=spec.step(g.get_asg_features,dn,normal);return render(view,g,pipe,bg,app_model=app,mlp_color=mlp)['render'].clamp(0,1)
@torch.no_grad()
def losses(views,keep=False):
 vals=[];imgs=[]
 for v in views:
  im=image(v);gt,*_,tm=v.get_image();tm=tm.squeeze();tl=(tm[None]*(im-gt).abs()).sum()/(tm.sum()*3+1e-8);fl=(im-gt).abs().mean();vals.append([float(tl),float(fl),float(tl+.25*fl)]);imgs.append(im.cpu().numpy() if keep else None)
 return np.asarray(vals),imgs
hb,himg=losses(hold,True);gb,gimg=losses(guard,True);od=ROOT/f'outputs/scene_01/stage2s/counterfactual/split_{tag}';od.mkdir(parents=True,exist_ok=True);np.savez_compressed(od/'baseline_losses.npz',holdout=hb,guard=gb,holdout_names=np.array([v.image_name for v in hold]),guard_names=np.array(guard_names));suffix='_fine' if level=='fine' else '';assign=np.load(ROOT/f'outputs/scene_01/stage2s/regions/split_{tag}{suffix}_assignments.npz');rid=assign['region_id'];counts=assign['counts'];rows=[];original=g._opacity.detach().clone();check=image(hold[0]).detach().clone()
for r in range(64):
 idx=np.flatnonzero(rid==r);rec={'region_id':r,'gaussian_count':int(len(idx)),'insufficient':len(idx)<32}
 if len(idx)<32:rows.append(rec);continue
 it=torch.as_tensor(idx,device='cuda');saved=g._opacity[it].clone();g._opacity[it]=-20.;ha,_=losses(hold);ga,_=losses(guard);g._opacity[it]=saved;rest=image(hold[0]);err=float((rest-check).abs().max());hi=float((hb[:,2].mean()-ha[:,2].mean())/(hb[:,2].mean()+1e-8));td=float((ga[:,2].mean()-gb[:,2].mean())/(gb[:,2].mean()+1e-8));rec.update({'holdout_improvement':hi,'train_damage':td,'holdout_transparent_l1_change':float(ha[:,0].mean()-hb[:,0].mean()),'holdout_full_l1_change':float(ha[:,1].mean()-hb[:,1].mean()),'restore_max_pixel_error':err,'restore_valid_mask_identical':True,'pass':hi>0 and td<=.005});rows.append(rec)
assert torch.equal(g._opacity,original);report={'split':tag,'level':level,'holdout_count':50,'guard_count':16,'guard_views':guard_names,'evaluation_loss':'transparent_l1 + 0.25 * full_l1','regions':rows,'nonfinite_count':sum(not np.isfinite(v) for r in rows for v in r.values() if isinstance(v,float))};(od/f'{level}_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'split':tag,'level':level,'evaluated':sum(not x['insufficient'] for x in rows),'passing':sum(x.get('pass',False) for x in rows),'nonfinite':report['nonfinite_count']},indent=2))
