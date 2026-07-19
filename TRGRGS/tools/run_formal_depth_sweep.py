#!/usr/bin/env python3
import sys,json,cv2,numpy as np,torch
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'compat'),str(ROOT/'third_party/TSGS')]
from trgr.camera_geometry import load_colmap_cameras
from trgr.transparency_mask_audit import renderer_transparency_tensor
from trgr.formal_depth_sweep import *
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera as TCamera
from gaussian_renderer import render_depth
from types import SimpleNamespace
scene_path=ROOT/'data/translab/scene_01';model=ROOT/'checkpoints/baseline/scene_01';outdir=ROOT/'outputs/scene_01/depth_sweep';outdir.mkdir(parents=True,exist_ok=True)
probe=json.load(open(ROOT/'outputs/scene_01/probe_views_r2.json'));assert probe['status'] in ('PASS_GT_FREE_PROBE_SELECTION','PASS_LIMITED_CANDIDATE_POOL') and len(probe['probe_ids'])==8
allcams={c.name:c for c in load_colmap_cameras(scene_path,ROOT/'third_party/TSGS')};g=GaussianModel(3,24);g.load_ply(str(model/'point_cloud/iteration_30000/point_cloud.ply'));pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False);bg=torch.zeros(3,device='cuda')
repeat={'valid_masks_identical':True,'max_depth_difference':0.,'nonfinite_count':0};records=[]
for name in probe['probe_ids']:
 c=allcams[name];R=c.world_to_camera[:3,:3].T;T=c.world_to_camera[:3,3];fovx=2*np.arctan(c.width/(2*c.K[0,0]));fovy=2*np.arctan(c.height/(2*c.K[1,1]));raw,trans,mask=renderer_transparency_tensor(scene_path,name)
 image=Image.open(scene_path/'images'/name)
 # dataset_readers multiplies the thresholded transparency PNG by image alpha
 # before Camera/process_image performs its bilinear resize.
 raw_trans=(raw/255>.5).astype(np.float32)
 if len(image.split())>3: raw_trans*=np.asarray(image.split()[3],np.float32)/255.
 cam=TCamera(c.name,R,T,float(fovx),float(fovy),800,800,str(scene_path/'images'/name),image,Path(name).stem,list(allcams).index(name),data_device='cuda',transparencies_map=raw_trans)
 actual_branch=(cam.transparencies_map.squeeze().detach().cpu().numpy()>.5)
 if not np.array_equal(actual_branch,mask): raise RuntimeError(f'renderer transparency mismatch: {name}')
 def one(start,end):return render_depth(cam,g,pipe,bg,transparencies_map=cam.transparencies_map,start_threshold=float(start),end_threshold=float(end),window_size=.03)['out_transparency_depth'].squeeze().detach().cpu().numpy().astype(np.float32)
 depths=[]
 for s,e in zip(STARTS,ENDS):
  a,b=one(s,e),one(s,e);va=np.isfinite(a)&(a>0);vb=np.isfinite(b)&(b>0);repeat['valid_masks_identical']&=bool(np.array_equal(va,vb));finite=np.isfinite(a)&np.isfinite(b);diff=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else float('inf');repeat['max_depth_difference']=max(repeat['max_depth_difference'],diff);repeat['nonfinite_count']+=int((~np.isfinite(a)).sum()+(~np.isfinite(b)).sum());depths.append(a)
 default=one(0,.2);depths=np.stack(depths);metrics=compute_dispersion(depths);payload={'depths':depths.astype(np.float16),'taus':TAUS,'start_thresholds':STARTS,'end_thresholds':ENDS,'baseline_default_depth':default.astype(np.float16),'baseline_default_valid':(np.isfinite(default)&(default>0)).astype(np.uint8),'transparent_mask':mask.astype(np.uint8),'view_name':np.array(name),'renderer_semantics_name':np.array(RENDERER_SEMANTICS_NAME),**metrics};np.savez_compressed(outdir/f'{Path(name).stem}.npz',**payload);records.append((name,depths,default,mask,metrics))

def colorize(a, valid=None):
 a=np.asarray(a,np.float32);valid=(np.isfinite(a)&(a>0)) if valid is None else valid
 out=np.zeros((*a.shape,3),np.uint8)
 if valid.any():
  lo,hi=np.quantile(a[valid],[.02,.98]);x=np.clip((a-lo)/(max(hi-lo,1e-8)),0,1);out=cv2.applyColorMap((x*255).astype(np.uint8),cv2.COLORMAP_TURBO);out[~valid]=0
 return out
for name,d,default,m,x in records:
 tiles=[colorize(default),cv2.cvtColor((m*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)]
 tiles += [colorize(q) for q in d]
 tiles += [colorize(x['depth_spread_abs'],m>0),colorize(x[DISPERSION_NAME],m>0)]
 grid=np.vstack([np.hstack(tiles[:5]),np.hstack(tiles[5:10])])
 cv2.imwrite(str(outdir/f'{Path(name).stem}_depth_grid.png'),grid)
# aggregate GT-free gate
inside=[];disps=[];spreads=[];tau_cov=np.zeros(6);zeros=0;total=0;nonfinite=0
for name,d,default,m,x in records:
 pix=m>0;method=pix&(x['num_valid_hypotheses']>=3);inside.append(method.sum()/max(pix.sum(),1));disps.extend(x[DISPERSION_NAME][method]);spreads.extend(x['depth_spread_abs'][method]);tau_cov+=x['valid'][:,pix].mean(1);zeros+=int((x['num_valid_hypotheses'][pix]==0).sum());total+=int(pix.sum());nonfinite+=int((~np.isfinite(d)).sum())
tau_cov/=8;disps=np.asarray(disps);spreads=np.asarray(spreads);coverage=float(np.mean(inside));sens=float((spreads>.0010963053256825854).mean()) if len(spreads) else 0.;failed=[]
if not repeat['valid_masks_identical'] or repeat['max_depth_difference']>1e-6:failed.append('FAIL_RENDER_NONDETERMINISTIC')
if coverage<.9:failed.append('FAIL_DEPTH_VALIDITY')
if not len(disps) or disps.std()<=1e-4:failed.append('FAIL_DISPERSION_COLLAPSE')
if nonfinite:failed.append('FAIL_NONFINITE')
if zeros/max(total,1)>=.05:failed.append('FAIL_DEPTH_VALIDITY')
if sens<.05:failed.append('FAIL_THRESHOLD_INSENSITIVE')
if (tau_cov>=.8).sum()<4:failed.append('FAIL_HYPOTHESIS_COVERAGE')
status='PASS_GT_FREE_DEPTH_SWEEP' if not failed else failed[0]
report={'status':status,'failed_checks':sorted(set(failed)),'depth_sweep_executed':True,'probe_ids':probe['probe_ids'],'parameters':{'taus':TAUS.tolist(),'half_window':.025,'window_size':.03,'baseline_default':{'start_threshold':0.,'end_threshold':.2,'window_size':.03}},'repeatability':repeat,'transparent_mask_three_hypothesis_coverage':coverage,'dispersion':{'mean':float(disps.mean()) if len(disps) else 0,'std':float(disps.std()) if len(disps) else 0,'p50':float(np.median(disps)) if len(disps) else 0,'p90':float(np.quantile(disps,.9)) if len(disps) else 0,'p95':float(np.quantile(disps,.95)) if len(disps) else 0},'all_zero_ratio':zeros/max(total,1),'nonfinite_ratio':nonfinite/(8*6*800*800),'threshold_sensitivity_ratio':sens,'tau_valid_coverage':tau_cov.tolist(),'renderer_semantics_name':RENDERER_SEMANTICS_NAME,'renderer_semantics':{'controlled_cuda_reference_match':None,'output_inside_selected_window':None,'note':'The production CUDA renderer was executed; a synthetic per-contributor CUDA reference harness is not exposed by TSGS.'},'monotonicity_gate':False,'gt_used':False}
(ROOT/'reports/stage1_gt_free_depth_sweep_r2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if status=='PASS_GT_FREE_DEPTH_SWEEP' else 2)
