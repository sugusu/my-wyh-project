#!/usr/bin/env python3
import hashlib,json,re
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];out={};failed=[]
def ply_count(p):
 with open(p,'rb') as f:
  for line in f:
   s=line.decode('ascii').strip()
   if s.startswith('element vertex '):return int(s.split()[-1])
   if s=='end_header':break
 return 0
for tag in ('a','b'):
 model=ROOT/f'checkpoints/split_{tag}/scene_01';log=(model/'training_log.log').read_text(errors='replace');matches=re.findall(r'\[ITER 30000\] Evaluating test: L1 ([\d.eE+-]+) PSNR ([\d.eE+-]+) SSIM ([\d.eE+-]+) LPIPS ([\d.eE+-]+)',log);assert matches;L,P,S,LP=map(float,matches[-1]);rd=model/'test/ours_30000/renders';gd=model/'test/ours_30000/gt';tl=[];sq=[]
 for p in sorted(rd.glob('*.png')):
  a=np.asarray(Image.open(p).convert('RGB'),np.float32)/255.;b=np.asarray(Image.open(gd/p.name).convert('RGB'),np.float32)/255.;raw=np.asarray(Image.open(ROOT/f'data/stage2s/split_{tag}/scene_01/transparent_masks'/p.name).convert('L'),np.float32)/255.;m=cv2.resize((raw>.5).astype(np.uint8),(a.shape[1],a.shape[0]),interpolation=cv2.INTER_LINEAR)>.5;d=np.abs(a-b)[m];tl.append(d.mean());sq.append(np.mean((a[m]-b[m])**2))
 ply=model/'point_cloud/iteration_30000/point_cloud.ply';mesh=model/'mesh/tsdf_fusion_post_30000.ply';n=ply_count(ply);finite=all(np.isfinite(x) for x in [L,P,S,LP,*tl,*sq]);rec={'status':'PASS' if ply.is_file() and mesh.stat().st_size>0 and finite and P>=35 and S>=.98 and LP<=.05 and n>=10000 else 'FAIL','holdout_count':len(tl),'L1':L,'PSNR':P,'SSIM':S,'LPIPS':LP,'transparent_mask_L1':float(np.mean(tl)),'transparent_mask_PSNR':float(np.mean([-10*np.log10(x+1e-12) for x in sq])),'gaussian_count':n,'mesh_bytes':mesh.stat().st_size,'finite':finite,'checkpoint_sha256':hashlib.sha256(ply.read_bytes()).hexdigest(),'mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((ROOT/f'configs/split_{tag}_scene01.yaml').read_bytes()).hexdigest()};out[f'split_{tag}']=rec
 if rec['status']!='PASS':failed.append(tag)
report={'status':'PASS_SPLIT_MODEL_PARITY' if not failed else 'FAIL_SPLIT_MODEL_PARITY','failed_splits':failed,**out};(ROOT/'reports/stage2s_split_model_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if not failed else 2)
