#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'compat'),str(ROOT/'third_party/TSGS')]
from scene.cameras import Camera
from gaussian_renderer import render_depth
class ControlledGaussians:
 active_sh_degree=0;max_sh_degree=0
 def __init__(self):
  self.xyz=torch.tensor([[0.,0.,1.],[0.,0.,1.01],[0.,0.,2.]],device='cuda')
  self.opacity=torch.tensor([[.2],[.5],[.4]],device='cuda')
  self.scaling=torch.full((3,3),np.log(.02),device='cuda')
  self.rotation=torch.tensor([[1.,0.,0.,0.]]*3,device='cuda')
 @property
 def get_xyz(self):return self.xyz
 @property
 def get_opacity(self):return self.opacity
 @property
 def get_scaling(self):return self.scaling.exp()
 @property
 def get_rotation(self):return self.rotation
 def get_normal(self,cam):return torch.tensor([[0.,0.,-1.]]*3,device='cuda')
pc=ControlledGaussians();im=Image.fromarray(np.zeros((64,64,3),np.uint8));cam=Camera(0,np.eye(3),np.zeros(3),1.,1.,64,64,'controlled.png',im,'controlled',0,data_device='cuda',transparencies_map=np.ones((64,64),np.float32));pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False);kw=dict(transparencies_map=cam.transparencies_map,start_threshold=0.,end_threshold=.7,window_size=.03,override_color=torch.zeros((3,3),device='cuda'))
a=render_depth(cam,pc,pipe,torch.zeros(3,device='cuda'),**kw)['out_transparency_depth'].detach().cpu().numpy();b=render_depth(cam,pc,pipe,torch.zeros(3,device='cuda'),**kw)['out_transparency_depth'].detach().cpu().numpy();d=float(a.reshape(64,64)[32,32]);expected=(.2*1.+.4*1.01)/.6
report={'status':'PASS_RENDERER_SEMANTICS' if 1.-1e-5<=d<=1.01+1e-5 and np.array_equal(a,b) else 'FAIL_RENDERER_SEMANTICS','controlled_scene_depths':[1.,1.01,2.],'controlled_opacities':[.2,.5,.4],'start_threshold':0.,'end_threshold':.7,'window_size':.03,'cpu_reference_close_cluster_depth':expected,'cuda_center_depth':d,'output_inside_selected_window':bool(1.-1e-5<=d<=1.01+1e-5),'repeat_exact':bool(np.array_equal(a,b))}
(ROOT/'reports/stage1_renderer_cuda_semantics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['status']=='PASS_RENDERER_SEMANTICS' else 2)
