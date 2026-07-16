#!/usr/bin/env python3
"""Stage 3.2.5: Representation-dependent optical drift confirmation"""
import sys, os, csv, math, hashlib, json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation"
os.makedirs(f"{OUTPUT}", exist_ok=True)

sys.path.insert(0, "/data/wyh/repos/TSGS")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, f"{BASE}/benchmark")

import torch, trimesh
from torch.nn import functional as F
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from deformations.twist import deform_points as twist_def

device = "cuda"
log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

# ═══════════════════════════════════════════════════════════
# 0. Setup
# ═══════════════════════════════════════════════════════════
mesh = trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N=len(mesh.vertices)
verts=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
spacing=1.5/40; scale_p=torch.full((N,3),spacing,device=device); scale_p[:,2]=spacing*0.1
rot_init=torch.zeros(N,4,device=device); rot_init[:,0]=1.0

cam_cfgs=[{"pos":[0,-3.5,1.5],"target":[0,0,0],"up":[0,0,1],"id":0},
          {"pos":[3.0,0,2.0],"target":[0,0,0],"up":[0,0,1],"id":4},
          {"pos":[0,3.5,1.5],"target":[0,0,0],"up":[0,0,-1],"id":8}]

def build_cam(cfg):
    pa=np.array(cfg["pos"],dtype=np.float32); ta=np.array(cfg["target"],dtype=np.float32); ua=np.array(cfg["up"],dtype=np.float32)
    fwd=ta-pa; fwd/=np.linalg.norm(fwd)
    rt=np.cross(ua,fwd); rt/=np.linalg.norm(rt); nu=np.cross(fwd,rt)
    Rw=np.eye(3,dtype=np.float32); Rw[0,:]=rt; Rw[1,:]=nu; Rw[2,:]=fwd
    T=-Rw@pa; R=Rw.T
    fx=256/(2*math.tan(math.radians(45/2)))
    cam=Camera(colmap_id=cfg["id"],R=R,T=T,FoVx=focal2fov(fx,256),FoVy=focal2fov(fx,256),image_width=256,image_height=256,image_path="",image_PIL=None,image_name=f"cam_{cfg['id']:03d}",uid=cfg["id"],preload_img=False,data_device="cpu")
    cam.original_image=torch.zeros(3,256,256); return cam

film_cams=[build_cam(c) for c in cam_cfgs]
bg_color=torch.zeros(3,device=device)
pipe=type('obj',(object,),{"debug":False,"convert_SHs_python":False,"compute_cov3D_python":False})()

class Adapter:
    def __init__(self,xyz,scale,rot,tau_raw,color_raw):
        self._xyz=xyz; self._scaling=torch.log(scale.clamp(min=1e-8))
        self._rotation=rot; self._tau_raw=tau_raw; self._color_raw=color_raw
        self.active_sh_degree=0; self.max_sh_degree=0; self.use_app=False
    @property
    def get_xyz(self): return self._xyz
    @property
    def get_scaling(self): return torch.exp(self._scaling)
    @property
    def get_rotation(self): return self._rotation/self._rotation.norm(dim=1,keepdim=True).clamp(min=1e-8)
    @property
    def get_opacity(self): return 1-torch.exp(-F.softplus(self._tau_raw))
    @property
    def get_transparency(self): return torch.full((self._xyz.shape[0],1),0.5,device=self._xyz.device)
    @property
    def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)

def white_pass(adapter,cam):
    r2=render(cam,adapter,pipe,bg_color,app_model=None,override_color=torch.ones_like(torch.sigmoid(adapter._color_raw)),return_plane=False,return_depth_normal=False)
    return r2["render"].mean(dim=0,keepdim=True).clamp(0,1), r2["radii"]

# ═══════════════════════════════════════════════════════════
# 1. Build frozen checkpoint (train canonical model)
# ═══════════════════════════════════════════════════════════
hdr("1. Training canonical checkpoint")
tr_can=torch.full((N,1),0.0,device=device,requires_grad=True)
cr_can=torch.zeros(N,3,device=device,requires_grad=True)
opt=torch.optim.Adam([tr_can,cr_can],lr=1e-2)
GT_DYN=f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/repaired_render/tau1.0_dynamic"
BG=f"{BASE}/experiments/stage3_2_fixed_optical_necessity/background_only"

for it in range(5000):
    opt.zero_grad(); loss=0
    for ci,cam in enumerate(film_cams):
        cid=[0,4,8][ci]
        adpt=Adapter(verts,scale_p,rot_init,tr_can,cr_can)
        r1=render(cam,adpt,pipe,bg_color,app_model=None,override_color=torch.sigmoid(cr_can),return_plane=False,return_depth_normal=False)
        C=r1["render"]
        r2=render(cam,adpt,pipe,bg_color,app_model=None,override_color=torch.ones_like(torch.sigmoid(cr_can)),return_plane=False,return_depth_normal=False)
        A=r2["render"].mean(dim=0,keepdim=True).clamp(0,1)
        bg=torch.tensor(np.array(Image.open(f"{BG}/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1)
        pred=(C+(1-A)*bg).clamp(0,1)
        gt=torch.tensor(np.array(Image.open(f"{GT_DYN}/canonical_cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1)
        loss+=(pred-gt).abs().mean()+0.2*(1-((2*pred*gt+0.01)/(pred**2+gt**2+0.01)).mean())
    loss.backward(); opt.step()
    if it%1000==0: log(f"  iter {it}: loss={loss.item():.6f}")

can_tau_raw=tr_can.detach().clone(); can_color_raw=cr_can.detach().clone()
ckpt_path=f"{OUTPUT}/canonical_checkpoint.pt"
torch.save({"tau_raw":can_tau_raw,"color_raw":can_color_raw}, ckpt_path)
log(f"Checkpoint saved. tau mean={F.softplus(can_tau_raw).mean().item():.4f}")

# ═══════════════════════════════════════════════════════════
# 2. Geometry helpers
# ═══════════════════════════════════════════════════════════
verts_t=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
z_range=(verts_t[:,2].min().item(),verts_t[:,2].max().item())

def get_state(name):
    if name=="canonical": return verts,scale_p,rot_init
    s=float(name.split("_")[1])
    if name.startswith("stretch"):
        dv=verts_t.clone(); dv[:,0]*=s; return dv,scale_p,rot_init
    elif name.startswith("biaxial"):
        dv=verts_t.clone(); dv[:,0]*=s; dv[:,1]*=s; return dv,scale_p*s,rot_init
    elif name.startswith("twist"):
        dv=twist_def(verts_t,int(name.split("_")[1]),z_range); return dv,scale_p,rot_init

def get_Js(name):
    if name=="canonical": return 1.0
    s=float(name.split("_")[1])
    return s if name.startswith("stretch") else s*s if name.startswith("biaxial") else 1.0

# ═══════════════════════════════════════════════════════════
# 3. Q1: Tau-regime direction flip
# ═══════════════════════════════════════════════════════════
hdr("2. Q1: Tau-regime sweep")
tau_scales=[0.25,0.5,1.0,2.0,4.0,8.0]
q1_states=["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","twist_60"]
q1_rows=[]

for ts in tau_scales:
    tau_adj=can_tau_raw.clone()*ts
    for st in ["canonical"]+q1_states:
        dv,sc,rt=get_state(st) if st!="canonical" else (verts,scale_p,rot_init)
        gm=Adapter(dv,sc,rt,tau_adj,can_color_raw.clone())
        A_vals=[]
        for ci,cam in enumerate(film_cams):
            cid=[0,4,8][ci]
            A,_=white_pass(gm,cam)
            diff=(torch.tensor(np.array(Image.open(f"{BG}/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1).abs().mean(dim=0)>0.01).float()
            interior=binary_erosion(binary_dilation(diff.cpu().numpy(),iterations=2),iterations=5)
            if interior.sum()>0: A_vals.append(A[0,interior].mean().item())
        A_m=np.mean(A_vals) if A_vals else 0
        te=-math.log(1-max(A_m,1e-10))
        q1_rows.append({"tau_scale":ts,"state":st,"Js":get_Js(st) if st!="canonical" else 1,"A":A_m,"tau_eff":te})

# Compute ratios
for ts in tau_scales:
    can_te=next(r["tau_eff"] for r in q1_rows if r["tau_scale"]==ts and r["state"]=="canonical")
    for st in q1_states:
        r=next(x for x in q1_rows if x["tau_scale"]==ts and x["state"]==st)
        r["ratio"]=r["tau_eff"]/max(can_te,1e-10)

log("stretch_2.00 tau response:")
for ts in tau_scales:
    r=next(x for x in q1_rows if x["tau_scale"]==ts and x["state"]=="stretch_2.00")
    log(f"  tau={ts:.2f}x: tau_eff={r['tau_eff']:.4f} ratio={r['ratio']:.4f}")

s2_ratios=[r["ratio"] for ts in tau_scales for r in q1_rows if r["tau_scale"]==ts and r["state"]=="stretch_2.00" and "ratio" in r]
flip=any(r>1.05 for r in s2_ratios) and any(r<0.95 for r in s2_ratios)
log(f"  Direction flip: {'YES' if flip else 'NO'}")

# ═══════════════════════════════════════════════════════════
# 4. Q2: Contributor decomposition (simplified)
# ═══════════════════════════════════════════════════════════
hdr("3. Q2: Contributor decomposition (white-pass proxy)")
# Use white-pass radii as a proxy for contributor count
# and A/tau_eff as the optical budget
contrib_rows=[]
for ts in [0.5,1.0,4.0]:
    tau_adj=can_tau_raw.clone()*ts
    for vname,vx,vsc in [("G0_canonical",verts,scale_p),("G1_position",verts_t.clone()*2.0,scale_p),("G3_full",verts_t.clone()*2.0,scale_p)]:
        # Actually use proper G1/G3 from get_state
        pass

# For stretch_2.00, use G1 and G0 at different tau scales
log("stretch_2.00 G1 position-only vs G0:")
for ts in [0.5,1.0,4.0]:
    tau_adj=can_tau_raw.clone()*ts
    # G0
    gm0=Adapter(verts,scale_p,rot_init,tau_adj,can_color_raw.clone())
    # G1: position only
    dv=verts_t.clone(); dv[:,0]*=2.0
    gm1=Adapter(dv,scale_p,rot_init,tau_adj,can_color_raw.clone())
    
    A0_vals=[]; A1_vals=[]
    for ci,cam in enumerate(film_cams):
        cid=[0,4,8][ci]
        A0,_=white_pass(gm0,cam)
        A1,_=white_pass(gm1,cam)
        diff=(torch.tensor(np.array(Image.open(f"{BG}/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1).abs().mean(dim=0)>0.01).float()
        interior=binary_erosion(binary_dilation(diff.cpu().numpy(),iterations=2),iterations=5)
        if interior.sum()>0:
            A0_vals.append(A0[0,interior].mean().item())
            A1_vals.append(A1[0,interior].mean().item())
    te0=-math.log(1-max(np.mean(A0_vals),1e-10))
    te1=-math.log(1-max(np.mean(A1_vals),1e-10))
    log(f"  tau={ts:.1f}x: G0_te={te0:.4f} G1_te={te1:.4f} ratio={te1/max(te0,1e-10):.4f}")

# ═══════════════════════════════════════════════════════════
# 5. Q3: Sampling resolution (21x21, 41x41, 81x81)
# ═══════════════════════════════════════════════════════════
hdr("4. Q3: Calibrated sampling resolution")
# Get target tau_eff from 41x41 frozen checkpoint at canonical
target_te=np.mean([r["tau_eff"] for r in q1_rows if r["tau_scale"]==1.0 and r["state"]=="canonical"])
log(f"Target tau_eff: {target_te:.4f}")

sampling_rows=[]
for res_name,divs in [("21x21",20),("81x81",80)]:
    N_r=(divs+1)**2
    sp=1.5/divs
    verts_r=torch.tensor([[-0.75+sp*i,-0.75+sp*j,0.0] for i in range(divs+1) for j in range(divs+1)],dtype=torch.float32,device=device)
    sc_r=torch.full((N_r,3),sp,device=device); sc_r[:,2]=sp*0.1
    ro_r=torch.zeros(N_r,4,device=device); ro_r[:,0]=1.0
    
    # Binary search for global tau
    lo,hi=0.01,10.0
    for _ in range(30):
        mid=(lo+hi)/2
        tau_test=torch.full((N_r,1),mid,device=device)
        class Adptr:
            def __init__(self):
                self._xyz=verts_r; self._scaling=torch.log(sc_r.clamp(min=1e-8))
                self._rotation=ro_r; self._tau_raw=tau_test; self._color_raw=torch.zeros(N_r,3,device=device)
                self.active_sh_degree=0; self.max_sh_degree=0; self.use_app=False
            @property
            def get_xyz(self): return self._xyz
            @property
            def get_scaling(self): return torch.exp(self._scaling)
            @property
            def get_rotation(self): return self._rotation/self._rotation.norm(dim=1,keepdim=True).clamp(min=1e-8)
            @property
            def get_opacity(self): return 1-torch.exp(-F.softplus(self._tau_raw))
            @property
            def get_transparency(self): return torch.full((N_r,1),0.5,device=device)
            @property
            def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)
        gm=Adptr()
        A_vals=[]
        for ci,cam in enumerate(film_cams):
            cid=[0,4,8][ci]; A,_=white_pass(gm,cam)
            diff=(torch.tensor(np.array(Image.open(f"{BG}/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1).abs().mean(dim=0)>0.01).float()
            interior=binary_erosion(binary_dilation(diff.cpu().numpy(),iterations=2),iterations=5)
            if interior.sum()>0: A_vals.append(A[0,interior].mean().item())
        te=np.mean([-math.log(1-max(a,1e-10)) for a in A_vals]) if A_vals else 0
        if te>target_te: hi=mid
        else: lo=mid
    
    global_tau=(lo+hi)/2
    log(f"  {res_name}: global_tau={global_tau:.4f}, achieved_te={te:.4f} vs target={target_te:.4f}")
    sampling_rows.append({"resolution":res_name,"global_tau":global_tau,"achieved_te":te,"target_te":target_te})

log("\n=== Stage 3.2.5 complete ===")
