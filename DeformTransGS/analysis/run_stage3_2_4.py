#!/usr/bin/env python3
"""Stage 3.2.4: Gaussian optical-budget conservation audit with frozen checkpoint"""
import sys, os, csv, math, hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_2_4_optical_budget_audit"
os.makedirs(OUTPUT, exist_ok=True)

sys.path.insert(0, "/data/wyh/repos/TSGS")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, f"{BASE}/benchmark")

import torch, trimesh
from torch.nn import functional as F
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from deformations.twist import deform_points as twist_def

device="cuda"
log_lines=[]
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

# ═══════════════════════════════════════════════════════════
# 0. Setup
# ═══════════════════════════════════════════════════════════
mesh = trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N=len(mesh.vertices)
verts=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
spacing=1.5/40; tang_scale=spacing*1.0
scale_p=torch.full((N,3),tang_scale,device=device); scale_p[:,2]=tang_scale*0.1
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
# 1. Train frozen canonical checkpoint (5000 iter for near-perfect fit)
# ═══════════════════════════════════════════════════════════
hdr("1. Training frozen canonical checkpoint")
tr_can=torch.full((N,1),0.0,device=device,requires_grad=True)
cr_can=torch.zeros(N,3,device=device,requires_grad=True)
opt_can=torch.optim.Adam([tr_can,cr_can],lr=1e-2)
GT_DYN=f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/repaired_render/tau1.0_dynamic"
BG=f"{BASE}/experiments/stage3_2_fixed_optical_necessity/background_only"

for it in range(5000):
    opt_can.zero_grad(); loss=0
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
    loss.backward(); opt_can.step()
    if it%1000==0: log(f"  iter {it}: loss={loss.item():.6f}")

can_tau_raw=tr_can.detach().clone()
can_color_raw=cr_can.detach().clone()
ckpt_path=f"{OUTPUT}/canonical_checkpoint.pt"
torch.save({"tau_raw":can_tau_raw,"color_raw":can_color_raw}, ckpt_path)
log(f"Checkpoint saved: {ckpt_path}")
log(f"tau mean={F.softplus(can_tau_raw).mean().item():.4f}")

# ═══════════════════════════════════════════════════════════
# 2. Geometry variants with material-point sampling
# ═══════════════════════════════════════════════════════════
hdr("2. Material point analysis")
verts_t=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
z_range=(verts_t[:,2].min().item(),verts_t[:,2].max().item())

def get_state(name):
    if name=="canonical": return verts, scale_p, rot_init
    s=float(name.split("_")[1])
    if name.startswith("stretch"):
        dv=verts_t.clone(); dv[:,0]*=s; return dv, scale_p, rot_init
    elif name.startswith("biaxial"):
        dv=verts_t.clone(); dv[:,0]*=s; dv[:,1]*=s; return dv, scale_p*s, rot_init
    elif name.startswith("twist"):
        dv=twist_def(verts_t,int(name.split("_")[1]),z_range); return dv, scale_p, rot_init

def get_Js(name):
    if name=="canonical": return 1.0
    s=float(name.split("_")[1])
    return s if name.startswith("stretch") else s*s if name.startswith("biaxial") else 1.0

# For each state, compute G0-G3 with material-point sampling
states_test=["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","twist_60"]
all_rows=[]

for st in states_test:
    dv,sc,rt=get_state(st)
    Js=get_Js(st)
    
    variants=[("G0_canonical",verts,scale_p,rot_init),
              ("G1_position",dv, scale_p,rot_init),
              ("G2_covariance",verts,sc,rot_init),
              ("G3_full",dv,sc,rot_init)]
    
    for vname,vx,vsc,vrot in variants:
        gm=Adapter(vx,vsc,vrot,can_tau_raw.clone(),can_color_raw.clone())
        
        for ci,cam in enumerate(film_cams):
            cid=[0,4,8][ci]
            A,radii=white_pass(gm,cam)
            # Get valid material points
            diff=(torch.tensor(np.array(Image.open(f"{BG}/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1).abs().mean(dim=0)>0.01).float()
            sheet_mask=binary_dilation(diff.cpu().numpy(),iterations=2)
            interior=binary_erosion(sheet_mask,iterations=5)
            
            if interior.sum()>0:
                A_interior=A[0,interior].mean().item()
                te=-math.log(1-max(A_interior,1e-10))
                radii_valid=radii[radii>0].float()
                radii_m=radii_valid.mean().item() if len(radii_valid)>0 else 0
                radii_med=radii_valid.median().item() if len(radii_valid)>0 else 0
            else:
                A_interior=0; te=0; radii_m=0; radii_med=0
            
            all_rows.append({"state":st,"variant":vname,"cam":cid,"Js":Js,"A":A_interior,"tau_eff":te,"radii_mean":radii_m,"radii_median":radii_med})

# Summary by variant
log("")
for vname in ["G0_canonical","G1_position","G2_covariance","G3_full"]:
    for st in states_test:
        vals=[r["tau_eff"] for r in all_rows if r["state"]==st and r["variant"]==vname]
        g0=[r["tau_eff"] for r in all_rows if r["state"]==st and r["variant"]=="G0_canonical"]
        if vals and g0:
            ratio=np.mean(vals)/max(np.mean(g0),1e-10)
            log(f"  {st:20s} {vname:15s}: tau_eff={np.mean(vals):.4f} ratio={ratio:.4f}")

# Direction resolution: stretch_2.00 G1
g1s=[r for r in all_rows if r["state"]=="stretch_2.00" and r["variant"]=="G1_position"]
g0s=[r for r in all_rows if r["state"]=="stretch_2.00" and r["variant"]=="G0_canonical"]
if g1s and g0s:
    log(f"\n  Position-only stretch_2.00: tau_eff_ratio={np.mean([r['tau_eff'] for r in g1s])/max(np.mean([r['tau_eff'] for r in g0s]),1e-10):.4f}")

# ═══════════════════════════════════════════════════════════
# 3. Tau-level and scale-policy ablation
# ═══════════════════════════════════════════════════════════
hdr("3. Tau/Scale ablation")
ablate_rows=[]
for st in ["stretch_1.50","stretch_2.00","biaxial_1.50","twist_60","canonical"]:
    dv,sc,rt=get_state(st) if st!="canonical" else (verts,scale_p,rot_init)
    Js=get_Js(st) if st!="canonical" else 1.0
    for tau_mul in [0.5,1.0,2.0,4.0]:
        tau_adj=can_tau_raw.clone()*tau_mul
        for sc_mul in [0.75,1.0,1.5]:
            sc_adj=sc*torch.tensor([sc_mul,sc_mul,sc_mul*0.1],device=device) if sc_mul!=1.0 else sc
            gm=Adapter(dv,sc_adj,rot_init,tau_adj,can_color_raw.clone())
            A_vals=[]
            for ci,cam in enumerate(film_cams):
                cid=[0,4,8][ci]
                A,_=white_pass(gm,cam)
                diff=(torch.tensor(np.array(Image.open(f"{BG}/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1).abs().mean(dim=0)>0.01).float()
                interior=binary_erosion(binary_dilation(diff.cpu().numpy(),iterations=2),iterations=5)
                if interior.sum()>0: A_vals.append(A[0,interior].mean().item())
            A_m=np.mean(A_vals) if A_vals else 0
            ablate_rows.append({"state":st,"Js":Js,"tau_mul":tau_mul,"sc_mul":sc_mul,"A":A_m,"tau_eff":-math.log(1-max(A_m,1e-10))if A_m<1 else 10})

can_ref=[r for r in ablate_rows if r["state"]=="canonical" and r["tau_mul"]==1.0 and r["sc_mul"]==1.0]
can_ref_te=can_ref[0]["tau_eff"] if can_ref else 1
log(f"\nReference canonical tau_eff={can_ref_te:.4f}")
for st in ["stretch_2.00","biaxial_1.50","twist_60"]:
    for tau_mul in [0.5,1.0,2.0,4.0]:
        for sc_mul in [0.75,1.0,1.5]:
            r=[x for x in ablate_rows if x["state"]==st and x["tau_mul"]==tau_mul and x["sc_mul"]==sc_mul]
            if r: log(f"  {st:15s} tau={tau_mul:.1f}x sc={sc_mul:.2f}x: tau_eff={r[0]['tau_eff']:.4f} ratio={r[0]['tau_eff']/can_ref_te:.4f}")

log("\n=== Stage 3.2.4 complete ===")
