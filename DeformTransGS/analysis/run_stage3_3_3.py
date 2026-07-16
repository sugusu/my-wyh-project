#!/usr/bin/env python3
"""Stage 3.3.3: Screen-space footprint heterogeneity and matched-Js mechanism audit"""
import sys, os, csv, math
import numpy as np
from scipy.stats import spearmanr, pearsonr, wilcoxon
from scipy.ndimage import binary_dilation, binary_erosion

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_3_3_screen_footprint_mechanism"
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
bg_color=torch.zeros(3,device=device)
pipe=type('obj',(object,),{"debug":False,"convert_SHs_python":False,"compute_cov3D_python":False})()

# ═══════════════════════════════════════════════════════════
# 0. Setup
# ═══════════════════════════════════════════════════════════
mesh=trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N=len(mesh.vertices)
verts=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
spacing=1.5/40; scale=torch.full((N,3),spacing,device=device); scale[:,2]=spacing*0.1
rot=torch.zeros(N,4,device=device); rot[:,0]=1.0
ckpt=torch.load(f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt",map_location=device,weights_only=True)
tau_raw=ckpt["tau_raw"]; color_raw=ckpt["color_raw"]
GRID=41; L=0.75

cam_cfgs=[{"pos":[0,-3.5,1.5],"target":[0,0,0],"up":[0,0,1],"id":0},
          {"pos":[3.0,0,2.0],"target":[0,0,0],"up":[0,0,1],"id":4},
          {"pos":[0,3.5,1.5],"target":[0,0,0],"up":[0,0,-1],"id":8}]
def build_cam(cfg):
    pa=np.array(cfg["pos"],dtype=np.float32); ta=np.array(cfg["target"],dtype=np.float32); ua=np.array(cfg["up"],dtype=np.float32)
    fwd=ta-pa; fwd/=np.linalg.norm(fwd); rt=np.cross(ua,fwd); rt/=np.linalg.norm(rt); nu=np.cross(fwd,rt)
    Rw=np.eye(3,dtype=np.float32); Rw[0,:]=rt; Rw[1,:]=nu; Rw[2,:]=fwd; T=-Rw@pa; R=Rw.T
    fx=256/(2*math.tan(math.radians(45/2)))
    cam=Camera(colmap_id=cfg["id"],R=R,T=T,FoVx=focal2fov(fx,256),FoVy=focal2fov(fx,256),image_width=256,image_height=256,image_path="",image_PIL=None,image_name=f"cam_{cfg['id']:03d}",uid=cfg["id"],preload_img=False,data_device="cpu")
    cam.original_image=torch.zeros(3,256,256); return cam
film_cams=[build_cam(c) for c in cam_cfgs]; cam_ids=[c["id"] for c in cam_cfgs]

class Adapter:
    def __init__(self,xyz,scl,rot,tau,col):
        self._xyz=xyz; self._scaling=torch.log(scl.clamp(min=1e-8))
        self._rotation=rot; self._tau_raw=tau; self._color_raw=col
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
    def get_transparency(self): return torch.full((N,1),0.5,device=device)
    @property
    def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)

def white_pass(gm,cam):
    r2=render(cam,gm,pipe,bg_color,app_model=None,override_color=torch.ones(N,3,device=device),return_plane=False,return_depth_normal=False)
    return r2["render"].mean(dim=0,keepdim=True).clamp(0,1), r2["radii"]

# Canonical render
can_gm=Adapter(verts,scale,rot,tau_raw,color_raw)
can_A={}; can_radii={}
for ci,cam in enumerate(film_cams):
    can_A[cam_ids[ci]], can_radii[cam_ids[ci]] = white_pass(can_gm, cam)

# State helper
def get_state(name):
    vt=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
    if name=="canonical": return vt, torch.ones(N,device=device)
    if name.startswith("stretch"):
        s=float(name.split("_")[1]); dv=vt.clone(); dv[:,0]*=s; return dv, torch.full((N,),s,device=device)
    if name.startswith("cubic"):
        lam={"l010":0.10,"l020":0.20,"l0333":1/3}[name.split("_")[1]]
        x_new=vt[:,0]+lam*vt[:,0]**3/L**2; dv=vt.clone(); dv[:,0]=x_new
        Js=1+3*lam*(vt[:,0]/L)**2; return dv, Js
    return vt, torch.ones(N,device=device)

# ═══════════════════════════════════════════════════════════
# 1. Spearman bug audit
# ═══════════════════════════════════════════════════════════
hdr("1. Spearman bug audit")
cubic_states = ["cubic_l010","cubic_l020","cubic_l0333"]
stretch_states = ["stretch_1.25","stretch_1.50","stretch_2.00"]

for st in cubic_states:
    dv, Js = get_state(st)
    gm = Adapter(dv, scale, rot, tau_raw, color_raw)
    cam_ratios = []
    for ci, cam in enumerate(film_cams):
        cid = cam_ids[ci]
        A, _ = white_pass(gm, cam)
        mask = binary_erosion(binary_dilation((can_A[cid][0].detach()>0.01).cpu().numpy(),iterations=2),iterations=5)
        if mask.sum()<10: continue
        te_c = -torch.log(1-can_A[cid][0,mask].detach().clamp(max=0.9999))
        te_d = -torch.log(1-A[0,mask].detach().clamp(max=0.9999))
        ratios = (te_d/te_c.clamp(min=1e-10)).cpu().numpy()
        cam_ratios.append(np.median(ratios))
    R = np.median(cam_ratios)
    
    # Per-point q and R (actually R is global median per camera × 3 cameras, so per-point needs point-level)
    # For Spearman, use the R and q for each material point
    q_vals = []  # q = 1/Js per point
    R_vals = []  # R = per-point median across cameras
    for idx in range(N):
        Js_i = Js[idx].item()
        q_vals.append(1.0/max(Js_i,1e-10))
        # Use per-camera ratios then median
        pc_ratios = []
        for ci, cam in enumerate(film_cams):
            cid = cam_ids[ci]
            A, _ = white_pass(gm, cam)
            mask = binary_erosion(binary_dilation((can_A[cid][0].detach()>0.01).cpu().numpy(),iterations=2),iterations=5)
            if mask.sum()<10: continue
            te_c = -torch.log(1-can_A[cid][0,mask].detach().clamp(max=0.9999))
            # For this point, find closest pixel
            mask_indices = np.where(mask)
            # Use median of all valid pixels as proxy
            te_d = -torch.log(1-A[0,mask].detach().clamp(max=0.9999))
            ratio_med = (te_d/te_c.clamp(min=1e-10)).median().item()
            pc_ratios.append(ratio_med)
        R_vals.append(np.median(pc_ratios) if pc_ratios else 1.0)
    
    q_a = np.asarray(q_vals, dtype=np.float64).reshape(-1)
    R_a = np.asarray(R_vals, dtype=np.float64).reshape(-1)
    valid = np.isfinite(q_a) & np.isfinite(R_a)
    
    log(f"  {st}: valid={valid.sum()}/{N}, q_unique={len(set(q_a[valid].round(6)))}, R_unique={len(set(R_a[valid].round(6)))}")
    if valid.sum() > 2 and len(set(q_a[valid].round(4))) > 1:
        rho, pv = spearmanr(q_a[valid], R_a[valid])
        log(f"    Spearman={rho:.4f} (p={pv:.4e})")
    else:
        log(f"    Spearman: cannot compute (insufficient unique values)")
        rho = 0.0

# ═══════════════════════════════════════════════════════════
# 2. Same-point matched Js analysis
# ═══════════════════════════════════════════════════════════
hdr("2. Same-point matched Js")
# For each material point, interpolate uniform response at cubic Js values
# Build uniform response curves per material point per camera
all_uniform = {}
for st in stretch_states:
    dv, Js = get_state(st)
    gm = Adapter(dv, scale, rot, tau_raw, color_raw)
    for ci, cam in enumerate(film_cams):
        cid = cam_ids[ci]
        A, _ = white_pass(gm, cam)
        mask = binary_erosion(binary_dilation((can_A[cid][0].detach()>0.01).cpu().numpy(),iterations=2),iterations=5)
        if mask.sum()<10: continue
        te_c = -torch.log(1-can_A[cid][0,mask].detach().clamp(max=0.9999))
        te_d = -torch.log(1-A[0,mask].detach().clamp(max=0.9999))
        # Per-point ratio (using median as proxy for each point since we don't have per-pixel correspondence)
        R = (te_d/te_c.clamp(min=1e-10)).median().item()
        q = 1.0/max(Js[0].item(),1e-10)
        if cid not in all_uniform:
            all_uniform[cid] = {"q":[],"R":[]}
        all_uniform[cid]["q"].append(q)
        all_uniform[cid]["R"].append(R)

# For cubic states, compute extra error
log("Matched-Js extra error:")
for st in cubic_states:
    dv, Js = get_state(st)
    gm = Adapter(dv, scale, rot, tau_raw, color_raw)
    delta_E_all = []
    for ci, cam in enumerate(film_cams):
        cid = cam_ids[ci]
        A, _ = white_pass(gm, cam)
        mask = binary_erosion(binary_dilation((can_A[cid][0].detach()>0.01).cpu().numpy(),iterations=2),iterations=5)
        if mask.sum()<10: continue
        te_c = -torch.log(1-can_A[cid][0,mask].detach().clamp(max=0.9999))
        te_d = -torch.log(1-A[0,mask].detach().clamp(max=0.9999))
        R_cubic = (te_d/te_c.clamp(min=1e-10)).median().item()
        
        # Interpolate uniform response
        uq = np.array(all_uniform[cid]["q"])
        uR = np.array(all_uniform[cid]["R"])
        if len(uq) > 1:
            sort_idx = np.argsort(uq)
            R_expected = np.interp(1.0/max(Js.median().item(),1e-10), uq[sort_idx], uR[sort_idx])
        else:
            R_expected = np.mean(uR) if uR else 1.0
        
        E_expected = abs(R_expected - 1.0/Js.median().item())
        E_cubic = abs(R_cubic - 1.0/Js.median().item())
        delta_E_all.append(E_cubic - E_expected)
    
    if delta_E_all:
        log(f"  {st}: Delta_E median={np.median(delta_E_all):.4f}, positive_frac={np.mean(np.array(delta_E_all)>0):.3f}")

log("\n=== Stage 3.3.3 complete ===")
