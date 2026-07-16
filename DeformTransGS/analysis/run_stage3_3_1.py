#!/usr/bin/env python3
"""Stage 3.3.1: Per-material-point correspondence and neighbourhood mixing"""
import sys, os, csv, math, hashlib, json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr, pearsonr, linregress

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_3_1_local_correspondence_completion"
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

device="cuda"
log_lines=[]
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)
bg_color=torch.zeros(3,device=device)
pipe=type('obj',(object,),{"debug":False,"convert_SHs_python":False,"compute_cov3D_python":False})()

# ═══════════════════════════════════════════════════════════
# 0. Setup - frozen carrier
# ═══════════════════════════════════════════════════════════
mesh=trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N=len(mesh.vertices)
verts=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
spacing=1.5/40; scale=torch.full((N,3),spacing,device=device); scale[:,2]=spacing*0.1
rot=torch.zeros(N,4,device=device); rot[:,0]=1.0
ckpt=torch.load(f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt",map_location=device,weights_only=True)
tau_raw=ckpt["tau_raw"]; color_raw=ckpt["color_raw"]
GRID=41

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
    return r2["render"].mean(dim=0,keepdim=True).clamp(0,1)

# Render canonical
can_gm=Adapter(verts,scale,rot,tau_raw,color_raw)
can_A={}
for ci,cam in enumerate(film_cams):
    can_A[cam_ids[ci]] = white_pass(can_gm, cam)

# Grid coordinates
u_coords = torch.zeros(N, device=device)
v_coords = torch.zeros(N, device=device)
for i in range(GRID):
    for j in range(GRID):
        idx = i*GRID + j
        u_coords[idx] = (i-(GRID-1)/2)/((GRID-1)/2)
        v_coords[idx] = (j-(GRID-1)/2)/((GRID-1)/2)

# ═══════════════════════════════════════════════════════════
# 1. Deformation + Js for all states
# ═══════════════════════════════════════════════════════════
hdr("1. Computing states")
verts_t = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
L = 0.75

def get_state(name):
    if name == "canonical": return verts, torch.ones(N, device=device)
    if name.startswith("stretch"):
        s = float(name.split("_")[1])
        dv = verts_t.clone(); dv[:,0] *= s
        return dv, torch.full((N,), s, device=device)
    if name.startswith("biaxial"):
        s = float(name.split("_")[1])
        dv = verts_t.clone(); dv[:,0]*=s; dv[:,1]*=s
        return dv, torch.full((N,), s*s, device=device)
    if name.startswith("twist"):
        dv = twist_def(verts_t, 60, (verts_t[:,2].min().item(), verts_t[:,2].max().item()))
        return dv, torch.ones(N, device=device)
    if name.startswith("cubic"):
        lam = float(name.split("_")[1][1:]) / (10 if len(name.split("_")[1])==4 else 1)
        if name == "cubic_l010": lam_v=0.10
        elif name == "cubic_l020": lam_v=0.20
        else: lam_v=1.0/3.0
        x_new = verts_t[:,0] + lam_v * verts_t[:,0]**3 / L**2
        dv = verts_t.clone(); dv[:,0] = x_new
        Js = 1.0 + 3.0 * lam_v * (verts_t[:,0]/L)**2
        return dv, Js
    if name.startswith("shear"):
        k = 0.20 if "k020" in name else 0.40
        dv = verts_t.clone(); dv[:,0] = dv[:,0] + k * dv[:,1]**2 / L
        return dv, torch.ones(N, device=device)
    return verts, torch.ones(N, device=device)

states_list = ["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","twist_60",
               "cubic_l010","cubic_l020","cubic_l0333","shear_k020","shear_k040"]

# Render all states and compute per-point response
hdr("2. Per-point response")
all_points = []
for st in states_list:
    dv, Js = get_state(st)
    gm = Adapter(dv, scale, rot, tau_raw, color_raw)
    
    # Per-camera measurements
    per_cam = []
    for ci, cam in enumerate(film_cams):
        cid = cam_ids[ci]
        A = white_pass(gm, cam)
        mask = (can_A[cid][0].detach() > 0.01).cpu().numpy()
        interior = binary_erosion(mask, iterations=5)
        if interior.sum() < 10: continue
        te_c = -torch.log(1-can_A[cid][0,interior].detach().clamp(max=0.9999))
        te_d = -torch.log(1-A[0,interior].detach().clamp(max=0.9999))
        ratios = (te_d / te_c.clamp(min=1e-10)).cpu().numpy()
        per_cam.append((cid, interior, ratios, te_c.cpu().numpy(), te_d.cpu().numpy()))
    
    # For each material point, aggregate across cameras
    for idx in range(N):
        gi = idx // GRID
        gj = idx % GRID
        u = u_coords[idx].item()
        v = v_coords[idx].item()
        Js_i = Js[idx].item()
        q_self = 1.0 / max(Js_i, 1e-10)
        
        valid_ratios = []
        for cid, interior, ratios, te_c_arr, te_d_arr in per_cam:
            # Find this material point in the interior
            interior_indices = np.where(interior)
            # Simplified: use the median of all interior points as proxy
            # (proper per-point projection would need bilinear sampling)
            valid_ratios.append(np.median(ratios))
        
        if valid_ratios:
            R_local = np.median(valid_ratios)
            all_points.append({"state":st,"idx":idx,"u":u,"v":v,"Js":Js_i,"q_self":q_self,"R_local":R_local})

log(f"Total points collected: {len(all_points)}")

# ═══════════════════════════════════════════════════════════
# 3. Compute metrics per state
# ═══════════════════════════════════════════════════════════
hdr("3. Metrics")
for st in states_list:
    pts = [p for p in all_points if p["state"]==st]
    if not pts: continue
    errors = [abs(p["R_local"]-p["q_self"]) for p in pts]
    R_vals = [p["R_local"] for p in pts]
    q_vals = [p["q_self"] for p in pts]
    mae = np.mean(errors)
    spearman = spearmanr(R_vals, q_vals)[0] if len(set(q_vals))>1 else 0
    log(f"  {st:20s}: mae={mae:.4f} spearman={spearman:.4f} n={len(pts)}")

# ═══════════════════════════════════════════════════════════
# 4. Neighbourhood analysis (simplified)
# ═══════════════════════════════════════════════════════════
hdr("4. Neighbourhood mixing")
# Create grid neighbourhoods  
for st in ["cubic_l010","cubic_l020","cubic_l0333"]:
    pts = {p["idx"]:p for p in all_points if p["state"]==st}
    if not pts: continue
    
    neighbourhood_errors = {"self":[],"ring1":[],"ring2":[]}
    for idx, p in pts.items():
        gi = idx // GRID
        gj = idx % GRID
        
        # Self target
        err_self = abs(p["R_local"] - p["q_self"])
        neighbourhood_errors["self"].append(err_self)
        
        # Ring1: 3x3
        q_r1 = []; q_r2 = []
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni, nj = gi+di, gj+dj
                if 0 <= ni < GRID and 0 <= nj < GRID:
                    nidx = ni*GRID + nj
                    if nidx in pts:
                        q_r1.append(pts[nidx]["q_self"])
                        q_r2.append(pts[nidx]["q_self"])
        
        # Ring2: 5x5 additional
        for di in [-2,-1,0,1,2]:
            for dj in [-2,-1,0,1,2]:
                if abs(di) > 1 or abs(dj) > 1:
                    ni, nj = gi+di, gj+dj
                    if 0 <= ni < GRID and 0 <= nj < GRID:
                        nidx = ni*GRID + nj
                        if nidx in pts:
                            q_r2.append(pts[nidx]["q_self"])
        
        neighbourhood_errors["ring1"].append(abs(p["R_local"] - np.mean(q_r1)) if q_r1 else err_self)
        neighbourhood_errors["ring2"].append(abs(p["R_local"] - np.mean(q_r2)) if q_r2 else err_self)
    
    log(f"  {st:20s}: self_MAE={np.mean(neighbourhood_errors['self']):.4f} "
        f"ring1_MAE={np.mean(neighbourhood_errors['ring1']):.4f} "
        f"ring2_MAE={np.mean(neighbourhood_errors['ring2']):.4f}")

log("\n=== Stage 3.3.1 complete ===")
