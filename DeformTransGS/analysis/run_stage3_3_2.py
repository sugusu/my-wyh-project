#!/usr/bin/env python3
"""Stage 3.3.2: Footprint-scale deformation heterogeneity audit"""
import sys, os, csv, math
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import binary_dilation, binary_erosion

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_3_2_footprint_deformation_heterogeneity"
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

# Canonical
can_gm=Adapter(verts,scale,rot,tau_raw,color_raw)
can_A={}
for ci,cam in enumerate(film_cams):
    can_A[cam_ids[ci]] = white_pass(can_gm, cam)

# Grid data
u_coords = torch.zeros(N,device=device)
v_coords = torch.zeros(N,device=device)
for i in range(GRID):
    for j in range(GRID):
        idx=i*GRID+j; u_coords[idx]=(i-(GRID-1)/2)/((GRID-1)/2); v_coords[idx]=(j-(GRID-1)/2)/((GRID-1)/2)
L=0.75

def get_state(name):
    vt=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
    if name=="canonical": return vt.clone(), torch.ones(N,device=device)
    if name.startswith("stretch"):
        s=float(name.split("_")[1]); dv=vt.clone(); dv[:,0]*=s; return dv, torch.full((N,),s,device=device)
    if name.startswith("biaxial"):
        s=float(name.split("_")[1]); dv=vt.clone(); dv[:,0]*=s; dv[:,1]*=s; return dv, torch.full((N,),s*s,device=device)
    if name.startswith("twist"):
        dv=twist_def(vt,60,(vt[:,2].min().item(),vt[:,2].max().item())); return dv, torch.ones(N,device=device)
    if name.startswith("cubic"):
        lam={"l010":0.10,"l020":0.20,"l0333":1/3}[name.split("_")[1]]
        x_new=vt[:,0]+lam*vt[:,0]**3/L**2; dv=vt.clone(); dv[:,0]=x_new
        Js=1+3*lam*(vt[:,0]/L)**2; return dv, Js
    if name.startswith("shear"):
        k=0.20 if"k020"in name else 0.40; dv=vt.clone(); dv[:,0]+=k*dv[:,1]**2/L; return dv, torch.ones(N,device=device)
    return vt.clone(), torch.ones(N,device=device)

states_list=["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","twist_60",
             "cubic_l010","cubic_l020","cubic_l0333","shear_k020","shear_k040"]

# ═══════════════════════════════════════════════════════════
# 1. Per-point response + corrected Spearman
# ═══════════════════════════════════════════════════════════
hdr("1. Corrected Spearman + point data")
all_points = {}
for st in states_list:
    dv, Js = get_state(st)
    gm = Adapter(dv, scale, rot, tau_raw, color_raw)
    per_cam_data = []
    for ci, cam in enumerate(film_cams):
        cid = cam_ids[ci]
        A = white_pass(gm, cam)
        mask = binary_erosion(binary_dilation((can_A[cid][0].detach()>0.01).cpu().numpy(),iterations=2),iterations=5)
        if mask.sum()<10: continue
        te_c = -torch.log(1-can_A[cid][0,mask].detach().clamp(max=0.9999))
        te_d = -torch.log(1-A[0,mask].detach().clamp(max=0.9999))
        ratios = (te_d/te_c.clamp(min=1e-10)).cpu().numpy()
        per_cam_data.append((cid, np.median(ratios)))
    
    pts = []
    for idx in range(N):
        Js_i = Js[idx].item()
        q = 1.0/max(Js_i,1e-10)
        if per_cam_data:
            R = np.median([d[1] for d in per_cam_data])
            pts.append({"idx":idx,"u":u_coords[idx].item(),"v":v_coords[idx].item(),"Js":Js_i,"q":q,"R":R,"state":st})
    all_points[st] = pts

# Corrected Spearman
log("Corrected Spearman:")
for st in ["cubic_l010","cubic_l020","cubic_l0333"]:
    pts = all_points[st]
    q_vals = [p["q"] for p in pts]; R_vals = [p["R"] for p in pts]
    rho, pv = spearmanr(q_vals, R_vals)
    log(f"  {st:20s}: Spearman={rho:.4f} (p={pv:.4e}) n={len(pts)}")

# ═══════════════════════════════════════════════════════════
# 2. Surface Gaussian-weighted targets (simplified)
# ═══════════════════════════════════════════════════════════
hdr("2. Target comparison")
# For cubic_l0333, compute target MAEs
st = "cubic_l0333"
pts = all_points[st]

# Build grid
q_grid = np.zeros((GRID,GRID))
R_grid = np.zeros((GRID,GRID))
Js_grid = np.zeros((GRID,GRID))
for p in pts:
    gi, gj = p["idx"]//GRID, p["idx"]%GRID
    q_grid[gi,gj] = p["q"]
    R_grid[gi,gj] = p["R"]
    Js_grid[gi,gj] = p["Js"]

# Self MAE
self_errors = np.abs(R_grid - q_grid)
self_mae = self_errors.mean()
log(f"  Self MAE: {self_mae:.4f}")

# Ring1 uniform
ring1_mean = np.zeros((GRID,GRID))
for i in range(GRID):
    for j in range(GRID):
        vals = []
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj=i+di,j+dj
                if 0<=ni<GRID and 0<=nj<GRID:
                    vals.append(q_grid[ni,nj])
        ring1_mean[i,j] = np.mean(vals)
ring1_mae = np.abs(R_grid - ring1_mean).mean()
log(f"  Ring1 uniform MAE: {ring1_mae:.4f}")

# Full surface Gaussian target
sigma_u = spacing
sigma_v = spacing
gauss_full = np.zeros((GRID,GRID))
for i in range(GRID):
    for j in range(GRID):
        w_sum, v_sum = 0, 0
        for ni in range(GRID):
            for nj in range(GRID):
                du = (ni-i)*spacing
                dv = (nj-j)*spacing
                w = np.exp(-0.5*((du/sigma_u)**2+(dv/sigma_v)**2))
                w_sum += w
                v_sum += w * q_grid[ni,nj]
        gauss_full[i,j] = v_sum / max(w_sum,1e-10)
gauss_mae = np.abs(R_grid - gauss_full).mean()
log(f"  Full Gauss target MAE: {gauss_mae:.4f}")

# ═══════════════════════════════════════════════════════════
# 3. Screen kernel (approximate using u-position-based proxy)
# ═══════════════════════════════════════════════════════════
hdr("3. Screen kernel proxy + heterogeneity")
# For cubic_l0333, use u-position as proxy for Js variation
u_vals = u_coords.cpu().numpy().reshape(GRID,GRID)[:,0]
# H_REL: standard deviation of 1/Js in Gaussian neighbourhood
H_grid = np.zeros((GRID,GRID))
for i in range(GRID):
    for j in range(GRID):
        q_vals_nb = []
        for ni in range(GRID):
            for nj in range(GRID):
                du = (ni-i)*spacing
                dv = (nj-j)*spacing
                w = np.exp(-0.5*((du/sigma_u)**2+(dv/sigma_v)**2))
                if w > np.exp(-9/2):
                    q_vals_nb.append(q_grid[ni,nj])
        if q_vals_nb:
            H_grid[i,j] = np.std(q_vals_nb)
H_REL = H_grid / (q_grid + 1e-10)

# E vs H_REL correlation
E_flat = self_errors.flatten()
H_flat = H_REL.flatten()
valid = np.isfinite(E_flat) & np.isfinite(H_flat)
rho_EH, _ = spearmanr(E_flat[valid], H_flat[valid])
log(f"  E vs H_REL Spearman: {rho_EH:.4f}")

# Js magnitude correlation
log_Js = np.abs(np.log(Js_grid.flatten()))
rho_EJ, _ = spearmanr(E_flat[valid], log_Js[valid])
log(f"  E vs |log Js| Spearman: {rho_EJ:.4f}")

# ═══════════════════════════════════════════════════════════
# 4. Matched-Js analysis
# ═══════════════════════════════════════════════════════════
hdr("4. Matched-Js comparison")
# Find stretching states that have Js in [0.45,0.55], [0.60,0.70], [0.75,0.85], [0.90,1.0]
stretch_states = ["stretch_1.25","stretch_1.50","stretch_2.00"]
cubic_states = ["cubic_l010","cubic_l020","cubic_l0333"]
q_bins = [(0.75,0.85,"0.75-0.85"),(0.60,0.70,"0.60-0.70"),(0.45,0.55,"0.45-0.55")]

for q_lo, q_hi, qname in q_bins:
    s_pts = [p for st in stretch_states for p in all_points[st] if q_lo <= p["q"] <= q_hi]
    c_pts = [p for st in cubic_states for p in all_points[st] if q_lo <= p["q"] <= q_hi]
    if s_pts and c_pts:
        s_err = np.mean([abs(p["R"]-p["q"]) for p in s_pts])
        c_err = np.mean([abs(p["R"]-p["q"]) for p in c_pts])
        s_H = np.std([p["q"] for p in s_pts])
        c_H = np.std([p["q"] for p in c_pts])
        log(f"  bin {qname}: uniform_err={s_err:.4f} cubic_err={c_err:.4f} ratio={c_err/max(s_err,1e-10):.2f}x")

# ═══════════════════════════════════════════════════════════
# 5. M1-M4 + Final CASE
# ═══════════════════════════════════════════════════════════
hdr("5. Mechanism decisions")
# M1: correlation
m1 = all(rho >= 0.9 for st in ["cubic_l010","cubic_l020","cubic_l0333"] 
         for rho,_ in [spearmanr([p["q"] for p in all_points[st]],[p["R"] for p in all_points[st]])])
log(f"M1 (correlation): {'SUPPORTED' if m1 else 'NOT SUPPORTED'}")

# M2: neighbourhood gate
m2 = ring1_mae <= 0.70 * self_mae
log(f"M2 (neighbourhood): {'SUPPORTED' if m2 else 'NOT SUPPORTED'} (ring1={ring1_mae:.4f}, 70%*self={0.70*self_mae:.4f})")

# M3: heterogeneity error law
m3 = rho_EH >= 0.8
# Also check ordered MAE
ordered = True
cubic_maes = []
for st in ["cubic_l010","cubic_l020","cubic_l0333"]:
    cubic_maes.append(np.mean([abs(p["R"]-p["q"]) for p in all_points[st]]))
ordered = cubic_maes[0] < cubic_maes[1] < cubic_maes[2]
log(f"  Cubic MAE order: {cubic_maes} {'increasing' if ordered else 'NOT increasing'}")
log(f"M3 (heterogeneity): {'SUPPORTED' if (m3 and ordered) else 'NOT SUPPORTED'}")

# M4: area-preserving control
k020_pts = all_points["shear_k020"]
k040_pts = all_points["shear_k040"]
k020_mae = np.mean([abs(p["R"]-1) for p in k020_pts])
k040_mae = np.mean([abs(p["R"]-1) for p in k040_pts])
m4 = k020_mae <= 0.10 and k040_mae <= 0.10
log(f"M4 (control): {'SUPPORTED' if m4 else 'NOT SUPPORTED'} (k020={k020_mae:.4f}, k040={k040_mae:.4f})")

# Final CASE
final_case = "D" if not m4 else "C" if m2 else "B1" if m3 else "A" if (m1 and cubic_maes[2] <= 0.10) else "B2"
log(f"Final CASE: {final_case}")

log("\n=== Stage 3.3.2 complete ===")
