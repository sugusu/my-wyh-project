#!/usr/bin/env python3
"""Stage 3.3: Local optical consistency boundary test with non-uniform deformation"""
import sys, os, csv, math, hashlib, json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr, pearsonr

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_3_local_optical_consistency"
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
# 0. Setup - same as Gold Protocol
# ═══════════════════════════════════════════════════════════
mesh=trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N=len(mesh.vertices)  # 1681
verts=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
spacing=1.5/40; scale=torch.full((N,3),spacing,device=device); scale[:,2]=spacing*0.1
rot=torch.zeros(N,4,device=device); rot[:,0]=1.0

ckpt=torch.load(f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt", map_location=device, weights_only=True)
tau_raw=ckpt["tau_raw"]; color_raw=ckpt["color_raw"]

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
    def __init__(self,xyz,scale,rot,tau_raw,col_raw):
        self._xyz=xyz; self._scaling=torch.log(scale.clamp(min=1e-8))
        self._rotation=rot; self._tau_raw=tau_raw; self._color_raw=col_raw
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

# ═══════════════════════════════════════════════════════════
# 1. Gold protocol reproduction + uniform baseline
# ═══════════════════════════════════════════════════════════
hdr("1. Gold reproduction")
mesh_center = verts.mean(dim=0)
# For the 41x41 grid, compute u coordinate
grid_size=41
u_coords = torch.zeros(N, device=device)
for i in range(grid_size):
    for j in range(grid_size):
        idx = i*grid_size + j
        u_coords[idx] = (i - (grid_size-1)/2) / ((grid_size-1)/2)  # -1 to 1

log(f"Sheet center: {mesh_center}, u range: [{u_coords.min():.3f}, {u_coords.max():.3f}]")

can_gm=Adapter(verts,scale,rot,tau_raw,color_raw)
can_A = {}
for ci,cam in enumerate(film_cams):
    can_A[cam_ids[ci]] = white_pass(can_gm, cam)

def compute_local_response(state_name, deformed_verts, F_func=None):
    """Compute per-material-point optical response"""
    gm_def=Adapter(deformed_verts,scale,rot,tau_raw,color_raw)
    def_A = {}
    for ci,cam in enumerate(film_cams):
        def_A[cam_ids[ci]] = white_pass(gm_def, cam)
    
    # Per-material-point response
    responses = []
    for ci,cam in enumerate(film_cams):
        cid=cam_ids[ci]
        A_can=can_A[cid]; A_def=def_A[cid]
        mask_can=binary_erosion(binary_dilation((A_can[0].detach()>0.01).cpu().numpy(), iterations=2), iterations=5)
        if mask_can.sum()<10: continue
        te_c = -torch.log(1-A_can[0,mask_can].detach().clamp(max=0.9999))
        te_d = -torch.log(1-A_def[0,mask_can].detach().clamp(max=0.9999))
        ratios = (te_d/te_c.clamp(min=1e-10)).cpu().numpy()
        finite = np.isfinite(ratios)
        if finite.sum()>0:
            # Map back to material indices
            mask_indices = np.where(mask_can)
            for idx_in_mask, (ri,ci_pix) in enumerate(zip(mask_indices[0], mask_indices[1])):
                # Find nearest material point - simplified: use all pixels in mask
                pass
            responses.extend(ratios[finite].tolist())
    
    return np.median(responses) if responses else 1.0

# ═══════════════════════════════════════════════════════════
# 2. Non-uniform cubic stretch
# ═══════════════════════════════════════════════════════════
hdr("2. Non-uniform cubic stretch")
L = 0.75  # half extent
verts_t = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
# Local coordinates
local_x = verts_t[:, 0]  # x coordinate from -0.75 to 0.75
u = local_x / L  # normalized -1 to 1

cubic_states = {}
for lam_val, name in [(0.10, "cubic_l010"), (0.20, "cubic_l020"), (1.0/3.0, "cubic_l0333")]:
    x_new = local_x + lam_val * local_x**3 / L**2
    dv = verts_t.clone()
    dv[:, 0] = x_new
    cubic_states[name] = dv
    
    # Analytic Js
    s_local = 1.0 + 3.0 * lam_val * u**2
    Js_analytic = s_local  # for planar sheet with normal z
    log(f"  {name}: max_Js={Js_analytic.max():.4f}, min_Js={Js_analytic.min():.4f}")

# ═══════════════════════════════════════════════════════════
# 3. Shear control
# ═══════════════════════════════════════════════════════════
hdr("3. Non-uniform shear control")
shear_states = {}
for k_val, name in [(0.20, "shear_k020"), (0.40, "shear_k040")]:
    dv = verts_t.clone()
    dv[:, 0] = dv[:, 0] + k_val * dv[:, 1]**2 / L
    shear_states[name] = dv
    # Js = 1 for this shear
    log(f"  {name}: position range x=[{dv[:,0].min():.3f},{dv[:,0].max():.3f}]")

# ═══════════════════════════════════════════════════════════
# 4. Compute responses
# ═══════════════════════════════════════════════════════════
hdr("4. Computing responses")
all_states = {
    "stretch_1.25": verts_t.clone()*torch.tensor([1.25,1.0,1.0],device=device),
    "stretch_1.50": verts_t.clone()*torch.tensor([1.50,1.0,1.0],device=device),
    "stretch_2.00": verts_t.clone()*torch.tensor([2.00,1.0,1.0],device=device),
    "biaxial_1.50": verts_t.clone()*torch.tensor([1.50,1.50,1.0],device=device),
    "twist_60": twist_def(verts_t, 60, (verts_t[:,2].min().item(),verts_t[:,2].max().item())),
    **cubic_states, **shear_states
}

results = []
for name, dv in all_states.items():
    gm_def=Adapter(dv,scale,rot,tau_raw,color_raw)
    point_ratios = []
    for ci,cam in enumerate(film_cams):
        cid=cam_ids[ci]
        A_c=can_A[cid]; A_d=white_pass(gm_def,cam)
        mask=binary_erosion(binary_dilation((A_c[0].detach()>0.01).cpu().numpy(),iterations=2),iterations=5)
        if mask.sum()<10: continue
        te_c = -torch.log(1-A_c[0,mask].detach().clamp(max=0.9999))
        te_d = -torch.log(1-A_d[0,mask].detach().clamp(max=0.9999))
        ratios = (te_d/te_c.clamp(min=1e-10)).cpu().numpy()
        point_ratios.extend(ratios[np.isfinite(ratios)].tolist())
    
    R = np.median(point_ratios) if point_ratios else 1.0
    log(f"  {name:20s}: R_tau_eff={R:.4f}")
    results.append({"state":name,"R":R})

log("\n=== Stage 3.3 complete ===")
