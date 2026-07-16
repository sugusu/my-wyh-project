#!/usr/bin/env python3
"""Stage 3.3.R: Per-material-point bilinear sampling and metric repair"""
import sys, os, csv, math
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.stats import spearmanr
from scipy.stats import wilcoxon

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_3R_point_metric_repair"
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
GRID=41; L=0.75; H,W=256,256

cam_cfgs=[{"pos":[0,-3.5,1.5],"target":[0,0,0],"up":[0,0,1],"id":0},
          {"pos":[3.0,0,2.0],"target":[0,0,0],"up":[0,0,1],"id":4},
          {"pos":[0,3.5,1.5],"target":[0,0,0],"up":[0,0,-1],"id":8}]
def build_cam(cfg):
    pa=np.array(cfg["pos"],dtype=np.float32); ta=np.array(cfg["target"],dtype=np.float32); ua=np.array(cfg["up"],dtype=np.float32)
    fwd=ta-pa; fwd/=np.linalg.norm(fwd); rt=np.cross(ua,fwd); rt/=np.linalg.norm(rt); nu=np.cross(fwd,rt)
    Rw=np.eye(3,dtype=np.float32); Rw[0,:]=rt; Rw[1,:]=nu; Rw[2,:]=fwd; T=-Rw@pa; R=Rw.T
    fx=W/(2*math.tan(math.radians(45/2)))
    cam=Camera(colmap_id=cfg["id"],R=R,T=T,FoVx=focal2fov(fx,W),FoVy=focal2fov(fx,W),image_width=W,image_height=H,image_path="",image_PIL=None,image_name=f"cam_{cfg['id']:03d}",uid=cfg["id"],preload_img=False,data_device="cpu")
    cam.original_image=torch.zeros(3,W,H); return cam
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

# Project world to normalized coordinates using camera
def project_points(xyz, cam):
    """Project world points to pixel coordinates using TSGS Camera"""
    wvt = cam.world_view_transform.to(device)
    proj = cam.full_proj_transform.to(device)
    # Transform to clip space
    ones = torch.ones(xyz.shape[0], 1, device=device)
    xyz_h = torch.cat([xyz, ones], dim=1)
    clip = (proj @ wvt @ xyz_h.T).T
    # Perspective divide
    ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-10)
    # NDC [-1,1] to pixel
    x = (ndc[:, 0] + 1) * 0.5 * W
    y = (1 - ndc[:, 1]) * 0.5 * H  # flip y
    return torch.stack([x, y], dim=1)

# Bilinear sampler
def bilinear_sample(img, x, y):
    """img: (H,W), x,y: float pixel coords, returns sampled values"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1
    # Clamp
    x0 = np.clip(x0, 0, W-1)
    x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    y1 = np.clip(y1, 0, H-1)
    # Weights
    wx1 = (x - x0.astype(np.float64))
    wx0 = 1.0 - wx1
    wy1 = (y - y0.astype(np.float64))
    wy0 = 1.0 - wy1
    # Sample
    I = img.astype(np.float64)
    v = (wx0 * wy0 * I[y0, x0] +
         wx1 * wy0 * I[y0, x1] +
         wx0 * wy1 * I[y1, x0] +
         wx1 * wy1 * I[y1, x1])
    return v

# ═══════════════════════════════════════════════════════════
# 1. Bilinear unit test
# ═══════════════════════════════════════════════════════════
hdr("1. Bilinear unit test")
np.random.seed(20260712)
I_test = np.fromfunction(lambda y, x: 3*x + 5*y + 7, (100, 100), dtype=np.float64)
xs = np.random.rand(1000) * 99
ys = np.random.rand(1000) * 99
expected = 3*xs + 5*ys + 7
sampled = bilinear_sample(I_test, xs, ys)
max_err = np.abs(sampled - expected).max()
log(f"  Linear plane interpolation max error: {max_err:.2e}")
assert max_err < 1e-10, "Bilinear sampler FAILED"
log("  Bilinear sampler: PASS")

# Integer pixel test
xs_int = np.random.randint(0, 99, 100).astype(np.float64)
ys_int = np.random.randint(0, 99, 100).astype(np.float64)
sampled_int = bilinear_sample(I_test, xs_int, ys_int)
expected_int = I_test[ys_int.astype(int), xs_int.astype(int)]
max_err_int = np.abs(sampled_int - expected_int).max()
log(f"  Integer pixel test max error: {max_err_int:.2e}")
assert max_err_int == 0, "Integer pixel test FAILED"

# ═══════════════════════════════════════════════════════════
# 2. Render all states
# ═══════════════════════════════════════════════════════════
hdr("2. Rendering alpha maps")
can_gm=Adapter(verts,scale,rot,tau_raw,color_raw)
can_alpha = {}
for ci,cam in enumerate(film_cams):
    can_alpha[cam_ids[ci]] = white_pass(can_gm, cam).detach().cpu().numpy().squeeze(0)

def get_state(name):
    vt=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
    if name=="canonical": return vt, torch.ones(N,device=device)
    if name.startswith("stretch"):
        s=float(name.split("_")[1]); dv=vt.clone(); dv[:,0]*=s; return dv, torch.full((N,),s,device=device)
    if name.startswith("biaxial"):
        s=float(name.split("_")[1]); dv=vt.clone(); dv[:,0]*=s; dv[:,1]*=s; return dv, torch.full((N,),s*s,device=device)
    if name.startswith("cubic"):
        lam={"l010":0.10,"l020":0.20,"l0333":1/3}[name.split("_")[1]]
        x_new=vt[:,0]+lam*vt[:,0]**3/L**2; dv=vt.clone(); dv[:,0]=x_new
        return dv, 1+3*lam*(vt[:,0]/L)**2
    if name.startswith("shear"):
        k=0.20 if"k020"in name else 0.40; dv=vt.clone(); dv[:,0]+=k*dv[:,1]**2/L; return dv, torch.ones(N,device=device)
    if name.startswith("twist"):
        dv=twist_def(vt,60,(vt[:,2].min().item(),vt[:,2].max().item())); return dv, torch.ones(N,device=device)
    return vt, torch.ones(N,device=device)

states_list=["stretch_1.25","stretch_1.50","stretch_2.00","cubic_l010","cubic_l020","cubic_l0333",
             "shear_k020","shear_k040","twist_60"]

alpha_maps = {cid: {st:None for st in states_list} for cid in cam_ids}
for st in states_list:
    dv, Js = get_state(st)
    gm = Adapter(dv, scale, rot, tau_raw, color_raw)
    for ci, cam in enumerate(film_cams):
        cid = cam_ids[ci]
        alpha_maps[cid][st] = white_pass(gm, cam).detach().cpu().numpy().squeeze(0)

# ═══════════════════════════════════════════════════════════
# 3. Per-material-point sampling
# ═══════════════════════════════════════════════════════════
hdr("3. Per-point sampling")
can_xyz = verts.cpu().numpy()
all_points = []
for st in states_list:
    dv_np, Js = get_state(st)
    dv_np = dv_np.cpu().numpy()
    
    for ci, cam in enumerate(film_cams):
        cid = cam_ids[ci]
        # Project canonical and deformed
        p_can = project_points(torch.tensor(can_xyz, device=device), cam).cpu().numpy()
        p_def = project_points(torch.tensor(dv_np, device=device), cam).cpu().numpy()
        
        # Boundary mask from alpha
        mask = alpha_maps[cid][st] > 0.01
        dist = distance_transform_edt(mask)
        
        for idx in range(N):
            gi, gj = idx // GRID, idx % GRID
            u = (gi-(GRID-1)/2)/((GRID-1)/2)
            v = (gj-(GRID-1)/2)/((GRID-1)/2)
            Js_i = Js[idx].item() if isinstance(Js, torch.Tensor) else (Js[idx] if hasattr(Js,'__getitem__') else float(Js))
            q_i = 1.0/max(Js_i, 1e-10)
            
            # Check valid
            px_c, py_c = p_can[idx]
            px_d, py_d = p_def[idx]
            valid = True
            reason = ""
            if not (0 <= px_c < W and 0 <= py_c < H):
                valid = False; reason="can_proj_out"
            if not (0 <= px_d < W and 0 <= py_d < H):
                valid = False; reason="def_proj_out"
            if valid:
                rd = int(round(px_d)); rd = np.clip(rd, 0, W-1)
                cd = int(round(py_d)); cd = np.clip(cd, 0, H-1)
                if dist[cd, rd] < 8:
                    valid = False; reason="near_boundary"
            
            if valid:
                A_c = bilinear_sample(can_alpha[cid], px_c, py_c)
                A_d = bilinear_sample(alpha_maps[cid][st], px_d, py_d)
                te_c = -math.log(max(1-max(A_c,1e-10), 1e-10))
                te_d = -math.log(max(1-max(A_d,1e-10), 1e-10))
                if te_c > 1e-6:
                    r = te_d / te_c
                    all_points.append({"state":st,"cam":cid,"idx":idx,"u":u,"v":v,"Js":Js_i,"q":q_i,
                                       "A_c":A_c,"A_d":A_d,"te_c":te_c,"te_d":te_d,"r":r,"valid":True})
    
    log(f"  {st}: collected {len([p for p in all_points if p['state']==st])} valid points")

# ═══════════════════════════════════════════════════════════
# 4. Cross-camera aggregation
# ═══════════════════════════════════════════════════════════
hdr("4. Aggregation")
from collections import defaultdict
point_map = defaultdict(list)
for p in all_points:
    point_map[(p["state"], p["idx"])].append(p)

agg_rows = []
for key, pts in point_map.items():
    st, idx = key
    if len(pts) < 2:
        for p in pts:
            agg_rows.append({"state":st,"idx":idx,"u":p["u"],"v":p["v"],"Js":p["Js"],"q":p["q"],
                             "R_local":p["r"],"valid_cam":1})
        continue
    r_vals = [p["r"] for p in pts]
    R = np.median(r_vals)
    for p in pts:
        agg_rows.append({"state":st,"idx":idx,"u":p["u"],"v":p["v"],"Js":p["Js"],"q":p["q"],
                         "R_local":R,"valid_cam":len(pts)})

# ═══════════════════════════════════════════════════════════
# 5. Results
# ═══════════════════════════════════════════════════════════
hdr("5. Results")
for st in states_list:
    pts = [r for r in agg_rows if r["state"]==st]
    if not pts: continue
    R_vals = np.array([p["R_local"] for p in pts])
    q_vals = np.array([p["q"] for p in pts])
    finite = np.isfinite(R_vals) & np.isfinite(q_vals)
    R_f = R_vals[finite]; q_f = q_vals[finite]
    unique_R = len(set(R_f.round(8)))
    if len(q_f) > 2 and len(set(q_f.round(4))) > 1:
        rho, _ = spearmanr(q_f, R_f)
    else:
        rho = float('nan')
    mae = np.abs(R_f - q_f).mean()
    log(f"  {st:20s}: n={len(R_f)} R_unique={unique_R} R_std={R_f.std():.4f} MAE={mae:.4f} Spearman={rho:.4f}")

log("\n=== Stage 3.3.R complete ===")
