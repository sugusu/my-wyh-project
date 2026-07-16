#!/usr/bin/env python3
"""Stage 3.2.2: Fixed-Tau Paradox Audit - independent recheck and alpha analysis"""
import sys, os, csv, math, hashlib, json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_2_2_fixed_tau_paradox"
os.makedirs(f"{OUTPUT}/alpha_maps", exist_ok=True)
os.makedirs(f"{OUTPUT}/fresh_e1", exist_ok=True)

sys.path.insert(0, "/data/wyh/repos/TSGS")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, f"{BASE}/benchmark")

import torch
from torch.nn import functional as F
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from deformations.twist import deform_points as twist_def
import trimesh

device = "cuda"
log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

# ═══════════════════════════════════════════════════════════
# 0. Setup - same as Stage 3.2.1 validated path
# ═══════════════════════════════════════════════════════════
mesh = trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N = len(mesh.vertices)
verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
spacing = 1.5 / 40
scale_p = torch.full((N, 3), spacing, device=device); scale_p[:, 2] = spacing * 0.1
rot_init = torch.zeros(N, 4, device=device); rot_init[:, 0] = 1.0

camera_cfgs = [{"pos": [0,-3.5,1.5],"target":[0,0,0],"up":[0,0,1],"id":0},
               {"pos":[3.0,0,2.0],"target":[0,0,0],"up":[0,0,1],"id":4},
               {"pos":[0,3.5,1.5],"target":[0,0,0],"up":[0,0,-1],"id":8}]

def build_cam(cfg):
    pa = np.array(cfg["pos"],dtype=np.float32); ta = np.array(cfg["target"],dtype=np.float32); ua = np.array(cfg["up"],dtype=np.float32)
    fwd = ta - pa; fwd = fwd / np.linalg.norm(fwd)
    rt = np.cross(ua, fwd); rt = rt / np.linalg.norm(rt); nu = np.cross(fwd, rt)
    R_w2c = np.eye(3, dtype=np.float32); R_w2c[0,:]=rt; R_w2c[1,:]=nu; R_w2c[2,:]=fwd
    T = -R_w2c @ pa; R = R_w2c.T
    fx = 256/(2*math.tan(math.radians(45/2)))
    cam = Camera(colmap_id=cfg["id"],R=R,T=T,FoVx=focal2fov(fx,256),FoVy=focal2fov(fx,256),
                 image_width=256,image_height=256,image_path="",image_PIL=None,
                 image_name=f"cam_{cfg['id']:03d}",uid=cfg["id"],preload_img=False,data_device="cpu")
    cam.original_image = torch.zeros(3,256,256); return cam

film_cams = [build_cam(c) for c in camera_cfgs]
bg_color = torch.zeros(3, device=device)
pipe = type('obj', (object,), {"debug":False,"convert_SHs_python":False,"compute_cov3D_python":False})()

class Adapter:
    def __init__(self, xyz, scale, rot, tau_raw, color_raw):
        self._xyz = xyz; self._scaling = torch.log(scale.clamp(min=1e-8))
        self._rotation = rot; self._tau_raw = tau_raw; self._color_raw = color_raw
        self.active_sh_degree=0; self.max_sh_degree=0; self.use_app=False
    @property
    def get_xyz(self): return self._xyz
    @property
    def get_scaling(self): return torch.exp(self._scaling)
    @property
    def get_rotation(self): return self._rotation / self._rotation.norm(dim=1,keepdim=True).clamp(min=1e-8)
    @property
    def get_opacity(self): return 1 - torch.exp(-F.softplus(self._tau_raw))
    @property
    def get_transparency(self): return torch.full((N,1),0.5,device=self._xyz.device)
    @property
    def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)

def render_two_pass(adapter, cam, bg_img=None):
    r1 = render(cam, adapter, pipe, bg_color, app_model=None, override_color=torch.sigmoid(adapter._color_raw), return_plane=False, return_depth_normal=False)
    C = r1["render"]
    white = torch.ones_like(torch.sigmoid(adapter._color_raw))
    r2 = render(cam, adapter, pipe, bg_color, app_model=None, override_color=white, return_plane=False, return_depth_normal=False)
    A = r2["render"].mean(dim=0, keepdim=True).clamp(0,1)
    if bg_img is not None:
        return (C + (1-A)*bg_img).clamp(0,1), C, A, r2["render"]
    return C, C, A, r2["render"]

def white_pass(adapter, cam):
    r2 = render(cam, adapter, pipe, bg_color, app_model=None, override_color=torch.ones_like(torch.sigmoid(adapter._color_raw)), return_plane=False, return_depth_normal=False)
    return r2["render"].mean(dim=0, keepdim=True).clamp(0,1)

# ═══════════════════════════════════════════════════════════
# 1. Load GT and background
# ═══════════════════════════════════════════════════════════
hdr("1. Loading GT")
GT_DYN = f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/repaired_render/tau1.0_dynamic"
BG_ONLY = f"{OUTPUT}/../background_only"
if not os.path.exists(BG_ONLY):
    BG_ONLY = f"{BASE}/experiments/stage3_2_fixed_optical_necessity/background_only"

gt_dyn = {}; bg_only = {}
state_files = ["canonical","stretch_1.10","stretch_1.25","stretch_1.50","stretch_2.00",
               "biaxial_1.10","biaxial_1.25","biaxial_1.50","twist_30","twist_60"]

for cid in [0,4,8]:
    gt_dyn[cid] = {}
    for st in state_files:
        p = f"{GT_DYN}/{st}_cam{cid:03d}.png"
        gt_dyn[cid][st] = torch.tensor(np.array(Image.open(p).convert("RGB")).astype(np.float32)/255.0, device=device).permute(2,0,1)
    bg_only[cid] = torch.tensor(np.array(Image.open(f"{BG_ONLY}/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0, device=device).permute(2,0,1)

# GT_DYN vs GT_FIXED gate recheck
GT_FIX = f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/repaired_render/tau1.0_fixed"
log("GT Gate independent recheck:")
gate_rows = []
for st in state_files:
    for ci, cid in enumerate([0,4,8]):
        gt_d = gt_dyn[cid][st]
        gt_f = torch.tensor(np.array(Image.open(f"{GT_FIX}/{st}_cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0, device=device).permute(2,0,1)
        diff = (gt_d - bg_only[cid]).abs().max(dim=0).values > 0.01
        mask = torch.tensor(binary_dilation(diff.cpu().numpy(), iterations=2), device=device)
        if mask.sum() < 10: continue
        mae = ((gt_d - gt_f).abs() * mask).sum() / mask.sum()
        gate_rows.append({"state":st,"cam":cid,"mae":mae.item()})
for st in state_files:
    vals = [r["mae"] for r in gate_rows if r["state"]==st]
    if vals: log(f"  {st:20s}: MAE={np.mean(vals):.6f}")

# ═══════════════════════════════════════════════════════════
# 2. Fresh E1
# ═══════════════════════════════════════════════════════════
hdr("2. Fresh E1 re-render")

# Train canonical model first
tr_can = torch.full((N,1), 0.0, device=device, requires_grad=True)
cr_can = torch.zeros(N,3, device=device, requires_grad=True)
opt_can = torch.optim.Adam([tr_can, cr_can], lr=1e-2)
for it in range(3000):
    opt_can.zero_grad(); loss=0
    for ci, cam in enumerate(film_cams):
        cid = [0,4,8][ci]
        adpt = Adapter(verts, scale_p, rot_init, tr_can, cr_can)
        pred,_,_,_ = render_two_pass(adpt, cam, bg_only[cid])
        gt = gt_dyn[cid]["canonical"]
        loss += (pred-gt).abs().mean() + 0.2*(1-((2*pred*gt+0.01)/(pred**2+gt**2+0.01)).mean())
    loss.backward(); opt_can.step()
    if it % 1000 == 0: log(f"  canonical iter {it}: loss={loss.item():.6f}")

can_tau_raw = tr_can.detach().clone()
can_color_raw = cr_can.detach().clone()

# Geometry + deformation
verts_t = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
z_range = (verts_t[:,2].min().item(), verts_t[:,2].max().item())

def get_state(name):
    if name == "canonical": return verts, scale_p, rot_init
    s = float(name.split("_")[1])
    if name.startswith("stretch"):
        dv = verts_t.clone(); dv[:,0] *= s; F_s = torch.diag(torch.tensor([s,1.0,1.0],device=device))
        return dv, scale_p, rot_init
    elif name.startswith("biaxial"):
        dv = verts_t.clone(); dv[:,0]*=s; dv[:,1]*=s
        return dv, scale_p*s, rot_init
    elif name.startswith("twist"):
        dv = twist_def(verts_t, int(name.split("_")[1]), z_range)
        return dv, scale_p, rot_init

e1_rows = []
for st in state_files:
    dv, sc, rt = get_state(st)
    adpt = Adapter(dv, sc, rt, can_tau_raw.clone(), can_color_raw.clone())
    for ci, cam in enumerate(film_cams):
        cid = [0,4,8][ci]
        pred,C_film,A,A_rgb = render_two_pass(adpt, cam, bg_only[cid])
        gt_d = gt_dyn[cid][st]
        diff = (gt_d - bg_only[cid]).abs().max(dim=0).values > 0.01
        mask = torch.tensor(binary_dilation(diff.cpu().numpy(), iterations=2), device=device)
        if mask.sum() < 10: continue
        mae_dynamic = ((pred-gt_d).abs()*mask).sum()/mask.sum()
        e1_rows.append({"state":st,"cam":cid,"mae_vs_dynamic":mae_dynamic.item()})

log("Fresh E1 vs GT_DYNAMIC:")
for st in state_files:
    vals = [r["mae_vs_dynamic"] for r in e1_rows if r["state"]==st]
    if vals: log(f"  {st:20s}: MAE={np.mean(vals):.6f}")

# ═══════════════════════════════════════════════════════════
# 3. White-pass alpha analysis
# ═══════════════════════════════════════════════════════════
hdr("3. Alpha field analysis")
alpha_rows = []
for st in state_files:
    dv, sc, rt = get_state(st)
    adpt = Adapter(dv, sc, rt, can_tau_raw.clone(), can_color_raw.clone())
    for ci, cam in enumerate(film_cams):
        cid = [0,4,8][ci]
        A = white_pass(adpt, cam)
        # Save alpha map
        Image.fromarray((A.squeeze(0).detach().cpu().numpy()*255).astype(np.uint8)).save(f"{OUTPUT}/alpha_maps/{st}_cam{cid:03d}.png")
        diff = (gt_dyn[cid][st] - bg_only[cid]).abs().max(dim=0).values > 0.01
        mask = binary_dilation(diff.cpu().numpy(), iterations=2)
        interior = binary_erosion(mask, iterations=3)
        if interior.sum() > 0:
            A_interior = A[0, interior].mean().item()
            tau_eff = (-torch.log(A[0, interior].clamp(min=1e-6))).mean().item()
        else:
            A_interior = 0; tau_eff = 0
        alpha_rows.append({"state":st,"cam":cid,"A_mean":A.mean().item(),"A_interior":A_interior,"tau_eff":tau_eff})

for st in state_files:
    vals = [r["A_interior"] for r in alpha_rows if r["state"]==st]
    if vals: log(f"  {st:20s}: A_interior={np.mean(vals):.6f}")

# ═══════════════════════════════════════════════════════════
# 4. Geometry component ablation
# ═══════════════════════════════════════════════════════════
hdr("4. Geometry component ablation")
ablate_states = ["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","twist_60"]
ablate_rows = []

for st in ablate_states:
    dv, sc, rt = get_state(st)
    Js = float(st.split("_")[1]) if st.startswith("stretch") else (
          float(st.split("_")[1])**2 if st.startswith("biaxial") else 1.0)
    
    for label, pos, cov in [("G0_canonical", verts, scale_p),
                            ("G1_position", dv, scale_p),
                            ("G2_covariance", verts, sc),
                            ("G3_full", dv, sc)]:
        adpt = Adapter(pos, cov, rot_init, can_tau_raw.clone(), can_color_raw.clone())
        A_interiors = []
        for ci, cam in enumerate(film_cams):
            cid = [0,4,8][ci]
            A = white_pass(adpt, cam)
            diff = (gt_dyn[cid]["canonical"] - bg_only[cid]).abs().max(dim=0).values > 0.01
            mask = binary_dilation(diff.cpu().numpy(), iterations=2)
            interior = binary_erosion(mask, iterations=3)
            if interior.sum() > 0:
                A_interiors.append(A[0, interior].mean().item())
        if A_interiors:
            tau_eff = -math.log(max(np.mean(A_interiors), 1e-6))
            ratio = tau_eff / 0.693  # ratio vs canonical effective tau at alpha=0.5
            ablate_rows.append({"state":st,"Js":Js,"variant":label,"A_mean":np.mean(A_interiors),"tau_eff":tau_eff,"ratio":ratio})

for r in ablate_rows:
    log(f"  {r['state']:20s} {r['variant']:15s}: A={r['A_mean']:.4f} tau_eff={r['tau_eff']:.4f} ratio={r['ratio']:.4f}")

# ═══════════════════════════════════════════════════════════
# 5. Oracle tau diagnostic
# ═══════════════════════════════════════════════════════════
hdr("5. Oracle tau diagnostic")
oracle_rows = []
for st in ["stretch_1.50","stretch_2.00","biaxial_1.50","twist_60"]:
    dv, sc, rt = get_state(st)
    Js = float(st.split("_")[1]) if st.startswith("stretch") else (
          float(st.split("_")[1])**2 if st.startswith("biaxial") else 1.0)
    h_ratio = 1.0 / max(Js, 1e-8)
    
    # Fixed tau (E1)
    adpt_fixed = Adapter(dv, sc, rt, can_tau_raw.clone(), can_color_raw.clone())
    
    # Oracle tau
    tau_oracle_raw = can_tau_raw.clone()
    tau_oracle = F.softplus(tau_oracle_raw) * h_ratio
    # Convert back to raw: solve softplus(raw) = tau_oracle
    # softplus(x) = log(1+exp(x)), so x = log(exp(tau_oracle)-1)
    tau_oracle_raw_new = torch.log(torch.exp(tau_oracle.clamp(max=10)) - 1).clamp(min=-10)
    adpt_oracle = Adapter(dv, sc, rt, tau_oracle_raw_new, can_color_raw.clone())
    
    for ci, cam in enumerate(film_cams):
        cid = [0,4,8][ci]
        pred_fix,_,_,_ = render_two_pass(adpt_fixed, cam, bg_only[cid])
        pred_ora,_,_,_ = render_two_pass(adpt_oracle, cam, bg_only[cid])
        gt_d = gt_dyn[cid][st]
        diff = (gt_d - bg_only[cid]).abs().max(dim=0).values > 0.01
        mask = torch.tensor(binary_dilation(diff.cpu().numpy(), iterations=2), device=device)
        if mask.sum() < 10: continue
        mae_fix = ((pred_fix-gt_d).abs()*mask).sum()/mask.sum()
        mae_ora = ((pred_ora-gt_d).abs()*mask).sum()/mask.sum()
        oracle_rows.append({"state":st,"cam":cid,"fixed_mae":mae_fix.item(),"oracle_mae":mae_ora.item()})

for st in ["stretch_1.50","stretch_2.00","biaxial_1.50","twist_60"]:
    f = [r["fixed_mae"] for r in oracle_rows if r["state"]==st]
    o = [r["oracle_mae"] for r in oracle_rows if r["state"]==st]
    if f: log(f"  {st:20s}: fixed={np.mean(f):.6f} oracle={np.mean(o):.6f}")

# ═══════════════════════════════════════════════════════════
# 6. Save
# ═══════════════════════════════════════════════════════════
hdr("6. Summary")
log("\n=== Stage 3.2.2 complete ===")
