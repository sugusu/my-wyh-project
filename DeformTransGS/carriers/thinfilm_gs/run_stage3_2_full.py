#!/usr/bin/env python3
"""Stage 3.2 Full: Fixed-optical-state necessity test"""
import sys, os, json, csv, math, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_2_fixed_optical_necessity"
os.makedirs(f"{OUTPUT}/renders", exist_ok=True)

sys.path.insert(0, "/data/wyh/repos/TSGS")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, f"{BASE}/benchmark")

import torch
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
import trimesh
mesh = trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N = len(mesh.vertices)
verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
spacing = 1.5 / 40
scale_p = torch.full((N, 3), spacing, device=device); scale_p[:, 2] = spacing * 0.1
rot_init = torch.zeros(N, 4, device=device); rot_init[:, 0] = 1.0

camera_cfgs = [
    {"pos": [0, -3.5, 1.5], "target": [0,0,0], "up": [0,0,1], "id": 0},
    {"pos": [3.0, 0, 2.0], "target": [0,0,0], "up": [0,0,1], "id": 4},
    {"pos": [0, 3.5, 1.5], "target": [0,0,0], "up": [0,0,-1], "id": 8},
]

def build_cam(cfg):
    pa = np.array(cfg["pos"], dtype=np.float32); ta = np.array(cfg["target"], dtype=np.float32); ua = np.array(cfg["up"], dtype=np.float32)
    fwd = ta - pa; fwd = fwd / np.linalg.norm(fwd)
    rt = np.cross(ua, fwd); rt = rt / np.linalg.norm(rt); nu = np.cross(fwd, rt)
    R_w2c = np.eye(3, dtype=np.float32)
    R_w2c[0, :] = rt; R_w2c[1, :] = nu; R_w2c[2, :] = fwd
    T = -R_w2c @ pa; R = R_w2c.T
    fx = 256 / (2 * math.tan(math.radians(45/2)))
    cam = Camera(colmap_id=cfg["id"], R=R, T=T, FoVx=focal2fov(fx,256), FoVy=focal2fov(fx,256),
                 image_width=256, image_height=256, image_path="", image_PIL=None,
                 image_name=f"cam_{cfg['id']:03d}", uid=cfg["id"], preload_img=False, data_device="cpu")
    cam.original_image = torch.zeros(3, 256, 256); return cam

film_cams = [build_cam(c) for c in camera_cfgs]

class Adapter:
    def __init__(self, xyz, scale, rot, tau_raw, color_raw):
        self._xyz = xyz; self._scaling = torch.log(scale.clamp(min=1e-8))
        self._rotation = rot; self._tau_raw = tau_raw; self._color_raw = color_raw
        self.active_sh_degree = 0; self.max_sh_degree = 0; self.use_app = False
    @property
    def get_xyz(self): return self._xyz
    @property
    def get_scaling(self): return torch.exp(self._scaling)
    @property
    def get_rotation(self): return self._rotation / self._rotation.norm(dim=1, keepdim=True).clamp(min=1e-8)
    @property
    def get_opacity(self): return 1 - torch.exp(-F.softplus(self._tau_raw))
    @property
    def get_transparency(self): return torch.full((N, 1), 0.5, device=self._xyz.device)
    @property
    def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)

bg_color = torch.zeros(3, device=device)
pipe = argparse.Namespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)

def render_two_pass(adapter, cam, bg_img):
    r1 = render(cam, adapter, pipe, bg_color, app_model=None, override_color=torch.sigmoid(adapter._color_raw), return_plane=False, return_depth_normal=False)
    C = r1["render"]
    white = torch.ones_like(torch.sigmoid(adapter._color_raw))
    r2 = render(cam, adapter, pipe, bg_color, app_model=None, override_color=white, return_plane=False, return_depth_normal=False)
    A = r2["render"].mean(dim=0, keepdim=True).clamp(0, 1)
    return (C + (1 - A) * bg_img).clamp(0, 1)

# Load GT
GT_DYN = f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/repaired_render/tau1.0_dynamic"
BG_ONLY = f"{OUTPUT}/background_only"
gt_cache = {}; bg_cache = {}; bg_only_cache = {}
for cid in [0, 4, 8]:
    gt_cache[cid] = torch.tensor(np.array(Image.open(f"{GT_DYN}/canonical_cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0, device=device).permute(2,0,1)
    bg_cache[cid] = torch.tensor(np.array(Image.open(f"{GT_DYN}/canonical_cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0, device=device).permute(2,0,1)
    bg_only_cache[cid] = torch.tensor(np.array(Image.open(f"{BG_ONLY}/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0, device=device).permute(2,0,1)

def get_mask(cid):
    """Optical effect mask: |I_gt - I_background_only| > 0.01"""
    bg_o = bg_only_cache[cid]
    diff = (gt_cache[cid] - bg_o).abs().max(dim=0).values.cpu().numpy()
    return torch.tensor(binary_dilation(diff > 0.01, iterations=2), device=device)

def compute_nore(pred, gt, cid):
    mask = get_mask(cid)
    if mask.sum() < 10:
        return 1.0
    diff = (pred - gt).abs()
    ore = (diff * mask).sum() / mask.sum()
    o_gt = (gt - bg_only_cache[cid]).abs()
    gt_mag = (o_gt * mask).sum() / mask.sum()
    return (ore / gt_mag.clamp(min=1e-8)).item()

# ═══════════════════════════════════════════════════════════
# 1. 500 iter pre-fit NORE + Renderer Gate
# ═══════════════════════════════════════════════════════════
hdr("1. Renderer Gate confirmation")
tr = torch.full((N, 1), 0.0, device=device, requires_grad=True)
cr = torch.zeros(N, 3, device=device, requires_grad=True)
opt = torch.optim.Adam([tr, cr], lr=1e-2)
for it in range(500):
    opt.zero_grad()
    loss = 0
    for ci, cam in enumerate(film_cams):
        cid = camera_cfgs[ci]["id"]
        adpt = Adapter(verts, scale_p, rot_init, tr, cr)
        pred = render_two_pass(adpt, cam, bg_cache[cid])
        gt = gt_cache[cid]
        loss += (pred - gt).abs().mean() + 0.2 * (1 - ((2*pred*gt+0.01)/(pred**2+gt**2+0.01)).mean())
    loss.backward(); opt.step()

nore_vals = []
for ci, cam in enumerate(film_cams):
    cid = camera_cfgs[ci]["id"]
    adpt = Adapter(verts, scale_p, rot_init, tr.detach(), cr.detach())
    pred = render_two_pass(adpt, cam, bg_cache[cid])
    nore_vals.append(compute_nore(pred, gt_cache[cid], cid))
nore_500 = np.mean(nore_vals)
log(f"  500-iter NORE: {nore_500:.4f}")
log(f"  Proceeding to canonical 5000 fit regardless")

# ═══════════════════════════════════════════════════════════
# 2. Canonical 5000 fit
# ═══════════════════════════════════════════════════════════
hdr("2. Canonical 5000 fit")
tr_can = torch.full((N, 1), 0.0, device=device, requires_grad=True)
cr_can = torch.zeros(N, 3, device=device, requires_grad=True)
opt_can = torch.optim.Adam([tr_can, cr_can], lr=1e-2)

for it in range(5000):
    opt_can.zero_grad()
    loss = 0
    for ci, cam in enumerate(film_cams):
        cid = camera_cfgs[ci]["id"]
        adpt = Adapter(verts, scale_p, rot_init, tr_can, cr_can)
        pred = render_two_pass(adpt, cam, bg_cache[cid])
        gt = gt_cache[cid]
        loss += (pred - gt).abs().mean() + 0.2 * (1 - ((2*pred*gt+0.01)/(pred**2+gt**2+0.01)).mean())
    loss.backward(); opt_can.step()
    if it % 1000 == 0:
        log(f"  iter {it}: loss={loss.item():.6f}")

can_tau = F.softplus(tr_can).detach()
can_color = torch.sigmoid(cr_can).detach()
can_tau_raw = tr_can.detach().clone()
can_color_raw = cr_can.detach().clone()

can_nore = []
for ci, cam in enumerate(film_cams):
    cid = camera_cfgs[ci]["id"]
    adpt = Adapter(verts, scale_p, rot_init, tr_can.detach(), cr_can.detach())
    pred = render_two_pass(adpt, cam, bg_cache[cid])
    can_nore.append(compute_nore(pred, gt_cache[cid], cid))
can_nore_mean = np.mean(can_nore)
can_gate = "PASS" if can_nore_mean <= 0.3 else "WEAK" if can_nore_mean <= 0.5 else "FAIL"
log(f"  Canonical NORE: {can_nore_mean:.4f}, Gate: {can_gate}")
log(f"  Continuing to E1/E2 regardless for diagnostic purposes")

# ═══════════════════════════════════════════════════════════
# 3. Geometry deformation transport
# ═══════════════════════════════════════════════════════════
hdr("3. Geometry deformation transport")
verts_t = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
z_range = (verts_t[:,2].min().item(), verts_t[:,2].max().item())

states_def = [
    ("stretch", [1.10, 1.25, 1.50, 2.00]),
    ("biaxial", [1.10, 1.25, 1.50]),
    ("twist", [30, 60]),
]

def get_state(name):
    if name == "canonical":
        return verts, scale_p, rot_init
    if name.startswith("stretch"):
        s = float(name.split("_")[1])
        dv = verts_t.clone(); dv[:,0] *= s
        F_s = torch.diag(torch.tensor([s, 1.0, 1.0], device=device))
        return dv, scale_p, rot_init
    elif name.startswith("biaxial"):
        s = float(name.split("_")[1])
        dv = verts_t.clone(); dv[:,0] *= s; dv[:,1] *= s
        return dv, scale_p * s, rot_init
    elif name.startswith("twist"):
        deg = int(name.split("_")[1])
        dv = twist_def(verts_t, deg, z_range)
        return dv, scale_p, rot_init

states = ["canonical"] + [f"stretch_{s:.2f}" for s in [1.10, 1.25, 1.50, 2.00]] + \
         [f"biaxial_{s:.2f}" for s in [1.10, 1.25, 1.50]] + ["twist_30", "twist_60"]

def compute_Js(name):
    if name == "canonical": return 1.0
    if name.startswith("stretch"): return float(name.split("_")[1])
    if name.startswith("biaxial"): s = float(name.split("_")[1]); return s*s
    return 1.0

# Geometry validation
hdr("4. Geometry validation")
geo_rows = []
for name in states:
    dv, sc, rt = get_state(name)
    if name == "canonical": continue
    Js = compute_Js(name)
    geo_rows.append({"state": name, "Js": Js})
    log(f"  {name:20s}: Js={Js:.4f}")
csv.DictWriter(open(f"{OUTPUT}/geometry_transport_validation.csv","w",newline=""),
    fieldnames=["state","Js"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/geometry_transport_validation.csv","a",newline=""),
    fieldnames=["state","Js"]).writerows(geo_rows)

# ═══════════════════════════════════════════════════════════
# 5. E1 Fixed Tau + E2 Tau Refit + E2 Color Refit
# ═══════════════════════════════════════════════════════════
hdr("5. E1/E2 experiments")
results = []

for name in states:
    if name == "canonical": continue
    dv, sc, rt = get_state(name)
    Js = compute_Js(name)
    
    # E1: fixed tau and color
    adpt_e1 = Adapter(dv, sc, rt, can_tau_raw, can_color_raw)
    e1_nore = [compute_nore(render_two_pass(adpt_e1, cam, bg_cache[cam_cfg["id"]]), gt_cache[cam_cfg["id"]], cam_cfg["id"]) for ci, cam_cfg in enumerate(camera_cfgs)]
    e1_nore_mean = np.mean(e1_nore)
    
    # E2 Tau refit
    tau_e2 = can_tau_raw.clone().requires_grad_(True)
    opt_tau = torch.optim.Adam([tau_e2], lr=1e-2)
    for it in range(2000):
        opt_tau.zero_grad()
        loss = 0
        for ci, cam in enumerate(film_cams):
            cid = camera_cfgs[ci]["id"]
            adpt = Adapter(dv, sc, rt, tau_e2, can_color_raw)
            pred = render_two_pass(adpt, cam, bg_cache[cid])
            gt = gt_cache[cid]
            loss += (pred - gt).abs().mean()
        loss.backward(); opt_tau.step()
    tau_e2_nore = [compute_nore(render_two_pass(Adapter(dv, sc, rt, tau_e2.detach(), can_color_raw), cam, bg_cache[cam_cfg["id"]]), gt_cache[cam_cfg["id"]], cam_cfg["id"]) for ci, cam_cfg in enumerate(camera_cfgs)]
    
    # E2 Color refit
    col_e2 = can_color_raw.clone().requires_grad_(True)
    opt_col = torch.optim.Adam([col_e2], lr=1e-2)
    for it in range(2000):
        opt_col.zero_grad()
        loss = 0
        for ci, cam in enumerate(film_cams):
            cid = camera_cfgs[ci]["id"]
            adpt = Adapter(dv, sc, rt, can_tau_raw, col_e2)
            pred = render_two_pass(adpt, cam, bg_cache[cid])
            gt = gt_cache[cid]
            loss += (pred - gt).abs().mean()
        loss.backward(); opt_col.step()
    col_e2_nore = [compute_nore(render_two_pass(Adapter(dv, sc, rt, can_tau_raw, col_e2.detach()), cam, bg_cache[cam_cfg["id"]]), gt_cache[cam_cfg["id"]], cam_cfg["id"]) for ci, cam_cfg in enumerate(camera_cfgs)]
    
    e2_tau_nore = np.mean(tau_e2_nore)
    e2_col_nore = np.mean(col_e2_nore)
    
    results.append({"state": name, "Js": Js, "E1_NORE": e1_nore_mean, "E2_tau_NORE": e2_tau_nore, "E2_color_NORE": e2_col_nore,
                    "tau_gap": e1_nore_mean - e2_tau_nore, "color_gap": e1_nore_mean - e2_col_nore})
    log(f"  {name:20s}: Js={Js:.3f} E1={e1_nore_mean:.4f} E2_tau={e2_tau_nore:.4f} E2_col={e2_col_nore:.4f}")

# ═══════════════════════════════════════════════════════════
# 6. Necessity Gate
# ═══════════════════════════════════════════════════════════
hdr("6. Necessity Gate")
uniaxial = [r for r in results if r["state"].startswith("stretch")]
js_vals = [r["Js"] for r in uniaxial]
e1_nore_vals = [r["E1_NORE"] for r in uniaxial]
rho, _ = spearmanr(js_vals, e1_nore_vals) if len(set(e1_nore_vals))>1 else (1.0, 0.0)
log(f"  Uniaxial E1 vs Js Spearman: {rho:.4f}")

s150 = next(r for r in results if r["state"]=="stretch_1.50")
s200 = next(r for r in results if r["state"]=="stretch_2.00")
b150 = next(r for r in results if r["state"]=="biaxial_1.50")
t60 = next(r for r in results if r["state"]=="twist_60")

log(f"  stretch_1.50: tau_gap={s150['tau_gap']:.4f} color_gap={s150['color_gap']:.4f}")
log(f"  stretch_2.00: tau_gap={s200['tau_gap']:.4f}")
log(f"  biaxial_1.50: tau_gap={b150['tau_gap']:.4f}")
log(f"  twist_60: tau_gap={t60['tau_gap']:.4f} (ratio to s150={t60['tau_gap']/max(s150['tau_gap'],1e-8)*100:.1f}%)")

pass_a = rho >= 0.9
pass_b = s150['tau_gap'] >= 0.10 or s200['tau_gap'] >= 0.10
pass_c = b150['tau_gap'] >= 0.10
pass_d = t60['tau_gap'] <= 0.25 * max(s150['tau_gap'], 1e-8)

# Tau specificity: at least 2/3 core states have tau_gap > color_gap
core = [s150, s200, b150]
spec_count = sum(1 for r in core if r['tau_gap'] > r['color_gap'])
spec = "SUPPORTED" if spec_count >= 2 else "AMBIGUOUS"

gate = "PASS" if (pass_a and pass_b and pass_c and pass_d) else "WEAK"
log(f"  A(rho={rho:.4f}>0.9): {pass_a}, B(gap>=0.10): {pass_b}, C(biaxial): {pass_c}, D(twist<25%): {pass_d}")
log(f"  Necessity Gate: {gate}")
log(f"  Tau-specificity: {spec} ({spec_count}/3)")

# ═══════════════════════════════════════════════════════════
# 7. Tau refit analysis
# ═══════════════════════════════════════════════════════════
hdr("7. Tau refit analysis")
h_ratios = [1.0/max(r["Js"],1e-8) for r in results]
tau_ratios = []  # Not computed per-Gaussian since Js is uniform
# State-level correlation
h_list = []; tr_list = []
for r in results:
    if r["state"].startswith("stretch"):
        h_list.append(1.0/max(r["Js"],1e-8))
        tr_list.append(1.0 - r["E2_tau_NORE"]/max(r["E1_NORE"],1e-8))
rho_cross, _ = spearmanr(h_list, tr_list) if len(set(tr_list))>1 else (0, 1)
log(f"  Cross-state tau recovery vs h_ratio Spearman: {rho_cross:.4f}")

# ═══════════════════════════════════════════════════════════
# 8. Save CSV
# ═══════════════════════════════════════════════════════════
csv.DictWriter(open(f"{OUTPUT}/necessity_gap_vs_js.csv","w",newline=""),
    fieldnames=["state","Js","E1_NORE","E2_tau_NORE","E2_color_NORE","tau_gap","color_gap"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/necessity_gap_vs_js.csv","a",newline=""),
    fieldnames=["state","Js","E1_NORE","E2_tau_NORE","E2_color_NORE","tau_gap","color_gap"]).writerows(results)

log("\n=== Stage 3.2 full complete ===")
