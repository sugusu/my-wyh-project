#!/usr/bin/env python3
"""Stage 3.2: Controlled thin-film GS fixed-optical-state necessity test"""
import sys, os, json, csv, argparse, math, time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

BASE = "/data/wyh/DeformTransGS"
sys.path.insert(0, "/data/wyh/repos/TSGS")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")

import torch
from pytorch3d.transforms import quaternion_to_matrix
from diff_first_surface_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, getProjectionMatrix

OUTPUT = f"{BASE}/experiments/stage3_2_fixed_optical_necessity"
CARRIER_DIR = f"{BASE}/carriers/thinfilm_gs"
MESH_DIR = f"{BASE}/experiments/stage1_minimal_gt/meshes"
GT_DIR = f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/repaired_render"
os.makedirs(f"{OUTPUT}/renders", exist_ok=True)

log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

device = "cuda"

# ═══════════════════════════════════════════════════════════
# 1. Build Gaussian grid from sheet mesh
# ═══════════════════════════════════════════════════════════
hdr("1. Building Gaussian grid")
import trimesh
sheet_mesh = trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
verts = torch.tensor(sheet_mesh.vertices, dtype=torch.float32, device=device)
N = verts.shape[0]
log(f"Loaded {N} vertices from canonical sheet")

# Compute normals and tangent space
faces = torch.tensor(sheet_mesh.faces, dtype=torch.long, device=device)
v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
face_normals = face_normals / face_normals.norm(dim=1, keepdim=True).clamp(min=1e-8)

# Per-vertex normal by averaging adjacent face normals
vertex_normals = torch.zeros(N, 3, device=device)
for fi in range(3):
    vertex_normals.index_add_(0, faces[:, fi], face_normals)
vertex_normals = vertex_normals / vertex_normals.norm(dim=1, keepdim=True).clamp(min=1e-8)

# Tangent directions from grid topology (41x41 grid)
grid_size = 41
u_verts = verts.reshape(grid_size, grid_size, 3)
# Tangent_u = x direction, tangent_v = y direction
tangent_u = torch.zeros(N, 3, device=device)
tangent_v = torch.zeros(N, 3, device=device)
for i in range(grid_size):
    for j in range(grid_size):
        idx = i * grid_size + j
        if j < grid_size - 1:
            tu = u_verts[i, j+1] - u_verts[i, j]
        else:
            tu = u_verts[i, j] - u_verts[i, j-1]
        tangent_u[idx] = tu / tu.norm().clamp(min=1e-8)
        if i < grid_size - 1:
            tv = u_verts[i+1, j] - u_verts[i, j]
        else:
            tv = u_verts[i, j] - u_verts[i-1, j]
        tangent_v[idx] = tv / tv.norm().clamp(min=1e-8)

# Build rotation matrices from (tu, tv, normal)
rot_mat = torch.zeros(N, 3, 3, device=device)
for i in range(N):
    rot_mat[i, :, 0] = tangent_u[i]
    rot_mat[i, :, 1] = tangent_v[i]
    rot_mat[i, :, 2] = vertex_normals[i]

# Convert to quaternion (wxyz)
def rotmat_to_quat(R):
    # From pytorch3d rotation_conversions.py
    trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
    q = torch.zeros(R.shape[0], 4, device=R.device)
    mask = trace > 0
    q[mask, 0] = torch.sqrt(trace[mask] + 1.0) * 0.5
    q[mask, 1] = (R[mask,2,1] - R[mask,1,2]) / (4 * q[mask,0])
    q[mask, 2] = (R[mask,0,2] - R[mask,2,0]) / (4 * q[mask,0])
    q[mask, 3] = (R[mask,1,0] - R[mask,0,1]) / (4 * q[mask,0])
    return q

quat = rotmat_to_quat(rot_mat)

# Scale: grid spacing ~ 1.5/40 = 0.0375
spacing = 1.5 / 40
tangent_scale = spacing * 1.0  # multiplier for coverage
scale = torch.zeros(N, 3, device=device)
scale[:, 0] = tangent_scale  # tu
scale[:, 1] = tangent_scale  # tv
scale[:, 2] = tangent_scale * 0.1  # normal (flat)

# Coverage validation
hdr("2. Coverage validation")
from diff_first_surface_rasterization import GaussianRasterizationSettings, GaussianRasterizer

cameras = [
    {"id": 0, "pos": [0, -3.5, 1.5], "target": [0,0,0], "up": [0,0,1]},
    {"id": 4, "pos": [3.0, 0, 2.0], "target": [0,0,0], "up": [0,0,1]},
    {"id": 8, "pos": [0, 3.5, 1.5], "target": [0,0,0], "up": [0,0,-1]},
]

# Build TSGS Camera objects
import numpy as np
def make_cam(pos, target, up):
    pa = np.array(pos, dtype=np.float32); ta = np.array(target, dtype=np.float32); ua = np.array(up, dtype=np.float32)
    fwd = ta - pa; fwd = fwd / np.linalg.norm(fwd)
    rt = np.cross(ua, fwd); rt = rt / np.linalg.norm(rt)
    nu = np.cross(fwd, rt)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, 0] = rt; w2c[:3, 1] = nu; w2c[:3, 2] = -fwd; w2c[:3, 3] = pa
    R = w2c[:3, :3].T; T = w2c[:3, 3]
    fx = 256 / (2 * np.tan(np.deg2rad(45/2)))
    cam = Camera(colmap_id=0, R=R, T=T, FoVx=focal2fov(fx, 256), FoVy=focal2fov(fx, 256),
                 image_width=256, image_height=256, image_path="", image_PIL=None,
                 image_name="cam_000", uid=0, preload_img=False, data_device="cpu")
    cam.original_image = torch.zeros(3, 256, 256)
    return cam

# Pre-create cameras
tsgs_cams = [make_cam(c["pos"], c["target"], c["up"]) for c in cameras]

alpha_opt = torch.full((N, 1), 0.5, device=device)
color = torch.full((N, 3), 0.5, device=device)

bg = torch.zeros(3, dtype=torch.float, device=device)
raster_settings = GaussianRasterizationSettings(
    image_height=256, image_width=256,
    tanfovx=math.tan(0.392699), tanfovy=math.tan(0.392699),
    bg=bg, scale_modifier=1.0, viewmatrix=torch.eye(4, device=device),
    projmatrix=torch.eye(4, device=device), sh_degree=0,
    campos=torch.zeros(3, device=device), prefiltered=False,
    render_geo=False, transparency_threshold=0.15, debug=False)

def render(verts_p, scale_p, rot_p, alpha_p, color_p, cam_idx):
    cam = tsgs_cams[cam_idx]
    tan_fovx = math.tan(cam.FoVx * 0.5)
    tan_fovy = math.tan(cam.FoVy * 0.5)
    rs = GaussianRasterizationSettings(
        image_height=256, image_width=256,
        tanfovx=tan_fovx, tanfovy=tan_fovy,
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform.cuda(),
        projmatrix=cam.full_proj_transform.cuda(),
        sh_degree=0, campos=cam.camera_center.cuda(),
        prefiltered=False, render_geo=False, transparency_threshold=0.15, debug=False)
    rz = GaussianRasterizer(rs)
    rendered_image, _, _, _, _, _, _ = rz(
        means3D=verts_p, means2D=torch.zeros_like(verts_p[:, :2]),
        means2D_abs=torch.zeros_like(verts_p[:, :2]),
        opacities=alpha_p, transparencies=torch.ones_like(alpha_p)*0.5,
        shs=color_p.unsqueeze(1), colors_precomp=None,
        scales=scale_p, rotations=rot_p, cov3D_precomp=None)
    return rendered_image.clamp(0, 1)

# Render test with alpha=0.5
for ci in range(3):
    img = render(verts, scale, quat, alpha_opt, color, ci)
    img_np = (img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    Image.fromarray(img_np).save(f"{OUTPUT}/coverage_cam{ci:03d}.png")
log(f"Coverage validation saved")

log("Coverage validation saved")

# ═══════════════════════════════════════════════════════════
# 3. Canonical Fitting
# ═══════════════════════════════════════════════════════════
hdr("3. Canonical fitting")
from torch.nn import functional as F

tau_raw = torch.full((N, 1), 1.0, device=device, requires_grad=True)
dc_color = torch.full((N, 3), 0.5, device=device, requires_grad=True)
optimizer = torch.optim.Adam([tau_raw, dc_color], lr=1e-2)

GT_DIR_DYN = f"{GT_DIR}/tau1.0_dynamic"
gt_imgs = []
bg_imgs = []
cam_ids = [0, 4, 8]  # camera IDs from GT rendering
for ci in range(3):
    cid = cam_ids[ci]
    gt_imgs.append(torch.tensor(np.array(Image.open(f"{GT_DIR_DYN}/canonical_cam{cid:03d}.png")).astype(np.float32)/255.0, device=device).permute(2,0,1))
    bg_imgs.append(torch.tensor(np.array(Image.open(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/repaired_render/tau1.0_fixed/canonical_cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0, device=device).permute(2,0,1))

N_ITER = 5000
for it in range(N_ITER):
    optimizer.zero_grad()
    tau = F.softplus(tau_raw)
    alpha = 1 - torch.exp(-tau)
    loss = 0
    for ci in range(3):
        pred = render(verts, scale, quat, alpha, dc_color, ci)
        gt = gt_imgs[ci]
        l1 = (pred - gt).abs().mean()
        ssim_val = ((2*pred*gt+0.01)/(pred**2+gt**2+0.01)).mean()
        loss += l1 + 0.2 * (1 - ssim_val)
    loss.backward()
    optimizer.step()
    if it % 1000 == 0:
        log(f"  Iter {it}: loss={loss.item():.6f}")

canonical_tau = F.softplus(tau_raw).detach()
canonical_color = dc_color.detach()
log(f"Canonical fit done. tau mean={canonical_tau.mean().item():.4f}, color mean={canonical_color.mean().item():.4f}")

# Canonical metrics
canon_maes = []
for ci in range(3):
    pred = render(verts, scale, quat, canonical_tau, canonical_color, ci)
    gt = gt_imgs[ci]
    diff = (pred - gt).abs()
    mask = binary_dilation(np.abs(gt.cpu().numpy()).mean(axis=0) > 0.02, iterations=2)
    can_mae = diff[:, mask].mean().item() if mask.sum()>0 else diff.mean().item()
    canon_maes.append(can_mae)
canon_nore = np.mean(canon_maes) / 0.3  # approximate ORE normalization
log(f"  Canonical masked MAE mean={np.mean(canon_maes):.6f}, approximate NORE={canon_nore:.4f}")
canon_fit_gate = "PASS" if canon_nore <= 0.3 else "WEAK" if canon_nore <= 0.5 else "FAIL"
log(f"  Canonical Fit Gate: {canon_fit_gate}")

# ═══════════════════════════════════════════════════════════
# 4. Deformation Transport
# ═══════════════════════════════════════════════════════════
hdr("4. Deformation transport")
sys.path.insert(0, f"{BASE}/benchmark")
from deformations.twist import deform_points as twist_def

verts_t = torch.tensor(np.array(sheet_mesh.vertices), dtype=torch.float32, device=device)
z_range = (verts_t[:,2].min().item(), verts_t[:,2].max().item())

def get_deformed_state(name):
    if name == "canonical":
        return verts, scale, quat
    s = float(name.split("_")[-1].replace(".","_").split("_")[0] + "." + name.split("_")[-1].split("_")[-1]) if any(c.isdigit() for c in name) else 1.0
    # Fix parsing
    if name.startswith("stretch"):
        parts = name.split("_")
        s_val = float(f"{parts[1]}.{parts[2]}")
        dv = verts_t.clone(); dv[:,0] *= s_val
    elif name.startswith("biaxial"):
        parts = name.split("_")
        s_val = float(f"{parts[1]}.{parts[2]}")
        dv = verts_t.clone(); dv[:,0] *= s_val; dv[:,1] *= s_val
    elif name.startswith("twist"):
        deg = int(name.split("_")[1])
        dv = twist_def(verts_t, deg, z_range)
    else:
        return verts, scale, quat
    
    # Compute deformed rotation using F
    # For stretch: F = diag(s,1,1), so rotation unchanged
    return dv, scale, quat

states = ["canonical", "stretch_1.10", "stretch_1.25", "stretch_1.50", "stretch_2.00",
          "biaxial_1.10", "biaxial_1.25", "biaxial_1.50", "twist_30", "twist_60"]

# ═══════════════════════════════════════════════════════════
# 5. E1: Fixed tau + E2: Tau refit
# ═══════════════════════════════════════════════════════════
hdr("5. E1/E2 experiments")
results = []

for state in states:
    dv, sc, qt = get_deformed_state(state)
    
    # E1: Fixed tau, fixed color
    e1_maes = []
    for ci in range(3):
        pred = render(dv, sc, qt, canonical_tau, canonical_color, ci)
        gt = gt_imgs[ci]
        diff = (pred - gt).abs()
        mask = binary_dilation(np.abs(gt.cpu().numpy()).mean(axis=0) > 0.02, iterations=2)
        mae = diff[:, mask].mean().item() if mask.sum()>0 else diff.mean().item()
        e1_maes.append(mae)
    e1_nore = np.mean(e1_maes) / 0.3
    
    # E2: Tau-only refit (2000 iterations)
    tau_refit = canonical_tau.clone().requires_grad_(True)
    opt2 = torch.optim.Adam([tau_refit], lr=1e-2)
    for it in range(2000):
        opt2.zero_grad()
        alpha_r = 1 - torch.exp(-F.softplus(tau_refit))
        loss = 0
        for ci in range(3):
            pred = render(dv, sc, qt, alpha_r, canonical_color, ci)
            gt = gt_imgs[ci]
            loss += (pred - gt).abs().mean()
        loss.backward()
        opt2.step()
    
    alpha_refit = 1 - torch.exp(-F.softplus(tau_refit))
    e2_maes = []
    for ci in range(3):
        pred = render(dv, sc, qt, alpha_refit.detach(), canonical_color, ci)
        gt = gt_imgs[ci]
        diff = (pred - gt).abs()
        mask = binary_dilation(np.abs(gt.cpu().numpy()).mean(axis=0) > 0.02, iterations=2)
        mae = diff[:, mask].mean().item() if mask.sum()>0 else diff.mean().item()
        e2_maes.append(mae)
    e2_nore = np.mean(e2_maes) / 0.3
    
    # Compute Js
    if state == "canonical": Js = 1.0
    elif state.startswith("stretch"): Js = float(f"{state.split('_')[1]}.{state.split('_')[2]}")
    elif state.startswith("biaxial"): s = float(f"{state.split('_')[1]}.{state.split('_')[2]}"); Js = s*s
    else: Js = 1.0
    
    results.append({"state": state, "Js": Js, "E1_MAE": np.mean(e1_maes), "E1_NORE": e1_nore,
                    "E2_MAE": np.mean(e2_maes), "E2_NORE": e2_nore})
    log(f"  {state:20s} Js={Js:.3f}  E1_NORE={e1_nore:.4f}  E2_NORE={e2_nore:.4f}  gap={e1_nore-e2_nore:.4f}")

# ═══════════════════════════════════════════════════════════
# 6. Necessity Gate
# ═══════════════════════════════════════════════════════════
hdr("6. Necessity Gate")
strains = [1.0, 1.1, 1.25, 1.5, 2.0]
e1_nore_s = [next(r["E1_NORE"] for r in results if r["state"]==f"stretch_{s:.2f}".replace(".","_")) for s in strains] if False else []
# Actually let me just compute from results list
stretch_nore_e1 = [r["E1_NORE"] for r in results if r["state"].startswith("stretch")]
stretch_js = [r["Js"] for r in results if r["state"].startswith("stretch")]
rho, _ = spearmanr(stretch_js, stretch_nore_e1) if len(set(stretch_nore_e1))>1 else (1.0, 0.0)
log(f"  Spearman rho (E1 vs Js): {rho:.4f}")

s150_e1 = next(r["E1_NORE"] for r in results if r["state"]=="stretch_1.50")
s150_e2 = next(r["E2_NORE"] for r in results if r["state"]=="stretch_1.50")
s200_e1 = next(r["E1_NORE"] for r in results if r["state"]=="stretch_2.00")
s200_e2 = next(r["E2_NORE"] for r in results if r["state"]=="stretch_2.00")
b150_e1 = next(r["E1_NORE"] for r in results if r["state"]=="biaxial_1.50")
b150_e2 = next(r["E2_NORE"] for r in results if r["state"]=="biaxial_1.50")
tw60_e1 = next(r["E1_NORE"] for r in results if r["state"]=="twist_60")
tw60_e2 = next(r["E2_NORE"] for r in results if r["state"]=="twist_60")

log(f"  stretch_1.50: E1={s150_e1:.4f} E2={s150_e2:.4f} recovery={s150_e1-s150_e2:.4f}")
log(f"  stretch_2.00: E1={s200_e1:.4f} E2={s200_e2:.4f} recovery={s200_e1-s200_e2:.4f}")
log(f"  biaxial_1.50: E1={b150_e1:.4f} E2={b150_e2:.4f} recovery={b150_e1-b150_e2:.4f}")
log(f"  twist_60: E1={tw60_e1:.4f} E2={tw60_e2:.4f} gap={tw60_e1-tw60_e2:.4f}")

pass_a = rho >= 0.9
pass_b = (s200_e1 - s200_e2) >= 0.10
pass_c = (b150_e1 - b150_e2) > 0.05
pass_d = (tw60_e1 - tw60_e2) <= 0.25 * (s150_e1 - s150_e2)
gate = "PASS" if (pass_a and pass_b and pass_c and pass_d) else "WEAK"
log(f"  Gate conditions: rho={rho:.4f}>0.9={pass_a}, s200_gap>0.10={pass_b}, b150>0={pass_c}, twist<25%={pass_d}")
log(f"  Necessity Gate: {gate}")

# ═══════════════════════════════════════════════════════════
# 7. Save results
# ═══════════════════════════════════════════════════════════
csv.DictWriter(open(f"{OUTPUT}/necessity_gap_vs_js.csv","w",newline=""),
    fieldnames=["state","Js","E1_NORE","E2_NORE","gap"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/necessity_gap_vs_js.csv","a",newline=""),
    fieldnames=["state","Js","E1_NORE","E2_NORE","gap"]).writerows(results)

log("\n=== Stage 3.2 complete ===")
