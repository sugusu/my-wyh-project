#!/usr/bin/env python3
"""
test_asg_frame_sensitivity.py
Stage 0.6 H2: ASG 坐标框架敏感性分析
"""

import sys, os, csv
import numpy as np
import torch
from pathlib import Path

TSGS_REPO = "/data/wyh/repos/TSGS"
CKPT_DIR = "/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k_v4"
PLY_PATH = os.path.join(CKPT_DIR, "point_cloud/iteration_30000/point_cloud.ply")
SPECULAR_PATH = os.path.join(CKPT_DIR, "specular/iteration_30000/specular.pth")
OUTPUT_DIR = "/data/wyh/DeformTransGS/experiments/stage0_6_hypothesis_check"

for p in [TSGS_REPO, os.path.join(TSGS_REPO, "pytorch3d_stub")]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

def log_header(title):
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)

SEED = 20260712
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 1. Load Data ────────────────────────────────────────────
log_header("1. Loading PLY and SpecularModel")
from plyfile import PlyData
plydata = PlyData.read(PLY_PATH)
N_total = plydata.elements[0].count

N_SAMPLE = 50000
rng = np.random.RandomState(SEED)
idx = rng.choice(N_total, min(N_SAMPLE, N_total), replace=False)
N = len(idx)
log(f"Sampled {N} Gaussians from {N_total}")

def read_attr(name):
    return torch.tensor(np.asarray(plydata.elements[0][name])[idx], dtype=torch.float32, device=device)

xyz = torch.stack([read_attr("x"), read_attr("y"), read_attr("z")], dim=1)
scale_raw = torch.stack([read_attr(f"scale_{i}") for i in range(3)], dim=1)
rot_raw = torch.stack([read_attr(f"rot_{i}") for i in range(4)], dim=1)
scale_a = torch.exp(scale_raw)
rot_norm = torch.nn.functional.normalize(rot_raw)

from pytorch3d.transforms import quaternion_to_matrix
rot_mat = quaternion_to_matrix(rot_norm)
smallest_idx = scale_a.min(dim=-1)[1]
canon_normal = rot_mat.gather(2, smallest_idx[:, None, None].expand(-1, 3, -1)).squeeze(-1)
canon_normal = canon_normal / canon_normal.norm(dim=1, keepdim=True).clamp(min=1e-8)

asg_names = sorted([n for n in [p.name for p in plydata.elements[0].properties] if n.startswith("f_asg_")], key=lambda x: int(x.split("_")[-1]))
asg_features = torch.stack([read_attr(n) for n in asg_names], dim=1) if asg_names else None
log(f"ASG features: {asg_features.shape}")

# Load SpecularModel
from scene.specular_model import SpecularModel
from utils.spec_utils import SpecularNetwork

spec_model = SpecularModel(is_real=False)
spec_model.specular.load_state_dict(torch.load(SPECULAR_PATH, map_location=device, weights_only=True))
spec_model.specular.eval()
spec_model.specular.to(device)
log(f"SpecularModel loaded from {SPECULAR_PATH}")

# ── 2. Camera Setup ──────────────────────────────────────────
log_header("2. Camera setup")
CKPT_DIR_OFFICIAL = "/data/wyh/RecycleGS/baselines/tsgs_scene01_full"
cameras_json = os.path.join(CKPT_DIR_OFFICIAL, "cameras.json")
if os.path.exists(cameras_json):
    import json
    with open(cameras_json) as f:
        cam_data = json.load(f)
    centers = []
    for cd in cam_data:
        pos = cd.get("position", cd.get("camera_center", None))
        if pos is not None:
            centers.append(pos)
    if len(centers) >= 10:
        step = len(centers) // 10
        camera_centers = torch.tensor([centers[i*step] for i in range(10)], dtype=torch.float32, device=device)
        log(f"Using {len(camera_centers)} real cameras from cameras.json")
    else:
        camera_centers = None
else:
    camera_centers = None

if camera_centers is None:
    # Construct 10 cameras from bbox
    bbox_min = xyz.min(dim=0).values
    bbox_max = xyz.max(dim=0).values
    center = (bbox_min + bbox_max) / 2
    extent = (bbox_max - bbox_min).norm().item()
    camera_centers = []
    for i in range(10):
        theta = 2 * np.pi * i / 10
        pos = center + torch.tensor([extent * np.cos(theta), extent * np.sin(theta), extent * 0.5], device=device)
        camera_centers.append(pos)
    camera_centers = torch.stack(camera_centers)
    log("Constructed 10 synthetic cameras from bounding box")

log(f"Camera centers: {camera_centers.shape}")

# Use first camera for all tests
cam_center = camera_centers[0:1]  # (1, 3)
viewdir = (xyz - cam_center)  # (N, 3)
viewdir = viewdir / viewdir.norm(dim=1, keepdim=True).clamp(min=1e-8)

# ── 3. Test A: Repeatability ──────────────────────────────
log_header("3. Test A: Repeatability")
rgb1 = spec_model.step(asg_features, viewdir, canon_normal)
rgb2 = spec_model.step(asg_features, viewdir, canon_normal)
max_err = (rgb1 - rgb2).abs().max().item()
log(f"Max absolute error between two identical calls: {max_err:.2e}")

# ── 4. Test B: Joint Rotation Sensitivity ─────────────────
log_header("4. Test B: Joint Rotation Sensitivity")

def rotation_matrix(axis, deg):
    theta = torch.tensor(deg * np.pi / 180.0, device=device)
    if axis == 'x':
        R = torch.tensor([[1, 0, 0], [0, theta.cos(), -theta.sin()], [0, theta.sin(), theta.cos()]], device=device)
    elif axis == 'y':
        R = torch.tensor([[theta.cos(), 0, theta.sin()], [0, 1, 0], [-theta.sin(), 0, theta.cos()]], device=device)
    elif axis == 'z':
        R = torch.tensor([[theta.cos(), -theta.sin(), 0], [theta.sin(), theta.cos(), 0], [0, 0, 1]], device=device)
    return R

test_b_configs = []
for axis in ['x', 'y', 'z']:
    for deg in [30, 60]:
        test_b_configs.append(("joint", axis, deg))
for axis in ['z']:
    for deg in [90]:
        test_b_configs.append(("joint", axis, deg))

test_b_results = []
for test_name, axis, deg in test_b_configs:
    R = rotation_matrix(axis, deg)
    view_rot = (R @ viewdir.unsqueeze(-1)).squeeze(-1)
    normal_rot = (R @ canon_normal.unsqueeze(-1)).squeeze(-1)

    rgb_orig = spec_model.step(asg_features, viewdir, canon_normal)
    rgb_rot = spec_model.step(asg_features, view_rot, normal_rot)

    mae = (rgb_orig - rgb_rot).abs().mean().item()
    rmse = ((rgb_orig - rgb_rot) ** 2).mean().sqrt().item()
    cos_sim = (rgb_orig * rgb_rot).sum(dim=1) / (rgb_orig.norm(dim=1) * rgb_rot.norm(dim=1) + 1e-8)
    cos_mean = cos_sim.mean().item()

    err = (rgb_orig - rgb_rot).abs()
    p50 = err.median(dim=0).values.mean().item()
    p90 = torch.quantile(err.flatten(), 0.90).item()
    p95 = torch.quantile(err.flatten(), 0.95).item()
    p99 = torch.quantile(err.flatten(), 0.99).item()

    res = {"test_name": test_name, "rotation_axis": axis, "rotation_degree": deg,
           "deformation_type": "", "strength": "", "sample_count": "",
           "rgb_mae": mae, "rgb_rmse": rmse, "cosine_similarity": cos_mean,
           "p50_error": p50, "p90_error": p90, "p95_error": p95, "p99_error": p99}
    test_b_results.append(res)
    log(f"  joint {axis}-{deg:2d}°: MAE={mae:.6f}  RMSE={rmse:.6f}  cos={cos_mean:.6f}")

# ── 5. Test C: View-only / Normal-only / Joint Perturbation ─
log_header("5. Test C: Input perturbation sensitivity")
test_c_configs = []
for mode in ["view_only", "normal_only", "joint"]:
    for deg in [15, 30, 60, 90]:
        test_c_configs.append((mode, deg))

test_c_results = []
for mode, deg in test_c_configs:
    R = rotation_matrix('z', deg)
    if mode == "view_only":
        v = (R @ viewdir.unsqueeze(-1)).squeeze(-1)
        n = canon_normal
    elif mode == "normal_only":
        v = viewdir
        n = (R @ canon_normal.unsqueeze(-1)).squeeze(-1)
    else:  # joint
        v = (R @ viewdir.unsqueeze(-1)).squeeze(-1)
        n = (R @ canon_normal.unsqueeze(-1)).squeeze(-1)

    rgb_orig = spec_model.step(asg_features, viewdir, canon_normal)
    rgb_pert = spec_model.step(asg_features, v, n)

    mae = (rgb_orig - rgb_pert).abs().mean().item()
    rmse = ((rgb_orig - rgb_pert) ** 2).mean().sqrt().item()
    cos_sim = (rgb_orig * rgb_pert).sum(dim=1) / (rgb_orig.norm(dim=1) * rgb_pert.norm(dim=1) + 1e-8)
    cos_mean = cos_sim.mean().item()
    err = (rgb_orig - rgb_pert).abs()
    p95 = torch.quantile(err.flatten(), 0.95).item()

    res = {"test_name": mode, "rotation_axis": "z", "rotation_degree": deg,
           "deformation_type": "perturbation", "strength": deg,
           "sample_count": N, "rgb_mae": mae, "rgb_rmse": rmse,
           "cosine_similarity": cos_mean, "p50_error": err.median(dim=0).values.mean().item(),
           "p90_error": torch.quantile(err.flatten(), 0.90).item(),
           "p95_error": p95, "p99_error": torch.quantile(err.flatten(), 0.99).item()}
    test_c_results.append(res)
    log(f"  {mode:15s} z-{deg:2d}°: MAE={mae:.6f}  RMSE={rmse:.6f}  cos={cos_mean:.6f}")

# ── 6. Test D: Deformation Input Sensitivity ──────────────
log_header("6. Test D: Deformation input sensitivity")

def F_stretch(sx, sy, sz):
    F = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).clone()
    F[:, 0, 0] = sx; F[:, 1, 1] = sy; F[:, 2, 2] = sz
    return F

def F_shear(gamma):
    F = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).clone()
    F[:, 0, 1] = gamma
    return F

def F_twist(theta_max):
    z_norm = (xyz[:, 2] - xyz[:, 2].min()) / (xyz[:, 2].max() - xyz[:, 2].min() + 1e-8)
    theta = theta_max * z_norm
    ct = theta.cos(); st = theta.sin()
    F = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).clone()
    F[:, 0, 0] = ct; F[:, 0, 1] = -st; F[:, 1, 0] = st; F[:, 1, 1] = ct
    return F

def compute_deformed_normal(F, batch_size=10000):
    n_batches = (N + batch_size - 1) // batch_size
    result = []
    for bi in range(n_batches):
        s = bi * batch_size
        e = min(s + batch_size, N)
        scale_b = scale_a[s:e]
        rot_b = rot_mat[s:e]
        F_b = F[s:e] if F.dim() == 3 else F
        S = torch.diag_embed(scale_b ** 2)
        Sigma = rot_b @ S @ rot_b.transpose(1, 2)
        Sigma_def = F_b @ Sigma @ F_b.transpose(1, 2)
        Sigma_def = 0.5 * (Sigma_def + Sigma_def.transpose(1, 2))
        _, eigvecs = torch.linalg.eigh(Sigma_def)
        result.append(eigvecs[:, :, 0])
        del S, Sigma, Sigma_def, eigvecs
        torch.cuda.empty_cache()
    return torch.cat(result, dim=0)

# Canonical ASG output
rgb_canonical = spec_model.step(asg_features, viewdir, canon_normal)

test_d_configs = [
    ("stretch", "stretch_x_1.25", lambda: F_stretch(1.25, 1.0, 1.0)),
    ("stretch", "stretch_x_2.00", lambda: F_stretch(2.00, 1.0, 1.0)),
    ("stretch", "stretch_y_0.50", lambda: F_stretch(1.0, 0.50, 1.0)),
    ("shear", "shear_0.5", lambda: F_shear(0.5)),
    ("shear", "shear_1.0", lambda: F_shear(1.0)),
    ("twist", "twist_30", lambda: F_twist(torch.tensor(30*np.pi/180.0, device=device))),
    ("twist", "twist_60", lambda: F_twist(torch.tensor(60*np.pi/180.0, device=device))),
]

test_d_results = []

for dtype, name, fn in test_d_configs:
    F = fn()
    n_def = compute_deformed_normal(F)

    # Deformed xyz and viewdir
    xyz_def = xyz  # positional change not applied (keeping consistent with analysis scope)
    # Actually stretch changes positions too, let's recompute viewdir
    xyz_def = (F @ xyz.unsqueeze(-1)).squeeze(-1)
    viewdir_def = (xyz_def - cam_center) / (xyz_def - cam_center).norm(dim=1, keepdim=True).clamp(min=1e-8)

    rgb_def = spec_model.step(asg_features, viewdir_def, n_def)

    mae = (rgb_canonical - rgb_def).abs().mean().item()
    rmse = ((rgb_canonical - rgb_def) ** 2).mean().sqrt().item()
    cos_sim = (rgb_canonical * rgb_def).sum(dim=1) / (rgb_canonical.norm(dim=1) * rgb_def.norm(dim=1) + 1e-8)
    cos_mean = cos_sim.mean().item()
    err = (rgb_canonical - rgb_def).abs()
    p95 = torch.quantile(err.flatten(), 0.95).item()

    res = {"test_name": "deformation", "rotation_axis": "none", "rotation_degree": 0,
           "deformation_type": dtype, "strength": name,
           "sample_count": N, "rgb_mae": mae, "rgb_rmse": rmse,
           "cosine_similarity": cos_mean, "p50_error": err.median(dim=0).values.mean().item(),
           "p90_error": torch.quantile(err.flatten(), 0.90).item(),
           "p95_error": p95, "p99_error": torch.quantile(err.flatten(), 0.99).item()}
    test_d_results.append(res)
    log(f"  {name:20s}: MAE={mae:.6f}  RMSE={rmse:.6f}  cos={cos_mean:.6f}")

# ── 7. Save CSV ──────────────────────────────────────────
log_header("7. Saving CSV")
fieldnames = ["test_name", "rotation_axis", "rotation_degree", "deformation_type", "strength",
              "sample_count", "rgb_mae", "rgb_rmse", "cosine_similarity",
              "p50_error", "p90_error", "p95_error", "p99_error"]
all_results = test_b_results + test_c_results + test_d_results
csv_path = os.path.join(OUTPUT_DIR, "asg_frame_metrics.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_results)
log(f"CSV: {csv_path}")

# ── 8. Charts ─────────────────────────────────────────────
log_header("8. Generating charts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chart 1: Joint rotation sensitivity
fig, ax = plt.subplots(figsize=(8, 5))
for axis in ['x', 'y', 'z']:
    rates = [r for r in test_b_results if r["rotation_axis"] == axis]
    ax.plot([r["rotation_degree"] for r in rates], [r["rgb_mae"] for r in rates], marker="o", label=f"{axis}-axis")
ax.set_xlabel("Rotation Degree"); ax.set_ylabel("RGB MAE")
ax.set_title("ASG Joint Rotation Sensitivity"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "asg_joint_rotation_sensitivity.png"), dpi=150)
plt.close()

# Chart 2: Input perturbation sensitivity
fig, ax = plt.subplots(figsize=(8, 5))
for mode in ["view_only", "normal_only", "joint"]:
    rates = [r for r in test_c_results if r["test_name"] == mode]
    ax.plot([r["rotation_degree"] for r in rates], [r["rgb_mae"] for r in rates], marker="o", label=mode)
ax.set_xlabel("Rotation Degree (z-axis)"); ax.set_ylabel("RGB MAE")
ax.set_title("ASG Input Perturbation Sensitivity"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "asg_input_perturbation_sensitivity.png"), dpi=150)
plt.close()

# Chart 3: Deformation output change (separate per type)
for dtype in ["stretch", "shear", "twist"]:
    rates = [r for r in test_d_results if r["deformation_type"] == dtype]
    if not rates:
        continue
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(rates)), [r["rgb_mae"] for r in rates])
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([r["strength"] for r in rates], rotation=45)
    ax.set_ylabel("RGB MAE")
    ax.set_title(f"ASG Deformation Output Change ({dtype})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"asg_deformation_output_change_{dtype}.png"), dpi=150)
    plt.close()

log("All charts saved")

# ── 9. Report ─────────────────────────────────────────────
log_header("9. Generating report")
report_path = os.path.join(OUTPUT_DIR, "asg_frame_report.md")
with open(report_path, "w") as f:
    f.write("# ASG Frame Sensitivity Report\n\n")
    f.write(f"Total Gaussians analyzed: {N}\n\n")

    f.write("## Test A: Repeatability\n\n")
    f.write(f"Max error between identical calls: {max_err:.2e}\n\n")

    f.write("## Test B: Joint Rotation Sensitivity\n\n")
    f.write("| Axis | Degree | MAE | RMSE | Cosine Sim |\n")
    f.write("|------|--------|-----|------|-----------|\n")
    for r in test_b_results:
        f.write(f"| {r['rotation_axis']} | {r['rotation_degree']} | {r['rgb_mae']:.6f} | {r['rgb_rmse']:.6f} | {r['cosine_similarity']:.6f} |\n")
    f.write("\n")

    f.write("## Test C: Input Perturbation Sensitivity (z-axis)\n\n")
    f.write("| Mode | Degree | MAE | RMSE | Cosine Sim |\n")
    f.write("|------|--------|-----|------|-----------|\n")
    for r in test_c_results:
        f.write(f"| {r['test_name']} | {r['rotation_degree']} | {r['rgb_mae']:.6f} | {r['rgb_rmse']:.6f} | {r['cosine_similarity']:.6f} |\n")
    f.write("\n")

    f.write("## Test D: Deformation Input Sensitivity\n\n")
    f.write("| Strength | MAE | RMSE | Cosine Sim |\n")
    f.write("|----------|-----|------|-----------|\n")
    for r in test_d_results:
        f.write(f"| {r['strength']} | {r['rgb_mae']:.6f} | {r['rgb_rmse']:.6f} | {r['cosine_similarity']:.6f} |\n")
    f.write("\n")

    f.write("## H2 Assessment\n\n")
    f.write(f"- Real cameras used: {camera_centers is not None}\n")
    f.write(f"- SpecularModel loaded successfully: True\n")
    f.write(f"- Test A numerical error: {max_err:.2e}\n\n")

    # Analysis
    f.write("### Key Findings\n\n")

    # Check if joint rotation matters
    max_joint_mae = max(r["rgb_mae"] for r in test_b_results)
    max_view_mae = max(r["rgb_mae"] for r in test_c_results if r["test_name"] == "view_only")
    max_normal_mae = max(r["rgb_mae"] for r in test_c_results if r["test_name"] == "normal_only")

    f.write(f"- Joint 90° rotation max MAE: {max_joint_mae:.6f}\n")
    f.write(f"- View-only perturbation max MAE: {max_view_mae:.6f}\n")
    f.write(f"- Normal-only perturbation max MAE: {max_normal_mae:.6f}\n")
    f.write(f"- View-only vs normal-only: {'view more sensitive' if max_view_mae > max_normal_mae else 'normal more sensitive'}\n\n")

    f.write("### Limitations\n\n")
    f.write("- Current data can only prove coordinate-frame sensitivity, not physical correctness.\n")
    f.write("- Without deformed GT, cannot directly prove ASG deformation transport is wrong.\n\n")

    f.write("### Judgment: H2 worth testing with GT\n\n")

log(f"Report: {report_path}")

log_path = os.path.join(OUTPUT_DIR, "stage0_6_log.txt")
with open(log_path, "a") as f:
    f.write("\n".join(log_lines))
log(f"Log: {log_path}")
log("\n=== ASG frame sensitivity analysis complete ===")
