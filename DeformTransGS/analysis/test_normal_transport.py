#!/usr/bin/env python3
"""
test_normal_transport.py
Stage 0.6 H1: 法向传输分析
Subsampled to manage GPU memory.
"""

import sys, os, csv
import numpy as np
import torch
from pathlib import Path

TSGS_REPO = "/data/wyh/repos/TSGS"
CHECKPOINT_PLY = "/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k_v4/point_cloud/iteration_30000/point_cloud.ply"
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

# ── 1. Load PLY raw data ──────────────────────────────────
log_header("1. Loading PLY data")
from plyfile import PlyData
plydata = PlyData.read(CHECKPOINT_PLY)
N_total = plydata.elements[0].count
log(f"Total Gaussians in PLY: {N_total}")

# Subsample
MAX_SAMPLES = 200000
rng = np.random.RandomState(20260712)
if N_total > MAX_SAMPLES:
    idx = rng.choice(N_total, MAX_SAMPLES, replace=False)
else:
    idx = np.arange(N_total)
N = len(idx)
log(f"Subsampled to {N} Gaussians")

def read_ply_attr(name):
    return torch.tensor(np.asarray(plydata.elements[0][name])[idx], dtype=torch.float32, device=device)

xyz = torch.stack([read_ply_attr("x"), read_ply_attr("y"), read_ply_attr("z")], dim=1)
scale_raw = torch.stack([read_ply_attr(f"scale_{i}") for i in range(3)], dim=1)
rot_raw = torch.stack([read_ply_attr(f"rot_{i}") for i in range(4)], dim=1)
opacity_raw = read_ply_attr("opacity").unsqueeze(1)
transparency_raw = read_ply_attr("transparency").unsqueeze(1)

# Load features if needed
f_dc_vals = torch.stack([read_ply_attr(f"f_dc_{i}") for i in range(3)], dim=1)  # (N, 3)
f_rest_names = sorted([n for n in [p.name for p in plydata.elements[0].properties] if n.startswith("f_rest_")], key=lambda x: int(x.split("_")[-1]))
f_rest_vals = torch.stack([read_ply_attr(n) for n in f_rest_names], dim=1) if f_rest_names else None

asg_names = sorted([n for n in [p.name for p in plydata.elements[0].properties] if n.startswith("f_asg_")], key=lambda x: int(x.split("_")[-1]))
asg_vals = torch.stack([read_ply_attr(n) for n in asg_names], dim=1) if asg_names else None

log(f"xyz: {xyz.shape}, scale_raw: {scale_raw.shape}, rot_raw: {rot_raw.shape}")
log(f"ASG features: {asg_vals.shape if asg_vals is not None else 'None'}")

# Activated attributes
scale_a = torch.exp(scale_raw)  # (N, 3)
rot_norm = torch.nn.functional.normalize(rot_raw)  # (N, 4) wxyz
opacity = torch.sigmoid(opacity_raw)
transparency = torch.sigmoid(transparency_raw)

# Rotation matrix from wxyz quaternion
from pytorch3d.transforms import quaternion_to_matrix
rot_mat = quaternion_to_matrix(rot_norm)  # (N, 3, 3)

# Canonical smallest axis
smallest_idx = scale_a.min(dim=-1)[1]  # (N,)
canon_normal_raw = rot_mat.gather(2, smallest_idx[:, None, None].expand(-1, 3, -1)).squeeze(-1)
canon_normal = canon_normal_raw / canon_normal_raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
log(f"Canonical normal computed")

# Canonical three axes
a0 = rot_mat[:, :, 0]  # (N, 3)
a1 = rot_mat[:, :, 1]
a2 = rot_mat[:, :, 2]

# ── 2. Deformation Functions ────────────────────────────────
def F_stretch(sx, sy, sz):
    F = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).clone()
    F[:, 0, 0] = sx
    F[:, 1, 1] = sy
    F[:, 2, 2] = sz
    return F

def F_shear(gamma):
    F = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).clone()
    F[:, 0, 1] = gamma
    return F

def F_twist(theta_max):
    z_norm = (xyz[:, 2] - xyz[:, 2].min()) / (xyz[:, 2].max() - xyz[:, 2].min() + 1e-8)
    theta = theta_max * z_norm
    ct = theta.cos()
    st = theta.sin()
    F = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).clone()
    F[:, 0, 0] = ct
    F[:, 0, 1] = -st
    F[:, 1, 0] = st
    F[:, 1, 1] = ct
    return F

# ── 3. Core Analysis Function ──────────────────────────────
def analyze_deformation(F):
    B = 10000  # batch size
    n_batches = (N + B - 1) // B

    n_tsgs_list = []
    n_transport_list = []
    axis_switched_list = []
    angle_list = []

    for bi in range(n_batches):
        s = bi * B
        e = min(s + B, N)

        scale_b = scale_a[s:e]
        rot_mat_b = rot_mat[s:e]
        canon_normal_b = canon_normal[s:e]
        a0_b = a0[s:e]; a1_b = a1[s:e]; a2_b = a2[s:e]
        smallest_idx_b = smallest_idx[s:e]
        F_b = F[s:e]

        S = torch.diag_embed(scale_b ** 2)
        Sigma = rot_mat_b @ S @ rot_mat_b.transpose(1, 2)
        Sigma_def = F_b @ Sigma @ F_b.transpose(1, 2)
        Sigma_def = 0.5 * (Sigma_def + Sigma_def.transpose(1, 2))

        eigvals, eigvecs = torch.linalg.eigh(Sigma_def)
        n_tsgs_b = eigvecs[:, :, 0]

        F_inv_T = torch.linalg.inv(F_b).transpose(1, 2)
        n_transport_b = (F_inv_T @ canon_normal_b.unsqueeze(-1)).squeeze(-1)
        n_transport_b = n_transport_b / n_transport_b.norm(dim=1, keepdim=True).clamp(min=1e-8)

        a0_t = (F_inv_T @ a0_b.unsqueeze(-1)).squeeze(-1)
        a1_t = (F_inv_T @ a1_b.unsqueeze(-1)).squeeze(-1)
        a2_t = (F_inv_T @ a2_b.unsqueeze(-1)).squeeze(-1)
        a0_t = a0_t / a0_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
        a1_t = a1_t / a1_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
        a2_t = a2_t / a2_t.norm(dim=1, keepdim=True).clamp(min=1e-8)

        scores = torch.stack([
            (n_tsgs_b * a0_t).sum(dim=1).abs(),
            (n_tsgs_b * a1_t).sum(dim=1).abs(),
            (n_tsgs_b * a2_t).sum(dim=1).abs(),
        ], dim=1)
        matched = scores.argmax(dim=1)
        switched = matched != smallest_idx_b

        dot = (n_tsgs_b * n_transport_b).sum(dim=1).abs().clamp(max=1.0)
        angle_b = torch.acos(dot) * 180.0 / np.pi

        n_tsgs_list.append(n_tsgs_b)
        n_transport_list.append(n_transport_b)
        axis_switched_list.append(switched)
        angle_list.append(angle_b)

        del S, Sigma, Sigma_def, eigvals, eigvecs, F_inv_T
        del n_tsgs_b, n_transport_b, a0_t, a1_t, a2_t, scores, matched, dot, angle_b
        torch.cuda.empty_cache()

    n_tsgs = torch.cat(n_tsgs_list, dim=0)
    n_transport = torch.cat(n_transport_list, dim=0)
    axis_switched = torch.cat(axis_switched_list, dim=0)
    angle = torch.cat(angle_list, dim=0)

    return n_tsgs, n_transport, axis_switched, angle, None

# ── 4. Run All Deformations ────────────────────────────────
log_header("2. Running deformation tests")

configs = [
    ("stretch", "identity",         lambda: F_stretch(1.0, 1.0, 1.0)),
    ("stretch", "stretch_x_1.25",   lambda: F_stretch(1.25, 1.0, 1.0)),
    ("stretch", "stretch_x_1.50",   lambda: F_stretch(1.50, 1.0, 1.0)),
    ("stretch", "stretch_x_2.00",   lambda: F_stretch(2.00, 1.0, 1.0)),
    ("stretch", "stretch_y_0.75",   lambda: F_stretch(1.0, 0.75, 1.0)),
    ("stretch", "stretch_y_0.50",   lambda: F_stretch(1.0, 0.50, 1.0)),
    ("stretch", "stretch_y_0.25",   lambda: F_stretch(1.0, 0.25, 1.0)),
    ("shear",   "shear_0.1",        lambda: F_shear(0.1)),
    ("shear",   "shear_0.25",       lambda: F_shear(0.25)),
    ("shear",   "shear_0.5",        lambda: F_shear(0.5)),
    ("shear",   "shear_1.0",        lambda: F_shear(1.0)),
]

results = []

for dtype, name, fn in configs:
    F = fn()
    n_tsgs, n_transport, axis_switched, angle, eigvals = analyze_deformation(F)
    switch_count = axis_switched.sum().item()
    switch_rate = switch_count / N * 100
    res = {
        "deformation_type": dtype,
        "name": name,
        "switch_count": switch_count,
        "switch_rate": switch_rate,
        "angle_mean": angle.mean().item(),
        "angle_median": angle.median().item(),
        "angle_p90": torch.quantile(angle, 0.90).item(),
        "angle_p95": torch.quantile(angle, 0.95).item(),
        "angle_p99": torch.quantile(angle, 0.99).item(),
        "angle_max": angle.max().item(),
    }
    results.append(res)
    log(f"  {name:20s} switch={switch_rate:7.3f}%  angle_mean={res['angle_mean']:7.3f}°  p95={res['angle_p95']:7.3f}°")

# Twist separately (more computation)
for deg in [15, 30, 60, 90]:
    name = f"twist_{deg}"
    theta = torch.tensor(deg * np.pi / 180.0, device=device)
    F = F_twist(theta)
    n_tsgs, n_transport, axis_switched, angle, eigvals = analyze_deformation(F)
    switch_count = axis_switched.sum().item()
    switch_rate = switch_count / N * 100
    res = {
        "deformation_type": "twist",
        "name": name,
        "switch_count": switch_count,
        "switch_rate": switch_rate,
        "angle_mean": angle.mean().item(),
        "angle_median": angle.median().item(),
        "angle_p90": torch.quantile(angle, 0.90).item(),
        "angle_p95": torch.quantile(angle, 0.95).item(),
        "angle_p99": torch.quantile(angle, 0.99).item(),
        "angle_max": angle.max().item(),
    }
    results.append(res)
    log(f"  {name:20s} switch={switch_rate:7.3f}%  angle_mean={res['angle_mean']:7.3f}°  p95={res['angle_p95']:7.3f}°")

del n_tsgs, n_transport, axis_switched, angle, eigvals
torch.cuda.empty_cache()

csv_path = os.path.join(OUTPUT_DIR, "normal_transport_metrics.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
log(f"\nCSV: {csv_path}")

# ── 5. Scale Ratio Analysis ────────────────────────────────
log_header("3. Scale ratio analysis")
scale_sorted = scale_a.sort(dim=-1).values
ratio = scale_sorted[:, 1] / (scale_sorted[:, 0] + 1e-8)

bins = [(1.0, 1.1, "[1.0,1.1)"), (1.1, 1.25, "[1.1,1.25)"),
        (1.25, 1.5, "[1.25,1.5)"), (1.5, 2.0, "[1.5,2.0)"),
        (2.0, float("inf"), "[2.0,inf)")]

F_strong = F_stretch(2.0, 1.0, 1.0)
_, _, axis_switched_strong, _, _ = analyze_deformation(F_strong)

log("Scale ratio vs switch rate (stretch_x_2.00):")
for lo, hi, label in bins:
    if hi == float("inf"):
        mask = ratio >= lo
    else:
        mask = (ratio >= lo) & (ratio < hi)
    cnt = mask.sum().item()
    sr = axis_switched_strong[mask].sum().item() / cnt * 100 if cnt > 0 else 0
    log(f"  {label:15s}: n={cnt:6d}  switch_rate={sr:6.2f}%")

# ── 6. Temporal Analysis ───────────────────────────────────
log_header("4. Temporal analysis")
torch.manual_seed(20260712)
temporal_N = min(50000, N)
temporal_idx = torch.randperm(N, device=device)[:temporal_N]

def temporal_sequence(steps, deform_fn):
    B = 10000
    all_n = []
    for s in steps:
        F = deform_fn(s)
        sub_n = []
        for bi in range((N + B - 1) // B):
            bi_s = bi * B
            bi_e = min(bi_s + B, N)
            scale_b = scale_a[bi_s:bi_e]
            rot_mat_b = rot_mat[bi_s:bi_e]
            F_b = F[bi_s:bi_e]

            S = torch.diag_embed(scale_b ** 2)
            Sigma = rot_mat_b @ S @ rot_mat_b.transpose(1, 2)
            Sigma_def = F_b @ Sigma @ F_b.transpose(1, 2)
            Sigma_def = 0.5 * (Sigma_def + Sigma_def.transpose(1, 2))
            _, eigvecs = torch.linalg.eigh(Sigma_def)
            sub_n.append(eigvecs[:, :, 0])
            del S, Sigma, Sigma_def, eigvecs
        n_full = torch.cat(sub_n, dim=0)
        all_n.append(n_full[temporal_idx])
        del n_full, sub_n
        torch.cuda.empty_cache()
    return torch.stack(all_n, dim=0)

strech_steps = torch.linspace(1.0, 2.0, 51, device=device)
log("Computing stretch temporal sequence (51 steps)...")
stretch_seq = temporal_sequence(strech_steps, lambda s: F_stretch(s, 1.0, 1.0))

shear_steps = torch.linspace(0, 1.0, 51, device=device)
log("Computing shear temporal sequence (51 steps)...")
shear_seq = temporal_sequence(shear_steps, lambda g: F_shear(g))

def compute_jumps(seq):
    diffs = []
    for t in range(1, seq.shape[0]):
        dot = (seq[t] * seq[t-1]).sum(dim=1).abs().clamp(max=1.0)
        diffs.append(torch.acos(dot) * 180.0 / np.pi)
    return torch.stack(diffs, dim=0)

stretch_jumps = compute_jumps(stretch_seq)
shear_jumps = compute_jumps(shear_seq)

def report_jumps(jumps, name):
    m = jumps.mean().item()
    p95 = torch.quantile(jumps, 0.95).item()
    p99 = torch.quantile(jumps, 0.99).item()
    mx = jumps.max().item()
    r15 = (jumps > 15).float().mean().item() * 100
    r30 = (jumps > 30).float().mean().item() * 100
    log(f"  {name}: mean={m:.3f}° p95={p95:.3f}° p99={p99:.3f}° max={mx:.3f}°")
    log(f"         >15°: {r15:.3f}%  >30°: {r30:.3f}%")
    return {"mean": m, "p95": p95, "p99": p99, "max": mx, "rate_15": r15, "rate_30": r30}

stretch_stats = report_jumps(stretch_jumps, "stretch")
shear_stats = report_jumps(shear_jumps, "shear")

# ── 7. Charts ─────────────────────────────────────────────
log_header("5. Generating charts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stretch_r = [r for r in results if r["deformation_type"] == "stretch"]
shear_r = [r for r in results if r["deformation_type"] == "shear"]
twist_r = [r for r in results if r["deformation_type"] == "twist"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot([r["name"] for r in stretch_r], [r["switch_rate"] for r in stretch_r], marker="o")
ax1.set_title("Stretch: Axis Switch Rate")
ax1.set_ylabel("Switch Rate (%)")
ax1.tick_params(axis="x", rotation=45)
ax2.plot([r["name"] for r in shear_r], [r["switch_rate"] for r in shear_r], marker="o")
ax2.set_title("Shear: Axis Switch Rate")
ax2.set_ylabel("Switch Rate (%)")
ax2.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "normal_axis_switch_rate.png"), dpi=150)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot([r["name"] for r in stretch_r], [r["angle_mean"] for r in stretch_r], marker="o", label="mean")
ax1.plot([r["name"] for r in stretch_r], [r["angle_p95"] for r in stretch_r], marker="s", label="p95")
ax1.set_title("Stretch: Normal Angular Error")
ax1.set_ylabel("Angle (deg)"); ax1.legend(); ax1.tick_params(axis="x", rotation=45)
ax2.plot([r["name"] for r in shear_r], [r["angle_mean"] for r in shear_r], marker="o", label="mean")
ax2.plot([r["name"] for r in shear_r], [r["angle_p95"] for r in shear_r], marker="s", label="p95")
ax2.set_title("Shear: Normal Angular Error")
ax2.set_ylabel("Angle (deg)"); ax2.legend(); ax2.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "normal_angular_error.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
bin_centers = []; bin_rates = []
for lo, hi, label in bins:
    if hi == float("inf"):
        mask = ratio >= lo
        center = lo + 0.5
    else:
        mask = (ratio >= lo) & (ratio < hi)
        center = (lo + hi) / 2
    cnt = mask.sum().item()
    sr = axis_switched_strong[mask].sum().item() / cnt * 100 if cnt > 0 else 0
    bin_centers.append(center); bin_rates.append(sr)
    ax.text(center, sr, f"{sr:.1f}%", ha="center", va="bottom")
ax.bar(bin_centers, bin_rates, width=0.3)
ax.set_xlabel("Canonical Scale Ratio (2nd / smallest)")
ax.set_ylabel("Axis Switch Rate (%)")
ax.set_title("Stretch_x_2.00: Scale Ratio vs Switch Rate")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scale_ratio_vs_switch_rate.png"), dpi=150)
plt.close()

s_steps = strech_steps[1:].cpu().numpy()
jump_rate_s = (stretch_jumps > 15).float().mean(dim=1).cpu().numpy() * 100
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(s_steps, jump_rate_s)
ax.set_xlabel("Stretch Strength s")
ax.set_ylabel(">15° Normal Jump Rate (%)")
ax.set_title("Temporal: Stretch 1.0→2.0")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "temporal_normal_jump_rate.png"), dpi=150)
plt.close()
log("All charts saved")

# ── 8. Save worst examples ─────────────────────────────────
log_header("6. Saving worst examples")
# Use global indices to avoid confusion
all_jumps_flat = stretch_jumps.reshape(-1)
worst_vals, worst_flat_idx = all_jumps_flat.topk(min(100, len(all_jumps_flat)))
worst_t = worst_flat_idx // stretch_jumps.shape[1]
worst_g = worst_flat_idx % stretch_jumps.shape[1]
# Map temporal_idx back to original PLY index (since we subsampled)
orig_temporal_idx = torch.from_numpy(idx).to(device)[temporal_idx]
global_worst = orig_temporal_idx[worst_g]

examples = {
    "canonical_index": global_worst.cpu().numpy(),
    "xyz": xyz[worst_g].cpu().numpy(),
    "canonical_scale": scale_a[worst_g].cpu().numpy(),
    "canonical_rotation": rot_norm[worst_g].cpu().numpy(),
    "canonical_smallest_axis": canon_normal[worst_g].cpu().numpy(),
    "strength_before": (1.0 + worst_t.cpu().numpy() / 50.0 * 1.0).astype(np.float32),
    "strength_after": (1.0 + (worst_t.cpu().numpy() + 1) / 50.0 * 1.0).astype(np.float32),
    "normal_before": stretch_seq[worst_t, torch.arange(len(worst_t), device=device), :].cpu().numpy(),
    "normal_after": stretch_seq[worst_t + 1, torch.arange(len(worst_t), device=device), :].cpu().numpy(),
}
npz_path = os.path.join(OUTPUT_DIR, "normal_switch_examples.npz")
np.savez_compressed(npz_path, **examples)
log(f"Saved worst 100 examples to {npz_path}")

# ── 9. Report ─────────────────────────────────────────────
log_header("7. Generating report")
report_path = os.path.join(OUTPUT_DIR, "normal_transport_report.md")
max_switch = max(r["switch_rate"] for r in results)
max_angle = max(r["angle_mean"] for r in results)
max_jump_15 = max(stretch_stats["rate_15"], shear_stats["rate_15"])
max_jump_30 = max(stretch_stats["rate_30"], shear_stats["rate_30"])

if max_switch > 5.0:
    h1 = "H1 supported"
elif max_switch > 1.0:
    h1 = "H1 partially supported"
else:
    h1 = "H1 not supported"

with open(report_path, "w") as f:
    f.write("# Normal Transport Analysis Report\n\n")
    f.write(f"Total Gaussians analyzed: {N}\n\n")
    f.write("## Canonical Scale Ratio Distribution\n\n")
    for lo, hi, label in bins:
        if hi == float("inf"):
            mask = ratio >= lo
        else:
            mask = (ratio >= lo) & (ratio < hi)
        f.write(f"- {label}: {mask.sum().item()} ({mask.sum().item()/N*100:.2f}%)\n")
    f.write("\n## Deformation Test Results\n\n")
    f.write("| Name | Switch Rate | Mean | Median | P90 | P95 | P99 | Max |\n")
    f.write("|------|------------|------|--------|-----|-----|-----|-----|\n")
    for r in results:
        f.write(f"| {r['name']} | {r['switch_rate']:.3f}% | {r['angle_mean']:.3f}° | ")
        f.write(f"{r['angle_median']:.3f}° | {r['angle_p90']:.3f}° | {r['angle_p95']:.3f}° | ")
        f.write(f"{r['angle_p99']:.3f}° | {r['angle_max']:.3f}° |\n")
    f.write(f"\n## Temporal Analysis\n\n")
    f.write(f"### Stretch 1.0→2.0\n")
    for k, v in stretch_stats.items():
        f.write(f"- {k}: {v}\n")
    f.write(f"\n### Shear 0→1.0\n")
    for k, v in shear_stats.items():
        f.write(f"- {k}: {v}\n")
    f.write(f"\n## H1 Assessment\n\n")
    f.write(f"Maximum axis switch rate: {max_switch:.3f}%\n")
    f.write(f"Maximum mean normal angle difference: {max_angle:.3f}°\n")
    f.write(f"Maximum >15° temporal jump rate: {max_jump_15:.3f}%\n")
    f.write(f"Maximum >30° temporal jump rate: {max_jump_30:.3f}%\n\n")
    f.write(f"**Judgment: {h1}**\n")

log(f"Report: {report_path}")

log_path = os.path.join(OUTPUT_DIR, "stage0_6_log.txt")
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))
log(f"Log: {log_path}")
log("\n=== Normal transport analysis complete ===")
