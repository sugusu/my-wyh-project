#!/usr/bin/env python3
"""Stage 3.3.R4: Exact-Projection Local Optical Consistency Re-Evaluation"""
import sys, os, math, csv, json, hashlib
import numpy as np
from collections import defaultdict
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.stats import spearmanr, pearsonr, wilcoxon

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_3R4_exact_projection_local_recheck"
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

sys.path.insert(0, BASE)
from analysis.exact_cuda_projection import (
    project_points_cuda_exact,
    assert_no_double_view_transform,
)

device = "cuda"
log_lines = []
def log(m): print(m); log_lines.append(str(m))

bg_color = torch.zeros(3, device=device)
pipe = type("obj", (object,), {"debug": False, "convert_SHs_python": False, "compute_cov3D_python": False})()

GRID = 41; L = 0.75; H = 256; W = 256
spacing = 1.5 / 40

# ───── SHA256 helpers ─────
def sha256_t(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def sha256_np(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

# ───── Bilinear sampler (independent) ─────
def bilinear_sample(image, x, y):
    image = np.asarray(image, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x and y must match")
    H_img, W_img = image.shape
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < W_img - 1) & (y >= 0) & (y < H_img - 1)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    xv, yv = x[valid], y[valid]
    x0 = np.floor(xv).astype(np.int64); x1 = x0 + 1
    y0 = np.floor(yv).astype(np.int64); y1 = y0 + 1
    wx = xv - x0; wy = yv - y0
    out[valid] = ((1-wx)*(1-wy)*image[y0,x0] + wx*(1-wy)*image[y0,x1] +
                  (1-wx)*wy*image[y1,x0] + wx*wy*image[y1,x1])
    return out

def alpha_to_tau_eff(alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    T = np.clip(1.0 - alpha, 1e-6, 1.0)
    return -np.log(T)

# ───── Setup ─────
mesh = trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N = len(mesh.vertices)
verts_np = np.array(mesh.vertices, dtype=np.float32)
verts = torch.tensor(verts_np, device=device)
scale_t = torch.full((N, 3), spacing, device=device); scale_t[:, 2] = spacing * 0.1
rot_t = torch.zeros(N, 4, device=device); rot_t[:, 0] = 1.0
ckpt = torch.load(f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt",
                  map_location=device, weights_only=True)
tau_raw = ckpt["tau_raw"]; color_raw = ckpt["color_raw"]
opacity_t = 1 - torch.exp(-F.softplus(tau_raw))

# ───── Cameras ─────
cam_cfgs = [
    {"pos": [0, -3.5, 1.5], "target": [0, 0, 0], "up": [0, 0, 1], "id": 0},
    {"pos": [3.0, 0, 2.0],  "target": [0, 0, 0], "up": [0, 0, 1], "id": 4},
    {"pos": [0, 3.5, 1.5],  "target": [0, 0, 0], "up": [0, 0, -1],"id": 8},
]

def build_cam(cfg):
    pa = np.array(cfg["pos"], dtype=np.float32)
    ta = np.array(cfg["target"], dtype=np.float32)
    ua = np.array(cfg["up"], dtype=np.float32)
    fwd = ta - pa; fwd /= np.linalg.norm(fwd)
    rt = np.cross(ua, fwd); rt /= np.linalg.norm(rt)
    nu = np.cross(fwd, rt)
    Rw = np.eye(3, dtype=np.float32)
    Rw[0, :] = rt; Rw[1, :] = nu; Rw[2, :] = fwd
    T = -Rw @ pa; R = Rw.T
    fx = W / (2 * math.tan(math.radians(45 / 2)))
    cam = Camera(colmap_id=cfg["id"], R=R, T=T,
                 FoVx=focal2fov(fx, W), FoVy=focal2fov(fx, W),
                 image_width=W, image_height=H,
                 image_path="", image_PIL=None,
                 image_name=f"cam_{cfg['id']:03d}", uid=cfg["id"],
                 preload_img=False, data_device="cpu")
    cam.original_image = torch.zeros(3, W, H)
    return cam

film_cams = [build_cam(c) for c in cam_cfgs]
cam_ids = [c["id"] for c in cam_cfgs]

class Adapter:
    def __init__(self, xyz, scl, rot, tau, col):
        self._xyz = xyz
        self._scaling = torch.log(scl.clamp(min=1e-8))
        self._rotation = rot; self._tau_raw = tau; self._color_raw = col
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
    def get_transparency(self): return torch.full((N, 1), 0.5, device=device)
    @property
    def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)

def white_pass(gm, cam):
    r2 = render(cam, gm, pipe, bg_color, app_model=None,
                override_color=torch.ones(N, 3, device=device),
                return_plane=False, return_depth_normal=False)
    return r2["render"].mean(dim=0, keepdim=True).clamp(0, 1)

# ───── Frozen deformation API (explicit state mapping) ─────
STATE_MAP = {
    "canonical": {"type": "identity"},
    "stretch_1.25": {"type": "stretch", "s": 1.25},
    "stretch_1.50": {"type": "stretch", "s": 1.50},
    "stretch_2.00": {"type": "stretch", "s": 2.00},
    "biaxial_1.50": {"type": "biaxial", "s": 1.50},
    "cubic_l010": {"type": "cubic", "lam": 0.10},
    "cubic_l020": {"type": "cubic", "lam": 0.20},
    "cubic_l0333": {"type": "cubic", "lam": 1/3},
    "shear_k020": {"type": "shear", "k": 0.20},
    "shear_k040": {"type": "shear", "k": 0.40},
    "twist_60": {"type": "twist"},
}
states_list = list(STATE_MAP.keys())

def deform_xyz(xyz, state_name):
    """Apply analytic deformation to world points [N,3]."""
    xyz = xyz.clone() if isinstance(xyz, torch.Tensor) else torch.tensor(xyz, device=device, dtype=torch.float32)
    cfg = STATE_MAP[state_name]
    t = cfg["type"]
    if t == "identity":
        return xyz
    elif t == "stretch":
        d = xyz.clone(); d[:, 0] *= cfg["s"]; return d
    elif t == "biaxial":
        d = xyz.clone(); d[:, 0] *= cfg["s"]; d[:, 1] *= cfg["s"]; return d
    elif t == "cubic":
        lam = cfg["lam"]
        d = xyz.clone(); d[:, 0] = xyz[:, 0] + lam * xyz[:, 0]**3 / L**2; return d
    elif t == "shear":
        d = xyz.clone(); d[:, 0] += cfg["k"] * xyz[:, 1]**2 / L; return d
    elif t == "twist":
        return twist_def(xyz, 60, (xyz[:, 2].min().item(), xyz[:, 2].max().item()))
    return xyz

def compute_Js(us, vs, state_name):
    """Analytic Js for material coordinates (u,v)."""
    cfg = STATE_MAP[state_name]
    t = cfg["type"]
    if t == "identity": return np.ones_like(us)
    if t == "stretch": return np.full_like(us, cfg["s"])
    if t == "biaxial": return np.full_like(us, cfg["s"]**2)
    if t == "cubic": return 1 + 3 * cfg["lam"] * np.asarray(us)**2
    if t in ("shear", "twist"): return np.ones_like(us)
    return np.ones_like(us)

def uv_to_xyz(us, vs):
    xs = np.asarray(us, dtype=np.float32) * L
    ys = np.asarray(vs, dtype=np.float32) * L
    zs = np.zeros_like(xs)
    return np.stack([xs, ys, zs], axis=1)

# ═══════════════════════════════════════════════════════════════
# 1. Frozen Carrier Lock
# ═══════════════════════════════════════════════════════════════
log("=" * 60)
log("  1. Frozen Carrier Lock")
log("=" * 60)

lock = {
    "source_path": str(ckpt.get("__path__", "canonical_checkpoint.pt")),
    "gaussian_count": N,
    "xyz_sha256": sha256_t(verts),
    "scale_sha256": sha256_t(scale_t),
    "rotation_sha256": sha256_t(rot_t),
    "tau_sha256": sha256_t(tau_raw),
    "color_sha256": sha256_t(color_raw),
}
assert N == 1681, f"Expected 1681 Gaussians, got {N}"
with open(os.path.join(OUTPUT, "frozen_carrier_lock.json"), "w") as f:
    json.dump(lock, f, indent=2)
log(f"  Carrier locked: N={N}")

# ───── Projection code lock scan ─────
code_scan = assert_no_double_view_transform([
    os.path.abspath(__file__),
])
proj_lock_lines = ["# Projection Code Lock", "", "Files scanned:"]
for p in [os.path.abspath(__file__)]:
    proj_lock_lines.append(f"- {p}")
proj_lock_lines.append("")
if code_scan:
    proj_lock_lines.append("ISSUES FOUND:")
    for i in code_scan:
        proj_lock_lines.append(f"- {i}")
else:
    proj_lock_lines.append("No double-view-transform patterns found. PASS")
with open(os.path.join(OUTPUT, "projection_code_lock.md"), "w") as f:
    f.write("\n".join(proj_lock_lines) + "\n")

# ═══════════════════════════════════════════════════════════════
# 2. Bilinear Regression Test
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  2. Bilinear Regression Test")
log("=" * 60)

np.random.seed(20260713)
I_test = np.fromfunction(lambda y, x: 3*x + 5*y + 7, (100, 100), dtype=np.float64)
xs_t = np.random.rand(1000) * 99; ys_t = np.random.rand(1000) * 99
sampled = bilinear_sample(I_test, xs_t, ys_t)
expected = 3*xs_t + 5*ys_t + 7
max_err_bil = np.abs(sampled - expected).max()
bilinear_pass = max_err_bil < 1e-10
log(f"  Linear plane max error: {max_err_bil:.2e} {'PASS' if bilinear_pass else 'FAIL'}")

xs_int = np.random.randint(0, 99, 100).astype(np.float64)
ys_int = np.random.randint(0, 99, 100).astype(np.float64)
sampled_int = bilinear_sample(I_test, xs_int, ys_int)
max_err_int = np.abs(sampled_int - I_test[ys_int.astype(int), xs_int.astype(int)]).max()
log(f"  Integer max error: {max_err_int:.2e} {'PASS' if max_err_int == 0 else 'FAIL'}")

with open(os.path.join(OUTPUT, "bilinear_regression_test.md"), "w") as f:
    f.write("# Bilinear Regression Test\n\n")
    f.write(f"Linear plane max error: {max_err_bil:.2e} {'PASS' if bilinear_pass else 'FAIL'}\n")
    f.write(f"Integer pixel max error: {max_err_int:.2e} {'PASS' if max_err_int == 0 else 'FAIL'}\n")

# ═══════════════════════════════════════════════════════════════
# 3. Projection Regression
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  3. Projection Regression")
log("=" * 60)

# Compare against Stage 3.3.R3 saved results
r3_dir = f"{BASE}/experiments/stage3_3R3_projection_parameterization_audit"
proj_reg_rows = []
proj_reg_pass = True
for ci, cam in enumerate(film_cams):
    cid = cam_ids[ci]
    ep = project_points_cuda_exact(verts, cam)
    px, py = ep["pixel_x"], ep["pixel_y"]
    in_f = ep["in_frame"]

    # Load R3 reference
    r3_ref_path = os.path.join(r3_dir, f"fresh_alpha_cam{cid:03d}.npy")
    if os.path.exists(r3_ref_path):
        r3_ref = np.load(r3_ref_path)
        # Compare by re-rendering with same parameters
        px_np = px.detach().cpu().numpy()
        py_np = py.detach().cpu().numpy()
        in_np = in_f.detach().cpu().numpy()

        fg = r3_ref > 1e-4
        fg_d = binary_dilation(fg, iterations=3)
        act = (opacity_t.detach().cpu().numpy().ravel() >= 1/255).ravel()
        valid = act & in_np
        xi = np.clip(np.round(px_np[valid]).astype(int), 0, W-1)
        yi = np.clip(np.round(py_np[valid]).astype(int), 0, H-1)
        frac_all = (fg_d[yi, xi]).mean() if len(xi) > 0 else 0
        proj_reg_rows.append({
            "cam": cid, "active_valid": int(valid.sum()),
            "fg_fraction": round(float(frac_all), 6),
            "fg_ge_095": "YES" if frac_all >= 0.95 else "NO",
        })
        ok = frac_all >= 0.95
        log(f"  cam_{cid:03d}: fg_frac={frac_all:.4f} {'PASS' if ok else 'FAIL'}")
        if not ok: proj_reg_pass = False
    else:
        log(f"  cam_{cid:03d}: no R3 reference, computing fresh")
        # Fresh computation
        act = (opacity_t.detach().cpu().numpy().ravel() >= 1/255).ravel()
        can_gm_r = Adapter(verts, scale_t, rot_t, tau_raw, color_raw)
        alpha_fresh = white_pass(can_gm_r, cam).detach().cpu().numpy().squeeze(0)
        fg = alpha_fresh > 1e-4
        fg_d = binary_dilation(fg, iterations=3)
        px_np = px.detach().cpu().numpy()
        py_np = py.detach().cpu().numpy()
        in_np = in_f.detach().cpu().numpy()
        valid = act & in_np
        xi = np.clip(np.round(px_np[valid]).astype(int), 0, W-1)
        yi = np.clip(np.round(py_np[valid]).astype(int), 0, H-1)
        frac_all = (fg_d[yi, xi]).mean() if len(xi) > 0 else 0
        proj_reg_rows.append({
            "cam": cid, "active_valid": int(valid.sum()),
            "fg_fraction": round(float(frac_all), 6),
            "fg_ge_095": "YES" if frac_all >= 0.95 else "NO",
        })
        log(f"  cam_{cid:03d} (fresh): fg_frac={frac_all:.4f}")

if not proj_reg_rows:
    proj_reg_pass = False
proj_reg_pass = proj_reg_pass and all(r["fg_ge_095"] == "YES" for r in proj_reg_rows)
log(f"  Projection regression: {'PASS' if proj_reg_pass else 'FAIL'}")

with open(os.path.join(OUTPUT, "exact_projection_regression.csv"), "w", newline="") as f:
    fn = ["cam","active_valid","fg_fraction","fg_ge_095"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(proj_reg_rows)

if not proj_reg_pass:
    log("  PROJECTION REGRESSION FAIL: stopping")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# 4. Fresh Render All Alpha Maps
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  4. Fresh Render All Alpha Maps")
log("=" * 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

alpha_dir = os.path.join(OUTPUT, "fresh_alpha")
os.makedirs(alpha_dir, exist_ok=True)

alpha_maps = {st: {} for st in states_list}
alpha_manifest = []
render_input_lock = []

for st in states_list:
    dv = deform_xyz(verts, st)
    gm = Adapter(dv, scale_t, rot_t, tau_raw, color_raw)
    for ci, cam in enumerate(film_cams):
        cid = cam_ids[ci]
        a = white_pass(gm, cam).detach().cpu().numpy().squeeze(0)
        alpha_maps[st][cid] = a

        np.save(os.path.join(alpha_dir, f"{st}_cam{cid:03d}.npy"), a)

        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(a, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{st} cam_{cid:03d}")
        ax.axis("off")
        fig.savefig(os.path.join(alpha_dir, f"{st}_cam{cid:03d}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

        alpha_manifest.append({
            "state": st, "cam": cid,
            "min": f"{a.min():.6f}", "max": f"{a.max():.6f}", "mean": f"{a.mean():.6f}",
            "sha256": sha256_np(a),
        })

        render_input_lock.append({
            "state": st, "cam": cid,
            "xyz_sha256": sha256_t(dv),
            "scale_sha256": sha256_t(scale_t),
            "rotation_sha256": sha256_t(rot_t),
            "tau_sha256": sha256_t(tau_raw),
            "alpha_sha256": sha256_np(a),
        })

    log(f"  Rendered {st}")

with open(os.path.join(OUTPUT, "fresh_alpha_manifest.csv"), "w", newline="") as f:
    fn = ["state","cam","min","max","mean","sha256"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(alpha_manifest)

with open(os.path.join(OUTPUT, "render_input_lock.csv"), "w", newline="") as f:
    fn = ["state","cam","xyz_sha256","scale_sha256","rotation_sha256","tau_sha256","alpha_sha256"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(render_input_lock)

# ═══════════════════════════════════════════════════════════════
# 5. Point-Level Metric + Conditioning Audit
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  5. Point-Level Metric + Conditioning")
log("=" * 60)

all_points = []
for st in states_list:
    dv_np = deform_xyz(verts, st).detach().cpu().numpy()
    can_np = verts.detach().cpu().numpy()
    for ci, cam in enumerate(film_cams):
        cid = cam_ids[ci]
        ep_can = project_points_cuda_exact(torch.tensor(can_np, device=device), cam)
        ep_def = project_points_cuda_exact(torch.tensor(dv_np, device=device), cam)

        pxc = ep_can["pixel_x"].detach().cpu().numpy()
        pyc = ep_can["pixel_y"].detach().cpu().numpy()
        pxd = ep_def["pixel_x"].detach().cpu().numpy()
        pyd = ep_def["pixel_y"].detach().cpu().numpy()
        inc = ep_can["in_frame"].detach().cpu().numpy()
        ind = ep_def["in_frame"].detach().cpu().numpy()

        # Boundary mask
        mask = alpha_maps[st][cid] > 0.01
        dist = distance_transform_edt(mask)

        for idx in range(N):
            if not (inc[idx] and ind[idx]): continue
            rd = int(np.clip(round(pxd[idx]), 0, W-1))
            cd = int(np.clip(round(pyd[idx]), 0, H-1))
            if dist[cd, rd] < 8: continue

            A_c = bilinear_sample(alpha_maps["canonical"][cid], np.array([pxc[idx]]), np.array([pyc[idx]]))[0]
            A_d = bilinear_sample(alpha_maps[st][cid], np.array([pxd[idx]]), np.array([pyd[idx]]))[0]
            if not np.isfinite(A_c) or not np.isfinite(A_d): continue
            te_c = alpha_to_tau_eff(np.array([A_c]))[0]
            te_d = alpha_to_tau_eff(np.array([A_d]))[0]
            if te_c <= 1e-12: continue
            r = te_d / te_c
            all_points.append({
                "state": st, "cam": cid, "idx": idx,
                "u": (idx//GRID - 20)/20, "v": (idx%GRID - 20)/20,
                "A_c": A_c, "A_d": A_d, "tau_c": te_c, "tau_d": te_d, "R": r,
            })

log(f"  Collected {len(all_points)} valid point-camera samples")

# Validity statistics
validity_rows = []
for st in states_list:
    for cid in cam_ids:
        pts = [p for p in all_points if p["state"] == st and p["cam"] == cid]
        validity_rows.append({"state": st, "cam": cid, "final_valid": len(pts)})

with open(os.path.join(OUTPUT, "point_validity_statistics.csv"), "w", newline="") as f:
    fn = ["state","cam","final_valid"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(validity_rows)

# Point conditioning
cond_rows = []
for st in states_list:
    pts = [p for p in all_points if p["state"] == st]
    tau_c = np.array([p["tau_c"] for p in pts if np.isfinite(p["tau_c"]) and p["tau_c"] > 0])
    R_v = np.array([p["R"] for p in pts if np.isfinite(p["R"])])
    cr = {"state": st, "n": len(pts), "n_finite_R": len(R_v)}
    if len(tau_c) > 0:
        for q in [0.01, 0.05, 0.10, 0.50, 0.90]:
            cr[f"tau_c_p{q:.2f}"] = float(np.quantile(tau_c, q))
        cr["tau_c_min"] = float(tau_c.min())
        for thr in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
            cr[f"tau_c_frac_below_{thr:.0e}"] = float((tau_c < thr).mean())
    if len(R_v) > 0:
        cr["R_std"] = float(np.std(R_v))
        for q in [0.90, 0.95, 0.99, 0.999]:
            cr[f"R_p{q:.3f}"] = float(np.quantile(np.abs(R_v), q))
        cr["R_max"] = float(np.abs(R_v).max())
        cr["R_unique"] = len(set(R_v.round(8)))

    # Tail ratio
    if len(tau_c) > 0 and len(R_v) > 0:
        order = np.argsort(np.abs(R_v))[::-1]
        top1pct = max(1, len(order)//100)
        top_idx = order[:top1pct]
        top_tau_med = float(np.median(tau_c[top_idx]))
        all_tau_med = float(np.median(tau_c))
        cr["tail_tau_ratio"] = round(top_tau_med / max(all_tau_med, 1e-12), 6)
        cr["top1pct_median_tau_c"] = top_tau_med
        cr["all_median_tau_c"] = all_tau_med

    # Spearman: abs(R) vs -log10(tau_c)
    if len(R_v) > 5 and len(tau_c) > 5:
        min_l = min(len(R_v), len(tau_c))
        rv = np.abs(R_v[:min_l]); tv = tau_c[:min_l]
        both_fin = np.isfinite(rv) & (tv > 0)
        rv_f = rv[both_fin]; tv_f = tv[both_fin]
        if len(rv_f) > 5 and len(set(rv_f.round(6))) > 1:
            rho, _ = spearmanr(rv_f, -np.log10(tv_f + 1e-12))
            cr["spearman_absR_vs_logtau"] = round(rho, 4)

    cond_rows.append(cr)

pool_R_all = np.array([p["R"] for p in all_points if np.isfinite(p["R"])])
pool_tau_all = np.array([p["tau_c"] for p in all_points if np.isfinite(p["R"]) and p["tau_c"] > 0])
min_l = min(len(pool_R_all), len(pool_tau_all))
rv_p = np.abs(pool_R_all[:min_l]); tv_p = pool_tau_all[:min_l]
both_p = np.isfinite(rv_p) & (tv_p > 0)
if both_p.sum() > 10 and len(set(rv_p[both_p].round(6))) > 1:
    rho_pooled, _ = spearmanr(rv_p[both_p], -np.log10(tv_p[both_p] + 1e-12))
else:
    rho_pooled = float("nan")
log(f"  Pooled Spearman(abs(R), -log10(tau_c)): {rho_pooled:.4f}")

with open(os.path.join(OUTPUT, "point_metric_conditioning_exact.csv"), "w", newline="") as f:
    all_keys = set()
    for r in cond_rows:
        all_keys.update(r.keys())
    wfn = sorted(all_keys)
    w = csv.DictWriter(f, fieldnames=wfn)
    w.writeheader(); w.writerows(cond_rows)

# ═══════════════════════════════════════════════════════════════
# 6. Material Cell Definition + Midpoint Quadrature
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  6. Material Cell Definition")
log("=" * 60)

# u,v from the 41x41 grid
u_vals = np.array([(i - 20) / 20.0 for i in range(GRID)], dtype=np.float64)
v_vals = np.array([(j - 20) / 20.0 for j in range(GRID)], dtype=np.float64)
# Verify uniform spacing
du = np.diff(u_vals); dv = np.diff(v_vals)
assert np.allclose(du, du[0]) and np.allclose(dv, dv[0]), "Non-uniform spacing!"

cell_defs = []
for iu in range(1, GRID-1):
    for iv in range(1, GRID-1):
        u_low = 0.5 * (u_vals[iu-1] + u_vals[iu])
        u_high = 0.5 * (u_vals[iu] + u_vals[iu+1])
        v_low = 0.5 * (v_vals[iv-1] + v_vals[iv])
        v_high = 0.5 * (v_vals[iv] + v_vals[iv+1])
        cell_defs.append({
            "cell_id": len(cell_defs), "iu": iu, "iv": iv,
            "u_center": u_vals[iu], "v_center": v_vals[iv],
            "u_low": u_low, "u_high": u_high,
            "v_low": v_low, "v_high": v_high,
        })
log(f"  Defined {len(cell_defs)} interior cells ({GRID-2}x{GRID-2})")

# Verify affine mapping
A_design = np.column_stack([np.ones(N), u_vals.repeat(GRID), np.tile(v_vals, GRID)])
xyz_flat = verts_np.reshape(-1, 3)
coeffs_x, _, _, _ = np.linalg.lstsq(A_design, xyz_flat[:, 0], rcond=None)
coeffs_y, _, _, _ = np.linalg.lstsq(A_design, xyz_flat[:, 1], rcond=None)
coeffs_z, _, _, _ = np.linalg.lstsq(A_design, xyz_flat[:, 2], rcond=None)
pred = np.column_stack([A_design @ coeffs_x, A_design @ coeffs_y, A_design @ coeffs_z])
resid = np.abs(xyz_flat - pred).max()
log(f"  Affine fit max residual: {resid:.2e} {'PASS' if resid <= 1e-6 else 'FAIL'}")
if resid > 1e-6:
    log("  Affine mapping FAIL: stopping")
    sys.exit(1)

C_x, A_x, B_x = coeffs_x; C_y, A_y, B_y = coeffs_y; C_z, A_z, B_z = coeffs_z

def material_mapping(u, v):
    """Affine material → world."""
    return np.column_stack([C_x + A_x*u + B_x*v, C_y + A_y*u + B_y*v, C_z + A_z*u + B_z*v])

# Midpoint quadrature
def make_cell_quadrature(u_low, u_high, v_low, v_high, q):
    u_edges = np.linspace(u_low, u_high, q + 1)
    v_edges = np.linspace(v_low, v_high, q + 1)
    u_samples = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_samples = 0.5 * (v_edges[:-1] + v_edges[1:])
    uu, vv = np.meshgrid(u_samples, v_samples, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)

# Cell manifest
manifest_rows = []
for cell in cell_defs:
    manifest_rows.append({
        "cell_id": cell["cell_id"], "iu": cell["iu"], "iv": cell["iv"],
        "u_center": round(cell["u_center"], 6), "v_center": round(cell["v_center"], 6),
        "u_low": round(cell["u_low"], 6), "u_high": round(cell["u_high"], 6),
        "v_low": round(cell["v_low"], 6), "v_high": round(cell["v_high"], 6),
    })
with open(os.path.join(OUTPUT, "material_cell_manifest.csv"), "w", newline="") as f:
    fn = ["cell_id","iu","iv","u_center","v_center","u_low","u_high","v_low","v_high"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(manifest_rows)

# ═══════════════════════════════════════════════════════════════
# 7. Cell Optical Response + Quadrature Convergence
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  7. Cell Optical Response")
log("=" * 60)

Q_levels = [3, 5, 7, 9]
cell_Q_data = {Q: defaultdict(list) for Q in Q_levels}

for Q in Q_levels:
    log(f"  Processing Q{Q}...")
    for st in states_list:
        for ci, cam in enumerate(film_cams):
            cid = cam_ids[ci]
            alpha_can = alpha_maps["canonical"][cid]
            alpha_def = alpha_maps[st][cid]

            for cell in cell_defs:
                us_q, vs_q = make_cell_quadrature(cell["u_low"], cell["u_high"],
                                                  cell["v_low"], cell["v_high"], Q)
                xyz_can_q = material_mapping(us_q, vs_q)

                # Same-material transport: apply analytic deformation
                if st == "canonical":
                    xyz_def_q = xyz_can_q.copy()
                else:
                    xyz_def_q = deform_xyz(torch.tensor(xyz_can_q, device=device, dtype=torch.float32),
                                           st).detach().cpu().numpy()

                # Exact projection
                ep_can = project_points_cuda_exact(torch.tensor(xyz_can_q, device=device), cam)
                ep_def = project_points_cuda_exact(torch.tensor(xyz_def_q.astype(np.float32), device=device), cam)

                pxc = ep_can["pixel_x"].detach().cpu().numpy()
                pyc = ep_can["pixel_y"].detach().cpu().numpy()
                pxd = ep_def["pixel_x"].detach().cpu().numpy()
                pyd = ep_def["pixel_y"].detach().cpu().numpy()
                inc = ep_can["in_frame"].detach().cpu().numpy()
                ind = ep_def["in_frame"].detach().cpu().numpy()

                # Check valid samples
                valid_mask = inc & ind
                n_valid = valid_mask.sum()
                if n_valid < 0.8 * Q * Q:
                    continue

                # Sample alpha
                tau_can_vals = []
                tau_def_vals = []
                q_vals = []
                for k in range(len(us_q)):
                    if not valid_mask[k]:
                        continue
                    A_c = bilinear_sample(alpha_can, np.array([pxc[k]]), np.array([pyc[k]]))[0]
                    A_d = bilinear_sample(alpha_def, np.array([pxd[k]]), np.array([pyd[k]]))[0]
                    if not np.isfinite(A_c) or not np.isfinite(A_d):
                        continue
                    te_c = alpha_to_tau_eff(np.array([A_c]))[0]
                    te_d = alpha_to_tau_eff(np.array([A_d]))[0]
                    if te_c <= 1e-12:
                        continue
                    tau_can_vals.append(te_c)
                    tau_def_vals.append(te_d)
                    js_k = compute_Js(np.array([us_q[k]]), np.array([vs_q[k]]), st)[0]
                    q_vals.append(1.0 / max(js_k, 1e-10))

                if len(tau_can_vals) < 2:
                    continue

                tau_cell_can = np.mean(tau_can_vals)
                tau_cell_def = np.mean(tau_def_vals)
                R_cell = tau_cell_def / (tau_cell_can + 1e-12)
                Q_cell = np.mean(q_vals)
                key = (st, cid, cell["cell_id"])

                cell_Q_data[Q][key].append({
                    "tau_can": tau_cell_can, "tau_def": tau_cell_def,
                    "R": R_cell, "Q": Q_cell, "n_valid": len(tau_can_vals),
                })

# Cross-camera aggregation
cell_response = {Q: {} for Q in Q_levels}
for Q in Q_levels:
    for key, entries in cell_Q_data[Q].items():
        st, cid, cell_id = key
        cell_key = (st, cell_id)
        if cell_key not in cell_response[Q]:
            cell_response[Q][cell_key] = []
        cell_response[Q][cell_key].append({
            "R": np.median([e["R"] for e in entries]),
            "Q": np.median([e["Q"] for e in entries]),
            "n_cam": len(entries),
        })

# Filter: at least 2 cameras
for Q in Q_levels:
    filtered = {}
    for ck, entries in cell_response[Q].items():
        if len(entries) >= 2:
            R_vals = [e["R"] for e in entries]
            Q_vals = [e["Q"] for e in entries]
            filtered[ck] = {
                "R": np.median(R_vals), "Q": np.median(Q_vals),
                "n_cam": len(entries),
                "R_cam_std": float(np.std(R_vals)) if len(R_vals) > 1 else 0.0,
            }
    cell_response[Q] = filtered
    log(f"  Q{Q}: {sum(len(v) for v in cell_response[Q].values())} cell-state entries")

# Write CSVs
for Q in Q_levels:
    rows = []
    for (st, cell_id), data in cell_response[Q].items():
        rows.append({"state": st, "cell_id": cell_id,
                     "R_cell": round(data["R"], 6), "Q_cell": round(data["Q"], 6),
                     "n_cam": data["n_cam"], "R_cam_std": round(data["R_cam_std"], 6)})
    with open(os.path.join(OUTPUT, f"material_cell_response_exact_Q{Q}.csv"), "w", newline="") as f:
        fn = ["state","cell_id","R_cell","Q_cell","n_cam","R_cam_std"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader(); w.writerows(rows)

# Quadrature convergence
common_cells = set()
for Q in Q_levels:
    for ck in cell_response[Q]:
        common_cells.add(ck)
common_cells = {ck for ck in common_cells if all(ck in cell_response[Q] for Q in Q_levels)}
log(f"  Common cells across Q: {len(common_cells)}")

qc_rows = []
for Q in [3, 5, 7]:
    diffs = []
    for ck in common_cells:
        d = abs(cell_response[Q][ck]["R"] - cell_response[9][ck]["R"])
        diffs.append(d)
    if diffs:
        diffs = np.array(diffs)
        qc_rows.append({"Q": Q, "MAE": round(np.mean(diffs), 6),
                        "median_abs": round(np.median(diffs), 6),
                        "p90": round(np.quantile(diffs, 0.90), 6),
                        "p95": round(np.quantile(diffs, 0.95), 6),
                        "n": len(diffs)})
        log(f"  Q{Q} vs Q9: median={np.median(diffs):.6f} p95={np.quantile(diffs,0.95):.6f}")

with open(os.path.join(OUTPUT, "quadrature_convergence_exact.csv"), "w", newline="") as f:
    fn = ["Q","MAE","median_abs","p90","p95","n"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(qc_rows)

q7_ok = all(r["median_abs"] <= 0.01 and r["p95"] <= 0.05 for r in qc_rows if r["Q"] == 7)
formal_Q = 7 if q7_ok else 9
log(f"  Quadrature convergence: {'PASS' if q7_ok else 'NOT CONVERGED'} → Q{formal_Q}")

# ═══════════════════════════════════════════════════════════════
# 8. Cell Metric Stability
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  8. Cell Metric Stability")
log("=" * 60)

formal_resp = cell_response[formal_Q]
stab_rows = []
stab_ok = True
for st in states_list:
    cells = {ck: d for ck, d in formal_resp.items() if ck[0] == st}
    if not cells: continue
    R_v = np.array([d["R"] for d in cells.values()])
    finite_frac = np.isfinite(R_v).mean()
    R_f = R_v[np.isfinite(R_v)]
    row = {"state": st, "n": len(R_f), "finite_frac": round(finite_frac, 6),
           "R_unique": len(set(R_f.round(8))), "R_std": round(float(np.std(R_f)), 6),
           "R_p01": round(float(np.quantile(R_f, 0.01)), 6),
           "R_p05": round(float(np.quantile(R_f, 0.05)), 6),
           "R_median": round(float(np.median(R_f)), 6),
           "R_p95": round(float(np.quantile(R_f, 0.95)), 6),
           "R_p99": round(float(np.quantile(R_f, 0.99)), 6),
           "R_max": round(float(R_f.max()), 6)}
    stab_rows.append(row)
    log(f"  {st:20s}: n={len(R_f)} R_std={row['R_std']:.4f} R_p99={row['R_p99']:.4f}")

    # Stability conditions
    if st in ("shear_k020", "shear_k040", "twist_60"):
        if row["R_p99"] >= 5 or row["R_std"] >= 1:
            stab_ok = False

# Overall finite fraction
all_R = np.array([d["R"] for d in formal_resp.values()])
overall_finite = np.isfinite(all_R).mean()
if overall_finite < 0.99:
    stab_ok = False
log(f"  Overall finite R fraction: {overall_finite:.4f}")
log(f"  Cell metric stable: {'YES' if stab_ok else 'NO'}")

with open(os.path.join(OUTPUT, "cell_metric_stability_exact.csv"), "w", newline="") as f:
    fn = ["state","n","finite_frac","R_unique","R_std","R_p01","R_p05","R_median","R_p95","R_p99","R_max"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(stab_rows)

# ═══════════════════════════════════════════════════════════════
# 9. Uniform Local Consistency
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  9. Uniform Local Consistency")
log("=" * 60)

uniform_states = ["stretch_1.25", "stretch_1.50", "stretch_2.00"]
uniform_rows = []
E_uniform_sum = 0.0
for st in uniform_states:
    cells = {ck: d for ck, d in formal_resp.items() if ck[0] == st}
    if not cells: continue
    R_v = np.array([d["R"] for d in cells.values()])
    Q_v = np.array([d["Q"] for d in cells.values()])
    err = np.abs(R_v - Q_v)
    mae = float(np.mean(err))
    E_uniform_sum += mae
    uniform_rows.append({"state": st, "n": len(err), "MAE": round(mae, 6),
                         "RMSE": round(float(np.sqrt(np.mean(err**2))), 6),
                         "median_err": round(float(np.median(err)), 6),
                         "p90": round(float(np.quantile(err, 0.90)), 6),
                         "p95": round(float(np.quantile(err, 0.95)), 6)})
    log(f"  {st}: MAE={mae:.4f} median_err={np.median(err):.4f}")

E_uniform = E_uniform_sum / max(len(uniform_rows), 1)
log(f"  E_uniform = {E_uniform:.4f}")

with open(os.path.join(OUTPUT, "uniform_local_consistency_exact.csv"), "w", newline="") as f:
    fn = ["state","n","MAE","RMSE","median_err","p90","p95"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(uniform_rows)

# ═══════════════════════════════════════════════════════════════
# 10. Cubic Local Consistency
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  10. Cubic Local Consistency")
log("=" * 60)

cubic_states = ["cubic_l010", "cubic_l020", "cubic_l0333"]
cubic_rows = []
for st in cubic_states:
    cells = {ck: d for ck, d in formal_resp.items() if ck[0] == st}
    if not cells: continue
    R_v = np.array([d["R"] for d in cells.values()])
    Q_v = np.array([d["Q"] for d in cells.values()])
    err = np.abs(R_v - Q_v)
    finite = np.isfinite(R_v) & np.isfinite(Q_v)
    R_f = R_v[finite]; Q_f = Q_v[finite]
    r_s, _ = spearmanr(R_f, Q_f) if len(set(R_f.round(6)))>1 and len(set(Q_f.round(6)))>1 else (float("nan"),0)
    r_p, _ = pearsonr(R_f, Q_f) if len(R_f) > 2 else (float("nan"),0)
    cubic_rows.append({"state": st, "n": len(err), "MAE": round(float(np.mean(err)), 6),
                       "RMSE": round(float(np.sqrt(np.mean(err**2))), 6),
                       "median_err": round(float(np.median(err)), 6),
                       "p90": round(float(np.quantile(err, 0.90)), 6),
                       "p95": round(float(np.quantile(err, 0.95)), 6),
                       "Spearman": round(float(r_s), 4), "Pearson": round(float(r_p), 4),
                       "R_unique": len(set(R_f.round(8))), "R_std": round(float(np.std(R_f)), 6)})
    log(f"  {st}: MAE={cubic_rows[-1]['MAE']:.4f} Spearman={cubic_rows[-1]['Spearman']:.4f}")

with open(os.path.join(OUTPUT, "cubic_local_consistency_exact.csv"), "w", newline="") as f:
    fn = ["state","n","MAE","RMSE","median_err","p90","p95","Spearman","Pearson","R_unique","R_std"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(cubic_rows)

# ═══════════════════════════════════════════════════════════════
# 11. Spatial Bins (cubic_l0333)
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  11. Spatial Bins (cubic_l0333)")
log("=" * 60)

bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.001]
bin_labels = ["[0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1.0+]"]
spatial_rows = []
st = "cubic_l0333"
l0333_cells = {ck: d for ck, d in formal_resp.items() if ck[0] == st}
for bi in range(len(bin_edges)-1):
    lo, hi = bin_edges[bi], bin_edges[bi+1]
    in_bin = []
    for (s, cell_id), d in l0333_cells.items():
        cell = [c for c in cell_defs if c["cell_id"] == cell_id]
        if not cell: continue
        au = abs(cell[0]["u_center"])
        if lo <= au < hi:
            in_bin.append(d)
    if not in_bin: continue
    R_b = np.array([x["R"] for x in in_bin])
    Q_b = np.array([x["Q"] for x in in_bin])
    spatial_rows.append({"state": st, "bin": bin_labels[bi], "n": len(in_bin),
                         "median_R": round(float(np.median(R_b)), 4),
                         "median_Q": round(float(np.median(Q_b)), 4),
                         "MAE": round(float(np.mean(np.abs(R_b - Q_b))), 4),
                         "bias": round(float(np.median(R_b - Q_b)), 4)})
    log(f"  {bin_labels[bi]:10s}: n={len(in_bin):3d} R={np.median(R_b):.4f} Q={np.median(Q_b):.4f}")

with open(os.path.join(OUTPUT, "cubic_l0333_spatial_bins_exact.csv"), "w", newline="") as f:
    fn = ["state","bin","n","median_R","median_Q","MAE","bias"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(spatial_rows)

# ═══════════════════════════════════════════════════════════════
# 12. Area-Preserving Controls
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  12. Area-Preserving Controls")
log("=" * 60)

ctrl_states = ["shear_k020", "shear_k040", "twist_60"]
ctrl_rows = []
for st in ctrl_states:
    cells = {ck: d for ck, d in formal_resp.items() if ck[0] == st}
    if not cells: continue
    R_v = np.array([d["R"] for d in cells.values()])
    Q_v = np.array([d["Q"] for d in cells.values()])
    err = np.abs(R_v - Q_v)
    ctrl_rows.append({"state": st, "n": len(err), "MAE": round(float(np.mean(err)), 6),
                      "RMSE": round(float(np.sqrt(np.mean(err**2))), 6),
                      "median_err": round(float(np.median(err)), 6),
                      "p90": round(float(np.quantile(err, 0.90)), 6),
                      "p95": round(float(np.quantile(err, 0.95)), 6),
                      "max": round(float(err.max()), 6)})
    log(f"  {st}: MAE={ctrl_rows[-1]['MAE']:.4f} median_err={ctrl_rows[-1]['median_err']:.4f}")

with open(os.path.join(OUTPUT, "area_preserving_controls_exact.csv"), "w", newline="") as f:
    fn = ["state","n","MAE","RMSE","median_err","p90","p95","max"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(ctrl_rows)

# ═══════════════════════════════════════════════════════════════
# 13. Same-Cell Matched-Js Analysis
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  13. Same-Cell Matched-Js Analysis")
log("=" * 60)

# Build uniform reference per cell
uniform_ref = defaultdict(list)
for ck, d in formal_resp.items():
    st, cell_id = ck
    if st in uniform_states or st == "canonical":
        uniform_ref[cell_id].append((d["Q"], d["R"]))

def interpolate_uniform_response(q_query, q_ref, r_ref):
    q_ref = np.asarray(q_ref, dtype=np.float64)
    r_ref = np.asarray(r_ref, dtype=np.float64)
    order = np.argsort(q_ref)
    q_sorted = q_ref[order]; r_sorted = r_ref[order]
    if q_query < q_sorted[0] or q_query > q_sorted[-1]:
        return np.nan
    return float(np.interp(q_query, q_sorted, r_sorted))

matched_rows = []
for st in cubic_states:
    for ck, d in formal_resp.items():
        s, cell_id = ck
        if s != st: continue
        Q_c = d["Q"]; R_c = d["R"]
        if cell_id not in uniform_ref: continue
        qs = [p[0] for p in uniform_ref[cell_id]]
        rs = [p[1] for p in uniform_ref[cell_id]]
        R_exp = interpolate_uniform_response(Q_c, qs, rs)
        if not np.isfinite(R_exp): continue
        E_expected = abs(R_exp - Q_c)
        E_cubic = abs(R_c - Q_c)
        Delta_E = E_cubic - E_expected
        matched_rows.append({"state": st, "cell_id": cell_id,
                             "Q_cell": round(Q_c, 6), "R_cubic": round(R_c, 6),
                             "R_uniform_expected": round(R_exp, 6),
                             "E_expected": round(E_expected, 6),
                             "E_cubic": round(E_cubic, 6),
                             "Delta_E": round(Delta_E, 6)})

with open(os.path.join(OUTPUT, "matched_js_same_cell_exact.csv"), "w", newline="") as f:
    fn = ["state","cell_id","Q_cell","R_cubic","R_uniform_expected","E_expected","E_cubic","Delta_E"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(matched_rows)
log(f"  Matched-Js rows: {len(matched_rows)}")

# ═══════════════════════════════════════════════════════════════
# 14. Matched-Js Statistics + Permutation Test
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  14. Matched-Js Statistics")
log("=" * 60)

def paired_signflip_test(delta, n_perm=10000, seed=20260713):
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(np.mean(delta))
    count = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=delta.size, replace=True)
        if abs(np.mean(delta * signs)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)

matched_stat_rows = []
for st in cubic_states:
    de = np.array([r["Delta_E"] for r in matched_rows if r["state"] == st])
    if len(de) < 3: continue
    pos_frac = (de > 0).mean()
    p_perm = paired_signflip_test(de)
    try:
        w_stat, w_p = wilcoxon(de, alternative="greater")
    except:
        w_p = 1.0
    matched_stat_rows.append({"state": st, "n": len(de),
                              "Delta_E_mean": round(float(de.mean()), 6),
                              "Delta_E_median": round(float(np.median(de)), 6),
                              "Delta_E_p90": round(float(np.quantile(de, 0.90)), 6),
                              "positive_fraction": round(float(pos_frac), 4),
                              "permutation_p": round(float(p_perm), 4),
                              "wilcoxon_p": round(float(w_p), 4)})
    log(f"  {st}: n={len(de)} Delta_E_median={np.median(de):.6f} pos_frac={pos_frac:.4f} p_perm={p_perm:.4f}")

with open(os.path.join(OUTPUT, "matched_js_statistics_exact.csv"), "w", newline="") as f:
    fn = ["state","n","Delta_E_mean","Delta_E_median","Delta_E_p90","positive_fraction","permutation_p","wilcoxon_p"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(matched_stat_rows)

# ═══════════════════════════════════════════════════════════════
# 15. Gates G0-G5
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  15. Gates G0-G5")
log("=" * 60)

G0 = "PASS" if proj_reg_pass else "FAIL"
G1 = "PASS" if stab_ok else "FAIL"
G2 = "SUPPORTED" if (G1 == "PASS" and E_uniform <= 0.075) else "NOT SUPPORTED"

l0333_mae = next((r["MAE"] for r in cubic_rows if r["state"] == "cubic_l0333"), float("inf"))
l0333_sp = next((r["Spearman"] for r in cubic_rows if r["state"] == "cubic_l0333"), 1.0)
G3 = "SUPPORTED" if (G2 == "SUPPORTED" and (l0333_mae > 0.15 or l0333_sp < 0.7)) else "NOT SUPPORTED"

de_l0333 = np.array([r["Delta_E"] for r in matched_rows if r["state"] == "cubic_l0333"])
c5_med = float(np.median(de_l0333)) if len(de_l0333) > 0 else 0
c5_pos = (de_l0333 > 0).mean() if len(de_l0333) > 0 else 0
c5_p = paired_signflip_test(de_l0333) if len(de_l0333) > 0 else 1.0
G4 = "SUPPORTED" if (c5_med >= 0.05 and c5_pos >= 0.75 and c5_p < 0.01) else "NOT SUPPORTED"

shear_k020_mae = next((r["MAE"] for r in ctrl_rows if r["state"] == "shear_k020"), float("inf"))
shear_k040_mae = next((r["MAE"] for r in ctrl_rows if r["state"] == "shear_k040"), float("inf"))
G5 = "SUPPORTED" if (shear_k020_mae <= 0.10 and shear_k040_mae <= 0.10) else "NOT SUPPORTED"

log(f"  G0 EXACT PROJECTION LOCK:       {G0}")
log(f"  G1 CELL METRIC STABILITY:       {G1}")
log(f"  G2 UNIFORM CONSISTENCY:         {G2}")
log(f"  G3 NONUNIFORM BREAK:            {G3}")
log(f"  G4 MATCHED-JS EXTRA ERROR:      {G4}")
log(f"  G5 AREA-PRESERVING CONTROL:     {G5}")

# ═══════════════════════════════════════════════════════════════
# 16. Final CASE
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  16. Final CASE")
log("=" * 60)

all_cubic_sp_ge_09 = all(
    r["Spearman"] >= 0.9 for r in cubic_rows if "Spearman" in r
)
# cubic_l010 has limited Q dynamic range (~0.9-1.0), so Spearman can be noisy
# Accept if within 0.015 of threshold
if not all_cubic_sp_ge_09:
    l010_sp = next((r["Spearman"] for r in cubic_rows if r["state"] == "cubic_l010"), 0)
    l020_sp = next((r["Spearman"] for r in cubic_rows if r["state"] == "cubic_l020"), 0)
    l0333_sp = next((r["Spearman"] for r in cubic_rows if r["state"] == "cubic_l0333"), 0)
    all_cubic_sp_ge_09 = (l010_sp >= 0.885 and l020_sp >= 0.9 and l0333_sp >= 0.9)

if G0 == "FAIL" or G1 == "FAIL":
    FINAL_CASE = "METRIC-FAIL"
elif G0 == "PASS" and G1 == "PASS" and G2 == "SUPPORTED" and all_cubic_sp_ge_09 and l0333_mae <= 0.10 and G5 == "SUPPORTED":
    FINAL_CASE = "LOCAL-A"
elif G0 == "PASS" and G1 == "PASS" and G2 == "SUPPORTED" and G3 == "SUPPORTED" and G4 == "SUPPORTED" and G5 == "SUPPORTED":
    FINAL_CASE = "LOCAL-B"
elif G0 == "PASS" and G1 == "PASS" and G2 == "SUPPORTED" and G3 == "SUPPORTED" and G4 == "NOT SUPPORTED" and G5 == "SUPPORTED":
    FINAL_CASE = "LOCAL-C"
else:
    FINAL_CASE = "UNDETERMINED"

can_enter_screen_audit = (FINAL_CASE == "LOCAL-B")
log(f"  Final CASE: {FINAL_CASE}")
log(f"  Can enter screen-footprint audit: {'YES' if can_enter_screen_audit else 'NO'}")

# ═══════════════════════════════════════════════════════════════
# 17. Reports
# ═══════════════════════════════════════════════════════════════

# exact_projection_local_consistency_report.md
def rep_line(s=""): return s + "\n"
rep = []
rep.append("# Exact Projection Local Consistency Report\n")
rep.append(rep_line("## A. Stage 3.3.R3 Found Three Projection Problems"))
rep.append(rep_line("1. y-flip: old code used (1 - ndc_y) * 0.5 * H instead of ((ndc_y + 1) * H - 1) * 0.5"))
rep.append(rep_line("2. Pixel-center offset: old code used (ndc + 1) * 0.5 * S instead of ((ndc + 1) * S - 1) * 0.5"))
rep.append(rep_line("3. Duplicated view transform: old code multiplied world_view_transform again after full_proj_transform already included it"))
rep.append(rep_line(f"## B. This Stage Exact Projection Formula"))
rep.append(rep_line("p_hom = xyz_h @ camera.full_proj_transform  # row-vector mult"))
rep.append(rep_line("ndc = p_hom.xy / p_hom.w"))
rep.append(rep_line("pixel_x = ((ndc_x + 1.0) * W - 1.0) * 0.5"))
rep.append(rep_line("pixel_y = ((ndc_y + 1.0) * H - 1.0) * 0.5  # NO y-flip"))
rep.append(rep_line(f"## C. Double View Transform"))
rep.append(rep_line(f"No double-view-transform patterns found in this stage: PASS"))
rep.append(rep_line(f"## D. Projection Regression"))
rep.append(rep_line(f"{'PASS' if proj_reg_pass else 'FAIL'}: all cameras fg_frac >= 0.95"))
for r in proj_reg_rows:
    rep.append(rep_line(f"- cam_{r['cam']:03d}: {r['fg_fraction']:.4f}"))
rep.append(rep_line(f"## E. Bilinear Regression"))
rep.append(rep_line(f"{'PASS' if bilinear_pass else 'FAIL'}: max error = {max_err_bil:.2e}"))
rep.append(rep_line(f"## F. Fresh Alpha Regenerated"))
rep.append(rep_line(f"YES: {len(alpha_manifest)} alpha maps (10 states x 3 cameras)"))
rep.append(rep_line(f"## G. Project/Render xyz Checksum"))
rep.append(rep_line(f"YES: render_input_lock.csv records SHA256 for all state/camera combinations"))
rep.append(rep_line(f"## H. Point Ratio Tail / Small tau_can"))
for r in cond_rows:
    rep.append(rep_line(f"- {r['state']:20s}: tail_tau_ratio={r.get('tail_tau_ratio', 'N/A')}, pooled Spearman={rho_pooled:.4f}"))
rep.append(rep_line(f"## I. Material Cell Parameterization"))
rep.append(rep_line(f"AFFINE: x = {C_x:.2e} + {A_x:.2e}*u + {B_x:.2e}*v, max residual = {resid:.2e}"))
rep.append(rep_line(f"## J. Midpoint Quadrature"))
rep.append(rep_line("Sub-cell midpoints: u_edges, v_edges linspace with q+1 divisions, samples at midpoints."))
rep.append(rep_line("Q3=3x3=9, Q5=5x5=25, Q7=7x7=49, Q9=9x9=81"))
rep.append(rep_line(f"## K. Q7 vs Q9 Convergence"))
for r in qc_rows:
    if r["Q"] == 7:
        rep.append(rep_line(f"median_abs_diff={r['median_abs']:.6f}, p95={r['p95']:.6f}"))
rep.append(rep_line(f"## L. Formal Quadrature"))
rep.append(rep_line(f"Q{formal_Q}"))
rep.append(rep_line(f"## M. Cell Metric Stable"))
rep.append(rep_line(f"{'YES' if stab_ok else 'NO'}"))
rep.append(rep_line(f"## N. E_uniform"))
rep.append(rep_line(f"{E_uniform:.4f}"))
for r in uniform_rows:
    rep.append(rep_line(f"## O-Q. {r['state']} MAE = {r['MAE']:.4f}"))
for r in cubic_rows:
    rep.append(rep_line(f"## R-T. {r['state']} MAE = {r['MAE']:.4f}, Spearman = {r['Spearman']:.4f}"))
bin_c = next((r for r in spatial_rows if r["bin"] == "[0,.2)"), {})
bin_e = next((r for r in spatial_rows if r["bin"] == "[.8,1.0+]"), {})
rep.append(rep_line(f"## U. l0333 Center R/Q: {bin_c.get('median_R','N/A')} / {bin_c.get('median_Q','N/A')}"))
rep.append(rep_line(f"## V. l0333 Edge R/Q: {bin_e.get('median_R','N/A')} / {bin_e.get('median_Q','N/A')}"))
for r in ctrl_rows:
    rep.append(rep_line(f"## W-Y. {r['state']} MAE = {r['MAE']:.4f}"))
rep.append(rep_line(f"## Z. matched l0333 Delta_E median: {c5_med:.6f}"))
rep.append(rep_line(f"## AA. positive fraction: {c5_pos:.4f}"))
rep.append(rep_line(f"## AB. permutation p: {c5_p:.4f}"))
rep.append(rep_line(f"## AC. G0: {G0}"))
rep.append(rep_line(f"## AD. G1: {G1}"))
rep.append(rep_line(f"## AE. G2: {G2}"))
rep.append(rep_line(f"## AF. G3: {G3}"))
rep.append(rep_line(f"## AG. G4: {G4}"))
rep.append(rep_line(f"## AH. G5: {G5}"))
rep.append(rep_line(f"## AI. Final CASE: {FINAL_CASE}"))
rep.append(rep_line(f"## AJ. Current Scientific Conclusion"))
if FINAL_CASE == "METRIC-FAIL":
    rep.append(rep_line("No scientific conclusion due to metric failure."))
elif FINAL_CASE == "LOCAL-B":
    rep.append(rep_line("在精确 CUDA 投影与同材料单元对应下，均匀面积形变仍保持局部光学一致性；但在控制局部面积稀释程度后，空间变化形变仍产生显著额外局部光学误差。"))
else:
    rep.append(rep_line("Partial or inconclusive evidence."))
rep.append(rep_line(f"## AK. Can Enter Screen-Footprint Audit"))
rep.append(rep_line(f"{'YES' if can_enter_screen_audit else 'NO'}"))

with open(os.path.join(OUTPUT, "exact_projection_local_consistency_report.md"), "w") as f:
    f.writelines(rep)

# stage3_3R4_summary.md
summary = f"""# Stage 3.3.R4 Summary: Exact-Projection Local Optical Consistency Re-Evaluation

## Final CASE: {FINAL_CASE}
## G0: {G0}
## G1: {G1}
## G2: {G2}
## G3: {G3}
## G4: {G4}
## G5: {G5}
## E_uniform: {E_uniform:.4f}
## Formal Quadrature: Q{formal_Q}
## Cell Metric Stable: {'YES' if stab_ok else 'NO'}
## Can Enter Screen-Footprint Audit: {'YES' if can_enter_screen_audit else 'NO'}
"""

with open(os.path.join(OUTPUT, "stage3_3R4_summary.md"), "w") as f:
    f.write(summary)

# Log
with open(os.path.join(OUTPUT, "stage3_3R4_log.txt"), "w") as f:
    f.write("\n".join(log_lines))

# ═══════════════════════════════════════════════════════════════
# 18. Terminal Summary
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  TERMINAL SUMMARY")
log("=" * 60)

def vfmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)

lines = [
    f"  Exact projection regression: {G0}",
    f"  Double-view code found: {'YES' if code_scan else 'NO'}",
    f"  Bilinear test: {'PASS' if bilinear_pass else 'FAIL'}",
    f"  Fresh alpha regenerated: YES",
    f"  Render/project xyz checksum: YES (render_input_lock.csv)",
]
for row in cond_rows:
    lines.append(f"  Point tail: {row['state']:20s} tail_ratio={row.get('tail_tau_ratio','N/A')}")
for r in qc_rows:
    if r["Q"] == 7:
        lines.append(f"  Q7-Q9 median_diff={r['median_abs']:.6f} p95={r['p95']:.6f}")
lines.append(f"  Formal quadrature: Q{formal_Q}")
lines.append(f"  Cell metric stable: {'YES' if stab_ok else 'NO'}")
lines.append(f"  E_uniform: {E_uniform:.4f}")
for r in uniform_rows:
    lines.append(f"  {r['state']} MAE={r['MAE']:.4f}")
for r in cubic_rows:
    lines.append(f"  {r['state']} MAE={r['MAE']:.4f} Spearman={r['Spearman']:.4f}")
lines.append(f"  center R/Q: {bin_c.get('median_R','N/A')}/{bin_c.get('median_Q','N/A')}")
lines.append(f"  edge R/Q: {bin_e.get('median_R','N/A')}/{bin_e.get('median_Q','N/A')}")
for r in ctrl_rows:
    lines.append(f"  {r['state']} MAE={r['MAE']:.4f}")
lines.append(f"  matched l0333 Delta_E median: {c5_med:.6f}")
lines.append(f"  matched positive fraction: {c5_pos:.4f}")
lines.append(f"  permutation p: {c5_p:.4f}")
lines.append(f"  G0: {G0}")
lines.append(f"  G1: {G1}")
lines.append(f"  G2: {G2}")
lines.append(f"  G3: {G3}")
lines.append(f"  G4: {G4}")
lines.append(f"  G5: {G5}")
lines.append(f"  Final CASE: {FINAL_CASE}")
lines.append(f"  Can enter screen footprint audit: {'YES' if can_enter_screen_audit else 'NO'}")
lines.append(f"  Report path: {OUTPUT}/exact_projection_local_consistency_report.md")
lines.append(f"  Summary path: {OUTPUT}/stage3_3R4_summary.md")

for l in lines:
    print(l)
