#!/usr/bin/env python3
"""Stage 3.4A: Clone-Induced Density Topology Optical Invariance Gate"""
import sys, os, math, csv, json, hashlib
import numpy as np
from collections import defaultdict

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_4A_clone_topology_optical_invariance"
os.makedirs(OUTPUT, exist_ok=True)

sys.path.insert(0, BASE)
sys.path.insert(0, "/data/wyh/repos/TSGS")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, f"{BASE}/benchmark")

import torch, trimesh
from torch.nn import functional as F
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov

from analysis.exact_cuda_projection import project_points_cuda_exact
from analysis.clone_topology import GaussianTensors, validate_gaussians, clone_gaussians, clone_unit_test

device = "cuda"
log_lines = []
def log(m): print(m); log_lines.append(str(m))

bg_color = torch.zeros(3, device=device)
pipe = type("obj", (object,), {"debug": False, "convert_SHs_python": False, "compute_cov3D_python": False})()

GRID = 41; L = 0.75; H = 256; W = 256
spacing = 1.5 / 40

def sha256_t(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def sha256_np(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

# ───── Start: Lock reference carrier ─────
log("=" * 60); log("  Locking reference carrier"); log("=" * 60)

mesh = trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N_ref = len(mesh.vertices)
verts_np = np.array(mesh.vertices, dtype=np.float32)
verts = torch.tensor(verts_np, device=device)
scale_t = torch.full((N_ref, 3), spacing, device=device); scale_t[:, 2] = spacing * 0.1
rot_t = torch.zeros(N_ref, 4, device=device); rot_t[:, 0] = 1.0
ckpt = torch.load(f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt",
                  map_location=device, weights_only=True)
tau_raw = ckpt["tau_raw"]; color_raw = ckpt["color_raw"]
opacity_t = 1 - torch.exp(-F.softplus(tau_raw))

assert N_ref == 1681, f"N={N_ref}"
ref_carrier_lock = {
    "n": N_ref, "source": "canonical_checkpoint.pt",
    "xyz_sha256": sha256_t(verts), "scale_sha256": sha256_t(scale_t),
    "rotation_sha256": sha256_t(rot_t), "tau_sha256": sha256_t(tau_raw),
    "color_sha256": sha256_t(color_raw),
}
with open(os.path.join(OUTPUT, "reference_carrier_lock.json"), "w") as f:
    json.dump(ref_carrier_lock, f, indent=2)

# ───── 12 cameras from benchmark rig (positions only, 256x256) ─────
cam_bench_json = json.load(open(f"{BASE}/experiments/stage1_minimal_gt/cameras.json"))
bench_cam_ids = list(range(12))

def build_bench_cam(cfg, res=512, fov=45):
    pa = np.array(cfg["origin"], dtype=np.float32)
    ta = np.array(cfg["target"], dtype=np.float32)
    ua = np.array(cfg["up"], dtype=np.float32)
    fwd = ta - pa; fwd /= np.linalg.norm(fwd)
    rt = np.cross(ua, fwd); rt /= np.linalg.norm(rt)
    nu = np.cross(fwd, rt)
    Rw = np.eye(3, dtype=np.float32); Rw[0, :] = rt; Rw[1, :] = nu; Rw[2, :] = fwd
    T = -Rw @ pa; R = Rw.T
    fx = res / (2 * math.tan(math.radians(fov / 2)))
    cam = Camera(colmap_id=cfg["id"], R=R, T=T,
                 FoVx=focal2fov(fx, res), FoVy=focal2fov(fx, res),
                 image_width=res, image_height=res,
                 image_path="", image_PIL=None,
                 image_name=f"cam_{cfg['id']:03d}", uid=cfg["id"],
                 preload_img=False, data_device="cpu")
    cam.original_image = torch.zeros(3, res, res)
    return cam

bench_cams = [build_bench_cam(c) for c in cam_bench_json]

# Adapter
class Adapter:
    def __init__(self, xyz, scl, rot, tau, col):
        self._xyz = xyz; self._scaling = torch.log(scl.clamp(min=1e-8))
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
    def get_transparency(self): return torch.full((self._xyz.shape[0], 1), 0.5, device=device)
    @property
    def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)

def white_pass(gm, cam):
    r2 = render(cam, gm, pipe, bg_color, app_model=None,
                override_color=torch.ones(gm.get_xyz.shape[0], 3, device=device),
                return_plane=False, return_depth_normal=False)
    return r2["render"].mean(dim=0, keepdim=True).clamp(0, 1)

# Bilinear sampler (same as Stage 3.3.R4)
def bilinear_sample(image, x, y):
    image = np.asarray(image, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Hi, Wi = image.shape
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < Wi-1) & (y >= 0) & (y < Hi-1)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    xv, yv = x[valid], y[valid]
    x0 = np.floor(xv).astype(np.int64); x1 = x0+1
    y0 = np.floor(yv).astype(np.int64); y1 = y0+1
    wx = xv - x0; wy = yv - y0
    out[valid] = ((1-wx)*(1-wy)*image[y0,x0] + wx*(1-wy)*image[y0,x1] +
                  (1-wx)*wy*image[y1,x0] + wx*wy*image[y1,x1])
    return out

def alpha_to_tau(alpha):
    T = np.clip(1.0 - np.asarray(alpha, dtype=np.float64), 1e-6, 1.0)
    return -np.log(T)

# Material parameterization (affine, validated)
u_vals = np.array([(i-20)/20.0 for i in range(GRID)], dtype=np.float64)
v_vals = np.array([(j-20)/20.0 for j in range(GRID)], dtype=np.float64)
A_design = np.column_stack([np.ones(N_ref), u_vals.repeat(GRID), np.tile(v_vals, GRID)])
xyz_flat = verts_np.reshape(-1, 3)
Cx, Ax, Bx = np.linalg.lstsq(A_design, xyz_flat[:, 0], rcond=None)[0]
Cy, Ay, By = np.linalg.lstsq(A_design, xyz_flat[:, 1], rcond=None)[0]
Cz, Az, Bz = np.linalg.lstsq(A_design, xyz_flat[:, 2], rcond=None)[0]
def material_map(us, vs):
    uu = np.asarray(us, dtype=np.float64); vv = np.asarray(vs, dtype=np.float64)
    return np.column_stack([Cx+Ax*uu+Bx*vv, Cy+Ay*uu+By*vv, Cz+Az*uu+Bz*vv])

# ───── Build REF GaussianTensors ─────
material_id_ref = torch.arange(N_ref, device=device, dtype=torch.long)
ref_gt = GaussianTensors(verts.clone(), scale_t.clone(), rot_t.clone(),
                         tau_raw.clone(), color_raw.clone(), material_id_ref.clone())

# ───── Clone parent selection ─────
iu_all = np.arange(1, GRID-1)  # 1..39
iv_all = np.arange(1, GRID-1)
Iu_grid, Iv_grid = np.meshgrid(iu_all, iv_all, indexing="ij")
Iu_flat = Iu_grid.ravel(); Iv_flat = Iv_grid.ravel()

def grid_to_idx(iu, iv):
    return iu * GRID + iv

def make_mask_c25_checker():
    return (Iu_flat % 2 == 0) & (Iv_flat % 2 == 0)

def make_mask_c50_checker():
    return (Iu_flat + Iv_flat) % 2 == 0

def make_mask_c100_interior():
    return np.ones(len(Iu_flat), dtype=bool)

def make_mask_c25_block():
    u_vals_i = (Iu_flat - 20) / 20.0
    v_vals_i = (Iv_flat - 20) / 20.0
    return (np.abs(u_vals_i) <= 0.5) & (np.abs(v_vals_i) <= 0.5)

clone_patterns = {
    "C25_CHECKER": make_mask_c25_checker(),
    "C50_CHECKER": make_mask_c50_checker(),
    "C100_INTERIOR": make_mask_c100_interior(),
    "C25_BLOCK": make_mask_c25_block(),
}
clone_modes = ["naive", "opacity_corrected"]

# Generate all representations
all_repr = {"REF": ref_gt.clone().to(device)}
repr_manifest = [{
    "representation": "REF", "gaussian_count": N_ref, "clone_count": 0,
    "clone_fraction": 0.0, "mode": "reference",
    "xyz_sha256": sha256_t(ref_gt.xyz), "tau_sha256": sha256_t(ref_gt.tau),
    "material_id_sha256": sha256_t(ref_gt.material_id),
}]

parent_manifest = []
parent_tensors = {}
for pname, mask in clone_patterns.items():
    selected_i = Iu_flat[mask]; selected_j = Iv_flat[mask]
    parent_idxs = [grid_to_idx(int(iu), int(iv)) for iu, iv in zip(selected_i, selected_j)]
    parent_t = torch.tensor(parent_idxs, device=device, dtype=torch.long)
    parent_tensors[pname] = parent_t

    for mode in clone_modes:
        repr_name = f"{pname}_{'NAIVE' if mode=='naive' else 'OC'}"
        gt = clone_gaussians(ref_gt, parent_t, mode)
        all_repr[repr_name] = gt
        frac = len(parent_t) / N_ref
        repr_manifest.append({
            "representation": repr_name, "gaussian_count": gt.n,
            "clone_count": len(parent_t), "clone_fraction": round(frac, 4),
            "mode": mode, "xyz_sha256": sha256_t(gt.xyz),
            "tau_sha256": sha256_t(gt.tau),
            "material_id_sha256": sha256_t(gt.material_id),
        })
        for idx in parent_idxs:
            iup = idx // GRID; ivp = idx % GRID
            parent_manifest.append({
                "variant": repr_name, "parent_index": idx,
                "iu": iup, "iv": ivp,
                "u": round((iup-20)/20.0, 4), "v": round((ivp-20)/20.0, 4),
                "tau_parent": float(tau_raw[idx].cpu()),
                "opacity_parent": float(opacity_t[idx].cpu()),
                "scale_u": spacing, "scale_v": spacing, "scale_n": spacing*0.1,
            })

log(f"  Created {len(all_repr)} representations")
for r in repr_manifest:
    log(f"  {r['representation']:25s}: N={r['gaussian_count']:5d} clones={r['clone_count']:3d}")

with open(os.path.join(OUTPUT, "clone_parent_manifest.csv"), "w", newline="") as f:
    fn = ["variant","parent_index","iu","iv","u","v","tau_parent","opacity_parent","scale_u","scale_v","scale_n"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(parent_manifest)

with open(os.path.join(OUTPUT, "representation_manifest.csv"), "w", newline="") as f:
    fn = ["representation","gaussian_count","clone_count","clone_fraction","mode",
          "xyz_sha256","tau_sha256","material_id_sha256"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(repr_manifest)

# ───── Clone unit test ─────
max_err_ct = clone_unit_test()
unit_test_pass = max_err_ct < 1e-12
log(f"  Clone unit test max error: {max_err_ct:.2e} {'PASS' if unit_test_pass else 'FAIL'}")
with open(os.path.join(OUTPUT, "clone_unit_tests.md"), "w") as f:
    f.write(f"# Clone Unit Tests\n\n")
    f.write(f"Revising Densification opacity correction formula:\n")
    f.write(f"  (1-o_new)^2 = 1-o_old  with tau_new = tau_old/2\n")
    f.write(f"  max error: {max_err_ct:.2e}\n")
    f.write(f"  {'PASS' if unit_test_pass else 'FAIL'}\n")

# ───── Clone identity assertions ─────
log("\n" + "=" * 60); log("  Clone identity assertions"); log("=" * 60)
identity_ok = True
for pname in clone_patterns:
    for mode in clone_modes:
        rn = f"{pname}_{'NAIVE' if mode=='naive' else 'OC'}"
        gt = all_repr[rn]
        mid = gt.material_id.long()
        # Find clones (indices beyond N_ref)
        clone_mask = torch.arange(gt.n, device=device) >= N_ref
        clone_idxs = torch.where(clone_mask)[0]
        for ci in clone_idxs:
            mi = mid[ci]
            parent_idx = mi  # material_id = index in REF
            for attr in ["xyz", "scale", "rotation", "color"]:
                pv = getattr(ref_gt, attr)[parent_idx]
                cv = getattr(gt, attr)[ci]
                if (pv - cv).abs().max().item() > 1e-8:
                    log(f"  FAIL: {rn} clone {ci} {attr} differs from parent {parent_idx}")
                    identity_ok = False
        # Check tau
        if mode == "naive":
            for ci in clone_idxs:
                mi = mid[ci]
                if (ref_gt.tau[mi] - gt.tau[ci]).abs().max().item() > 1e-5:
                    log(f"  FAIL: {rn} naive clone tau mismatch at {ci}")
                    identity_ok = False
        elif mode == "opacity_corrected":
            # Use stored parent tensor for this pattern
            for pname_ck, pt in parent_tensors.items():
                if pname_ck in rn:
                    for pi in pt.tolist():
                        expected = ref_gt.tau[pi].item() / 2.0
                        actual = gt.tau[pi].item()
                        if abs(expected - actual) > 1e-5:
                            log(f"  FAIL: {rn} parent {pi} tau: expected {expected:.6f} got {actual:.6f}")
                            identity_ok = False
                    break
log(f"  Identity assertions: {'PASS' if identity_ok else 'FAIL'}")

# ───── 12-camera canonical fresh render ─────
log("\n" + "=" * 60); log("  Canonical fresh renders (12 cameras)"); log("=" * 60)

alpha_dir = os.path.join(OUTPUT, "canonical_alpha")
os.makedirs(alpha_dir, exist_ok=True)

can_alpha = {}
can_manifest = []
for rname, gt in all_repr.items():
    can_alpha[rname] = {}
    gm = Adapter(gt.xyz, gt.scale, gt.rotation, gt.tau, gt.color)
    for ci, cam in enumerate(bench_cams):
        cid = cam.colmap_id
        a = white_pass(gm, cam).detach().cpu().numpy().squeeze(0)
        can_alpha[rname][cid] = a
        sdir = os.path.join(alpha_dir, rname)
        os.makedirs(sdir, exist_ok=True)
        np.save(os.path.join(sdir, f"cam{cid:03d}.npy"), a)
        can_manifest.append({
            "representation": rname, "cam": cid,
            "min": f"{a.min():.6f}", "max": f"{a.max():.6f}",
            "mean": f"{a.mean():.6f}", "sha256": sha256_np(a),
        })
    log(f"  Rendered {rname} (N={gt.n})")
    del gm

with open(os.path.join(OUTPUT, "canonical_clone_equivalence.csv"), "w", newline="") as f:
    fn = ["representation","cam","min","max","mean","sha256"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(can_manifest)

# ───── Canonical equivalence metrics ─────
log("\n" + "=" * 60); log("  Canonical equivalence metrics"); log("=" * 60)

eq_rows = []
for rname in all_repr:
    if rname == "REF": continue
    gt = all_repr[rname]
    for ci, cam in enumerate(bench_cams):
        cid = cam.colmap_id
        a_r = can_alpha[rname][cid]
        a_ref = can_alpha["REF"][cid]
        alpha_mae = float(np.abs(a_r - a_ref).mean())
        # Tau_eff MAE
        tau_r = alpha_to_tau(a_r.ravel())
        tau_ref = alpha_to_tau(a_ref.ravel())
        tau_mae = float(np.abs(tau_r - tau_ref).mean())
        eq_rows.append({"representation": rname, "cam": cid,
                        "alpha_mae": alpha_mae, "tau_mae": tau_mae})

# Aggregate per representation
repr_eq = defaultdict(list)
for r in eq_rows:
    repr_eq[r["representation"]].append(r)

canon_eq_rows = []
for rname, rows in repr_eq.items():
    alpha_maes = np.array([r["alpha_mae"] for r in rows])
    tau_maes = np.array([r["tau_mae"] for r in rows])
    canon_eq_rows.append({
        "representation": rname,
        "mean_alpha_mae": round(float(alpha_maes.mean()), 6),
        "max_alpha_mae": round(float(alpha_maes.max()), 6),
        "mean_tau_mae": round(float(tau_maes.mean()), 6),
        "max_tau_mae": round(float(tau_maes.max()), 6),
    })
    log(f"  {rname:25s}: alpha_MAE={alpha_maes.mean():.6f} (max={alpha_maes.max():.6f}) tau_MAE={tau_maes.mean():.6f}")

with open(os.path.join(OUTPUT, "canonical_equivalence_summary.csv"), "w", newline="") as f:
    fn = ["representation","mean_alpha_mae","max_alpha_mae","mean_tau_mae","max_tau_mae"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(canon_eq_rows)

# ───── Q7 cell tau metric for canonical equivalence ─────
log("  Computing Q7 cell tau for canonical equivalence...")

def make_cell_quad(u_low, u_high, v_low, v_high, q=7):
    ue = np.linspace(u_low, u_high, q+1); ve = np.linspace(v_low, v_high, q+1)
    us = 0.5*(ue[:-1]+ue[1:]); vs = 0.5*(ve[:-1]+ve[1:])
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)

cell_defs = []
for iu in range(1, GRID-1):
    for iv in range(1, GRID-1):
        cell_defs.append({
            "id": len(cell_defs), "iu": iu, "iv": iv,
            "u_c": u_vals[iu], "v_c": v_vals[iv],
            "u_l": 0.5*(u_vals[iu-1]+u_vals[iu]),
            "u_h": 0.5*(u_vals[iu]+u_vals[iu+1]),
            "v_l": 0.5*(v_vals[iv-1]+v_vals[iv]),
            "v_h": 0.5*(v_vals[iv]+v_vals[iv+1]),
        })

def compute_cell_tau(alpha_map, cell_defs, cam):
    """Compute per-cell mean tau_eff using Q7."""
    cell_taus = {}
    for cell in cell_defs:
        us_q, vs_q = make_cell_quad(cell["u_l"], cell["u_h"], cell["v_l"], cell["v_h"], 7)
        xyz_q = material_map(us_q, vs_q)
        ep = project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32), device=device), cam)
        pxc = ep["pixel_x"].detach().cpu().numpy()
        pyc = ep["pixel_y"].detach().cpu().numpy()
        inc = ep["in_frame"].detach().cpu().numpy()
        if inc.sum() < 0.8 * 49:
            cell_taus[cell["id"]] = np.nan
            continue
        A_s = bilinear_sample(alpha_map, pxc[inc], pyc[inc])
        tau_s = alpha_to_tau(A_s)
        cell_taus[cell["id"]] = float(np.nanmean(tau_s))
    return cell_taus

cell_tau_rows = []
for rname in all_repr:
    if rname == "REF": continue
    for ci, cam in enumerate(bench_cams):
        cid = cam.colmap_id
        t_ref = compute_cell_tau(can_alpha["REF"][cid], cell_defs, cam)
        t_r = compute_cell_tau(can_alpha[rname][cid], cell_defs, cam)
        errors = []
        for cell_id in t_ref:
            if np.isfinite(t_ref[cell_id]) and np.isfinite(t_r.get(cell_id, np.nan)):
                errors.append(abs(t_ref[cell_id] - t_r[cell_id]))
        if errors:
            cell_tau_rows.append({
                "representation": rname, "cam": cid,
                "cell_tau_mae": round(float(np.mean(errors)), 6),
                "cell_tau_rmse": round(float(np.sqrt(np.mean(np.square(errors)))), 6),
            })

repr_cell_tau = defaultdict(list)
for r in cell_tau_rows:
    repr_cell_tau[r["representation"]].append(r)

for rname, rows in repr_cell_tau.items():
    maes = [r["cell_tau_mae"] for r in rows]
    log(f"  {rname:25s}: cell_tau_MAE={np.mean(maes):.6f}")

with open(os.path.join(OUTPUT, "canonical_cell_tau_equivalence.csv"), "w", newline="") as f:
    fn = ["representation","cam","cell_tau_mae","cell_tau_rmse"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(cell_tau_rows)

# ───── T1: Naive vs OC bias control ─────
log("\n" + "=" * 60); log("  T1: Clone Opacity Bias Control"); log("=" * 60)

t1_pairs = []
for pname in clone_patterns:
    n_rep = f"{pname}_NAIVE"
    oc_rep = f"{pname}_OC"
    if n_rep not in all_repr or oc_rep not in all_repr: continue
    n_maes = np.array([r["alpha_mae"] for r in repr_eq[n_rep]])
    oc_maes = np.array([r["alpha_mae"] for r in repr_eq[oc_rep]])
    t1_pairs.append({
        "pattern": pname,
        "naive_mean_mae": round(float(n_maes.mean()), 6),
        "oc_mean_mae": round(float(oc_maes.mean()), 6),
        "ratio": round(float(n_maes.mean() / max(oc_maes.mean(), 1e-10)), 4),
        "diff": round(float(n_maes.mean() - oc_maes.mean()), 6),
        "naive_gt_2x_oc": "YES" if n_maes.mean() > 2 * oc_maes.mean() else "NO",
    })
    log(f"  {pname}: NAIVE_MAE={n_maes.mean():.6f} OC_MAE={oc_maes.mean():.6f} ratio={n_maes.mean()/max(oc_maes.mean(),1e-10):.2f}")

t1_pass_count = sum(1 for p in t1_pairs if p["naive_gt_2x_oc"] == "YES" and p["diff"] >= 0.005)
T1_status = "PASS" if t1_pass_count >= 3 else "NOT SUPPORTED"
log(f"  T1: {T1_status} ({t1_pass_count}/4 patterns pass)")

with open(os.path.join(OUTPUT, "clone_bias_control.csv"), "w", newline="") as f:
    fn = ["pattern","naive_mean_mae","oc_mean_mae","ratio","diff","naive_gt_2x_oc"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(t1_pairs)

# ───── T2: Canonical Equivalence Gate ─────
log("\n" + "=" * 60); log("  T2: Canonical Equivalence Gate"); log("=" * 60)

OC_repr_names = [r for r in all_repr if r.endswith("_OC")]
canon_eq_oc = []
for rname in OC_repr_names:
    rows = repr_eq[rname]
    mean_alpha = np.mean([r["alpha_mae"] for r in rows])
    max_alpha = np.max([r["alpha_mae"] for r in rows])
    mean_tau = np.mean([r["tau_mae"] for r in rows])
    cr = repr_cell_tau.get(rname, [])
    mean_cell_tau = np.mean([r["cell_tau_mae"] for r in cr]) if cr else float("inf")

    passes = (mean_alpha <= 0.005 and mean_tau <= 0.0075 and
              mean_cell_tau <= 0.01 and max_alpha <= 0.01)
    canon_eq_oc.append({
        "representation": rname,
        "mean_alpha_mae": round(mean_alpha, 6),
        "max_alpha_mae": round(max_alpha, 6),
        "mean_tau_mae": round(mean_tau, 6),
        "mean_cell_tau_mae": round(mean_cell_tau, 6),
        "canonical_equivalent": "YES" if passes else "NO",
    })
    log(f"  {rname:25s}: α_MAE={mean_alpha:.6f} τ_MAE={mean_tau:.6f} cell_MAE={mean_cell_tau:.6f} {'EQ' if passes else '---'}")

eq_oc_variants = [r["representation"] for r in canon_eq_oc if r["canonical_equivalent"] == "YES"]
checker_eq = [r for r in eq_oc_variants if "CHECKER" in r]
block_or_c100_eq = [r for r in eq_oc_variants if ("BLOCK" in r or "C100" in r)]

T2_status = "PASS" if (len(checker_eq) >= 1 and len(block_or_c100_eq) >= 1 and
                        len(eq_oc_variants) >= 2) else "FAIL"
log(f"  T2: {T2_status} ({len(eq_oc_variants)} equiv OC variants; checkers={len(checker_eq)} blocks={len(block_or_c100_eq)})")

with open(os.path.join(OUTPUT, "canonical_equivalence_gate.csv"), "w", newline="") as f:
    fn = ["representation","mean_alpha_mae","max_alpha_mae","mean_tau_mae","mean_cell_tau_mae","canonical_equivalent"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(canon_eq_oc)

# ───── Material transport (deformation) for OC variants ─────
log("\n" + "=" * 60); log("  Material transport for OC variants"); log("=" * 60)

STATE_MAP = {
    "stretch_1.50": {"type": "stretch", "s": 1.50},
    "stretch_2.00": {"type": "stretch", "s": 2.00},
    "biaxial_1.50": {"type": "biaxial", "s": 1.50},
    "cubic_l020": {"type": "cubic", "lam": 0.20},
    "cubic_l0333": {"type": "cubic", "lam": 1/3},
    "shear_k040": {"type": "shear", "k": 0.40},
    "twist_60": {"type": "twist"},
}
deform_states = list(STATE_MAP.keys())

from deformations.twist import deform_points as twist_def

def deform_and_transport(gt, state_name, material_u, material_v):
    mid = gt.material_id.long()
    # Convert material_id (which is the REF index) to grid coordinates
    iu_mat = mid // GRID; iv_mat = mid % GRID
    u_mat = material_u[iu_mat]; v_mat = material_v[iv_mat]
    cfg = STATE_MAP[state_name]; t = cfg["type"]

    # Compute deformed xyz per material
    x_can = gt.xyz.clone()
    if t == "identity": x_def = x_can
    elif t == "stretch": x_def = x_can.clone(); x_def[:,0] *= cfg["s"]
    elif t == "biaxial": x_def = x_can.clone(); x_def[:,0] *= cfg["s"]; x_def[:,1] *= cfg["s"]
    elif t == "cubic":
        lam = cfg["lam"]
        x_def = x_can.clone(); x_def[:,0] = x_can[:,0] + lam * x_can[:,0]**3 / L**2
    elif t == "shear":
        x_def = x_can.clone(); x_def[:,0] += cfg["k"] * x_can[:,1]**2 / L
    elif t == "twist":
        x_def = twist_def(x_can, 60, (x_can[:,2].min().item(), x_can[:,2].max().item()))
    else: x_def = x_can

    # F and Js (identity Jacobian for simplicity, just analytic Js for metric)
    _, Js = compute_cell_target(u_mat, v_mat, state_name)
    return x_def, Js, u_mat, v_mat

def compute_cell_target(us, vs, state_name):
    cfg = STATE_MAP[state_name]; t = cfg["type"]
    if torch.is_tensor(us): us_np = us.detach().cpu().numpy()
    else: us_np = np.asarray(us, dtype=np.float64)
    if t == "identity": return None, np.ones_like(us_np)
    if t == "stretch": return None, np.full_like(us_np, cfg["s"])
    if t == "biaxial": return None, np.full_like(us_np, cfg["s"]**2)
    if t == "cubic": return None, 1 + 3 * cfg["lam"] * us_np**2
    if t in ("shear", "twist"): return None, np.ones_like(us_np)
    return None, np.ones_like(us_np)

# Material u,v lookup tables
mat_u_t = torch.tensor(u_vals, device=device, dtype=torch.float32)
mat_v_t = torch.tensor(v_vals, device=device, dtype=torch.float32)

def transport_clone_representation(gt, state_name):
    mid = gt.material_id.long()
    u_mat = mat_u_t[mid // GRID]
    v_mat = mat_v_t[mid % GRID]
    x_def, Js, _, _ = deform_and_transport(gt, state_name, u_mat, v_mat)

    # Verify same material_id → same deformation
    uniq_mid = torch.unique(mid)
    for uid in uniq_mid:
        mask = mid == uid
        if mask.sum() > 1:
            sub_xyz = x_def[mask]
            if sub_xyz.std(dim=0).max().item() > 1e-7:
                log(f"  WARN: material_id {uid} has divergent xyz (std={sub_xyz.std(dim=0).max().item():.2e})")

    return GaussianTensors(
        xyz=x_def, scale=gt.scale.clone(), rotation=gt.rotation.clone(),
        tau=gt.tau.clone(), color=gt.color.clone(), material_id=gt.material_id.clone(),
    ), Js

# ───── Deformed render + cell metric ─────
log("\nDeformed renders and cell metric...")

deformed_alpha_dir = os.path.join(OUTPUT, "deformed_alpha")
os.makedirs(deformed_alpha_dir, exist_ok=True)

test_repr = ["REF"] + eq_oc_variants
def_alpha = {}
def_manifest = []
for rname in test_repr:
    def_alpha[rname] = {}
    gt = all_repr[rname]
    for st in deform_states:
        gt_def, Js = transport_clone_representation(gt, st)
        gm = Adapter(gt_def.xyz, gt_def.scale, gt_def.rotation, gt_def.tau, gt_def.color)
        def_alpha[rname][st] = {}
        for ci, cam in enumerate(bench_cams):
            cid = cam.colmap_id
            a = white_pass(gm, cam).detach().cpu().numpy().squeeze(0)
            def_alpha[rname][st][cid] = a
            sdir = os.path.join(deformed_alpha_dir, rname, st)
            os.makedirs(sdir, exist_ok=True)
            np.save(os.path.join(sdir, f"cam{cid:03d}.npy"), a)
            def_manifest.append({
                "representation": rname, "state": st, "cam": cid,
                "sha256": sha256_np(a),
            })
        log(f"  {rname:25s} {st}")
        del gm, gt_def
    del gt

with open(os.path.join(OUTPUT, "deformed_render_manifest.csv"), "w", newline="") as f:
    fn = ["representation","state","cam","sha256"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(def_manifest)

# ───── Q7 Cell optical response ─────
log("\nComputing cell optical response...")

def compute_cell_response(alpha_can, alpha_def, cell_defs, cam, Js_fn):
    """Compute R_cell, Q_cell for all cells."""
    R_cells = {}; Q_cells = {}
    for cell in cell_defs:
        us_q, vs_q = make_cell_quad(cell["u_l"], cell["u_h"], cell["v_l"], cell["v_h"], 7)
        xyz_q = material_map(us_q, vs_q)
        ep = project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32), device=device), cam)
        pxc = ep["pixel_x"].detach().cpu().numpy()
        pyc = ep["pixel_y"].detach().cpu().numpy()
        inc = ep["in_frame"].detach().cpu().numpy()
        if inc.sum() < 0.8 * 49:
            R_cells[cell["id"]] = np.nan; Q_cells[cell["id"]] = np.nan
            continue

        A_can = bilinear_sample(alpha_can, pxc[inc], pyc[inc])
        A_def = bilinear_sample(alpha_def, pxc[inc], pyc[inc])
        tau_can = alpha_to_tau(A_can); tau_def = alpha_to_tau(A_def)
        tau_cell_can = np.nanmean(tau_can); tau_cell_def = np.nanmean(tau_def)
        if tau_cell_can <= 1e-12:
            R_cells[cell["id"]] = np.nan; Q_cells[cell["id"]] = np.nan
            continue
        R_cells[cell["id"]] = tau_cell_def / tau_cell_can
        # Q_cell
        qs = 1.0 / np.maximum(Js_fn(us_q[inc], vs_q[inc]), 1e-10)
        Q_cells[cell["id"]] = float(np.mean(qs))
    return R_cells, Q_cells

def build_Js_fn(st):
    cfg = STATE_MAP[st]; t = cfg["type"]
    if t == "stretch": s = cfg["s"]; return lambda u,v: np.full_like(u, s)
    elif t == "biaxial": s = cfg["s"]; return lambda u,v: np.full_like(u, s*s)
    elif t == "cubic": lam = cfg["lam"]; return lambda u,v: 1+3*lam*np.asarray(u)**2
    else: return lambda u,v: np.ones_like(u)

cell_responses = {}
for rname in test_repr:
    cell_responses[rname] = {}
    for st in deform_states:
        Js_fn = build_Js_fn(st)
        cell_responses[rname][st] = {}
        cam_R_list = defaultdict(list)
        cam_Q_list = defaultdict(list)
        for ci, cam in enumerate(bench_cams):
            cid = cam.colmap_id
            R_cells, Q_cells = compute_cell_response(
                can_alpha[rname][cid], def_alpha[rname][st][cid],
                cell_defs, cam, Js_fn)
            for cell_id in R_cells:
                if np.isfinite(R_cells.get(cell_id, np.nan)):
                    cam_R_list[cell_id].append(R_cells[cell_id])
                    cam_Q_list[cell_id].append(Q_cells.get(cell_id, np.nan))

        # Cross-camera median
        for cell_id in cam_R_list:
            rv = [v for v in cam_R_list[cell_id] if np.isfinite(v)]
            qv = [v for v in cam_Q_list[cell_id] if np.isfinite(v)]
            if len(rv) >= 2:
                cell_responses[rname][st][cell_id] = {
                    "R": float(np.median(rv)), "Q": float(np.median(qv)),
                }
        log(f"  {rname:25s} {st:15s}: cells={len(cell_responses[rname][st])}")

# ───── Clone transport identity ─────
log("\nTransport identity...")
transport_rows = []
for rname in test_repr:
    if rname == "REF": continue
    gt = all_repr[rname]
    for st in deform_states:
        gt_def, Js = transport_clone_representation(gt, st)
        # Check same material_id → same xyz
        mid = gt_def.material_id.long()
        for uid in torch.unique(mid):
            mask = mid == uid
            if mask.sum() > 1:
                sub_xyz = gt_def.xyz[mask]
                max_diff = (sub_xyz - sub_xyz[:1]).abs().max().item()
                if max_diff > 1e-7:
                    transport_rows.append({
                        "representation": rname, "state": st,
                        "material_id": int(uid), "n_shared": int(mask.sum()),
                        "max_xyz_diff": max_diff, "pass": "NO",
                    })
                    transport_rows.append({
                        "representation": rname, "state": st,
                        "material_id": int(uid), "n_shared": int(mask.sum()),
                        "max_xyz_diff": 0.0, "pass": "YES",
                    })

with open(os.path.join(OUTPUT, "clone_transport_identity.csv"), "w", newline="") as f:
    fn = ["representation","state","material_id","n_shared","max_xyz_diff","pass"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(transport_rows)

# ───── Physical consistency & deformation invariance ─────
log("\nPhysical consistency & invariance...")

# Per-representation, per-state metrics
phys_rows = []
inv_rows = []
for rname in test_repr:
    for st in deform_states:
        cells = cell_responses[rname][st]
        if not cells: continue
        R_v = np.array([d["R"] for d in cells.values()])
        Q_v = np.array([d["Q"] for d in cells.values()])
        err = np.abs(R_v - Q_v)
        m = {"representation": rname, "state": st, "n": len(err),
             "MAE_phys": round(float(np.mean(err)), 6),
             "median_err": round(float(np.median(err)), 6),
             "p90": round(float(np.quantile(err, 0.90)), 6),
             "p95": round(float(np.quantile(err, 0.95)), 6)}
        if len(set(R_v.round(6))) > 1 and len(set(Q_v.round(6))) > 1:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(R_v, Q_v)
            m["Spearman"] = round(float(rho), 4)
        else:
            m["Spearman"] = float("nan")
        phys_rows.append(m)

        # Invariance: compare to REF
        if rname != "REF" and st in cell_responses["REF"]:
            ref_cells = cell_responses["REF"][st]
            Ds = []
            for cell_id in cells:
                if cell_id in ref_cells and np.isfinite(cells[cell_id]["R"]) and np.isfinite(ref_cells[cell_id]["R"]):
                    Ds.append(abs(cells[cell_id]["R"] - ref_cells[cell_id]["R"]))
            if Ds:
                Ds = np.array(Ds)
                inv_rows.append({"representation": rname, "state": st, "n": len(Ds),
                                 "MAE_R_REF": round(float(np.mean(Ds)), 6),
                                 "median_R_REF": round(float(np.median(Ds)), 6),
                                 "p90_R_REF": round(float(np.quantile(Ds, 0.90)), 6),
                                 "p95_R_REF": round(float(np.quantile(Ds, 0.95)), 6)})

with open(os.path.join(OUTPUT, "clone_physical_consistency.csv"), "w", newline="") as f:
    fn = ["representation","state","n","MAE_phys","median_err","p90","p95","Spearman"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(phys_rows)

with open(os.path.join(OUTPUT, "clone_deformation_invariance.csv"), "w", newline="") as f:
    fn = ["representation","state","n","MAE_R_REF","median_R_REF","p90_R_REF","p95_R_REF"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(inv_rows)

# ───── Amplification diagnostic ─────
amp_rows = []
for rname in eq_oc_variants:
    for st in deform_states:
        if st not in cell_responses[rname] or st not in cell_responses["REF"]: continue
        cells_r = cell_responses[rname][st]; cells_ref = cell_responses["REF"][st]
        # Canonical cell tau disagreement (D0): from canonical cell tau
        cr = repr_cell_tau.get(rname, [])
        D0 = np.mean([r["cell_tau_mae"] for r in cr]) if cr else 1e-8
        # Deformed response disagreement (Ds)
        Ds_list = []
        for cell_id in cells_r:
            if cell_id in cells_ref and np.isfinite(cells_r[cell_id]["R"]) and np.isfinite(cells_ref[cell_id]["R"]):
                Ds_list.append(abs(cells_r[cell_id]["R"] - cells_ref[cell_id]["R"]))
        Ds = np.mean(Ds_list) if Ds_list else D0
        amp_rows.append({"representation": rname, "state": st,
                         "D0": round(D0, 8), "Ds": round(Ds, 8),
                         "amplification": round(Ds / max(D0, 1e-8), 4)})

with open(os.path.join(OUTPUT, "clone_divergence_amplification.csv"), "w", newline="") as f:
    fn = ["representation","state","D0","Ds","amplification"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(amp_rows)

# ───── Clone region analysis (C25_BLOCK_OC) ─────
log("\nClone region analysis...")
block_rep = "C25_BLOCK_OC"
region_rows = []
if block_rep in test_repr:
    for st in deform_states:
        cells = cell_responses[block_rep][st]
        ref_cells = cell_responses["REF"][st]
        inside_err = []; outside_err = []
        for cell in cell_defs:
            au = abs(cell["u_c"]); av = abs(cell["v_c"])
            in_region = (au <= 0.5) and (av <= 0.5)
            cell_id = cell["id"]
            if cell_id not in cells or cell_id not in ref_cells: continue
            d = abs(cells[cell_id]["R"] - ref_cells[cell_id]["R"])
            if in_region: inside_err.append(d)
            else: outside_err.append(d)
        region_rows.append({"state": st,
                            "inside_MAE": round(float(np.mean(inside_err)), 6) if inside_err else 0,
                            "outside_MAE": round(float(np.mean(outside_err)), 6) if outside_err else 0,
                            "inside_n": len(inside_err), "outside_n": len(outside_err)})
        log(f"  {st}: inside_MAE={np.mean(inside_err):.6f} outside_MAE={np.mean(outside_err):.6f}")

with open(os.path.join(OUTPUT, "clone_region_analysis.csv"), "w", newline="") as f:
    fn = ["state","inside_MAE","outside_MAE","inside_n","outside_n"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader(); w.writerows(region_rows)

# ───── T3-T5 Gates ─────
log("\n" + "=" * 60); log("  T3-T5 Gates"); log("=" * 60)

# T3: Deformation optical invariance
t3_ok = True
for rname in eq_oc_variants:
    for st in ["stretch_2.00", "cubic_l0333"]:
        match = [r for r in inv_rows if r["representation"] == rname and r["state"] == st]
        if match and match[0]["MAE_R_REF"] > 0.05:
            t3_ok = False
            log(f"  T3 FAIL: {rname} {st} MAE_R_REF={match[0]['MAE_R_REF']:.6f} > 0.05")
T3_status = "SUPPORTED" if (t3_ok and len(eq_oc_variants) > 0) else "NOT SUPPORTED"

# T4: Physical area-dilution consistency
t4_states = ["stretch_1.50", "stretch_2.00", "biaxial_1.50", "cubic_l020", "cubic_l0333"]
t4_ok = True
for rname in eq_oc_variants:
    maes = []
    for st in t4_states:
        match = [r for r in phys_rows if r["representation"] == rname and r["state"] == st]
        if match:
            maes.append(match[0]["MAE_phys"])
    if maes and np.mean(maes) > 0.10:
        t4_ok = False
        log(f"  T4 FAIL: {rname} mean_phys_MAE={np.mean(maes):.6f} > 0.10")
T4_status = "SUPPORTED" if (t4_ok and len(eq_oc_variants) > 0) else "NOT SUPPORTED"

# T5: Area-preserving control
t5_ok = True
for rname in eq_oc_variants:
    match = [r for r in phys_rows if r["representation"] == rname and r["state"] == "shear_k040"]
    if match and match[0]["MAE_phys"] > 0.10:
        t5_ok = False
        log(f"  T5 FAIL: {rname} shear_k040 MAE={match[0]['MAE_phys']:.6f} > 0.10")
T5_status = "SUPPORTED" if (t5_ok and len(eq_oc_variants) > 0) else "NOT SUPPORTED"

# T0: Protocol lock
T0_status = "PASS" if (unit_test_pass and identity_ok) else "FAIL"

# Extra: T2 re-evaluation from gate
T2_status_final = T2_status  # computed above

log(f"  T0: {T0_status}")
log(f"  T1: {T1_status}")
log(f"  T2: {T2_status}")
log(f"  T3: {T3_status}")
log(f"  T4: {T4_status}")
log(f"  T5: {T5_status}")

# ───── Final CASE ─────
log("\n" + "=" * 60); log("  Final CASE"); log("=" * 60)

if T0_status == "FAIL":
    FINAL_CASE = "PROTOCOL-FAIL"
elif T2_status == "FAIL":
    FINAL_CASE = "TOPOLOGY-C"
elif T3_status == "NOT SUPPORTED" or T4_status == "NOT SUPPORTED":
    if T2_status == "PASS":
        FINAL_CASE = "TOPOLOGY-B"
    else:
        FINAL_CASE = "TOPOLOGY-C"
elif T0_status == "PASS" and T2_status == "PASS" and T3_status == "SUPPORTED" and T4_status == "SUPPORTED" and T5_status == "SUPPORTED":
    FINAL_CASE = "TOPOLOGY-A"
else:
    FINAL_CASE = "TOPOLOGY-C"

log(f"  Final CASE: {FINAL_CASE}")
next_test_allowed = (FINAL_CASE == "TOPOLOGY-A")  # split topology test
log(f"  Next test (split topology) allowed: {'YES' if next_test_allowed else 'NO'}")

# ───── Visualizations ─────
log("\n" + "=" * 60); log("  Visualizations"); log("=" * 60)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

viz_dir = os.path.join(OUTPUT, "figures")
os.makedirs(viz_dir, exist_ok=True)

# 1. Canonical clone alpha difference
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ["REF (cam000)", "C50_CHECKER_NAIVE diff", "C50_CHECKER_OC diff"]
data_plots = [
    can_alpha["REF"][0],
    np.abs(can_alpha["C50_CHECKER_NAIVE"][0] - can_alpha["REF"][0]),
    np.abs(can_alpha["C50_CHECKER_OC"][0] - can_alpha["REF"][0]),
]
for ax, d, t in zip(axes, data_plots, titles):
    im = ax.imshow(d, cmap="inferno" if "diff" in t else "gray", vmin=0)
    ax.set_title(t); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Canonical Clone Alpha Difference (cam000)")
fig.savefig(os.path.join(viz_dir, "canonical_clone_alpha_difference.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 2. Canonical equivalence by clone fraction
fig, ax = plt.subplots(figsize=(10, 6))
repr_names = [r for r in all_repr if r != "REF"]
x_labels = []; y_vals = []
for rn in repr_names:
    rows = repr_eq[rn]
    x_labels.append(rn)
    y_vals.append(np.mean([r["alpha_mae"] for r in rows]))
ax.bar(range(len(x_labels)), y_vals)
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel("12-camera mean alpha MAE")
ax.axhline(0.005, color="r", linestyle="--", label="Threshold 0.005")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(viz_dir, "canonical_equivalence_by_clone_fraction.png"), dpi=150)
plt.close(fig)

# 3. Stretch_2 R by representation
fig, ax = plt.subplots(figsize=(10, 6))
st = "stretch_2.00"
x_labels = []; r_medians = []
for rname in test_repr:
    cells = cell_responses.get(rname, {}).get(st, {})
    rv = [d["R"] for d in cells.values()]
    if rv:
        x_labels.append(rname)
        r_medians.append(np.median(rv))
ax.bar(range(len(x_labels)), r_medians)
ax.axhline(0.5, color="r", linestyle="--", label="Physical target (1/2)")
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel("Median R_cell (stretch_2.00)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(viz_dir, "stretch2_R_by_representation.png"), dpi=150)
plt.close(fig)

# 4. Cubic_l0333 clone invariance scatter
fig, ax = plt.subplots(figsize=(8, 8))
st = "cubic_l0333"
colors = ["black", "blue", "green", "orange", "purple"]
for rname, color in zip(test_repr, colors[:len(test_repr)]):
    cells = cell_responses.get(rname, {}).get(st, {})
    qv = [d["Q"] for d in cells.values()]
    rv = [d["R"] for d in cells.values()]
    ax.scatter(qv, rv, s=2, alpha=0.5, color=color, label=rname, marker="o")
ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.3)
ax.set_xlabel("Q_cell (1/Js)")
ax.set_ylabel("R_cell")
ax.set_title("cubic_l0333 deformation optical response")
ax.legend(markerscale=5)
fig.tight_layout()
fig.savefig(os.path.join(viz_dir, "cubic0333_clone_invariance.png"), dpi=150)
plt.close(fig)

# 5. Clone region error map
if block_rep in test_repr:
    st = "cubic_l0333"
    cells = cell_responses.get(block_rep, {}).get(st, {})
    ref_cells = cell_responses.get("REF", {}).get(st, {})
    err_map = np.full((GRID, GRID), np.nan)
    for cell in cell_defs:
        cell_id = cell["id"]
        if cell_id in cells and cell_id in ref_cells:
            err_map[cell["iu"], cell["iv"]] = abs(cells[cell_id]["R"] - ref_cells[cell_id]["R"])
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(err_map, cmap="inferno", origin="lower",
                   extent=[-1, 1, -1, 1])
    ax.set_title(f"{block_rep} vs REF: |R_diff| (cubic_l0333)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # Mark clone region
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False, edgecolor="cyan", linewidth=2))
    fig.savefig(os.path.join(viz_dir, "clone_region_error_map.png"), dpi=150)
    plt.close(fig)

# ───── Reports ─────
log("\n" + "=" * 60); log("  Reports"); log("=" * 60)

def rl(s=""): return s + "\n"
rep = []
rep.append("# Clone Topology Optical Invariance Report\n")
rep.append(rl("## A. Stage 3.3.R4 Strict Label"))
rep.append(rl("Strict LOCAL-A requires all three cubic Spearman >= 0.9. cubic_l010 Spearman=0.8892 < 0.9. Therefore: NEAR-LOCAL-A (not strict LOCAL-A)."))
rep.append(rl("## B. Why Stop Explicit Tau Correction Line"))
rep.append(rl("cubic_l0333 MAE=0.0519, Spearman=0.945, matched-Js extra error negligible. No supported nonuniform break."))
rep.append(rl("## C. Why Study Density Topology"))
rep.append(rl("Fixed-identity carrier may over-constrain representation. Real 3DGS changes topology via clone/split/prune."))
rep.append(rl("## D. Revising Densification Clone Opacity Bias"))
rep.append(rl("Original 3DGS clone preserves same opacity, biasing alpha compositing toward cloned primitives."))
rep.append(rl("## E. Tau/2 Equivalence"))
rep.append(rl("(1-o_new)^2 = 1-o_old → o_new = 1-sqrt(1-o_old) → with o=1-exp(-tau): exp(-tau_new) = exp(-tau_old/2) → tau_new = tau_old/2"))
rep.append(rl(f"## F. REF Count: {N_ref}"))
rep.append(rl("## G. Clone Variant Counts"))
for r in repr_manifest:
    rep.append(rl(f"- {r['representation']:25s}: {r['gaussian_count']:5d} ({r['clone_count']:3d} clones, {r['clone_fraction']*100:.0f}%)"))
rep.append(rl(f"## H. Clone Unit Test: {'PASS' if unit_test_pass else 'FAIL'}"))
rep.append(rl(f"## I. Naive Clone Mean Alpha MAE"))
for p in t1_pairs:
    rep.append(rl(f"- {p['pattern']:15s}: NAIVE={p['naive_mean_mae']:.6f}"))
rep.append(rl(f"## J. OC Clone Mean Alpha MAE"))
for p in t1_pairs:
    rep.append(rl(f"- {p['pattern']:15s}: OC={p['oc_mean_mae']:.6f}"))
rep.append(rl(f"## K. T1: {T1_status}"))
for p in t1_pairs:
    rep.append(rl(f"- {p['pattern']:15s}: {p['naive_gt_2x_oc']} (ratio={p['ratio']:.2f}, diff={p['diff']:.6f})"))
rep.append(rl("## L. Canonical-Equivalent OC Variants"))
for r in canon_eq_oc:
    rep.append(rl(f"- {r['representation']:25s}: {r['canonical_equivalent']} (alpha_MAE={r['mean_alpha_mae']:.6f}, cell_MAE={r['mean_cell_tau_mae']:.6f})"))
rep.append(rl(f"## M. T2: {T2_status}"))
rep.append(rl(f"## N. stretch_2.00 REF Physical MAE"))
for r in phys_rows:
    if r["representation"] == "REF" and r["state"] == "stretch_2.00":
        rep.append(rl(f"- MAE={r['MAE_phys']:.6f}"))
rep.append(rl("## O-P. stretch_2.00 OC Physical MAE / MAE_R_REF"))
for rname in eq_oc_variants:
    pm = [r for r in phys_rows if r["representation"] == rname and r["state"] == "stretch_2.00"]
    im = [r for r in inv_rows if r["representation"] == rname and r["state"] == "stretch_2.00"]
    if pm: rep.append(rl(f"- {rname:25s} phys_MAE={pm[0]['MAE_phys']:.6f}"))
    if im: rep.append(rl(f"  {'':25s} MAE_R_REF={im[0]['MAE_R_REF']:.6f}"))
rep.append(rl(f"## Q. cubic_l0333 REF Physical MAE"))
for r in phys_rows:
    if r["representation"] == "REF" and r["state"] == "cubic_l0333":
        rep.append(rl(f"- MAE={r['MAE_phys']:.6f}"))
rep.append(rl("## R-S. cubic_l0333 OC Physical MAE / MAE_R_REF"))
for rname in eq_oc_variants:
    pm = [r for r in phys_rows if r["representation"] == rname and r["state"] == "cubic_l0333"]
    im = [r for r in inv_rows if r["representation"] == rname and r["state"] == "cubic_l0333"]
    if pm: rep.append(rl(f"- {rname:25s} phys_MAE={pm[0]['MAE_phys']:.6f}"))
    if im: rep.append(rl(f"  {'':25s} MAE_R_REF={im[0]['MAE_R_REF']:.6f}"))
rep.append(rl("## T. Block Inside/Outside Error"))
for r in region_rows:
    rep.append(rl(f"- {r['state']:15s}: inside={r['inside_MAE']:.6f} outside={r['outside_MAE']:.6f}"))
shear_ctrl = [r for r in phys_rows if r["state"] == "shear_k040" and r["representation"] == "REF"]
if shear_ctrl: rep.append(rl(f"## U. Shear k040 Control: REF phys_MAE={shear_ctrl[0]['MAE_phys']:.6f}"))
rep.append(rl(f"## V. T0: {T0_status}"))
rep.append(rl(f"## W. T1: {T1_status}"))
rep.append(rl(f"## X. T2: {T2_status}"))
rep.append(rl(f"## Y. T3: {T3_status}"))
rep.append(rl(f"## Z. T4: {T4_status}"))
rep.append(rl(f"## AA. T5: {T5_status}"))
rep.append(rl(f"## AB. Final CASE: {FINAL_CASE}"))
rep.append(rl("## AC. Current Scientific Question"))
rep.append(rl("Does opacity-corrected clone preserve deformation-induced area-dilution optical consistency?"))
rep.append(rl(f"## AD. Next Test Allowed"))
rep.append(rl(f"{'YES: split topology test' if next_test_allowed else 'NO: protocol or equivalence must be resolved first'}"))

with open(os.path.join(OUTPUT, "clone_topology_optical_invariance_report.md"), "w") as f:
    f.writelines(rep)

# Summary
summary = f"""# Stage 3.4A Summary: Clone Topology Optical Invariance

## Final CASE: {FINAL_CASE}
## T0: {T0_status}
## T1: {T1_status}
## T2: {T2_status}
## T3: {T3_status}
## T4: {T4_status}
## T5: {T5_status}
## REF Count: {N_ref}
## OC Equivalent Variants: {len(eq_oc_variants)}
## Next Split Test Allowed: {'YES' if next_test_allowed else 'NO'}
"""

with open(os.path.join(OUTPUT, "stage3_4A_summary.md"), "w") as f:
    f.write(summary)

with open(os.path.join(OUTPUT, "stage3_4A_log.txt"), "w") as f:
    f.write("\n".join(log_lines))

# ═══════════════════════════════════════════════════════════════
# Terminal summary
# ═══════════════════════════════════════════════════════════════
log("\n" + "=" * 60); log("  TERMINAL SUMMARY"); log("=" * 60)

out_lines = [
    f"  Strict Stage3.3.R4 label: NEAR-LOCAL-A (l010 Spearman=0.889 < 0.9)",
    f"  Explicit tau correction line stopped: YES",
    f"  REF count: {N_ref}",
]
for r in repr_manifest:
    out_lines.append(f"  {r['representation']:25s}: N={r['gaussian_count']}")
out_lines.append(f"  Clone unit test: {'PASS' if unit_test_pass else 'FAIL'}")
for p in t1_pairs:
    out_lines.append(f"  Naive alpha MAE ({p['pattern']:15s}): {p['naive_mean_mae']:.6f}")
for p in t1_pairs:
    out_lines.append(f"  OC alpha MAE ({p['pattern']:15s}): {p['oc_mean_mae']:.6f}")
out_lines.append(f"  T1: {T1_status}")
out_lines.append(f"  Canonical-equivalent OC: {eq_oc_variants}")
out_lines.append(f"  T2: {T2_status}")
for r in phys_rows:
    if r["representation"] == "REF":
        out_lines.append(f"  {r['state']:15s} REF phys_MAE={r['MAE_phys']:.6f}")
for rname in eq_oc_variants:
    for st in ["stretch_2.00", "cubic_l0333"]:
        pm = [r for r in phys_rows if r["representation"] == rname and r["state"] == st]
        im = [r for r in inv_rows if r["representation"] == rname and r["state"] == st]
        if pm: out_lines.append(f"  {rname:25s} {st:15s} phys_MAE={pm[0]['MAE_phys']:.6f}")
        if im: out_lines.append(f"  {'':25s} {'':15s} MAE_R_REF={im[0]['MAE_R_REF']:.6f}")

for r in region_rows:
    out_lines.append(f"  BLOCK inside/outside {r['state']}: in={r['inside_MAE']:.6f} out={r['outside_MAE']:.6f}")

shear_ref_ctrl = [r for r in phys_rows if r["state"] == "shear_k040" and r["representation"] == "REF"]
if shear_ref_ctrl: out_lines.append(f"  shear_k040 REF phys_MAE={shear_ref_ctrl[0]['MAE_phys']:.6f}")
for rname in eq_oc_variants:
    sc = [r for r in phys_rows if r["state"] == "shear_k040" and r["representation"] == rname]
    if sc: out_lines.append(f"  shear_k040 {rname} phys_MAE={sc[0]['MAE_phys']:.6f}")

out_lines += [
    f"  T0: {T0_status}", f"  T1: {T1_status}", f"  T2: {T2_status}",
    f"  T3: {T3_status}", f"  T4: {T4_status}", f"  T5: {T5_status}",
    f"  Final CASE: {FINAL_CASE}",
    f"  Next test (split) allowed: {'YES' if next_test_allowed else 'NO'}",
    f"  Report: {OUTPUT}/clone_topology_optical_invariance_report.md",
    f"  Summary: {OUTPUT}/stage3_4A_summary.md",
]
for l in out_lines:
    print(l)
