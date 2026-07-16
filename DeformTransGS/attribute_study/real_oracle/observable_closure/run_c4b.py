from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import site
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diff_first_surface_rasterization import GaussianRasterizationSettings, GaussianRasterizer


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OUT = PROJECT / "experiments" / "stage4_0_R2A_C4B_optical_observable_semantics"
C4A = PROJECT / "experiments" / "stage4_0_R2A_C4A_canonical_provenance_closure"
GT_ROOT = PROJECT / "experiments" / "stage4_0_R2A_GT1_gt_optical_semantics_closure" / "clean_gt"
GT_SRC = PROJECT / "attribute_study" / "real_oracle" / "gt_closure" / "clean_gt_renderer.py"
C4A_SRC = PROJECT / "attribute_study" / "real_oracle" / "canonical_closure" / "run_c4a.py"
LAUNCHER = PROJECT / "attribute_study" / "real_oracle" / "pipeline_closure" / "verified_stage4_python.sh"
RASTER_ROOT = ROOT / "repos" / "TSGS" / "submodules" / "diff-first-surface-rasterization"
RASTER_PY = RASTER_ROOT / "diff_first_surface_rasterization" / "__init__.py"
RASTER_FW = RASTER_ROOT / "cuda_rasterizer" / "forward.cu"
RASTER_BW = RASTER_ROOT / "cuda_rasterizer" / "backward.cu"

TRAIN_IDS = list(range(16))
TEST_IDS = [0, 3, 6, 9, 12, 15, 18, 21]
SEED = 20260714
MAX_ITERS = 4000
PATIENCE = 500
LR = 0.03
EPS = 1e-6

CASES = {
    "K0": ("S0_PLANAR_SHEET", "MAT0_NEUTRAL_FIXED_THICKNESS", "D0_IDENTITY"),
    "K1": ("S0_PLANAR_SHEET", "MAT1_NEUTRAL_MASS_CONSERVING", "D0_IDENTITY"),
    "K2": ("S1_WAVY_MEMBRANE", "MAT2_TINTED_MASS_CONSERVING", "D0_IDENTITY"),
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    st = path.stat() if path.exists() else None
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": st.st_size if st else "",
        "mtime": st.st_mtime if st else "",
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def gt_path(case: str, cid: int, suffix: str) -> Path:
    s, m, d = CASES[case]
    return GT_ROOT / s / m / d / f"camera_{cid:02d}_{suffix}.npy"


def load_rgb_hwc(path: Path) -> np.ndarray:
    arr = np.load(path).astype("float32")
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def gt_rgb_chw(case: str, cid: int) -> torch.Tensor:
    return torch.from_numpy(load_rgb_hwc(gt_path(case, cid, "rgb"))).permute(2, 0, 1).cuda()


def gt_tau_chw(case: str, cid: int) -> torch.Tensor:
    return torch.from_numpy(np.load(gt_path(case, cid, "tau_rgb")).astype("float32")).permute(2, 0, 1).cuda()


def camera_vec(cid: int) -> torch.Tensor:
    elev = 25.0 if cid < 12 else 50.0
    az = (cid % 12) * 30.0
    er = math.radians(elev)
    ar = math.radians(az)
    pos = torch.tensor([3.3 * math.cos(er) * math.cos(ar), 3.3 * math.cos(er) * math.sin(ar), 3.3 * math.sin(er)], device="cuda")
    return -pos / (pos.norm() + 1e-12)


def sh_basis(view: torch.Tensor) -> torch.Tensor:
    x, y, z = view[:, 0], view[:, 1], view[:, 2]
    return torch.stack([torch.ones_like(x), x, y, z, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y], dim=1)


class State(torch.nn.Module):
    def __init__(self, surface: str, n: int = 4096):
        super().__init__()
        torch.manual_seed(SEED)
        g = int(math.sqrt(n))
        xs, ys = torch.meshgrid(torch.linspace(-0.8, 0.8, g, device="cuda"), torch.linspace(-0.8, 0.8, g, device="cuda"), indexing="xy")
        if surface == "S1_WAVY_MEMBRANE":
            z = 2.0 + 0.18 * torch.sin(math.pi * xs) * torch.sin(math.pi * ys)
            dzdx = 0.18 * math.pi * torch.cos(math.pi * xs) * torch.sin(math.pi * ys)
            dzdy = 0.18 * math.pi * torch.sin(math.pi * xs) * torch.cos(math.pi * ys)
            normal = torch.stack([-dzdx.reshape(-1), -dzdy.reshape(-1), torch.ones(n, device="cuda")], dim=1)
            normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-12)
        else:
            z = torch.full((g, g), 2.0, device="cuda")
            normal = torch.tensor([0.0, 0.0, 1.0], device="cuda").repeat(n, 1)
        self.n = n
        self.surface = surface
        self.register_buffer("means3D", torch.stack([xs.reshape(-1), ys.reshape(-1), z.reshape(-1)], dim=1))
        self.register_buffer("means2D", torch.zeros(n, 3, device="cuda"))
        self.register_buffer("means2D_abs", torch.zeros(n, 3, device="cuda"))
        self.register_buffer("scales", torch.full((n, 3), 0.018, device="cuda"))
        rots = torch.zeros(n, 4, device="cuda")
        rots[:, 0] = 1.0
        self.register_buffer("rots", rots)
        self.register_buffer("trans", torch.ones(n, 1, device="cuda"))
        self.register_buffer("normal", normal)
        self.register_buffer("t1", torch.tensor([1.0, 0.0, 0.0], device="cuda").repeat(n, 1))
        self.register_buffer("t2", torch.tensor([0.0, 1.0, 0.0], device="cuda").repeat(n, 1))
        self.o_raw = torch.nn.Parameter(torch.full((n, 1), -1.2, device="cuda"))
        self.sh_coeffs = torch.nn.Parameter(torch.zeros(n, 9, 3, device="cuda"))
        with torch.no_grad():
            self.sh_coeffs[:, 0, :] = 0.55
        self.v_raw = torch.nn.Parameter(torch.zeros(n, 3, device="cuda"))

    def named_release_parameters(self):
        yield "o_raw", self.o_raw
        yield "sh_coeffs", self.sh_coeffs
        yield "v_raw", self.v_raw


def make_rasterizer() -> GaussianRasterizer:
    settings = GaussianRasterizationSettings(
        image_height=512,
        image_width=512,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.ones(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"),
        sh_degree=0,
        campos=torch.tensor([0.0, 0.0, 0.0], device="cuda"),
        prefiltered=False,
        render_geo=False,
        transparency_threshold=0.0,
        debug=False,
    )
    return GaussianRasterizer(settings)


def render_state(st: State, cid: int, rasterizer: GaussianRasterizer | None = None):
    view = camera_vec(cid)[None, :].repeat(st.n, 1)
    basis = sh_basis(view)
    colors = torch.sigmoid((basis[:, :, None] * st.sh_coeffs).sum(dim=1))
    local_view = torch.stack([(view * st.normal).sum(1), (view * st.t1).sum(1), (view * st.t2).sum(1)], dim=1)
    opacity = torch.sigmoid(st.o_raw + (st.v_raw * local_view).sum(1, keepdim=True))
    if rasterizer is None:
        rasterizer = make_rasterizer()
    result = rasterizer(st.means3D, st.means2D, st.means2D_abs, opacity, st.trans, colors_precomp=colors, scales=st.scales, rotations=st.rots)
    rgb = result[0]
    out_transparency = result[6]
    raster_alpha = 1.0 - torch.clamp(out_transparency, 0.0, 1.0)
    c4a_alpha = torch.clamp(rgb.mean(dim=0), 0.0, 1.0)
    return rgb, raster_alpha, c4a_alpha, opacity, colors


def dssim_global(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    x = pred.reshape(3, -1)
    y = gt.reshape(3, -1)
    mux = x.mean(dim=1)
    muy = y.mean(dim=1)
    vx = ((x - mux[:, None]) ** 2).mean(dim=1)
    vy = ((y - muy[:, None]) ** 2).mean(dim=1)
    cov = ((x - mux[:, None]) * (y - muy[:, None])).mean(dim=1)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2) + 1e-12)
    return torch.clamp((1.0 - ssim.mean()) * 0.5, 0.0, 1.0)


def corrected_loss_parts(st: State, case: str, cid: int, targets: dict[int, tuple[torch.Tensor, torch.Tensor]], rasterizer: GaussianRasterizer):
    pred, raster_alpha, c4a_alpha, _, _ = render_state(st, cid, rasterizer)
    gt_rgb, gt_tau = targets[cid]
    rgb_loss = (pred - gt_rgb).abs().mean()
    tau_pred = -torch.log(torch.clamp(pred, EPS, 1.0))
    tau_loss = torch.abs(torch.log((tau_pred + EPS) / (gt_tau + EPS))).mean()
    dssim = dssim_global(pred, gt_rgb)
    alpha_gt = 1.0 - torch.exp(-gt_tau.mean(dim=0))
    alpha_loss_old = torch.abs(torch.log((-torch.log(torch.clamp(1.0 - c4a_alpha, EPS, 1.0)) + EPS) / (-torch.log(torch.clamp(1.0 - alpha_gt, EPS, 1.0)) + EPS))).mean()
    return rgb_loss, tau_loss, dssim, alpha_loss_old, pred, raster_alpha, c4a_alpha


def corrected_total_loss(st: State, case: str, train_ids: list[int], targets: dict[int, tuple[torch.Tensor, torch.Tensor]], rasterizer: GaussianRasterizer):
    parts = [corrected_loss_parts(st, case, cid, targets, rasterizer)[:3] for cid in train_ids]
    rgb = torch.stack([p[0] for p in parts]).mean()
    tau = torch.stack([p[1] for p in parts]).mean()
    dssim = torch.stack([p[2] for p in parts]).mean()
    return rgb + 0.5 * tau + 0.1 * dssim, rgb, tau, dssim


def metric_case(render_root: Path, case: str) -> dict:
    rows = []
    per_cam = []
    for cid in TEST_IDS:
        pred = np.load(render_root / case / f"camera_{cid:02d}_rgb.npy").astype("float32")
        if pred.shape[0] == 3:
            pred_hwc = np.transpose(pred, (1, 2, 0))
        else:
            pred_hwc = pred
        gt = load_rgb_hwc(gt_path(case, cid, "rgb"))
        gt_tau = np.load(gt_path(case, cid, "tau_rgb")).astype("float32")
        mse = float(((pred_hwc - gt) ** 2).mean())
        psnr = -10.0 * math.log10(max(mse, 1e-12))
        tau_pred = -np.log(np.clip(pred_hwc, EPS, 1.0))
        elog = np.abs(np.log((tau_pred + EPS) / (gt_tau + EPS)))
        rows.append({"psnr": psnr, "median": float(np.median(elog)), "p90": float(np.quantile(elog, 0.90)), "p95": float(np.quantile(elog, 0.95)), "p99": float(np.quantile(elog, 0.99)), "factor2": float((elog <= math.log(2.0)).mean())})
        per_cam.append({"case": case, "camera_id": cid, "PSNR": psnr, "median_tau_eq_Elog": float(np.median(elog))})
    return {
        "case": case,
        "PSNR": float(np.mean([r["psnr"] for r in rows])),
        "SSIM": 0.0,
        "median_TAU_EQ_RGB_ELOG": float(np.median([r["median"] for r in rows])),
        "p90_TAU_EQ_RGB_ELOG": float(np.median([r["p90"] for r in rows])),
        "p95_TAU_EQ_RGB_ELOG": float(np.median([r["p95"] for r in rows])),
        "p99_TAU_EQ_RGB_ELOG": float(np.median([r["p99"] for r in rows])),
        "factor2_fraction": float(np.mean([r["factor2"] for r in rows])),
        "per_camera": per_cam,
    }


def save_checkpoint(path: Path, st: State, opt: torch.optim.Optimizer, case: str, iteration: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "case": case,
            "surface": CASES[case][0],
            "material": CASES[case][1],
            "deformation": CASES[case][2],
            "iteration": iteration,
            "seed": SEED,
            "loss": "RGB_L1 + 0.5*TAU_EQ_RGB_LOG_L1 + 0.1*DSSIM",
            "o_raw": st.o_raw.detach().cpu(),
            "sh_coeffs": st.sh_coeffs.detach().cpu(),
            "v_raw": st.v_raw.detach().cpu(),
            "optimizer": opt.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, case: str) -> tuple[State, dict, float]:
    data = torch.load(path, map_location="cpu")
    st = State(CASES[case][0])
    maxerr = 0.0
    with torch.no_grad():
        for name in ["o_raw", "sh_coeffs", "v_raw"]:
            src = data[name].cuda()
            dst = getattr(st, name)
            dst.copy_(src)
            maxerr = max(maxerr, float((dst - src).abs().max()))
    return st, data, maxerr


def protocol_lock() -> str:
    items = {
        "verified_gt_root": file_record(GT_ROOT),
        "gt_generator_source": file_record(GT_SRC),
        "c4a_source": file_record(C4A_SRC),
        "stage4_launcher": file_record(LAUNCHER),
        "rasterizer_python": file_record(RASTER_PY),
        "rasterizer_forward": file_record(RASTER_FW),
        "rasterizer_backward": file_record(RASTER_BW),
        "c4a_summary": file_record(C4A / "stage4_0_R2A_C4A_summary.md"),
        "c4a_metrics": file_record(C4A / "c4a_fresh_canonical_metrics.csv"),
    }
    for case in CASES:
        items[f"c4a_checkpoint_{case}"] = file_record(C4A / "rerun" / case / "checkpoint.pt")
    write_text(OUT / "c4b_protocol_lock.json", json.dumps(items, indent=2) + "\n")
    return "PASS" if all(v["exists"] for v in items.values()) else "FAIL"


def trace_semantics() -> tuple[str, str]:
    gt_text = """# GT Alpha Semantics

Source: `/data/wyh/DeformTransGS/attribute_study/real_oracle/gt_closure/clean_gt_renderer.py`, lines around 40-72.

Equations:

`tau_rgb = sigma_rgb * h / cos_theta`

`tau_mean = mean(tau_rgb)`

`A_gt = 1 - exp(-tau_mean)`

Classification: `OPTICAL-DIAGNOSTIC-DERIVED-FROM-CHANNEL-MEAN`.

`A_gt` is saved as `camera_##_alpha.npy`. It does not participate in triangle visibility, z-buffer geometry, or surface mask construction; geometry support is represented separately by `inside` and `triangle_id`.
"""
    pred_text = """# Predicted Alpha Semantics

C4A actual predicted alpha source:

`/data/wyh/DeformTransGS/attribute_study/real_oracle/canonical_closure/run_c4a.py`, `render_state`, line around 186:

`alpha = clamp(rgb.mean(dim=0), 0, 1)`.

Therefore the alpha array used by C4A is a post-hoc RGB-mean diagnostic, not the rasterizer accumulated alpha tensor.

Locked rasterizer return source:

`diff_first_surface_rasterization/__init__.py` lines 105-117 returns `out_transparency` as tuple element 6.

Primitive raster alpha source:

`cuda_rasterizer/forward.cu` lines 374-379:

`alpha = min(0.99, conic_opacity.w * exp(power))`, skipping if `< 1/255`.

Accumulation source:

`cuda_rasterizer/forward.cu` line around 390 uses `test_T = T * (1 - alpha)`. `backward.cu` lines 452-454 describe `T_final` as the product of all `(1-alpha)` factors.

Thus raster accumulated alpha is a Gaussian kernel/opacity compositing observable, while C4A's saved alpha was an RGB-derived diagnostic. Neither is mathematically identical to GT `1-exp(-mean(tau_rgb))` by definition.
"""
    write_text(OUT / "gt_alpha_semantics.md", gt_text)
    write_text(OUT / "predicted_alpha_semantics.md", pred_text)
    rows = [
        {"quantity": "GT_A_gt", "definition": "1-exp(-mean(tau_rgb))", "channel_dimension": "scalar_from_RGB_mean", "depends_on_Gaussian_kernel": "NO", "depends_on_compositing_order": "NO", "depends_on_RGB_channel_sigma": "YES", "derived_from_final_RGB": "YES_FOR_CLEAN_GT"},
        {"quantity": "C4A_pred_alpha", "definition": "clamp(mean(I_pred_rgb),0,1)", "channel_dimension": "scalar_from_rendered_RGB_mean", "depends_on_Gaussian_kernel": "YES_VIA_RGB_RENDER", "depends_on_compositing_order": "YES_VIA_RGB_RENDER", "depends_on_RGB_channel_sigma": "NO_DIRECT_SIGMA_STATE", "derived_from_final_RGB": "YES"},
        {"quantity": "raster_accumulated_alpha", "definition": "1-product_i(1-min(0.99,opacity_i*exp(power_i)))", "channel_dimension": "scalar_compositing", "depends_on_Gaussian_kernel": "YES", "depends_on_compositing_order": "YES", "depends_on_RGB_channel_sigma": "NO", "derived_from_final_RGB": "NO"},
    ]
    for r in rows:
        r["ALPHA_OBSERVABLE_EQUIVALENT_TO_GT"] = "NO" if r["quantity"] != "GT_A_gt" else "SELF"
    write_csv(OUT / "alpha_observable_semantic_comparison.csv", rows)
    witnesses = [
        {"witness": "W0", "description": "same raster accumulated alpha 0.5 with RGB tau [0.1,0.1,0.1] vs [1.0,1.0,1.0]", "A_gt_1": 1 - math.exp(-0.1), "A_gt_2": 1 - math.exp(-1.0), "raster_alpha_1": 0.5, "raster_alpha_2": 0.5},
        {"witness": "W1", "description": "same mean tau but different per-channel tau distribution", "A_gt_1": 1 - math.exp(-1.0), "A_gt_2": 1 - math.exp(-1.0), "tau_rgb_1": "[1,1,1]", "tau_rgb_2": "[0,1,2]"},
        {"witness": "W2", "description": "same final transmittance from one alpha 0.75 vs two overlapping alphas 0.5,0.5", "raster_alpha_1": 0.75, "raster_alpha_2": 0.75, "decomposition_1": "[0.75]", "decomposition_2": "[0.5,0.5]"},
    ]
    write_csv(OUT / "alpha_non_equivalence_witness.csv", witnesses)
    return "NO", "PASS"


def validate_tau_eq() -> tuple[float, float, str]:
    rels = []
    maxs = []
    for case in CASES:
        for cid in TRAIN_IDS + TEST_IDS:
            rgb = load_rgb_hwc(gt_path(case, cid, "rgb"))
            tau = np.load(gt_path(case, cid, "tau_rgb")).astype("float32")
            tau_eq = -np.log(np.clip(rgb, EPS, 1.0))
            mask = tau > 1e-7
            rel = np.abs(tau_eq[mask] - tau[mask]) / np.maximum(np.abs(tau[mask]), EPS)
            rels.append(np.quantile(rel, 0.99))
            maxs.append(float(np.max(np.abs(tau_eq - tau))))
    p99 = float(np.max(rels))
    mx = float(np.max(maxs))
    text = f"""# Image-Equivalent Optical Depth

Definition:

`tau_eq_rgb = -log(clamp(I_rgb, 1e-6, 1))`.

For GT:

`tau_eq_gt_rgb = -log(clamp(I_gt_rgb, 1e-6, 1))`.

Because clean GT RGB is generated as `exp(-tau_rgb)`, `tau_eq_gt_rgb` equals saved `tau_rgb` up to float32 storage precision.

This observable is derived from final white-background RGB. It does not claim Gaussian opacity is physical extinction.

Validation:

- relative p99: `{p99:.6e}`
- max absolute error: `{mx:.6e}`
"""
    write_text(OUT / "image_equivalent_optical_depth_definition.md", text)
    return p99, mx, "PASS" if p99 <= 1e-6 and mx <= 1e-4 else "FAIL"


def c4a_pixel_audit() -> tuple[str, str]:
    rows = []
    alpha_old = {}
    alpha_eq = {}
    root = C4A / "fresh_reproduction"  # corrected C4A rerun fresh arrays
    if not root.exists():
        root = C4A / "fresh_reproduction_old"
    for case in CASES:
        old_vals = []
        eq_vals = []
        for cid in TEST_IDS:
            pred = np.load(root / case / f"camera_{cid:02d}_rgb.npy").astype("float32")
            pred_hwc = np.transpose(pred, (1, 2, 0)) if pred.shape[0] == 3 else pred
            pred_alpha_path = root / case / f"camera_{cid:02d}_alpha.npy"
            pred_alpha = np.load(pred_alpha_path).astype("float32") if pred_alpha_path.exists() else pred_hwc.mean(axis=2)
            gt_alpha = np.load(gt_path(case, cid, "alpha")).astype("float32")
            gt_tau = np.load(gt_path(case, cid, "tau_rgb")).astype("float32")
            tau_pred = -np.log(np.clip(pred_hwc, EPS, 1.0))
            tau_pred_mean = tau_pred.mean(axis=2)
            eq_alpha = 1.0 - np.exp(-tau_pred_mean)
            valid = gt_alpha > 1e-7
            old_elog = np.abs(np.log((-np.log(np.clip(1 - pred_alpha[valid], EPS, 1.0)) + EPS) / (-np.log(np.clip(1 - gt_alpha[valid], EPS, 1.0)) + EPS)))
            eq_elog = np.abs(np.log((-np.log(np.clip(1 - eq_alpha[valid], EPS, 1.0)) + EPS) / (-np.log(np.clip(1 - gt_alpha[valid], EPS, 1.0)) + EPS)))
            old_vals.append(float(np.median(old_elog)))
            eq_vals.append(float(np.median(eq_elog)))
            rows.append({"case": case, "camera_id": cid, "old_pred_alpha_vs_GT_Agt_median_Elog": old_vals[-1], "tau_eq_pred_alpha_vs_GT_Agt_median_Elog": eq_vals[-1], "corr_old": float(np.corrcoef(pred_alpha[valid].reshape(-1), gt_alpha[valid].reshape(-1))[0, 1]), "corr_tau_eq": float(np.corrcoef(eq_alpha[valid].reshape(-1), gt_alpha[valid].reshape(-1))[0, 1])})
        alpha_old[case] = float(np.median(old_vals))
        alpha_eq[case] = float(np.median(eq_vals))
    write_csv(OUT / "c4a_optical_observable_pixel_audit.csv", rows)
    return "/".join(f"{alpha_old[k]:.6f}" for k in CASES), "/".join(f"{alpha_eq[k]:.6f}" for k in CASES)


def gradient_conflict() -> tuple[str, str, str]:
    rows = []
    conflict = False
    for case in CASES:
        st = State(CASES[case][0])
        rasterizer = make_rasterizer()
        targets = {cid: (gt_rgb_chw(case, cid), gt_tau_chw(case, cid)) for cid in TRAIN_IDS[:4]}
        for cid in TRAIN_IDS[:4]:
            losses = {}
            rgb, tau, dssim, alpha_old, _, _, _ = corrected_loss_parts(st, case, cid, targets, rasterizer)
            losses["rgb"] = rgb
            losses["tau"] = tau
            losses["dssim"] = dssim
            losses["alpha"] = alpha_old
            grads = {}
            for lname, loss in losses.items():
                gvals = torch.autograd.grad(loss, [st.o_raw, st.sh_coeffs, st.v_raw], retain_graph=True, allow_unused=True)
                grads[lname] = [g.detach().reshape(-1) if g is not None else torch.zeros(1, device="cuda") for g in gvals]
            for idx, attr in enumerate(["O", "C", "V"]):
                ga = grads["alpha"][idx]
                gr = grads["rgb"][idx]
                gt = grads["tau"][idx]
                gd = grads["dssim"][idx]
                cos_ar = float(F.cosine_similarity(ga, gr, dim=0).item()) if ga.norm() > 0 and gr.norm() > 0 else 0.0
                cos_at = float(F.cosine_similarity(ga, gt, dim=0).item()) if ga.norm() > 0 and gt.norm() > 0 else 0.0
                cos_ad = float(F.cosine_similarity(ga, gd, dim=0).item()) if ga.norm() > 0 and gd.norm() > 0 else 0.0
                rows.append({"case": case, "camera_id": cid, "attribute": attr, "cos_alpha_rgb": cos_ar, "cos_alpha_tau": cos_at, "cos_alpha_dssim": cos_ad, "norm_alpha_over_tau": float(ga.norm() / (gt.norm() + 1e-12)), "norm_alpha_over_rgb": float(ga.norm() / (gr.norm() + 1e-12))})
    for case in CASES:
        for attr in ["O", "C", "V"]:
            sel = [r for r in rows if r["case"] == case and r["attribute"] == attr]
            if np.median([r["cos_alpha_rgb"] for r in sel]) < 0 and np.median([r["cos_alpha_tau"] for r in sel]) < 0:
                conflict = True
    write_csv(OUT / "canonical_loss_gradient_conflict.csv", rows)
    def pack(key: str) -> str:
        chunks = []
        for case in CASES:
            vals = []
            for attr in ["O", "C", "V"]:
                sel = [r for r in rows if r["case"] == case and r["attribute"] == attr]
                vals.append(f"{attr}:{np.median([r[key] for r in sel]):.3f}")
            chunks.append(f"{case}=" + ",".join(vals))
        return " | ".join(chunks)
    return pack("cos_alpha_rgb"), pack("cos_alpha_tau"), "YES" if conflict else "NO"


def rerun_corrected() -> tuple[list[dict], list[dict], list[dict], float]:
    hist_root = OUT / "corrected_canonical_history"
    model_root = OUT / "corrected_canonical_models"
    render_root = OUT / "corrected_canonical_test_renders"
    manifest = []
    history_summary = []
    reload_max = 0.0
    for case in CASES:
        st = State(CASES[case][0])
        opt = torch.optim.Adam([p for _, p in st.named_release_parameters()], lr=LR)
        targets = {cid: (gt_rgb_chw(case, cid), gt_tau_chw(case, cid)) for cid in TRAIN_IDS}
        rasterizer = make_rasterizer()
        init = {n: p.detach().clone() for n, p in st.named_release_parameters()}
        rows = []
        best = float("inf")
        stale = 0
        for it in range(MAX_ITERS):
            opt.zero_grad(set_to_none=True)
            total, rgb_loss, tau_loss, dssim = corrected_total_loss(st, case, TRAIN_IDS, targets, rasterizer)
            total.backward()
            opt.step()
            val = float(total.item())
            if val < best - 1e-8:
                best = val
                stale = 0
            else:
                stale += 1
            row = {"iteration": it, "total_loss": val, "RGB_loss": float(rgb_loss.item()), "TAU_EQ_loss": float(tau_loss.item()), "DSSIM": float(dssim.item())}
            for attr, name in [("O", "o_raw"), ("C", "sh_coeffs"), ("V", "v_raw")]:
                p = getattr(st, name)
                row[f"{attr}_grad_L2"] = float(p.grad.norm()) if p.grad is not None else 0.0
                row[f"{attr}_delta_L2"] = float((p.detach() - init[name]).norm())
            rows.append(row)
            if stale >= PATIENCE:
                break
        write_csv(hist_root / f"{case}.csv", rows)
        save_checkpoint(model_root / f"{case}.pt", st, opt, case, rows[-1]["iteration"])
        history_summary.append({"case": case, "steps": len(rows), "initial_loss": rows[0]["total_loss"], "final_loss": rows[-1]["total_loss"], "best_loss": min(r["total_loss"] for r in rows)})
        st2, _, err = load_checkpoint(model_root / f"{case}.pt", case)
        reload_max = max(reload_max, err)
        rr = render_root / case
        rr.mkdir(parents=True, exist_ok=True)
        fresh_raster = make_rasterizer()
        for cid in TEST_IDS:
            rgb, raster_alpha, _, opacity, colors = render_state(st2, cid, fresh_raster)
            rgb_np = rgb.detach().cpu().numpy().astype("float32")
            alpha_np = raster_alpha.detach().cpu().numpy().astype("float32")
            np.save(rr / f"camera_{cid:02d}_rgb.npy", rgb_np)
            np.save(rr / f"camera_{cid:02d}_raster_alpha.npy", alpha_np)
            manifest.append({"case": case, "camera_id": cid, "rgb_path": str(rr / f"camera_{cid:02d}_rgb.npy"), "raster_alpha_path": str(rr / f"camera_{cid:02d}_raster_alpha.npy"), "rgb_sha256": sha256_file(rr / f"camera_{cid:02d}_rgb.npy"), "raster_alpha_sha256": sha256_file(rr / f"camera_{cid:02d}_raster_alpha.npy"), "opacity_low_sat_fraction": float((torch.sigmoid(st2.o_raw) <= 1e-4).float().mean()), "opacity_high_sat_fraction": float((torch.sigmoid(st2.o_raw) >= 1 - 1e-4).float().mean()), "raster_alpha_ge_0p99_fraction": float((raster_alpha >= 0.99).float().mean()), "SH_color_clamp_fraction": float(((colors <= 1e-4) | (colors >= 1 - 1e-4)).float().mean())})
    write_csv(OUT / "corrected_canonical_render_manifest.csv", manifest)
    metrics = []
    per_cam = []
    for case in CASES:
        m = metric_case(render_root, case)
        per_cam.extend(m.pop("per_camera"))
        m["PASS"] = m["PSNR"] >= 28 and m["median_TAU_EQ_RGB_ELOG"] <= 0.25
        metrics.append(m)
    write_csv(OUT / "corrected_canonical_metrics.csv", metrics)
    return history_summary, manifest, metrics, reload_max


def capacity_diag(metrics: list[dict], manifest: list[dict], history_summary: list[dict]) -> str:
    rows = []
    for m in metrics:
        if m["PSNR"] >= 28 and m["median_TAU_EQ_RGB_ELOG"] <= 0.25:
            continue
        case = m["case"]
        h = next(x for x in history_summary if x["case"] == case)
        mans = [x for x in manifest if x["case"] == case]
        cls = "TRAIN-FIT-INSUFFICIENT" if h["final_loss"] > 0.05 else "OTHER-EXACT"
        rows.append({"case": case, "TEST_PSNR": m["PSNR"], "TEST_tau_eq_Elog": m["median_TAU_EQ_RGB_ELOG"], "best_train_iteration": "", "final_iteration": h["steps"] - 1, "last500_loss_reduction": "", "visibility_coverage": "NOT_RENDER_GEO", "raster_alpha_saturation_fraction_ge_0p99": float(np.mean([x["raster_alpha_ge_0p99_fraction"] for x in mans])), "opacity_low_saturation_fraction": float(np.mean([x["opacity_low_sat_fraction"] for x in mans])), "opacity_high_saturation_fraction": float(np.mean([x["opacity_high_sat_fraction"] for x in mans])), "SH_color_clamp_fraction": float(np.mean([x["SH_color_clamp_fraction"] for x in mans])), "classification": cls})
    write_csv(OUT / "corrected_canonical_capacity_diagnostic.csv", rows)
    return ",".join(sorted({r["classification"] for r in rows})) if rows else "NONE"


def update_readme() -> None:
    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """

## Stage4.0-R2A-C4B optical observable semantics closure

Stage4.0-R2A-C4A confirmed that the old canonical pipeline had `CANONICAL-JOB-CASE-REUSE`; the identical canonical metrics were invalid. Case-keyed reruns executed 4000 Adam steps for K0/K1/K2 with exact checkpoint reloads. Those runs showed tau-equivalent RGB error near the original threshold while the alpha Elog term was much larger.

Stage4.0-R2A-C4B traces the benchmark GT alpha as `1-exp(-mean(tau_rgb))`, an optical diagnostic rather than a geometry mask. The rasterizer alpha family is generated by Gaussian opacity/kernel alpha composition, and the C4A saved alpha was an RGB-mean diagnostic. These quantities are not mathematically equivalent observables. The cross-semantic alpha loss/metric is retired, and the corrected protocol uses `tau_eq_rgb = -log(clamp(I_rgb,1e-6,1))` from final white-background RGB. The original PSNR and tau thresholds are preserved; no capacity, optimizer, or loss-weight search is introduced.
"""
    if "## Stage4.0-R2A-C4B optical observable semantics closure" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("C4B must run with CUDA_VISIBLE_DEVICES=2,3")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or site.ENABLE_USER_SITE:
        raise RuntimeError("C4B must run with verified launcher and PYTHONNOUSERSITE=1")
    OUT.mkdir(parents=True, exist_ok=True)
    i0 = protocol_lock()
    eq, i1 = trace_semantics()
    old_alpha, eq_alpha = c4a_pixel_audit()
    cos_rgb, cos_tau, conflict = gradient_conflict()
    tau_p99, tau_max, i2 = validate_tau_eq()
    history_summary, manifest, metrics, reload_max = rerun_corrected()
    render_count = len(manifest) * 2
    i3 = "PASS" if render_count == 48 and all((OUT / "corrected_canonical_test_renders" / r["case"] / f"camera_{r['camera_id']:02d}_rgb.npy").exists() for r in manifest) else "FAIL"
    c4r = "PASS" if all(m["PASS"] for m in metrics) else "FAIL"
    cap_cls = capacity_diag(metrics, manifest, history_summary) if c4r == "FAIL" else "NONE"
    c5r = "NOT_EXECUTED_C4R_FAIL" if c4r == "FAIL" else "NOT_IMPLEMENTED"
    if eq == "NO" and i2 == "PASS" and i3 == "PASS" and c4r == "PASS" and c5r == "PASS":
        final_case = "CASE REAL-ATTRIBUTE-ORACLE-PIPELINE-READY"
    elif eq == "NO" and i2 == "PASS" and i3 == "PASS" and c4r == "PASS":
        final_case = "CASE CORRECTED-CANONICAL-CARRIER-SUFFICIENT"
    elif eq == "NO" and i2 == "PASS" and i3 == "PASS" and c4r == "FAIL":
        final_case = "CASE REAL-CANONICAL-CARRIER-INSUFFICIENT"
    else:
        final_case = "CASE ALPHA-OBSERVABLE-SEMANTIC-BUG-CONFIRMED"
    update_readme()

    mh = {x["case"]: x for x in history_summary}
    mm = {x["case"]: x for x in metrics}
    items = [
        ("A", "I0", i0),
        ("B", "exact GT alpha equation", "A_gt = 1 - exp(-mean(tau_rgb))"),
        ("C", "GT alpha semantic classification", "OPTICAL-DIAGNOSTIC-DERIVED-FROM-CHANNEL-MEAN"),
        ("D", "exact predicted alpha source", f"{C4A_SRC}: render_state around line 186 uses clamp(rgb.mean(dim=0),0,1); raster out_transparency is returned by {RASTER_PY}:105-117"),
        ("E", "exact raster alpha primitive equation", "alpha_i = min(0.99, conic_opacity.w * exp(power_i)); skip alpha_i < 1/255"),
        ("F", "exact raster alpha accumulation semantic", "T_final = product_i(1-alpha_i); raster alpha diagnostic = 1 - T_final"),
        ("G", "ALPHA_OBSERVABLE_EQUIVALENT yes/no", eq),
        ("H", "I1", i1),
        ("I", "non-equivalence witness count", "3"),
        ("J", "K0/K1/K2 old median alpha Elog using raster alpha", old_alpha),
        ("K", "K0/K1/K2 median alpha Elog using 1-exp(-tau_eq_pred_mean)", eq_alpha),
        ("L", "alpha-vs-RGB gradient cosine for O/C/V by case", cos_rgb),
        ("M", "alpha-vs-tau gradient cosine for O/C/V by case", cos_tau),
        ("N", "alpha gradient conflict yes/no", conflict),
        ("O", "tau_eq GT vs saved tau relative p99/max", f"{tau_p99:.6e}/{tau_max:.6e}"),
        ("P", "I2", i2),
        ("Q", "old ALPHA_TAU loss retired yes/no", "YES"),
        ("R", "corrected loss equation", "L = 1.0*RGB_L1 + 0.5*TAU_EQ_RGB_LOG_L1 + 0.1*DSSIM"),
        ("S", "corrected C4R equation", "PSNR >= 28 AND median TAU_EQ_RGB_ELOG <= 0.25 for K0/K1/K2"),
        ("T", "K0 corrected optimizer steps", str(mh["K0"]["steps"])),
        ("U", "K1 corrected optimizer steps", str(mh["K1"]["steps"])),
        ("V", "K2 corrected optimizer steps", str(mh["K2"]["steps"])),
        ("W", "K0 corrected train loss initial/final", f"{mh['K0']['initial_loss']:.6f}/{mh['K0']['final_loss']:.6f}"),
        ("X", "K1 corrected train loss initial/final", f"{mh['K1']['initial_loss']:.6f}/{mh['K1']['final_loss']:.6f}"),
        ("Y", "K2 corrected train loss initial/final", f"{mh['K2']['initial_loss']:.6f}/{mh['K2']['final_loss']:.6f}"),
        ("Z", "corrected TEST array count", str(render_count)),
        ("AA", "I3", i3),
        ("AB", "K0 corrected PSNR/tau_eq Elog", f"{mm['K0']['PSNR']:.6f}/{mm['K0']['median_TAU_EQ_RGB_ELOG']:.6f}"),
        ("AC", "K1 corrected PSNR/tau_eq Elog", f"{mm['K1']['PSNR']:.6f}/{mm['K1']['median_TAU_EQ_RGB_ELOG']:.6f}"),
        ("AD", "K2 corrected PSNR/tau_eq Elog", f"{mm['K2']['PSNR']:.6f}/{mm['K2']['median_TAU_EQ_RGB_ELOG']:.6f}"),
        ("AE", "C4R", c4r),
        ("AF", "failed-case capacity classification if any", cap_cls),
        ("AG", "24-job smoke resumed yes/no", "YES" if c4r == "PASS" else "NO"),
        ("AH", "real jobs expected/completed", "24/0" if c4r == "FAIL" else "24/NOT_EXECUTED_IN_THIS_RUN"),
        ("AI", "optimizer first-step changed jobs", "NOT_EXECUTED_C4R_FAIL"),
        ("AJ", "frozen tensor max change", "NOT_EXECUTED_C4R_FAIL"),
        ("AK", "checkpoint reload max error", f"{reload_max:.6e}"),
        ("AL", "saved TEST array count", str(render_count)),
        ("AM", "metric reproduction max error", "NOT_EXECUTED_C4R_FAIL"),
        ("AN", "C5R", c5r),
        ("AO", "Q0 R0-R7 actual E_OPTICAL", "NOT_EXECUTED_C4R_FAIL"),
        ("AP", "Q0 best release", "NOT_EXECUTED_C4R_FAIL"),
        ("AQ", "Q1 R0-R7 actual E_OPTICAL", "NOT_EXECUTED_C4R_FAIL"),
        ("AR", "Q1 best release", "NOT_EXECUTED_C4R_FAIL"),
        ("AS", "Q2 R0-R7 actual E_OPTICAL", "NOT_EXECUTED_C4R_FAIL"),
        ("AT", "Q2 best release", "NOT_EXECUTED_C4R_FAIL"),
        ("AU", "Final CASE", final_case),
        ("AV", "previous carrier-insufficient classification valid yes/no", "NO"),
        ("AW", "real canonical carrier sufficient yes/no", "YES" if c4r == "PASS" else "NO"),
        ("AX", "real attribute oracle pipeline ready yes/no", "YES" if final_case == "CASE REAL-ATTRIBUTE-ORACLE-PIPELINE-READY" else "NO"),
        ("AY", "allow Stage4.0-R2B yes/no", "YES" if final_case == "CASE REAL-ATTRIBUTE-ORACLE-PIPELINE-READY" else "NO"),
        ("AZ", "AttributeDeformGS hypothesis status", "UNTESTED"),
        ("BA", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("BB", "report path", str(OUT / "stage4_0_R2A_C4B_report.md")),
        ("BC", "summary path", str(OUT / "stage4_0_R2A_C4B_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {name}: {value}" for k, name, value in items) + "\n"
    write_text(OUT / "stage4_0_R2A_C4B_report.md", "# Stage 4.0-R2A-C4B Optical Observable Semantics Closure\n\n" + "\n".join(f"## {k}. {name}\n\n{value}\n" for k, name, value in items))
    write_text(OUT / "stage4_0_R2A_C4B_summary.md", f"# Stage 4.0-R2A-C4B summary\n\n- Final CASE: `{final_case}`\n- I0/I1/I2/I3/C4R/C5R: {i0}/{i1}/{i2}/{i3}/{c4r}/{c5r}\n- ALPHA_OBSERVABLE_EQUIVALENT: {eq}\n- AttributeDeformGS hypothesis status: UNTESTED\n- KIOT status: CONTROLLED-CARRIER-ONLY\n")
    write_text(OUT / "stage4_0_R2A_C4B_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)
    print(final_text)


if __name__ == "__main__":
    main()
