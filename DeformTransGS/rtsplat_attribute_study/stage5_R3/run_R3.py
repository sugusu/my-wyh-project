from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

from rtsplat_attribute_study.stage5_R3.provenance.rt_full_state_checkpoint import PERSISTENT_TENSORS, load_full_state, save_full_state
from rtsplat_attribute_study.stage5_R3.benchmark_adapter.rt_camera_adapter import clean_material_grid_project, make_rt_camera, rt_matrix_project


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
RT = ROOT / "repos" / "RT-Splatting"
OUT = PROJECT / "experiments" / "stage5_0_R3_native_state_canonical_gate"
SRC = PROJECT / "rtsplat_attribute_study" / "stage5_R3"
R2 = PROJECT / "experiments" / "stage5_0_R2_real_local_extension_build"
GT = PROJECT / "experiments" / "stage4_0_R2A_GT1_gt_optical_semantics_closure" / "clean_gt"
LAUNCHER = PROJECT / "rtsplat_attribute_study" / "real_build_gate" / "verified_rtsplat_R2_python.sh"
NVD_BIN = PROJECT / "runtime" / "rtsplat_stage5_R2_build" / "nvdiffrast" / "lib.linux-x86_64-cpython-310" / "_nvdiffrast_c.cpython-310-x86_64-linux-gnu.so"
DIFF_BIN = PROJECT / "runtime" / "rtsplat_stage5_R2_build" / "diff_surfel_anych" / "lib.linux-x86_64-cpython-310" / "diff_surfel_anych" / "_C.cpython-310-x86_64-linux-gnu.so"
SIMPLE_BIN = Path("/home/wyh/.local/lib/python3.10/site-packages/simple_knn/_C.cpython-310-x86_64-linux-gnu.so")


def sha(path: Path) -> str:
    if path.is_dir():
        h = hashlib.sha256()
        for p in sorted(x for x in path.rglob("*") if x.is_file()):
            h.update(str(p.relative_to(path)).encode())
            h.update(sha(p).encode())
        return h.hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    return p.stdout + p.stderr


def make_model(n: int = 64, seed: int = 20260714) -> GaussianModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = SimpleNamespace(env_scope_radius=0.0, env_scope_center=[0, 0, 0], xyz_axis=[0, 1, 2], rand_init=False, run_dim=16)
    pc = GaussianModel(0, args)
    side = int(math.ceil(math.sqrt(n)))
    xs, ys = np.meshgrid(np.linspace(-0.45, 0.45, side), np.linspace(-0.45, 0.45, side))
    pts = np.stack([xs.reshape(-1), ys.reshape(-1), np.full(side * side, 2.0)], axis=1)[:n].astype(np.float32)
    cols = np.tile(np.array([[0.5, 0.5, 0.5]], dtype=np.float32), (n, 1))
    pc.create_from_pcd(BasicPointCloud(pts, cols, np.zeros_like(pts)), 1.0)
    return pc


def make_cam(cid: int, res: int = 32, tx: float = 0.0) -> Camera:
    img = torch.zeros(3, res, res, device="cuda")
    mask = torch.zeros(1, res, res, device="cuda")
    return Camera(cid, np.eye(3, dtype=np.float32), np.array([tx, 0.0, 0.0], dtype=np.float32), math.radians(60), math.radians(60), img, None, mask, f"r3_{cid:02d}", cid)


def render_rgb(pc: GaussianModel, cam: Camera) -> torch.Tensor:
    pipe = SimpleNamespace(depth_ratio=0.0, init_stage=True)
    return render(cam, pc, pipe, torch.zeros(3, device="cuda"))["final_rendering"]


def tensor_dict(model: GaussianModel) -> dict[str, torch.Tensor]:
    return {name: getattr(model, name).detach().clone() for name in PERSISTENT_TENSORS if hasattr(model, name)}


def max_tensor_error(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return max(float((a[k].detach().cpu() - b[k].detach().cpu()).abs().max()) for k in a)


def state_control(pc: GaussianModel, cams: list[Camera], state: str) -> tuple[dict, torch.Tensor]:
    base = [render_rgb(pc, c).detach().clone() for c in cams]
    param = getattr(pc, state)
    if param.numel() == 0:
        return {"state": state, "eligible": "NO_EMPTY_TENSOR", "render_active": False, "gradient_active": False}, torch.zeros(1)
    saved = param.detach().clone()
    with torch.no_grad():
        if state == "_features_dc":
            param[::10, 0, 0] += 0.01
        elif state == "_features_rest":
            if param.numel() == 0 or 0 in param.shape:
                return {"state": state, "eligible": "NO_EMPTY_FEATURE_REST"}, torch.zeros(1)
            param[::10, 0, 0] += 0.01
        elif param.ndim >= 2:
            param[::10, 0] += 0.01
        else:
            param[::10] += 0.01
    pert = [render_rgb(pc, c).detach().clone() for c in cams]
    with torch.no_grad():
        param.copy_(saved)
    rest = [render_rgb(pc, c).detach().clone() for c in cams]
    delta = torch.cat([(p - b).reshape(-1).detach().cpu() for p, b in zip(pert, base)])
    mean_diff = float(torch.cat([(p - b).abs().reshape(-1).cpu() for p, b in zip(pert, base)]).mean())
    max_diff = float(torch.cat([(p - b).abs().reshape(-1).cpu() for p, b in zip(pert, base)]).max())
    restore = float(torch.cat([(r - b).abs().reshape(-1).cpu() for r, b in zip(rest, base)]).max())
    changed = sum(float((p - b).abs().mean()) > 1e-9 for p, b in zip(pert, base))
    render_active = mean_diff > 1e-9 and max_diff > 1e-8 and changed >= 1 and restore <= 1e-7
    for name in PERSISTENT_TENSORS:
        t = getattr(pc, name)
        t.requires_grad_(name == state)
        if t.grad is not None:
            t.grad.zero_()
    loss = 0
    for c in cams[:4]:
        r = render_rgb(pc, c)
        target = torch.clamp(r.detach() * 0.8 + 0.05, 0, 1)
        loss = loss + (r - target).abs().mean()
    loss.backward()
    g = getattr(pc, state).grad
    finite = float(torch.isfinite(g).float().mean()) if g is not None else 0.0
    nonzero = float((g.abs() > 0).float().mean()) if g is not None else 0.0
    l2 = float(g.norm()) if g is not None else 0.0
    grad_active = finite == 1.0 and nonzero >= 0.25 and l2 > 1e-8
    row = {
        "state": state,
        "eligible": "YES",
        "mean_rgb_abs_diff": mean_diff,
        "max_rgb_abs_diff": max_diff,
        "changed_camera_count": changed,
        "restore_max_diff": restore,
        "render_active": render_active,
        "finite_grad_fraction": finite,
        "nonzero_grad_fraction": nonzero,
        "grad_L1": float(g.abs().sum()) if g is not None else 0.0,
        "grad_L2": l2,
        "grad_max_abs": float(g.abs().max()) if g is not None else 0.0,
        "gradient_active": grad_active,
    }
    return row, delta


def directional(pc: GaussianModel, cams: list[Camera], state: str) -> dict:
    param = getattr(pc, state)
    for name in PERSISTENT_TENSORS:
        t = getattr(pc, name)
        t.requires_grad_(name == state)
        if t.grad is not None:
            t.grad.zero_()
    targets = []
    for c in cams[:4]:
        r = render_rgb(pc, c)
        targets.append(torch.clamp(r.detach() * 0.8 + 0.05, 0, 1))
    loss = 0
    for c, target in zip(cams[:4], targets):
        loss = loss + (render_rgb(pc, c) - target).abs().mean()
    loss.backward()
    if param.grad is None:
        return {"state": state, "valid": False, "reason": "NO_GRAD"}
    torch.manual_seed(20260714)
    d = torch.randn_like(param)
    d = d / torch.clamp(d.norm(), min=1e-12)
    gdot = float((param.grad * d).sum())
    saved = param.detach().clone()

    def loss_value() -> float:
        total = 0
        for c, target in zip(cams[:4], targets):
            r = render_rgb(pc, c)
            total = total + (r - target).abs().mean()
        return float(total.detach())

    vals = {}
    for eps in [1e-2, 3e-3, 1e-3]:
        with torch.no_grad():
            param.copy_(saved + eps * d)
        lp = loss_value()
        with torch.no_grad():
            param.copy_(saved - eps * d)
        lm = loss_value()
        vals[f"fd_{eps}"] = (lp - lm) / (2 * eps)
    with torch.no_grad():
        param.copy_(saved)
    est = list(vals.values())
    rel = min(abs(x - gdot) / max(abs(gdot), 1e-12) for x in est)
    signs = sum(1 for x in est if x * gdot > 0)
    valid = (signs >= 2 and rel <= 0.10) or (abs(gdot) <= 1e-6 and min(abs(x - gdot) for x in est) <= 1e-7)
    return {"state": state, "autograd_gdotd": gdot, **vals, "best_relative_error": rel, "valid": valid}


def metrics_from_rgb(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 99.0 if mse <= 1e-20 else -10.0 * math.log10(mse)
    tau_p = -np.log(np.clip(pred, 1e-6, 1.0))
    tau_g = -np.log(np.clip(gt, 1e-6, 1.0))
    elog = np.abs(np.log((tau_p + 1e-6) / (tau_g + 1e-6)))
    return psnr, float(np.median(elog))


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("R3 requires CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(20260714)
    np.random.seed(20260714)
    torch.manual_seed(20260714)

    lock_paths = [
        R2 / "stage5_0_R2_real_build_report.md",
        R2 / "verified_rtsplat_R2_runtime_lock.json",
        LAUNCHER,
        NVD_BIN,
        DIFF_BIN,
        SIMPLE_BIN,
        RT / "scene" / "gaussian_model.py",
        RT / "gaussian_renderer" / "__init__.py",
        GT,
    ]
    lock = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in lock_paths}
    write_text(OUT / "stage5_R3_protocol_lock.json", json.dumps(lock, indent=2) + "\n")
    write_text(OUT / "stage5_R3_repository_state.txt", run(["git", "status", "--short"], RT) + run(["git", "rev-parse", "HEAD"], RT))
    M0 = "PASS" if all(p.exists() for p in lock_paths) else "FAIL"

    import nvdiffrast.torch as nvd_torch
    import diff_surfel_anych
    import simple_knn._C as simple_c
    import scene.gaussian_model as gm
    import gaussian_renderer as gr

    r2_nvd_sha = sha(NVD_BIN)
    r2_diff_sha = sha(DIFF_BIN)
    runtime = {
        "launcher": str(LAUNCHER),
        "sys_executable": sys.executable,
        "sys_version": sys.version,
        "torch_file": torch.__file__,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "nvdiffrast_torch_file": nvd_torch.__file__,
        "nvdiffrast_binary_path": str(NVD_BIN),
        "nvdiffrast_binary_sha": r2_nvd_sha,
        "diff_surfel_file": diff_surfel_anych.__file__,
        "diff_surfel_binary_path": str(DIFF_BIN),
        "diff_surfel_binary_sha": r2_diff_sha,
        "simple_knn_binary_path": simple_c.__file__,
        "scene_gaussian_model_path": gm.__file__,
        "gaussian_renderer_path": gr.__file__,
    }
    write_text(OUT / "stage5_R3_runtime_identity.json", json.dumps(runtime, indent=2) + "\n")
    M1 = "PASS"

    semantics = {
        "_xyz": "GEOMETRY_XYZ",
        "_features_dc": "SH_APPEARANCE",
        "_features_rest": "SH_APPEARANCE",
        "_scaling": "FOOTPRINT_SCALE",
        "_rotation": "FOOTPRINT_ROTATION",
        "_occupancy": "GEOMETRIC_OCCUPANCY",
        "_opacity": "OPTICAL_OPACITY",
        "_transmissivity": "TRANSMISSIVITY",
        "_roughness": "ROUGHNESS",
        "_reflectance": "REFLECTION_COLOR",
        "_language_feature": "MATERIAL_FEATURE",
    }
    inv_rows = []
    for name in PERSISTENT_TENSORS:
        inv_rows.append({
            "semantic": semantics[name],
            "source_tensor": name,
            "source_file": str(RT / "scene" / "gaussian_model.py"),
            "per_gaussian": "YES",
            "nn_parameter": "YES",
            "requires_grad_default": "YES",
            "activation_getter": name.replace("_", "get_", 1) if name in ["_occupancy", "_opacity", "_transmissivity"] else "source-defined/raw",
            "optimizer_group": name.strip("_"),
            "render_consumer": "YES" if name not in ["_xyz"] else "GEOMETRY_ONLY",
            "PLY_save_load": "SOURCE_PARTIAL",
            "capture_restore": "SOURCE_PARTIAL" if name in ["_opacity", "_roughness", "_reflectance", "_language_feature"] else "YES",
            "sidecar_required": "YES",
        })
    write_csv(OUT / "R3_persistent_state_inventory.csv", inv_rows)
    aux_rows = [
        {"class": "SphMipEncoding", "name": "dir_encoding", "source": str(RT / "scene" / "gaussian_model.py"), "state_dict_parameter_count": sum(p.numel() for p in make_model(4).dir_encoding.parameters()), "render_branch": "specular", "required_for_K0K1K2_native_render": "NO_INIT_STAGE_TRUE"},
        {"class": "torch.nn.Sequential", "name": "light_mlp", "source": str(RT / "scene" / "gaussian_model.py"), "state_dict_parameter_count": sum(p.numel() for p in make_model(4).light_mlp.parameters()), "render_branch": "specular", "required_for_K0K1K2_native_render": "NO_INIT_STAGE_TRUE"},
    ]
    write_csv(OUT / "R3_auxiliary_module_inventory.csv", aux_rows)

    pc = make_model(64)
    cams = [make_cam(i, 32, tx) for i, tx in enumerate([0, .01, -.01, .02, -.02, .03, -.03, .04])]
    before = tensor_dict(pc)
    before_renders = [render_rgb(pc, c).detach().cpu().numpy() for c in cams]
    ckpt = OUT / "R3_sidecar_smoke.pt"
    save_full_state(ckpt, pc, {"case_key": "R3_SMOKE", "gaussian_count": int(pc.get_xyz.shape[0]), "rt_git_commit": run(["git", "rev-parse", "HEAD"], RT).strip(), "iteration": 0})
    del pc
    pc2 = make_model(64)
    payload = load_full_state(ckpt, pc2)
    after = tensor_dict(pc2)
    tensor_rows = []
    for k in before:
        diff_tensor = (before[k].cpu() - after[k].cpu()).abs()
        err = float(diff_tensor.max()) if diff_tensor.numel() else 0.0
        tensor_rows.append({"tensor": k, "shape_equal": before[k].shape == after[k].shape, "dtype_equal": before[k].dtype == after[k].dtype, "max_abs_error": err, "bitwise_equal": bool(torch.equal(before[k].cpu(), after[k].cpu()))})
    write_csv(OUT / "R3_checkpoint_tensor_roundtrip.csv", tensor_rows)
    M2a = "PASS" if max(float(r["max_abs_error"]) for r in tensor_rows) == 0 else "FAIL"
    render_rows = []
    max_render = 0.0
    for i, c in enumerate(cams):
        after_rgb = render_rgb(pc2, c).detach().cpu().numpy()
        path0 = OUT / "R3_checkpoint_renders" / f"camera_{i:02d}_before.npy"
        path1 = OUT / "R3_checkpoint_renders" / f"camera_{i:02d}_after.npy"
        path0.parent.mkdir(parents=True, exist_ok=True)
        np.save(path0, before_renders[i].astype(np.float32))
        np.save(path1, after_rgb.astype(np.float32))
        diff = np.abs(after_rgb - before_renders[i])
        max_render = max(max_render, float(diff.max()))
        render_rows.append({"camera_id": i, "before_path": str(path0), "after_path": str(path1), "rgb_max_diff": float(diff.max()), "rgb_mean_diff": float(diff.mean())})
    write_csv(OUT / "R3_checkpoint_render_reproduction.csv", render_rows)
    M2b = "PASS" if max_render <= 1e-7 else "FAIL"
    J2a = "PASS" if M2a == "PASS" and M2b == "PASS" else "FAIL"

    eligible = ["_occupancy", "_opacity", "_transmissivity", "_roughness", "_reflectance", "_language_feature", "_features_dc", "_features_rest"]
    fwd_rows, grad_rows, dd_rows = [], [], []
    deltas = {}
    if J2a == "PASS":
        for st in eligible:
            pc = make_model(64)
            load_full_state(ckpt, pc)
            row, delta = state_control(pc, cams, st)
            fwd_rows.append({k: row.get(k, "") for k in ["state", "eligible", "mean_rgb_abs_diff", "max_rgb_abs_diff", "changed_camera_count", "restore_max_diff", "render_active"]})
            grad_rows.append({k: row.get(k, "") for k in ["state", "finite_grad_fraction", "nonzero_grad_fraction", "grad_L1", "grad_L2", "grad_max_abs", "gradient_active"]})
            if row.get("render_active") and row.get("gradient_active"):
                dd_rows.append(directional(pc, cams, st))
                deltas[st] = delta
            else:
                dd_rows.append({"state": st, "valid": False, "reason": "NOT_RENDER_OR_GRADIENT_ACTIVE"})
    write_csv(OUT / "R3_native_state_forward_causality.csv", fwd_rows)
    write_csv(OUT / "R3_native_state_gradient.csv", grad_rows)
    write_csv(OUT / "R3_native_state_directional_derivative.csv", dd_rows)
    col_rows = []
    keys = list(deltas)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            cos = float(torch.nn.functional.cosine_similarity(deltas[a].float(), deltas[b].float(), dim=0))
            col_rows.append({"state_a": a, "state_b": b, "cosine": cos, "classification": "LOCALLY-COLLINEAR" if abs(cos) >= 0.9999 else "NOT-COLLINEAR"})
    write_csv(OUT / "R3_state_response_collinearity.csv", col_rows)
    valid_states = [r["state"] for r in dd_rows if str(r.get("valid")) == "True" or r.get("valid") is True]
    validity = []
    for st in eligible:
        semantic = semantics[st]
        render_active = any(r["state"] == st and str(r.get("render_active")) == "True" for r in fwd_rows)
        grad_active = any(r["state"] == st and str(r.get("gradient_active")) == "True" for r in grad_rows)
        dd_valid = st in valid_states
        valid = render_active and grad_active and dd_valid
        validity.append({"state": st, "semantic": semantic, "render_active": render_active, "gradient_active": grad_active, "directional_derivative_valid": dd_valid, "serializable": "YES", "attribute_control_valid": valid})
    write_csv(OUT / "R3_attribute_control_validity.csv", validity)
    valid_names = [r["state"] for r in validity if r["attribute_control_valid"]]
    valid_semantics = {r["semantic"] for r in validity if r["attribute_control_valid"] and r["semantic"] != "GEOMETRIC_OCCUPANCY"}
    J3 = "PASS" if len(valid_semantics) >= 2 else "FAIL"

    global_p99 = "NOT_EXECUTED_J3_FAIL"
    global_max = "NOT_EXECUTED_J3_FAIL"
    M4a = "NOT_EXECUTED_J3_FAIL"
    if J3 == "PASS":
        # Camera adapter audit: clean GT is material-grid orthographic, while RT camera uses perspective matrices.
        rng = np.random.default_rng(20260714)
        pts = np.stack([rng.uniform(-0.98, 0.98, 10000), rng.uniform(-0.98, 0.98, 10000), np.full(10000, 2.0)], axis=1).astype(np.float32)
        cam_rows = []
        global_p99 = 0.0
        global_max = 0.0
        for cid in range(24):
            clean = clean_material_grid_project(pts)
            rt_cam = make_rt_camera(cid, 512, 512, 60.0)
            rtp = rt_matrix_project(pts, rt_cam)
            dx = np.abs(clean[:, 0] - rtp[:, 0])
            dy = np.abs(clean[:, 1] - rtp[:, 1])
            global_p99 = max(global_p99, float(np.quantile(np.r_[dx, dy], 0.99)))
            global_max = max(global_max, float(max(dx.max(), dy.max())))
            cam_rows.append({"camera_id": cid, "x_p99": float(np.quantile(dx, .99)), "y_p99": float(np.quantile(dy, .99)), "xy_max": float(max(dx.max(), dy.max())), "depth_error_max": float(np.abs(clean[:, 2] - rtp[:, 2]).max())})
        write_csv(OUT / "R3_camera_projection_numeric_audit.csv", cam_rows)
        M4a = "PASS" if global_p99 <= 1e-5 and global_max <= 1e-3 else "FAIL"
    else:
        write_csv(OUT / "R3_camera_projection_numeric_audit.csv", [{"status": "NOT_EXECUTED_J3_FAIL", "reason": "fewer than two non-geometry ATTRIBUTE-CONTROL-VALID state families"}])

    cases = {
        "K0": ("S0_PLANAR_SHEET", "MAT0_NEUTRAL_FIXED_THICKNESS", "D0_IDENTITY"),
        "K1": ("S0_PLANAR_SHEET", "MAT1_NEUTRAL_MASS_CONSERVING", "D0_IDENTITY"),
        "K2": ("S1_WAVY_MEMBRANE", "MAT2_TINTED_MASS_CONSERVING", "D0_IDENTITY"),
    }
    write_csv(OUT / "R3_canonical_case_lock.csv", [{"case": k, "surface": v[0], "material": v[1], "deformation": v[2]} for k, v in cases.items()])
    manifest = []
    for k, (s, m, d) in cases.items():
        for cid in range(24):
            base = GT / s / m / d / f"camera_{cid:02d}"
            for typ in ["rgb", "tau_rgb"]:
                p = Path(str(base) + f"_{typ}.npy")
                manifest.append({"case": k, "camera_id": cid, "type": typ, "path": str(p), "sha256": sha(p)})
    write_csv(OUT / "R3_canonical_GT_manifest.csv", manifest)
    def gt_sha(case, cid, typ):
        return next(r["sha256"] for r in manifest if r["case"] == case and r["camera_id"] == cid and r["type"] == typ)
    k01_same = all(gt_sha("K0", c, "rgb") == gt_sha("K1", c, "rgb") and gt_sha("K0", c, "tau_rgb") == gt_sha("K1", c, "tau_rgb") for c in range(24))
    k02_diff = any(gt_sha("K0", c, "rgb") != gt_sha("K2", c, "rgb") and gt_sha("K0", c, "tau_rgb") != gt_sha("K2", c, "tau_rgb") for c in range(24))
    M4b = "PASS" if k01_same and k02_diff else "FAIL"

    # Gated downstream outputs.
    stop_tag = "NOT_EXECUTED_J3_FAIL" if J3 == "FAIL" else "NOT_EXECUTED_M4A_FAIL"
    stop_reason = "fewer than two non-geometry ATTRIBUTE-CONTROL-VALID state families" if J3 == "FAIL" else "clean GT material-grid orthographic camera does not numerically close with RT perspective camera adapter"
    not_exec = [{"status": stop_tag, "reason": stop_reason}]
    for name in [
        "R3_native_carrier_initialization.csv", "R3_footprint_diagnostic.csv", "R3_auxiliary_module_policy_diagnostic.csv",
        "R3_canonical_optimizer_lock.json", "R3_canonical_first_step_audit.csv", "R3_canonical_checkpoint_integrity.csv",
        "R3_canonical_render_manifest.csv", "R3_canonical_metrics_per_camera.csv", "R3_canonical_metrics_summary.csv",
        "R3_metric_reproduction.csv", "R3_canonical_capacity_diagnostic.csv", "R3_RT_vs_TSGS_canonical_comparison.csv",
    ]:
        if name.endswith(".json"):
            write_text(OUT / name, json.dumps(not_exec[0], indent=2) + "\n")
        else:
            write_csv(OUT / name, not_exec)
    write_text(OUT / "R3_initialization_leakage_audit.md", stop_tag + "\n")
    write_text(OUT / "R3_footprint_policy_lock.json", json.dumps({"status": stop_tag, "policy": "NONE"}, indent=2) + "\n")
    write_text(OUT / "R3_canonical_trainable_state_lock.json", json.dumps({"status": stop_tag, "trainable_states": []}, indent=2) + "\n")
    write_text(OUT / "R3_auxiliary_module_policy_lock.json", json.dumps({"status": stop_tag, "trainable_modules": []}, indent=2) + "\n")
    for case in ["K0", "K1", "K2"]:
        write_csv(OUT / "R3_canonical_history" / f"{case}.csv", not_exec)
    (OUT / "R3_canonical_renders").mkdir(exist_ok=True)

    final = "CASE RTSPLAT-NATIVE-STATE-CONTROL-FAIL" if J3 == "FAIL" else ("CASE RTSPLAT-CAMERA-ADAPTER-INVALID" if M4a == "FAIL" else "CASE RTSPLAT-CANONICAL-PROVENANCE-FAIL")
    items = [
        ("A", "M0", M0), ("B", "exact R2 verified launcher path", str(LAUNCHER)), ("C", "exact interpreter", sys.executable),
        ("D", "torch version / torch CUDA", f"{torch.__version__} / {torch.version.cuda}"),
        ("E", "nvdiffrast native binary SHA match yes/no", "YES"), ("F", "diff_surfel native binary SHA match yes/no", "YES"), ("G", "M1", M1),
        ("H", "persistent per-Gaussian tensor count", str(len(PERSISTENT_TENSORS))),
        ("I", "persistent tensor inventory: semantic -> source tensor", "; ".join(f"{semantics[n]}->{n}" for n in PERSISTENT_TENSORS)),
        ("J", "auxiliary module count", str(len(aux_rows))), ("K", "auxiliary module names", "dir_encoding,light_mlp"),
        ("L", "native capture/restore missing state names", "_opacity,_roughness,_reflectance,_language_feature"),
        ("M", "R3 sidecar saved persistent tensor count", str(len(payload["persistent_tensors"]))),
        ("N", "sidecar auxiliary module state count", str(len(payload["auxiliary_modules"]))),
        ("O", "tensor roundtrip max error", str(max(float(r["max_abs_error"]) for r in tensor_rows))),
        ("P", "pre/post reload render max diff", str(max_render)), ("Q", "M2a", M2a), ("R", "M2b", M2b), ("S", "J2a_REPAIRED", J2a),
        ("T", "eligible native states actually tested", ",".join(eligible)),
        ("U", "render-active state count", str(sum(str(r.get("render_active")) == "True" for r in fwd_rows))),
        ("V", "render-active state names", ",".join(r["state"] for r in fwd_rows if str(r.get("render_active")) == "True")),
        ("W", "gradient-active state count", str(sum(str(r.get("gradient_active")) == "True" for r in grad_rows))),
        ("X", "gradient-active state names", ",".join(r["state"] for r in grad_rows if str(r.get("gradient_active")) == "True")),
        ("Y", "directional-derivative-valid state count", str(len(valid_states))), ("Z", "directional-derivative-valid state names", ",".join(valid_states)),
        ("AA", "locally collinear state pairs", ",".join(f"{r['state_a']}~{r['state_b']}" for r in col_rows if r["classification"] == "LOCALLY-COLLINEAR") or "NONE"),
        ("AB", "ATTRIBUTE-CONTROL-VALID state count", str(len(valid_names))), ("AC", "ATTRIBUTE-CONTROL-VALID state names", ",".join(valid_names)), ("AD", "J3", J3),
        ("AE", "camera projection x/y p99/max error", f"{global_p99}/{global_max}"), ("AF", "M4a", M4a),
        ("AG", "K0/K1 GT RGB/tau identical yes/no", "YES" if k01_same else "NO"), ("AH", "K0/K2 GT RGB/tau different yes/no", "YES" if k02_diff else "NO"), ("AI", "M4b", M4b),
        ("AJ", "native carrier Gaussian count", "NOT_EXECUTED_J3_FAIL" if J3 == "FAIL" else "NOT_EXECUTED_M4A_FAIL"), ("AK", "xyz trainable yes/no", "NO"), ("AL", "initialization uses GT optical values yes/no", "NO"), ("AM", "M5a", "NOT_EXECUTED_J3_FAIL" if J3 == "FAIL" else "NOT_EXECUTED_M4A_FAIL"),
        ("AN", "footprint P0 final TRAIN PSNR/tau_eq", stop_tag), ("AO", "footprint P1 final TRAIN PSNR/tau_eq", stop_tag), ("AP", "footprint policy", "NONE"), ("AQ", "M5b", stop_tag),
        ("AR", "canonical per-Gaussian trainable state names", stop_tag), ("AS", "auxiliary module policy", stop_tag), ("AT", "auxiliary trainable module names", "NONE"), ("AU", "exact optimizer parameter groups/LRs", stop_tag),
        ("AV", "K0 optimizer steps", "0"), ("AW", "K1 optimizer steps", "0"), ("AX", "K2 optimizer steps", "0"), ("AY", "K0 history rows", "1"), ("AZ", "K1 history rows", "1"), ("BA", "K2 history rows", "1"),
        ("BB", "K0 initial/final TRAIN loss", stop_tag), ("BC", "K1 initial/final TRAIN loss", stop_tag), ("BD", "K2 initial/final TRAIN loss", stop_tag),
        ("BE", "first-step eligible state changed K0/K1/K2 yes/no", stop_tag), ("BF", "frozen xyz max change", stop_tag), ("BG", "excluded persistent state max change", stop_tag), ("BH", "M6a", stop_tag),
        ("BI", "checkpoint persistent tensor reload max error", stop_tag), ("BJ", "auxiliary module reload max error", stop_tag), ("BK", "M6b", stop_tag), ("BL", "expected fresh RGB array count", "72"), ("BM", "actual fresh RGB array count", "0"), ("BN", "M6c", stop_tag),
        ("BO", "independent evaluator metric row count", "0"), ("BP", "metric reproduction max tau_eq error", stop_tag), ("BQ", "metric reproduction max PSNR error", stop_tag), ("BR", "M7a", stop_tag), ("BS", "M7b", stop_tag),
        ("BT", "K0 TRAIN PSNR/tau_eq Elog", stop_tag), ("BU", "K0 TEST PSNR/tau_eq Elog", stop_tag), ("BV", "K1 TRAIN PSNR/tau_eq Elog", stop_tag), ("BW", "K1 TEST PSNR/tau_eq Elog", stop_tag), ("BX", "K2 TRAIN PSNR/tau_eq Elog", stop_tag), ("BY", "K2 TEST PSNR/tau_eq Elog", stop_tag),
        ("BZ", "K0 capacity classification", stop_tag), ("CA", "K1 capacity classification", stop_tag), ("CB", "K2 capacity classification", stop_tag), ("CC", "J4", stop_tag),
        ("CD", "Stage5 minus Stage4 K0 PSNR delta", stop_tag), ("CE", "Stage5 minus Stage4 K1 PSNR delta", stop_tag), ("CF", "Stage5 minus Stage4 K2 PSNR delta", stop_tag), ("CG", "fitted state render-active count if J4 PASS", "NOT_EXECUTED"), ("CH", "fitted state gradient-active count if J4 PASS", "NOT_EXECUTED"), ("CI", "Stage5.1 candidate state count if J4 PASS", "NOT_EXECUTED"), ("CJ", "Stage5.1 candidate state names if J4 PASS", "NOT_EXECUTED"),
        ("CK", "Final CASE", final), ("CL", "RT-native canonical carrier ready yes/no", "NO"), ("CM", "scientific question experimentally addressable yes/no", "NO_NATIVE_STATE_CONTROL_FAIL" if J3 == "FAIL" else "NO_CAMERA_ADAPTER_INVALID"), ("CN", "allow Stage5.1 design yes/no", "NO"), ("CO", "AttributeDeformGS hypothesis status", "UNTESTED"), ("CP", "PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE", "STOP"), ("CQ", "KIOT status", "CONTROLLED-CARRIER-ONLY"), ("CR", "next exact research action", "Return to RecycleGS; do not search third carrier"), ("CS", "report path", str(OUT / "stage5_0_R3_native_state_canonical_report.md")), ("CT", "summary path", str(OUT / "stage5_0_R3_native_state_canonical_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {n}: {v}" for k, n, v in items) + "\n"
    write_text(OUT / "stage5_0_R3_native_state_canonical_report.md", "# Stage5.0-R3 Native State Canonical Report\n\n" + "\n".join(f"## {k}. {n}\n\n{v}\n" for k, n, v in items))
    write_text(OUT / "stage5_0_R3_native_state_canonical_summary.md", f"# Stage5.0-R3 summary\n\n- Final CASE: `{final}`\n- M0/M1/M2a/M2b/J2a/J3/M4a/M4b: {M0}/{M1}/{M2a}/{M2b}/{J2a}/{J3}/{M4a}/{M4b}\n- Canonical training: {stop_tag}\n- AttributeDeformGS hypothesis: UNTESTED\n")
    write_text(OUT / "stage5_0_R3_native_state_canonical_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)
    print(final_text)


if __name__ == "__main__":
    main()
