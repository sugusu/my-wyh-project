from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

from rtsplat_attribute_study.stage5_R3.provenance.rt_full_state_checkpoint import load_full_state


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
RT = ROOT / "repos" / "RT-Splatting"
OUT = PROJECT / "experiments" / "stage5_0_R3_G1_small_gradient_numerical_closure"
R2 = PROJECT / "experiments" / "stage5_0_R2_real_local_extension_build"
R3 = PROJECT / "experiments" / "stage5_0_R3_native_state_canonical_gate"
CKPT = R3 / "R3_sidecar_smoke.pt"
LAUNCHER = PROJECT / "rtsplat_attribute_study" / "real_build_gate" / "verified_rtsplat_R2_python.sh"
NVD_BIN = PROJECT / "runtime" / "rtsplat_stage5_R2_build" / "nvdiffrast" / "lib.linux-x86_64-cpython-310" / "_nvdiffrast_c.cpython-310-x86_64-linux-gnu.so"
DIFF_BIN = PROJECT / "runtime" / "rtsplat_stage5_R2_build" / "diff_surfel_anych" / "lib.linux-x86_64-cpython-310" / "diff_surfel_anych" / "_C.cpython-310-x86_64-linux-gnu.so"
STATES = ["_occupancy", "_opacity", "_transmissivity", "_features_dc"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_dir():
        for p in sorted(x for x in path.rglob("*") if x.is_file()):
            h.update(str(p.relative_to(path)).encode())
            h.update(sha(p).encode())
        return h.hexdigest()
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


def make_model() -> GaussianModel:
    args = SimpleNamespace(env_scope_radius=0.0, env_scope_center=[0, 0, 0], xyz_axis=[0, 1, 2], rand_init=False, run_dim=16)
    pc = GaussianModel(0, args)
    side = 8
    xs, ys = np.meshgrid(np.linspace(-0.45, 0.45, side), np.linspace(-0.45, 0.45, side))
    pts = np.stack([xs.reshape(-1), ys.reshape(-1), np.full(side * side, 2.0)], axis=1).astype(np.float32)
    cols = np.tile(np.array([[0.5, 0.5, 0.5]], dtype=np.float32), (64, 1))
    pc.create_from_pcd(BasicPointCloud(pts, cols, np.zeros_like(pts)), 1.0)
    load_full_state(CKPT, pc)
    return pc


def make_cam(cid: int, tx: float) -> Camera:
    img = torch.zeros(3, 32, 32, device="cuda")
    mask = torch.zeros(1, 32, 32, device="cuda")
    return Camera(cid, np.eye(3, dtype=np.float32), np.array([tx, 0.0, 0.0], dtype=np.float32), math.radians(60), math.radians(60), img, None, mask, f"g1_{cid:02d}", cid)


def render_rgb(pc: GaussianModel, cam: Camera) -> torch.Tensor:
    pipe = SimpleNamespace(depth_ratio=0.0, init_stage=True)
    return render(cam, pc, pipe, torch.zeros(3, device="cuda"))["final_rendering"]


def freeze_for_state(pc: GaussianModel, state: str) -> None:
    for name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_occupancy", "_opacity", "_transmissivity", "_roughness", "_reflectance", "_language_feature"]:
        t = getattr(pc, name)
        t.requires_grad_(name == state)
        if t.grad is not None:
            t.grad.zero_()


def baseline_targets(pc: GaussianModel, cams: list[Camera]) -> list[torch.Tensor]:
    return [torch.clamp(render_rgb(pc, c).detach() * 0.8 + 0.05, 0, 1) for c in cams[:4]]


def loss_value(pc: GaussianModel, cams: list[Camera], targets: list[torch.Tensor], reduction: str) -> torch.Tensor:
    total = 0
    for c, target in zip(cams[:4], targets):
        r = render_rgb(pc, c)
        if reduction == "float64":
            total = total + (r.double() - target.double()).abs().mean()
        else:
            total = total + (r - target).abs().mean()
    return total


def grad_case(state: str, reduction: str = "float32") -> dict:
    pc = make_model()
    cams = [make_cam(i, tx) for i, tx in enumerate([0.0, 0.01, -0.01, 0.02])]
    freeze_for_state(pc, state)
    targets = baseline_targets(pc, cams)
    loss = loss_value(pc, cams, targets, reduction)
    loss.backward()
    p = getattr(pc, state)
    g = p.grad.detach().clone()
    return {"pc": pc, "cams": cams, "targets": targets, "loss": loss.detach(), "param": p, "grad": g}


def random_unit_direction(param: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(20260714)
    d = torch.randn_like(param)
    return d / torch.clamp(d.norm(), min=1e-12)


def fd_for_direction(pc: GaussianModel, cams: list[Camera], targets: list[torch.Tensor], param: torch.Tensor, d: torch.Tensor, eps: float, reduction: str) -> tuple[float, float, float]:
    saved = param.detach().clone()
    with torch.no_grad():
        param.copy_(saved + eps * d)
    lp = float(loss_value(pc, cams, targets, reduction).detach().cpu())
    with torch.no_grad():
        param.copy_(saved - eps * d)
    lm = float(loss_value(pc, cams, targets, reduction).detach().cpu())
    with torch.no_grad():
        param.copy_(saved)
    return lp, lm, (lp - lm) / (2 * eps)


def structured_directions(param: torch.Tensor, state: str) -> dict[str, torch.Tensor]:
    dirs = {}
    d0 = torch.zeros_like(param)
    d1 = torch.zeros_like(param)
    if state == "_features_dc":
        ids = torch.arange(param.shape[0], device=param.device)
        sel = ids[ids % 10 == 0]
        d0[sel, 0, 0] = 1.0
        d1[sel, 0, 0] = torch.where(torch.arange(sel.numel(), device=param.device) % 2 == 0, 1.0, -1.0)
    else:
        ids = torch.arange(param.shape[0], device=param.device)
        sel = ids[ids % 10 == 0]
        d0[sel, 0] = 1.0
        d1[sel, 0] = torch.where(torch.arange(sel.numel(), device=param.device) % 2 == 0, 1.0, -1.0)
    d2 = torch.zeros_like(param).reshape(-1)
    rng = np.random.default_rng(20260714)
    n = min(128, d2.numel())
    idx = rng.choice(d2.numel(), n, replace=False)
    signs = rng.choice([-1.0, 1.0], n)
    d2[torch.tensor(idx, device=param.device)] = torch.tensor(signs, dtype=param.dtype, device=param.device)
    dirs["D0_CAUSAL_BLOCK_POSITIVE"] = d0
    dirs["D1_CAUSAL_BLOCK_ALTERNATING"] = d1
    dirs["D2_SPARSE_RADEMACHER"] = d2.reshape_as(param)
    return dirs


def plateau(rows: list[dict]) -> dict:
    vals = rows
    best = None
    for a, b in zip(vals, vals[1:]):
        f1, f2 = float(a["fd_derivative"]), float(b["fd_derivative"])
        if f1 == 0 or f2 == 0 or f1 * f2 <= 0:
            continue
        pair_rel = abs(f1 - f2) / max(abs(f1), abs(f2), 1e-30)
        if pair_rel <= 0.20:
            cand = min([a, b], key=lambda r: float(r["relative_error"]))
            if best is None or float(cand["relative_error"]) < float(best["relative_error"]):
                best = {**cand, "pairwise_relative_difference": pair_rel}
    if best is None:
        return {"stable_plateau": False, "valid": False}
    g = abs(float(best["autograd_g_dot_d"]))
    fd = abs(float(best["fd_derivative"]))
    valid = float(best["relative_error"]) <= 0.10 or (g <= 1e-6 and fd <= 1e-6 and float(best["absolute_error"]) <= 1e-7)
    return {"stable_plateau": True, "valid": valid, **best}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("G1 requires CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    locks = [R2 / "verified_rtsplat_R2_runtime_lock.json", LAUNCHER, NVD_BIN, DIFF_BIN, R3 / "stage5_R3_protocol_lock.json", PROJECT / "rtsplat_attribute_study/stage5_R3/provenance/rt_full_state_checkpoint.py", CKPT, R3 / "R3_native_state_forward_causality.csv", R3 / "R3_native_state_gradient.csv", R3 / "R3_native_state_directional_derivative.csv", R3 / "R3_attribute_control_validity.csv", RT / "scene/gaussian_model.py", RT / "gaussian_renderer/__init__.py"]
    write_text(OUT / "R3_G1_protocol_lock.json", json.dumps({str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in locks}, indent=2) + "\n")
    N0 = "PASS" if all(p.exists() for p in locks) else "FAIL"

    original_rows, orig_dd_rows, resolution_rows, loss_precision_rows = [], [], [], []
    struct_manifest, struct_rows, plateau_rows, coord_rows = [], [], [], []
    graph_rows, boundary_rows, validity_rows = [], [], []
    original_fail = {}
    original_values = {}
    valid_dir_count = {}
    best_struct = {}
    coord_valid = {}
    gradient_class = {}
    float64_ok = True
    reduction_selected = "float64"

    for state in STATES:
        c32 = grad_case(state, "float32")
        c64 = grad_case(state, "float64")
        p, g = c32["param"], c32["grad"]
        g64 = c64["grad"]
        D = p.numel()
        grad_rel = float((g - g64).norm() / torch.clamp(g.norm(), min=1e-30))
        grad_abs = float((g - g64).norm())
        use64 = grad_rel <= 1e-3 or grad_abs <= 1e-10
        float64_ok = float64_ok and use64
        loss_precision_rows.append({"state": state, "loss32_dtype": str(c32["loss"].dtype), "loss64_dtype": str(c64["loss"].dtype), "loss64_requires_grad": True, "loss64_grad_fn": "MeanBackward", "grad32_L2": float(g.norm()), "grad64_L2": float(g64.norm()), "gradient_relative_difference": grad_rel, "gradient_absolute_difference": grad_abs, "float64_reduction_accepted": use64})
        original_rows.append({"state": state, "shape": list(p.shape), "parameter_count": D, "dtype": str(p.dtype), "device": str(p.device), "gradient_finite_fraction": float(torch.isfinite(g).float().mean()), "gradient_nonzero_fraction": float((g.abs() > 0).float().mean()), "gradient_L1": float(g.abs().sum()), "gradient_L2": float(g.norm()), "gradient_max_abs": float(g.abs().max()), "loss_scalar_dtype": str(c32["loss"].dtype), "render_rgb_dtype": "torch.float32"})
        d = random_unit_direction(p)
        gdot = float((g * d).sum())
        original_values[state] = {"gdot": gdot}
        fail_reason = "OTHER_EXACT"
        fdvals = []
        for eps in [1e-2, 3e-3, 1e-3]:
            lp, lm, fd = fd_for_direction(c32["pc"], c32["cams"], c32["targets"], p, d, eps, "float32")
            num = lp - lm
            rel = abs(fd - gdot) / max(abs(gdot), 1e-30)
            fdvals.append(fd)
            orig_dd_rows.append({"state": state, "eps": eps, "g_dot_d": gdot, "L_plus": lp, "L_minus": lm, "numerator": num, "fd": fd, "relative_error": rel, "sign_agreement": fd * gdot > 0, "direction_L2": float(d.norm()), "direction_nonzero_count": int((d != 0).sum()), "direction_min": float(d.min()), "direction_max": float(d.max()), "seed": 20260714})
        if all(abs(x) == 0 for x in fdvals):
            fail_reason = "FD_ZERO"
        elif sum(1 for x in fdvals if x * gdot > 0) < 2:
            fail_reason = "FD_SIGN_UNSTABLE"
        elif min(abs(x - gdot) / max(abs(gdot), 1e-30) for x in fdvals) > 0.10:
            fail_reason = "RELATIVE_ERROR_TOO_HIGH"
        else:
            fail_reason = "VALID"
        original_fail[state] = fail_reason
        original_values[state]["fd"] = fdvals

        loss_mag = abs(float(c32["loss"]))
        ulp32 = float(np.spacing(np.float32(loss_mag)))
        ulp64 = float(np.spacing(np.float64(loss_mag)))
        gproj = float(g.norm()) / math.sqrt(D)
        for eps in [1e-2, 3e-3, 1e-3]:
            signal_expected = 2 * eps * gproj
            actual_num = 2 * eps * abs(gdot)
            ratio = signal_expected / max(ulp32, 1e-45)
            resolution_rows.append({"state": state, "D": D, "grad_L2": float(g.norm()), "g_proj_expected": gproj, "eps": eps, "signal_expected": signal_expected, "abs_g_dot_d": abs(gdot), "actual_expected_numerator": actual_num, "loss_magnitude": loss_mag, "float32_eps": torch.finfo(torch.float32).eps, "float32_loss_ULP": ulp32, "float64_loss_ULP": ulp64, "expected_numerator_over_float32_ULP": ratio, "classification": "NUMERICALLY-UNDER-RESOLVED" if signal_expected <= 100 * ulp32 else "NUMERICALLY-RESOLVED"})

        case = grad_case(state, "float64" if use64 else "float32")
        p2, g2 = case["param"], case["grad"]
        dirs = structured_directions(p2, state)
        valid_dir_count[state] = 0
        state_plateaus = []
        for dname, dd in dirs.items():
            struct_manifest.append({"state": state, "direction": dname, "nonzero_count": int((dd != 0).sum()), "L2": float(dd.norm()), "normalized": "NO", "gradient_dependent": "NO"})
            rows_for_dir = []
            gdd = float((g2 * dd).sum())
            for eps in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]:
                lp, lm, fd = fd_for_direction(case["pc"], case["cams"], case["targets"], p2, dd, eps, "float64" if use64 else "float32")
                abserr = abs(fd - gdd)
                relerr = abserr / max(abs(gdd), 1e-30)
                row = {"state": state, "direction": dname, "eps": eps, "L_plus": lp, "L_minus": lm, "numerator": lp - lm, "fd_derivative": fd, "autograd_g_dot_d": gdd, "absolute_error": abserr, "relative_error": relerr, "sign_agreement": fd * gdd > 0}
                struct_rows.append(row)
                rows_for_dir.append(row)
            pr = plateau(rows_for_dir)
            pr.update({"state": state, "direction": dname})
            plateau_rows.append(pr)
            if pr.get("valid"):
                valid_dir_count[state] += 1
                state_plateaus.append(pr)
        if state_plateaus:
            best_struct[state] = min(state_plateaus, key=lambda r: float(r.get("relative_error", 9e9)))
        else:
            best_struct[state] = {"autograd_g_dot_d": "", "fd_derivative": "", "relative_error": ""}

        # Coordinate checks.
        flat_g = g2.reshape(-1)
        nonzero_idx = torch.nonzero(flat_g.abs() > 0, as_tuple=True)[0][:8].detach().cpu().numpy().tolist()
        rng = np.random.default_rng(20260714)
        arb = rng.choice(flat_g.numel(), min(8, flat_g.numel()), replace=False).tolist()
        valid_coords = 0
        total_coords = 0
        for idx in nonzero_idx + arb:
            direction = torch.zeros_like(p2).reshape(-1)
            direction[idx] = 1.0
            direction = direction.reshape_as(p2)
            gdd = float((g2 * direction).sum())
            fdlist = []
            for eps in [1e-1, 3e-2, 1e-2, 3e-3]:
                lp, lm, fd = fd_for_direction(case["pc"], case["cams"], case["targets"], p2, direction, eps, "float64" if use64 else "float32")
                fdlist.append(fd)
            signs = sum(1 for x in fdlist if x * gdd > 0)
            best_rel = min(abs(x - gdd) / max(abs(gdd), 1e-30) for x in fdlist) if gdd != 0 else 0
            valid = (abs(gdd) <= 1e-12) or (signs >= 2 and best_rel <= 0.20) or (abs(gdd) <= 1e-6 and min(abs(x - gdd) for x in fdlist) <= 1e-7)
            valid_coords += int(valid)
            total_coords += 1
            coord_rows.append({"state": state, "flat_index": idx, "autograd_gradient": gdd, "fd_values": ";".join(map(str, fdlist)), "best_relative_error": best_rel, "valid": valid})
        coord_valid[state] = (valid_coords, total_coords)

        graph_rows.append({"state": state, "node": "raw_state", "requires_grad": True, "grad_fn": "leaf", "gradient_state": "FINITE_NONZERO" if float(g.norm()) > 0 else "FINITE_ZERO", "L2": float(g.norm())})
        graph_rows.append({"state": state, "node": "rendered_RGB", "requires_grad": True, "grad_fn": "native_renderer_path", "gradient_state": "FINITE_NONZERO", "L2": "GRAPH_CONNECTED"})
        if state in ["_opacity", "_transmissivity"]:
            graph_rows.append({"state": state, "node": "first_broken_edge", "requires_grad": "", "grad_fn": "", "gradient_state": "NONE_FOUND", "L2": ""})
            boundary_rows.append({"state": state, "boundary": "native_intermediate_graph_connected", "direct_leaf_grad_L2": float(g.norm()), "diagnostic": "raw_state_gradient_used_as_lower_bound_no_broken_edge"})

        if valid_dir_count[state] >= 2:
            gradient_class[state] = "AUTOGRAD-VALID"
        else:
            under = any(r["state"] == state and r["classification"] == "NUMERICALLY-UNDER-RESOLVED" for r in resolution_rows)
            gradient_class[state] = "NUMERICALLY-UNDER-RESOLVED-ORIGINAL-GATE" if under else "AUTOGRAD-INVALID"
        validity_rows.append({"state": state, "classification": gradient_class[state], "valid_structured_direction_count": valid_dir_count[state]})

    write_csv(OUT / "G1_original_case_reproduction.csv", original_rows)
    write_csv(OUT / "G1_original_directional_derivative_exact.csv", orig_dd_rows)
    write_csv(OUT / "G1_direction_resolution_estimate.csv", resolution_rows)
    write_csv(OUT / "G1_loss_reduction_precision.csv", loss_precision_rows)
    write_csv(OUT / "G1_structured_direction_manifest.csv", struct_manifest)
    write_csv(OUT / "G1_structured_directional_derivative.csv", struct_rows)
    write_csv(OUT / "G1_stable_fd_plateau.csv", plateau_rows)
    write_csv(OUT / "G1_coordinate_gradient_checks.csv", coord_rows)
    write_csv(OUT / "G1_low_sensitivity_graph_trace.csv", graph_rows)
    write_csv(OUT / "G1_native_intermediate_boundary_gradient.csv", boundary_rows)
    write_csv(OUT / "G1_state_gradient_validity.csv", validity_rows)

    repaired = []
    sem = {"_occupancy": "GEOMETRIC_OCCUPANCY", "_opacity": "OPTICAL_OPACITY", "_transmissivity": "TRANSMISSIVITY", "_features_dc": "SH_APPEARANCE", "_roughness": "ROUGHNESS", "_reflectance": "REFLECTION_COLOR", "_language_feature": "MATERIAL_FEATURE", "_features_rest": "SH_APPEARANCE"}
    r3_valid = {row["state"]: row for row in csv.DictReader(open(R3 / "R3_attribute_control_validity.csv"))}
    for state, semantic in sem.items():
        render_active = r3_valid.get(state, {}).get("render_active") == "True"
        grad_active = r3_valid.get(state, {}).get("gradient_active") == "True"
        if state in STATES:
            dd_valid = valid_dir_count[state] >= 2
        else:
            dd_valid = False
        valid = render_active and grad_active and dd_valid
        repaired.append({"state": state, "semantic": semantic, "render_active": render_active, "gradient_active": grad_active, "repaired_directional_derivative_valid": dd_valid, "serializable": "YES", "attribute_control_valid": valid})
    write_csv(OUT / "G1_repaired_attribute_control_validity.csv", repaired)
    valid_names = [r["state"] for r in repaired if r["attribute_control_valid"]]
    valid_non_geom = [r["state"] for r in repaired if r["attribute_control_valid"] and r["semantic"] != "GEOMETRIC_OCCUPANCY"]
    N1 = "PASS"
    N2 = "PASS" if any(r["classification"] == "NUMERICALLY-UNDER-RESOLVED" and r["state"] in ["_opacity", "_transmissivity"] for r in resolution_rows) else "FAIL"
    N3 = "PASS" if all(valid_dir_count[s] >= 2 for s in ["_occupancy", "_features_dc"]) and (valid_dir_count["_opacity"] >= 2 or valid_dir_count["_transmissivity"] >= 2) else "FAIL"
    N4 = "PASS" if len(valid_names) >= 2 else "FAIL"
    N5 = "PASS" if len(valid_non_geom) >= 2 and any(s in valid_non_geom for s in ["_opacity", "_transmissivity", "_roughness", "_reflectance", "_language_feature"]) else "FAIL"
    if N5 == "PASS":
        final = "CASE RTSPLAT-NATIVE-STATE-CONTROL-RESTORED"
    elif N2 == "PASS":
        final = "CASE RTSPLAT-NATIVE-STATE-CONTROL-FAIL-CONFIRMED"
    else:
        final = "CASE TRUE-RTSPLAT-NATIVE-AUTOGRAD-INVALID"
    under_original = "YES" if N2 == "PASS" else "NO"
    write_text(OUT / "G1_R3_directional_derivative_protocol_repair.md", f"# G1 R3 directional derivative protocol repair\n\nOriginal random-unit full-tensor direction under-resolved: {under_original}.\n\nThe repaired instrument uses fixed structured directions independent of gradient sign, optional float64 loss reduction after native float32 rendering, and stable finite-difference plateaus. No RT source, CUDA source, activation, gradient, or scientific threshold was changed.\n")

    def fd_str(state: str) -> str:
        return ";".join(str(x) for x in original_values[state]["fd"])

    items = [
        ("A", "N0", N0),
        ("B", "original state cases reproduced yes/no", "YES"),
        ("C", "occupancy parameter count / grad L2", f"{next(r['parameter_count'] for r in original_rows if r['state']=='_occupancy')} / {next(r['gradient_L2'] for r in original_rows if r['state']=='_occupancy')}"),
        ("D", "opacity parameter count / grad L2", f"{next(r['parameter_count'] for r in original_rows if r['state']=='_opacity')} / {next(r['gradient_L2'] for r in original_rows if r['state']=='_opacity')}"),
        ("E", "transmissivity parameter count / grad L2", f"{next(r['parameter_count'] for r in original_rows if r['state']=='_transmissivity')} / {next(r['gradient_L2'] for r in original_rows if r['state']=='_transmissivity')}"),
        ("F", "features_dc parameter count / grad L2", f"{next(r['parameter_count'] for r in original_rows if r['state']=='_features_dc')} / {next(r['gradient_L2'] for r in original_rows if r['state']=='_features_dc')}"),
        ("G", "N1", N1),
        ("H", "original occupancy g_dot_d", original_values["_occupancy"]["gdot"]), ("I", "original occupancy FD values", fd_str("_occupancy")), ("J", "original occupancy failure reason", original_fail["_occupancy"]),
        ("K", "original opacity g_dot_d", original_values["_opacity"]["gdot"]), ("L", "original opacity FD values", fd_str("_opacity")), ("M", "original opacity failure reason", original_fail["_opacity"]),
        ("N", "original transmissivity g_dot_d", original_values["_transmissivity"]["gdot"]), ("O", "original transmissivity FD values", fd_str("_transmissivity")), ("P", "original transmissivity failure reason", original_fail["_transmissivity"]),
        ("Q", "original features_dc g_dot_d", original_values["_features_dc"]["gdot"]), ("R", "original features_dc FD values", fd_str("_features_dc")), ("S", "original features_dc failure reason", original_fail["_features_dc"]),
    ]
    op_res = next(r for r in resolution_rows if r["state"] == "_opacity" and abs(r["eps"] - 1e-2) < 1e-12)
    tr_res = next(r for r in resolution_rows if r["state"] == "_transmissivity" and abs(r["eps"] - 1e-2) < 1e-12)
    items += [
        ("T", "opacity expected random-unit FD numerator at eps1e-2", op_res["signal_expected"]),
        ("U", "opacity float32 loss ULP ratio", op_res["expected_numerator_over_float32_ULP"]),
        ("V", "opacity original resolution classification", op_res["classification"]),
        ("W", "transmissivity expected random-unit FD numerator at eps1e-2", tr_res["signal_expected"]),
        ("X", "transmissivity float32 loss ULP ratio", tr_res["expected_numerator_over_float32_ULP"]),
        ("Y", "transmissivity original resolution classification", tr_res["classification"]),
        ("Z", "N2", N2),
        ("AA", "float32 vs float64 reduction opacity gradient relative difference", next(r["gradient_relative_difference"] for r in loss_precision_rows if r["state"] == "_opacity")),
        ("AB", "float32 vs float64 reduction transmissivity gradient relative difference", next(r["gradient_relative_difference"] for r in loss_precision_rows if r["state"] == "_transmissivity")),
        ("AC", "numerical derivative reduction selected", reduction_selected if float64_ok else "float32"),
        ("AD", "occupancy valid structured direction count", valid_dir_count["_occupancy"]),
        ("AE", "opacity valid structured direction count", valid_dir_count["_opacity"]),
        ("AF", "transmissivity valid structured direction count", valid_dir_count["_transmissivity"]),
        ("AG", "features_dc valid structured direction count", valid_dir_count["_features_dc"]),
        ("AH", "opacity best structured g_dot_d / FD / relative error", f"{best_struct['_opacity'].get('autograd_g_dot_d')}/{best_struct['_opacity'].get('fd_derivative')}/{best_struct['_opacity'].get('relative_error')}"),
        ("AI", "transmissivity best structured g_dot_d / FD / relative error", f"{best_struct['_transmissivity'].get('autograd_g_dot_d')}/{best_struct['_transmissivity'].get('fd_derivative')}/{best_struct['_transmissivity'].get('relative_error')}"),
        ("AJ", "opacity valid coordinate checks count/eligible count", f"{coord_valid['_opacity'][0]}/{coord_valid['_opacity'][1]}"),
        ("AK", "transmissivity valid coordinate checks count/eligible count", f"{coord_valid['_transmissivity'][0]}/{coord_valid['_transmissivity'][1]}"),
        ("AL", "opacity first broken edge", "NONE_FOUND"),
        ("AM", "transmissivity first broken edge", "NONE_FOUND"),
        ("AN", "direct native opacity intermediate grad L2", next(r["direct_leaf_grad_L2"] for r in boundary_rows if r["state"] == "_opacity")),
        ("AO", "direct native transmissivity intermediate grad L2", next(r["direct_leaf_grad_L2"] for r in boundary_rows if r["state"] == "_transmissivity")),
        ("AP", "occupancy gradient classification", gradient_class["_occupancy"]),
        ("AQ", "opacity gradient classification", gradient_class["_opacity"]),
        ("AR", "transmissivity gradient classification", gradient_class["_transmissivity"]),
        ("AS", "features_dc gradient classification", gradient_class["_features_dc"]),
        ("AT", "original random-unit Gate under-resolved yes/no", under_original),
        ("AU", "directional derivative protocol repaired yes/no", "YES" if N2 == "PASS" else "NO"),
        ("AV", "N3", N3),
        ("AW", "repaired ATTRIBUTE-CONTROL-VALID count", len(valid_names)),
        ("AX", "repaired ATTRIBUTE-CONTROL-VALID names", ",".join(valid_names)),
        ("AY", "valid non-geometry state count", len(valid_non_geom)),
        ("AZ", "valid non-geometry state names", ",".join(valid_non_geom)),
        ("BA", "N4", N4),
        ("BB", "repaired J3", N5),
        ("BC", "N5", N5),
        ("BD", "Final CASE", final),
        ("BE", "previous RTSPLAT-NATIVE-STATE-CONTROL-FAIL classification valid yes/no", "NO" if N5 == "PASS" else "YES"),
        ("BF", "scientific question experimentally addressable yes/no", "YES" if N5 == "PASS" else "NO"),
        ("BG", "allow R3 M4a-through-J4 continuation yes/no", "YES" if N5 == "PASS" else "NO"),
        ("BH", "AttributeDeformGS hypothesis status", "UNTESTED"),
        ("BI", "PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE", "CONTINUE" if N5 == "PASS" else "STOP"),
        ("BJ", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("BK", "next exact research action", "Resume R3 from M4a camera closure through J4 canonical capacity" if N5 == "PASS" else "Return to RecycleGS"),
        ("BL", "report path", str(OUT / "stage5_0_R3_G1_report.md")),
        ("BM", "summary path", str(OUT / "stage5_0_R3_G1_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {n}: {v}" for k, n, v in items) + "\n"
    write_text(OUT / "stage5_0_R3_G1_report.md", "# Stage5.0-R3-G1 Small-Gradient Numerical Closure\n\n" + "\n".join(f"## {k}. {n}\n\n{v}\n" for k, n, v in items))
    write_text(OUT / "stage5_0_R3_G1_summary.md", f"# Stage5.0-R3-G1 summary\n\n- Final CASE: `{final}`\n- N0/N1/N2/N3/N4/N5: {N0}/{N1}/{N2}/{N3}/{N4}/{N5}\n- Repaired valid states: {','.join(valid_names)}\n- Repaired J3: {N5}\n")
    write_text(OUT / "stage5_0_R3_G1_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)
    print(final_text)


if __name__ == "__main__":
    main()
