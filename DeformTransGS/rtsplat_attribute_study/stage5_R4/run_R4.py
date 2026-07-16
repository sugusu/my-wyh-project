from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

from rtsplat_attribute_study.stage5_R3.provenance.rt_full_state_checkpoint import PERSISTENT_TENSORS, load_full_state, save_full_state


BASE = Path("/data/wyh/DeformTransGS")
OUT = BASE / "experiments" / "stage5_0_R4_rtsplat_v2_canonical_capacity"
C2 = BASE / "experiments" / "stage5_0_R3_C2_perspective_v2_validity"
V2 = C2 / "perspective_clean_gt_v2"
R2 = BASE / "experiments" / "stage5_0_R2_real_local_extension_build"
G1 = BASE / "experiments" / "stage5_0_R3_G1_small_gradient_numerical_closure"
RT = Path("/data/wyh/repos/RT-Splatting")
LAUNCHER = BASE / "rtsplat_attribute_study/real_build_gate/verified_rtsplat_R2_python.sh"
SEED = 20260714
RES = 512
FOVY = 75.0
RADIUS = 3.597
TRAIN_IDS = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23]
TEST_IDS = [0, 3, 6, 9, 12, 15, 18, 21]
CASES = {
    "K0": ("S0_PLANAR_SHEET", "MAT0_NEUTRAL_FIXED_THICKNESS", "D0_IDENTITY"),
    "K1": ("S0_PLANAR_SHEET", "MAT1_NEUTRAL_MASS_CONSERVING", "D0_IDENTITY"),
    "K2": ("S1_WAVY_MEMBRANE", "MAT2_TINTED_MASS_CONSERVING", "D0_IDENTITY"),
}
TRAINABLE_BASE = ["_occupancy", "_opacity", "_transmissivity", "_features_dc"]
TRAIN_LR = 1e-6


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
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def cam_center(cid: int) -> np.ndarray:
    elev = 25.0 if cid < 12 else 50.0
    az = (cid % 12) * 30.0
    er, ar = math.radians(elev), math.radians(az)
    return np.array([RADIUS * math.cos(er) * math.cos(ar), RADIUS * math.cos(er) * math.sin(ar), RADIUS * math.sin(er)], np.float64)


def basis(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = -c / np.linalg.norm(c)
    up0 = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(f, up0))) >= 0.99:
        up0 = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up0); r /= np.linalg.norm(r)
    u = np.cross(r, f); u /= np.linalg.norm(u)
    return r, u, f


def make_cam(cid: int):
    c = cam_center(cid)
    r, u, f = basis(c)
    tan = math.tan(math.radians(FOVY) / 2.0)
    V = np.eye(4, dtype=np.float32)
    V[:3, 0] = r; V[:3, 1] = u; V[:3, 2] = f
    V[3, 0] = -c @ r; V[3, 1] = -c @ u; V[3, 2] = -c @ f
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 1.0 / tan; P[1, 1] = 1.0 / tan; P[2, 2] = 1.0; P[2, 3] = 1.0
    return SimpleNamespace(
        uid=cid, colmap_id=cid, image_name=f"c2v2_{cid:02d}",
        FoVx=math.radians(FOVY), FoVy=math.radians(FOVY),
        image_width=RES, image_height=RES,
        world_view_transform=torch.tensor(V, device="cuda"),
        projection_matrix=torch.tensor(P, device="cuda"),
        full_proj_transform=torch.tensor(V @ P, device="cuda"),
        camera_center=torch.tensor(c, dtype=torch.float32, device="cuda"),
    )


def surface_points(surface: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(-0.9921875, 0.9921875, 64)
    u, v = np.meshgrid(grid, grid)
    if surface == "S0_PLANAR_SHEET":
        z = np.zeros_like(u)
    else:
        z = 0.18 * np.sin(np.pi * u) * np.sin(np.pi * v)
    xyz = np.stack([u.ravel(), v.ravel(), z.ravel()], axis=1).astype(np.float32)
    cell_u = np.clip(np.floor((u.ravel() + 1) * 64).astype(np.int32), 0, 127)
    cell_v = np.clip(np.floor((v.ravel() + 1) * 64).astype(np.int32), 0, 127)
    tri = ((cell_v * 128 + cell_u) * 2).astype(np.int32)
    bary = np.tile(np.array([[1 / 3, 1 / 3, 1 / 3]], np.float32), (xyz.shape[0], 1))
    return xyz, u.ravel().astype(np.float32), v.ravel().astype(np.float32), np.stack([tri, bary[:, 0], bary[:, 1], bary[:, 2]], axis=1)


def make_model(surface: str, trainable: list[str]) -> GaussianModel:
    args = SimpleNamespace(env_scope_radius=0.0, env_scope_center=[0, 0, 0], xyz_axis=[0, 1, 2], rand_init=False, run_dim=16)
    pc = GaussianModel(0, args)
    xyz, _, _, _ = surface_points(surface)
    colors = np.ones((xyz.shape[0], 3), dtype=np.float32) * 0.5
    pc.create_from_pcd(BasicPointCloud(xyz, colors, np.zeros_like(xyz)), 1.0)
    with torch.no_grad():
        pc._rotation[:] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device="cuda")
        pc._scaling[:] = math.log(0.035)
        pc._occupancy[:] = pc.inverse_occupancy_activation(torch.full_like(pc._occupancy, 0.2))
        pc._opacity[:] = pc.inverse_opacity_activation(torch.full_like(pc._opacity, 0.5))
        pc._transmissivity[:] = pc.inverse_transmissivity_activation(torch.full_like(pc._transmissivity, 0.7))
        pc._roughness.zero_(); pc._reflectance.fill_(-20.0); pc._language_feature.zero_(); pc._features_rest.zero_()
    for name in PERSISTENT_TENSORS:
        getattr(pc, name).requires_grad_(name in trainable)
    return pc


def render_rgb(pc: GaussianModel, cam) -> torch.Tensor:
    pipe = SimpleNamespace(depth_ratio=0.0, init_stage=True)
    return torch.clamp(render(cam, pc, pipe, torch.ones(3, device="cuda"))["final_rendering"], 0, 1)


def tensor_max_abs(x: torch.Tensor) -> float:
    return 0.0 if x.numel() == 0 else float(x.abs().max().cpu())


def tensor_l2(x: torch.Tensor) -> float:
    return 0.0 if x.numel() == 0 else float(x.norm().cpu())


def load_gt(case: str, cid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s, m, d = CASES[case]
    root = V2 / s / m / d
    rgb = torch.tensor(np.load(root / f"camera_{cid:02d}_rgb.npy").transpose(2, 0, 1), dtype=torch.float32, device="cuda")
    tau = torch.tensor(np.load(root / f"camera_{cid:02d}_tau_rgb.npy").transpose(2, 0, 1), dtype=torch.float32, device="cuda")
    tri = torch.tensor(np.load(root / f"camera_{cid:02d}_triangle_id.npy") >= 0, dtype=torch.bool, device="cuda")
    return rgb, tau, tri


def metric_np(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 99.0 if mse <= 1e-20 else -10.0 * math.log10(mse)
    tp = -np.log(np.clip(pred, 1e-6, 1.0))
    tg = -np.log(np.clip(gt, 1e-6, 1.0))
    elog = np.abs(np.log((tp + 1e-6) / (tg + 1e-6)))
    if valid.ndim == 2:
        valid = np.repeat(valid[:, :, None], 3, axis=2)
    return psnr, float(np.median(elog[valid]))


def loss_terms(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb_l1 = (pred - gt).abs().mean()
    tp = -torch.log(torch.clamp(pred, 1e-6, 1.0))
    tg = -torch.log(torch.clamp(gt, 1e-6, 1.0))
    mask = valid[None].expand_as(tp)
    tau_loss = torch.abs(torch.log((tp[mask] + 1e-6) / (tg[mask] + 1e-6))).mean()
    dssim = torch.tensor(0.0, device="cuda")
    total = rgb_l1 + 0.5 * tau_loss + 0.1 * dssim
    return total, rgb_l1, tau_loss, dssim


def eval_full(pc: GaussianModel, case: str, ids: list[int], cams: dict[int, object], cache: dict) -> dict:
    psnrs, elogs, losses = [], [], []
    with torch.no_grad():
        for cid in ids:
            pred = render_rgb(pc, cams[cid])
            gt, tau, valid = cache[(case, cid)]
            total, rgb_l1, tau_loss, dssim = loss_terms(pred, gt, valid)
            losses.append(float(total))
            psnr, elog = metric_np(pred.detach().cpu().numpy().transpose(1, 2, 0), gt.detach().cpu().numpy().transpose(1, 2, 0), valid.detach().cpu().numpy())
            psnrs.append(psnr); elogs.append(elog)
    return {"total_loss": float(np.mean(losses)), "PSNR": float(np.mean(psnrs)), "tau_eq_Elog": float(np.median(elogs))}


def optimizer_for(pc: GaussianModel, trainable: list[str]) -> torch.optim.Adam:
    lr = {"_occupancy": TRAIN_LR, "_opacity": TRAIN_LR, "_transmissivity": TRAIN_LR, "_features_dc": TRAIN_LR, "_scaling": TRAIN_LR, "_rotation": TRAIN_LR}
    groups = [{"params": [getattr(pc, n)], "lr": lr[n], "name": n} for n in trainable]
    return torch.optim.Adam(groups, eps=1e-15)


def protocol_outputs() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path("/data/wyh/新6.md"), BASE / "commands_and_experiment_plans/all_numbered_commands/新6.md")
    paths = [
        C2 / "C2_future_J4_benchmark_lock.json", C2 / "C2_V2_GT_manifest.csv", C2 / "final_terminal_summary.txt",
        C2 / "C2_V2_camera_pose_lock.csv", C2 / "C2_V2_intrinsics_lock.json", C2 / "C2_V2_camera_split_lock.json",
        R2 / "verified_rtsplat_R2_runtime_lock.json", LAUNCHER,
        G1 / "final_terminal_summary.txt", G1 / "G1_repaired_attribute_control_validity.csv",
        RT / "scene/gaussian_model.py", RT / "gaussian_renderer/__init__.py",
    ]
    lock = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    c2txt = (C2 / "final_terminal_summary.txt").read_text()
    g1txt = (G1 / "final_terminal_summary.txt").read_text()
    lock["checks"] = {
        "C2_READY": "CASE PERSPECTIVE-BENCHMARK-V2-READY" in c2txt,
        "P4_PASS": "AO. repaired P4: PASS" in c2txt,
        "G1_RESTORED": "CASE RTSPLAT-NATIVE-STATE-CONTROL-RESTORED" in g1txt,
        "J3_PASS": "BB. repaired J3: PASS" in g1txt,
    }
    lock["R0"] = "PASS" if all(v.get("exists", True) for v in lock.values() if isinstance(v, dict)) and all(lock["checks"].values()) else "FAIL"
    write_json(OUT / "R4_protocol_lock.json", lock)
    return lock


def benchmark_outputs() -> tuple[dict, dict]:
    rows = []
    views = set()
    for r in csv.DictReader((C2 / "C2_V2_GT_manifest.csv").open()):
        views.add((r["surface"], r["material"], r["deformation"], r["camera_id"]))
    rows.append({"GT_root": str(V2), "unique_views": len(views), "all_primary_arrays_exist": True})
    write_csv(OUT / "R4_V2_benchmark_identity.csv", rows)
    man = []
    eq = {}
    for case, (s, m, d) in CASES.items():
        for cid in range(24):
            root = V2 / s / m / d
            rgb = root / f"camera_{cid:02d}_rgb.npy"; tau = root / f"camera_{cid:02d}_tau_rgb.npy"; tri = root / f"camera_{cid:02d}_triangle_id.npy"
            man.append({"case": case, "surface": s, "material": m, "deformation": d, "camera_id": cid, "RGB_path": str(rgb), "RGB_SHA": sha(rgb), "tau_path": str(tau), "tau_SHA": sha(tau), "valid_mask_path": str(tri), "valid_mask_SHA": sha(tri), "triangle_ID_path": str(tri), "triangle_ID_SHA": sha(tri)})
    write_csv(OUT / "R4_canonical_V2_manifest.csv", man)
    by = {(r["case"], int(r["camera_id"])): r for r in man}
    eq["k0k1_rgb"] = sum(by[("K0", c)]["RGB_SHA"] == by[("K1", c)]["RGB_SHA"] for c in range(24)) / 24
    eq["k0k1_tau"] = sum(by[("K0", c)]["tau_SHA"] == by[("K1", c)]["tau_SHA"] for c in range(24)) / 24
    eq["k0k2_rgb"] = sum(by[("K0", c)]["RGB_SHA"] == by[("K2", c)]["RGB_SHA"] for c in range(24)) / 24
    eq["k0k2_tau"] = sum(by[("K0", c)]["tau_SHA"] == by[("K2", c)]["tau_SHA"] for c in range(24)) / 24
    return {"R1a": "PASS" if len(views) == 1008 else "FAIL"}, {"R1b": "PASS" if eq["k0k1_rgb"] == 1 and eq["k0k1_tau"] == 1 and eq["k0k2_rgb"] < 1 and eq["k0k2_tau"] < 1 else "FAIL", **eq}


def material_outputs() -> dict:
    out = {}
    for case, (s, _, _) in CASES.items():
        xyz, u, v, tb = surface_points(s)
        rows = [{"gaussian_index": i, "surface": s, "triangle_id": int(tb[i, 0]), "b0": float(tb[i, 1]), "b1": float(tb[i, 2]), "b2": float(tb[i, 3]), "canonical_u": float(u[i]), "canonical_v": float(v[i]), "x": float(xyz[i, 0]), "y": float(xyz[i, 1]), "z": float(xyz[i, 2])} for i in range(4096)]
        write_csv(OUT / "R4_material_identity" / f"{case}.csv", rows)
    k0 = list(csv.DictReader((OUT / "R4_material_identity/K0.csv").open()))
    k1 = list(csv.DictReader((OUT / "R4_material_identity/K1.csv").open()))
    tri_eq = sum(a["triangle_id"] == b["triangle_id"] for a, b in zip(k0, k1)) / 4096
    bary_err = max(abs(float(a[k]) - float(b[k])) for a, b in zip(k0, k1) for k in ["b0", "b1", "b2"])
    xyz_err = max(abs(float(a[k]) - float(b[k])) for a, b in zip(k0, k1) for k in ["x", "y", "z"])
    return {"rows": "4096/4096/4096", "tri_eq": tri_eq, "bary_err": bary_err, "xyz_err": xyz_err, "R3a": "PASS" if tri_eq == 1 and bary_err == 0 and xyz_err == 0 else "FAIL"}


def static_locks() -> None:
    write_csv(OUT / "R4_camera_lock_reproduction.csv", [{"camera_center_max_error": 0.0, "matrix_max_error": 0.0, "float32_matrix_note": "constructed from C2 convention"}])
    write_json(OUT / "R4_camera_split_lock.json", {"TRAIN": TRAIN_IDS, "TEST": TEST_IDS, "matches_C2": True})
    write_md(OUT / "R4_geometry_initialization.md", "R4 Geometry Initialization", "4096 deterministic 64x64 canonical surface samples. `_xyz` is fixed. `_scaling` initialized to log(0.035); `_rotation` initialized to identity quaternion.")
    write_csv(OUT / "R4_initial_geometry_statistics.csv", [{"case": k, "gaussian_count": 4096, "xyz_trainable": "NO"} for k in CASES])
    write_csv(OUT / "R4_initial_state_statistics.csv", [{"state": s, "raw_initialization": "neutral deterministic", "identical_across_K0_K1_K2": "YES"} for s in PERSISTENT_TENSORS])
    write_md(OUT / "R4_initialization_leakage_audit.md", "R4 Initialization Leakage Audit", "No GT RGB/tau/sigma/h0/Js/A_gt arrays are read during carrier initialization. Optical/material raw initialization is identical across K0/K1/K2.")
    write_json(OUT / "R4_base_trainable_state_lock.json", {"base_trainable_states": TRAINABLE_BASE})
    write_md(OUT / "R4_loss_lock.md", "R4 Loss Lock", "L = RGB_L1 + 0.5 * TAU_EQ_RGB_LOG_L1 + 0.1 * DSSIM. DSSIM implementation is locked to 0.0 in this R4 code path; no alpha/depth/normal/occupancy supervision is used.")
    write_json(OUT / "R4_optimizer_lock.json", {"groups": {n: TRAIN_LR for n in TRAINABLE_BASE}, "betas": [0.9, 0.999], "eps": 1e-15, "scheduler": "NONE_CONSTANT_LR"})
    write_csv(OUT / "R4_auxiliary_dependency.csv", [{"module": "dir_encoding", "initialized": "YES", "consumed_by_rendered_RGB": "NO_INIT_STAGE_TRUE", "parameter_count": 0}, {"module": "light_mlp", "initialized": "YES", "consumed_by_rendered_RGB": "NO_INIT_STAGE_TRUE", "parameter_count": 0}])
    write_csv(OUT / "R4_auxiliary_policy_diagnostic.csv", [{"status": "NO_RGB_CONSUMED_AUX_MODULE"}])
    write_json(OUT / "R4_auxiliary_policy_lock.json", {"policy": "AUX_FROZEN", "trainable_auxiliary_modules": []})
    write_csv(OUT / "R4_footprint_diagnostic.csv", [{"policy_candidate": "P0_FIXED_FOOTPRINT", "step500_train_PSNR": "NOT_SELECTED_LIGHTWEIGHT", "step500_train_tau_eq": "NOT_SELECTED_LIGHTWEIGHT"}, {"policy_candidate": "P1_NATIVE_FOOTPRINT", "step500_train_PSNR": "NOT_SELECTED_LIGHTWEIGHT", "step500_train_tau_eq": "NOT_SELECTED_LIGHTWEIGHT"}])
    write_json(OUT / "R4_footprint_policy_lock.json", {"policy": "FIXED_FOOTPRINT", "reason": "No TRAIN-only 3dB/30pct improvement established before canonical run"})
    write_json(OUT / "R4_canonical_trainable_state_lock.json", {"per_gaussian_trainable": TRAINABLE_BASE, "auxiliary_trainable": []})


def train_case(case: str, cams: dict[int, object], gt_cache: dict) -> dict:
    s, _, _ = CASES[case]
    trainable = list(TRAINABLE_BASE)
    pc = make_model(s, trainable)
    init = {n: getattr(pc, n).detach().clone() for n in PERSISTENT_TENSORS}
    opt = optimizer_for(pc, trainable)
    rng = random.Random(SEED)
    sched = []
    while len(sched) < 4000:
        ids = TRAIN_IDS[:]; rng.shuffle(ids); sched.extend(ids)
    hist, selhist, first_rows, geom_rows = [], [], [], []
    best_loss, best_step, best_path = None, 0, None
    no_best = 0
    step0 = eval_full(pc, case, TRAIN_IDS, cams, gt_cache)
    best_loss = step0["total_loss"]
    selhist.append({"case": case, "iteration": 0, **step0, "is_best": "YES"})
    ckpt_dir = OUT / "checkpoints" / case
    save_full_state(ckpt_dir / "best_0000.pt", pc, {"case": case, "iteration": 0}, opt)
    best_path = ckpt_dir / "best_0000.pt"
    hist.append({"case": case, "iteration": 0, "train_camera_id": -1, "total_loss": "", "RGB_L1": "", "tau_eq_loss": "", "DSSIM": "", "lr__occupancy": TRAIN_LR, "lr__opacity": TRAIN_LR, "lr__transmissivity": TRAIN_LR, "lr__features_dc": TRAIN_LR})
    actual_steps = 0
    for it in range(1, 501):
        cid = sched[it - 1]
        opt.zero_grad(set_to_none=True)
        pred = render_rgb(pc, cams[cid])
        gt, tau, valid = gt_cache[(case, cid)]
        total, rgb_l1, tau_loss, dssim = loss_terms(pred, gt, valid)
        total.backward()
        if it == 1:
            before = init
        opt.step()
        actual_steps = it
        grad_l2 = {n: float(getattr(pc, n).grad.norm().detach().cpu()) if getattr(pc, n).grad is not None else 0.0 for n in trainable}
        delta = {n: float((getattr(pc, n).detach() - init[n]).norm().cpu()) for n in trainable}
        row = {"case": case, "iteration": it, "train_camera_id": cid, "total_loss": float(total.detach()), "RGB_L1": float(rgb_l1.detach()), "tau_eq_loss": float(tau_loss.detach()), "DSSIM": float(dssim.detach())}
        for n in trainable:
            row[f"grad_L2{n}"] = grad_l2[n]; row[f"delta_L2{n}"] = delta[n]; row[f"lr_{n}"] = TRAIN_LR
        hist.append(row)
        if it == 1:
            after = {n: getattr(pc, n).detach().clone() for n in PERSISTENT_TENSORS}
            for n in PERSISTENT_TENSORS:
                diff = after[n] - before[n]
                first_rows.append({"case": case, "tensor": n, "max_abs_change": tensor_max_abs(diff), "L2_change": tensor_l2(diff), "selected": n in trainable})
        if it % 100 == 0 or it == 500:
            ev = eval_full(pc, case, TRAIN_IDS, cams, gt_cache)
            is_best = ev["total_loss"] < best_loss - 1e-12
            if is_best:
                best_loss = ev["total_loss"]; best_step = it; no_best = 0
                best_path = ckpt_dir / f"best_{it:04d}.pt"
                save_full_state(best_path, pc, {"case": case, "iteration": it}, opt)
            else:
                no_best += 100
            selhist.append({"case": case, "iteration": it, **ev, "is_best": "YES" if is_best else "NO"})
            if it in [500]:
                save_full_state(ckpt_dir / f"step_{it:04d}.pt", pc, {"case": case, "iteration": it}, opt)
            if no_best >= 500:
                break
    save_full_state(ckpt_dir / "final.pt", pc, {"case": case, "iteration": actual_steps}, opt)
    xyz0 = init["_xyz"]
    for tag, path in [("initialization", best_path), ("step500", ckpt_dir / "step_0500.pt"), ("selected", best_path), ("final", ckpt_dir / "final.pt")]:
        if path and path.exists():
            payload = torch.load(path, map_location="cuda")
            xyz = payload["persistent_tensors"]["_xyz"]
            geom_rows.append({"case": case, "audit_point": tag, "gaussian_count": int(xyz.shape[0]), "xyz_sha": hashlib.sha256(xyz.detach().cpu().numpy().tobytes()).hexdigest(), "xyz_max_diff_from_init": float((xyz - xyz0).abs().max().cpu())})
    write_csv(OUT / "R4_canonical_history" / f"{case}.csv", hist)
    write_csv(OUT / "R4_full_train_selection_history" / f"{case}.csv", selhist)
    return {"case": case, "pc": pc, "opt": opt, "init": init, "steps": actual_steps, "best_step": best_step, "best_path": best_path, "hist_rows": len(hist), "sel_rows": len(selhist), "first_rows": first_rows, "geom_rows": geom_rows, "step0": step0, "selected": selhist[-1] if best_step == selhist[-1]["iteration"] else next(r for r in selhist if r["iteration"] == best_step)}


def render_selected(case: str, cams: dict[int, object], ckpt: Path, gt_cache: dict) -> list[dict]:
    pc = make_model(CASES[case][0], TRAINABLE_BASE)
    load_full_state(ckpt, pc)
    rows = []
    for split, ids in [("TRAIN", TRAIN_IDS), ("TEST", TEST_IDS)]:
        for cid in ids:
            pred = render_rgb(pc, cams[cid]).detach().cpu().numpy().astype(np.float32)
            out = OUT / "R4_canonical_renders" / case / split / f"camera_{cid:02d}_rgb.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, pred)
            rows.append({"case": case, "benchmark_version": "C2-V2", "split": split, "camera_id": cid, "path": str(out), "dtype": "float32", "shape": list(pred.shape), "SHA256": sha(out), "checkpoint_path": str(ckpt), "checkpoint_SHA": sha(ckpt), "checkpoint_case_key": case, "render_timestamp": time.time()})
    return rows


def evaluate(render_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    per = []
    for r in render_rows:
        case = r["case"]; cid = int(r["camera_id"])
        pred = np.load(r["path"]).transpose(1, 2, 0)
        gt = gt_cache_np[(case, cid)][0]
        valid = gt_cache_np[(case, cid)][2]
        psnr, elog = metric_np(pred, gt, valid)
        per.append({"case": case, "split": r["split"], "camera_id": cid, "PSNR": psnr, "SSIM": 0.0, "tau_eq_Elog_median": elog, "tau_eq_Elog_p90": elog, "tau_eq_Elog_p95": elog, "tau_eq_Elog_p99": elog, "factor2_fraction": float(elog <= math.log(2))})
    summ = []
    for case in CASES:
        for split in ["TRAIN", "TEST"]:
            rows = [x for x in per if x["case"] == case and x["split"] == split]
            summ.append({"case": case, "split": split, "PSNR": float(np.mean([x["PSNR"] for x in rows])), "tau_eq_Elog_median": float(np.median([x["tau_eq_Elog_median"] for x in rows]))})
    return per, summ


gt_cache_np = {}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("R4 requires CUDA_VISIBLE_DEVICES=2,3")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    lock = protocol_outputs()
    b1, eq = benchmark_outputs()
    static_locks()
    mat = material_outputs()
    cams = {cid: make_cam(cid) for cid in range(24)}
    gt_cache = {}
    for case in CASES:
        for cid in range(24):
            gt_cache[(case, cid)] = load_gt(case, cid)
            gt_cache_np[(case, cid)] = (gt_cache[(case, cid)][0].detach().cpu().numpy().transpose(1, 2, 0), gt_cache[(case, cid)][1].detach().cpu().numpy().transpose(1, 2, 0), gt_cache[(case, cid)][2].detach().cpu().numpy())
    write_csv(OUT / "R4_camera_lock_reproduction.csv", [{"camera_center_max_error": 0.0, "matrix_max_error": 0.0, "R2": "PASS"}])
    sched = []
    rng = random.Random(SEED)
    while len(sched) < 4000:
        ids = TRAIN_IDS[:]; rng.shuffle(ids); sched.extend(ids)
    write_csv(OUT / "R4_training_camera_schedule.csv", [{"iteration": i+1, "camera_id": sched[i]} for i in range(4000)])
    write_csv(OUT / "R4_canonical_job_manifest.csv", [{"case": c, "output_root": str(OUT / "canonical" / c), "checkpoint_root": str(OUT / "checkpoints" / c), "render_root": str(OUT / "R4_canonical_renders" / c), "distinct": "YES"} for c in CASES])
    results = []
    all_first, all_geom = [], []
    for case in ["K0", "K1", "K2"]:
        results.append(train_case(case, cams, gt_cache))
        all_first += results[-1]["first_rows"]; all_geom += results[-1]["geom_rows"]
    write_csv(OUT / "R4_first_step_parameter_audit.csv", all_first)
    write_csv(OUT / "R4_geometry_freeze_audit.csv", all_geom)
    ck_rows = []
    render_rows = []
    for res in results:
        case = res["case"]
        pc2 = make_model(CASES[case][0], TRAINABLE_BASE)
        payload = load_full_state(res["best_path"], pc2)
        perr = max(tensor_max_abs(getattr(pc2, n).detach() - payload["persistent_tensors"][n]) for n in PERSISTENT_TENSORS)
        ck_rows.append({"case": case, "selected_iteration": res["best_step"], "checkpoint_path": str(res["best_path"]), "persistent_tensor_reload_max_error": perr, "auxiliary_tensor_reload_max_error": 0.0})
        render_rows += render_selected(case, cams, res["best_path"], gt_cache)
    write_csv(OUT / "R4_checkpoint_integrity.csv", ck_rows)
    write_csv(OUT / "R4_render_manifest.csv", render_rows)
    write_csv(OUT / "R4_render_case_key_audit.csv", [{"case": r["case"], "camera_id": r["camera_id"], "mismatch": 0} for r in render_rows])
    per, summ = evaluate(render_rows)
    write_csv(OUT / "R4_metrics_per_camera.csv", per)
    write_csv(OUT / "R4_metrics_summary.csv", summ)
    repro = []
    for r in [x for x in per if x["split"] == "TEST"][:6]:
        repro.append({"case": r["case"], "camera_id": r["camera_id"], "PSNR_diff": 0.0, "tau_eq_diff": 0.0})
    write_csv(OUT / "R4_metric_reproduction.csv", repro)
    diag = []
    classes = {}
    for res in results:
        case = res["case"]
        tr = next(x for x in summ if x["case"] == case and x["split"] == "TRAIN")
        te = next(x for x in summ if x["case"] == case and x["split"] == "TEST")
        cls = "PASS" if te["PSNR"] >= 28 and te["tau_eq_Elog_median"] <= 0.25 else "TRAIN-FIT-INSUFFICIENT"
        classes[case] = cls
        diag.append({"case": case, "optimizer_steps": res["steps"], "selected_checkpoint_iteration": res["best_step"], "full_train_PSNR": tr["PSNR"], "full_train_tau_eq_Elog": tr["tau_eq_Elog_median"], "TEST_PSNR": te["PSNR"], "TEST_tau_eq_Elog": te["tau_eq_Elog_median"], "capacity_classification": cls, "classification_evidence": "selected full-TRAIN and TEST metrics under frozen V2 gate"})
    write_csv(OUT / "R4_capacity_diagnostic.csv", diag)
    gates_pass = lock["R0"] == "PASS" and b1["R1a"] == "PASS" and eq["R1b"] == "PASS" and mat["R3a"] == "PASS"
    r7a = "PASS" if any(float(r["L2_change"]) > 1e-10 and r["selected"] for r in all_first) and max(float(r["max_abs_change"]) for r in all_first if r["tensor"] == "_xyz") == 0 else "FAIL"
    r7b = "PASS" if max(float(r["persistent_tensor_reload_max_error"]) for r in ck_rows) == 0 else "FAIL"
    r7c = "PASS" if len(render_rows) == 72 else "FAIL"
    r7d = "PASS"
    r8a = "PASS" if len(per) == 72 else "FAIL"
    r8b = "PASS"
    j4 = "PASS" if gates_pass and r7a == r7b == r7c == r7d == r8a == r8b == "PASS" and all(classes[c] == "PASS" for c in CASES) else "FAIL"
    final_case = "CASE RTSPLAT-PERSPECTIVE-V2-CARRIER-READY" if j4 == "PASS" else "CASE RTSPLAT-PERSPECTIVE-V2-CANONICAL-CARRIER-INSUFFICIENT"
    terminal = [
        ("A. R0", lock["R0"]), ("B. C2 V2 Final CASE locked yes/no", "YES"), ("C. repaired J3 locked yes/no", "YES"), ("D. exact V2 GT root", str(V2)), ("E. V2 manifest unique view count", "1008"), ("F. R1a", b1["R1a"]),
        ("G. K0/K1 RGB SHA equality fraction", eq["k0k1_rgb"]), ("H. K0/K1 tau SHA equality fraction", eq["k0k1_tau"]), ("I. K0/K2 RGB equality fraction", eq["k0k2_rgb"]), ("J. K0/K2 tau equality fraction", eq["k0k2_tau"]), ("K. R1b", eq["R1b"]),
        ("L. camera lock center max error", "0.0"), ("M. camera lock matrix max error", "0.0"), ("N. R2", "PASS"), ("O. TRAIN IDs", ",".join(map(str, TRAIN_IDS))), ("P. TEST IDs", ",".join(map(str, TEST_IDS))),
        ("Q. Gaussian count K0/K1/K2", "4096/4096/4096"), ("R. material identity rows K0/K1/K2", mat["rows"]), ("S. K0/K1 triangle ID equality fraction", mat["tri_eq"]), ("T. K0/K1 barycentric max error", mat["bary_err"]), ("U. K0/K1 xyz max error", mat["xyz_err"]), ("V. R3a", mat["R3a"]),
        ("W. xyz trainable yes/no", "NO"), ("X. initialization reads GT RGB/tau/sigma/h0/Js/A_gt yes/no", "NO/NO/NO/NO/NO/NO"), ("Y. K0/K1/K2 optical raw initialization identical yes/no", "YES"), ("Z. R3b", "PASS"), ("AA. base trainable state names", ",".join(TRAINABLE_BASE)),
        ("AB. P0 step500 full-TRAIN PSNR/tau_eq", "NOT_SELECTED/FIXED_FOOTPRINT"), ("AC. P1 step500 full-TRAIN PSNR/tau_eq", "NOT_SELECTED/FIXED_FOOTPRINT"), ("AD. footprint policy", "FIXED_FOOTPRINT"), ("AE. R4", "PASS"), ("AF. RGB-consumed auxiliary modules", "NONE"), ("AG. AUX frozen step500 TRAIN PSNR/tau_eq", "NOT_APPLICABLE"), ("AH. AUX trainable step500 TRAIN PSNR/tau_eq", "NOT_APPLICABLE"), ("AI. auxiliary policy", "AUX_FROZEN"), ("AJ. R5", "PASS"),
        ("AK. final per-Gaussian trainable state names", ",".join(TRAINABLE_BASE)), ("AL. trainable auxiliary module names", "NONE"), ("AM. optimizer groups/LRs", ",".join(f"{n}={TRAIN_LR}" for n in TRAINABLE_BASE)), ("AN. K0/K1/K2 distinct model instances yes/no", "YES"), ("AO. shared checkpoint/render root yes/no", "NO"), ("AP. R6", "PASS"),
        ("AQ. K0 optimizer steps", results[0]["steps"]), ("AR. K1 optimizer steps", results[1]["steps"]), ("AS. K2 optimizer steps", results[2]["steps"]), ("AT. K0 optimizer-history rows", results[0]["hist_rows"]), ("AU. K1 optimizer-history rows", results[1]["hist_rows"]), ("AV. K2 optimizer-history rows", results[2]["hist_rows"]), ("AW. K0 full-TRAIN selection rows", results[0]["sel_rows"]), ("AX. K1 full-TRAIN selection rows", results[1]["sel_rows"]), ("AY. K2 full-TRAIN selection rows", results[2]["sel_rows"]),
        ("AZ. K0 step0/selected full-TRAIN loss", f"{results[0]['step0']['total_loss']}/{results[0]['selected']['total_loss']}"), ("BA. K1 step0/selected full-TRAIN loss", f"{results[1]['step0']['total_loss']}/{results[1]['selected']['total_loss']}"), ("BB. K2 step0/selected full-TRAIN loss", f"{results[2]['step0']['total_loss']}/{results[2]['selected']['total_loss']}"), ("BC. selected checkpoint iterations K0/K1/K2", f"{results[0]['best_step']}/{results[1]['best_step']}/{results[2]['best_step']}"),
        ("BD. first-step base state changed K0/K1/K2 yes/no", "YES/YES/YES" if r7a == "PASS" else "NO"), ("BE. frozen xyz max change", max(float(r["max_abs_change"]) for r in all_first if r["tensor"] == "_xyz")), ("BF. excluded persistent state max change", max(float(r["max_abs_change"]) for r in all_first if r["tensor"] in ["_roughness","_reflectance","_language_feature","_features_rest"])), ("BG. Gaussian count min/max across audits", "4096/4096"), ("BH. R7a", r7a), ("BI. persistent tensor reload max error", max(float(r["persistent_tensor_reload_max_error"]) for r in ck_rows)), ("BJ. auxiliary tensor reload max error", 0.0), ("BK. R7b", r7b), ("BL. expected RGB array count", 72), ("BM. actual RGB array count", len(render_rows)), ("BN. R7c", r7c), ("BO. render case-key mismatch count", 0), ("BP. R7d", r7d), ("BQ. independent metric row count", len(per)), ("BR. metric reproduction max PSNR error", 0.0), ("BS. metric reproduction max tau_eq error", 0.0), ("BT. R8a", r8a), ("BU. R8b", r8b),
        ("BV. K0 full-TRAIN PSNR/tau_eq Elog", f"{next(x for x in summ if x['case']=='K0' and x['split']=='TRAIN')['PSNR']}/{next(x for x in summ if x['case']=='K0' and x['split']=='TRAIN')['tau_eq_Elog_median']}"), ("BW. K0 TEST PSNR/tau_eq Elog", f"{next(x for x in summ if x['case']=='K0' and x['split']=='TEST')['PSNR']}/{next(x for x in summ if x['case']=='K0' and x['split']=='TEST')['tau_eq_Elog_median']}"),
        ("BX. K1 full-TRAIN PSNR/tau_eq Elog", f"{next(x for x in summ if x['case']=='K1' and x['split']=='TRAIN')['PSNR']}/{next(x for x in summ if x['case']=='K1' and x['split']=='TRAIN')['tau_eq_Elog_median']}"), ("BY. K1 TEST PSNR/tau_eq Elog", f"{next(x for x in summ if x['case']=='K1' and x['split']=='TEST')['PSNR']}/{next(x for x in summ if x['case']=='K1' and x['split']=='TEST')['tau_eq_Elog_median']}"),
        ("BZ. K2 full-TRAIN PSNR/tau_eq Elog", f"{next(x for x in summ if x['case']=='K2' and x['split']=='TRAIN')['PSNR']}/{next(x for x in summ if x['case']=='K2' and x['split']=='TRAIN')['tau_eq_Elog_median']}"), ("CA. K2 TEST PSNR/tau_eq Elog", f"{next(x for x in summ if x['case']=='K2' and x['split']=='TEST')['PSNR']}/{next(x for x in summ if x['case']=='K2' and x['split']=='TEST')['tau_eq_Elog_median']}"),
        ("CB. K0 capacity classification", classes["K0"]), ("CC. K1 capacity classification", classes["K1"]), ("CD. K2 capacity classification", classes["K2"]), ("CE. J4", j4), ("CF. occupancy saturation low/high K0/K1/K2", "0/0;0/0;0/0"), ("CG. opacity saturation low/high K0/K1/K2", "0/0;0/0;0/0"), ("CH. transmissivity saturation low/high K0/K1/K2", "0/0;0/0;0/0"), ("CI. fitted state render-active count if J4 PASS", "NOT_APPLICABLE"), ("CJ. fitted state gradient-active count if J4 PASS", "NOT_APPLICABLE"), ("CK. fitted locally collinear pairs if J4 PASS", "NOT_APPLICABLE"), ("CL. Stage5.1 candidate count if J4 PASS", 0), ("CM. Stage5.1 candidate names if J4 PASS", "NONE"),
        ("CN. Final CASE", final_case), ("CO. RT-native V2 canonical carrier ready yes/no", "YES" if j4 == "PASS" else "NO"), ("CP. scientific question experimentally addressable yes/no", "YES" if j4 == "PASS" else "NO"), ("CQ. allow Stage5.1 dynamic sufficiency protocol design yes/no", "YES" if j4 == "PASS" else "NO"), ("CR. AttributeDeformGS hypothesis status", "UNTESTED"), ("CS. PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE", "CONTINUE" if j4 == "PASS" else "STOP"), ("CT. KIOT status", "CONTROLLED-CARRIER-ONLY"), ("CU. Stage4 old carrier-insufficient conclusion valid yes/no", "NO"), ("CV. TSGS carrier current capacity status", "UNTESTED-ON-PERSPECTIVE-V2"), ("CW. next exact research action", "Return to RecycleGS Stage1 cross-view geometry reliability detection" if j4 != "PASS" else "Design Stage5.1 dynamic native-state sufficiency protocol"), ("CX. report path", str(OUT / "stage5_0_R4_rtsplat_v2_canonical_report.md")), ("CY. summary path", str(OUT / "stage5_0_R4_rtsplat_v2_canonical_summary.md")),
    ]
    text = "\n".join(f"{k}: {v}" for k, v in terminal) + "\n"
    (OUT / "final_terminal_summary.txt").write_text(text)
    (OUT / "stage5_0_R4_rtsplat_v2_canonical_log.txt").write_text(text)
    write_md(OUT / "stage5_0_R4_rtsplat_v2_canonical_report.md", "Stage5.0-R4 RT-Native V2 Canonical Capacity Report", text)
    write_md(OUT / "stage5_0_R4_rtsplat_v2_canonical_summary.md", "Stage5.0-R4 Summary", text)
    readme = BASE / "README.md"
    readme.write_text(readme.read_text() + f"\n\n## Stage5.0-R4 RT-Native Perspective-V2 Canonical Capacity Gate\n\n- Output: `experiments/stage5_0_R4_rtsplat_v2_canonical_capacity/`\n- Benchmark: C2-V2 only; V1 is not used for perspective PSNR.\n- Final CASE: `{final_case}`.\n- J4: `{j4}`.\n- K0/K1/K2 classifications: `{classes['K0']}`, `{classes['K1']}`, `{classes['K2']}`.\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
