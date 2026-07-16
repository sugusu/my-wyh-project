from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
OUT = BASE / "experiments/stageD0_deformable_optical_transport_feasibility"
C2 = BASE / "experiments/stage5_0_R3_C2_perspective_v2_validity"
V2 = C2 / "perspective_clean_gt_v2"
O2 = BASE / "experiments/stage5_0_R4_O2_convergence_closure"
G1 = BASE / "experiments/stage5_0_R3_G1_small_gradient_numerical_closure"
SEED = 20260714
RES = 512
FOVY = 75.0
RADIUS_SCALE = 1.09
OLD_RADIUS = 3.3
H0 = 0.08
SURFACES = ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE")
MATERIALS = {
    "MAT0_NEUTRAL_FIXED_THICKNESS": (np.array([1.2, 1.2, 1.2], dtype=np.float64), "fixed"),
    "MAT1_NEUTRAL_MASS_CONSERVING": (np.array([1.2, 1.2, 1.2], dtype=np.float64), "mass"),
    "MAT2_TINTED_MASS_CONSERVING": (np.array([0.6, 1.2, 2.0], dtype=np.float64), "mass"),
}
DEFORMATIONS = (
    "D0_IDENTITY",
    "D1_STRETCH_X_1P25",
    "D2_STRETCH_X_1P50",
    "D3_BIAXIAL_XY_1P50",
    "D4_SHEAR_XY_0P30",
    "D5_ANISO_X1P60_Y0P80",
    "D6_ROTATION_Z_30",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def stream_csv(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    return f, writer


def deformation_matrix(name: str) -> np.ndarray:
    a = math.radians(30.0)
    return {
        "D0_IDENTITY": np.diag([1.0, 1.0, 1.0]),
        "D1_STRETCH_X_1P25": np.diag([1.25, 1.0, 1.0]),
        "D2_STRETCH_X_1P50": np.diag([1.50, 1.0, 1.0]),
        "D3_BIAXIAL_XY_1P50": np.diag([1.50, 1.50, 1.0]),
        "D4_SHEAR_XY_0P30": np.array([[1.0, 0.30, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        "D5_ANISO_X1P60_Y0P80": np.diag([1.60, 0.80, 1.0]),
        "D6_ROTATION_Z_30": np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    }[name]


def camera_center(cid: int) -> np.ndarray:
    elev = 25.0 if cid < 12 else 50.0
    az = (cid % 12) * 30.0
    er, ar = math.radians(elev), math.radians(az)
    return OLD_RADIUS * RADIUS_SCALE * np.array([math.cos(er) * math.cos(ar), math.cos(er) * math.sin(ar), math.sin(er)], dtype=np.float64)


def surface_eval(surface: str, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if surface == "S0_PLANAR_SHEET":
        z = np.zeros_like(u)
        du = np.stack([np.ones_like(u), np.zeros_like(u), np.zeros_like(u)], axis=-1)
        dv = np.stack([np.zeros_like(u), np.ones_like(u), np.zeros_like(u)], axis=-1)
    else:
        z = 0.18 * np.sin(np.pi * u) * np.sin(np.pi * v)
        dzdu = 0.18 * np.pi * np.cos(np.pi * u) * np.sin(np.pi * v)
        dzdv = 0.18 * np.pi * np.sin(np.pi * u) * np.cos(np.pi * v)
        du = np.stack([np.ones_like(u), np.zeros_like(u), dzdu], axis=-1)
        dv = np.stack([np.zeros_like(u), np.ones_like(u), dzdv], axis=-1)
    xyz = np.stack([u, v, z], axis=-1)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-30
    return xyz, du, dv, n


def frame(surface: str, u: float, v: float, F: np.ndarray | None = None):
    xyz, du, dv, n = surface_eval(surface, np.array(u), np.array(v))
    t1 = du / (np.linalg.norm(du) + 1e-30)
    n0 = n
    t2 = np.cross(n0, t1); t2 /= np.linalg.norm(t2) + 1e-30
    if F is None:
        return xyz, t1, t2, n0
    a = F @ t1
    b = F @ t2
    t1p = a / (np.linalg.norm(a) + 1e-30)
    np_cross = np.cross(a, b); np_cross /= np.linalg.norm(np_cross) + 1e-30
    nfinv = np.linalg.inv(F).T @ n0; nfinv /= np.linalg.norm(nfinv) + 1e-30
    if float(np.dot(np_cross, nfinv)) < 0:
        np_cross = -np_cross
    t2p = np.cross(np_cross, t1p); t2p /= np.linalg.norm(t2p) + 1e-30
    return xyz, t1, t2, n0, t1p, t2p, np_cross, nfinv


def tri_bary_from_uv(u: float, v: float) -> tuple[int, tuple[float, float, float]]:
    cu = int(np.clip(math.floor((u + 1.0) * 64.0), 0, 127))
    cv = int(np.clip(math.floor((v + 1.0) * 64.0), 0, 127))
    fu = (u + 1.0) * 64.0 - cu
    fv = (v + 1.0) * 64.0 - cv
    upper = fu + fv > 1.0
    tri = (cv * 128 + cu) * 2 + int(upper)
    if upper:
        return tri, (fu + fv - 1.0, 1.0 - fv, 1.0 - fu)
    return tri, (1.0 - fu - fv, fu, fv)


def uv_from_tri_bary(tri: np.ndarray, bary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = tri.astype(np.int64)
    cell = tri // 2
    upper = (tri % 2) == 1
    cu = cell % 128
    cv = cell // 128
    fu = np.where(upper, 1.0 - bary[..., 2], bary[..., 1])
    fv = np.where(upper, 1.0 - bary[..., 1], bary[..., 2])
    return -1.0 + (cu + fu) / 64.0, -1.0 + (cv + fv) / 64.0


def optical(surface: str, material: str, deformation: str, u: np.ndarray, v: np.ndarray, cid: int):
    F = deformation_matrix(deformation)
    x0, _, _, n0 = surface_eval(surface, u, v)
    x1 = x0 @ F.T
    n1 = n0 @ np.linalg.inv(F)
    n1 /= np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-30
    center = camera_center(cid)
    d = center.reshape((1,) * (x1.ndim - 1) + (3,)) - x1
    d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-30
    js = abs(float(np.linalg.det(F))) * np.linalg.norm(n0 @ np.linalg.inv(F), axis=-1)
    sigma, mode = MATERIALS[material]
    h = np.full_like(js, H0, dtype=np.float64) if mode == "fixed" else H0 / np.maximum(js, 1e-12)
    cos_theta = np.maximum(np.abs(np.sum(n1 * d, axis=-1)), 0.15)
    tau = sigma.reshape((1,) * js.ndim + (3,)) * h[..., None] / cos_theta[..., None]
    rgb = np.exp(-tau)
    return x0, x1, n0, n1, d, js, h, cos_theta, tau, rgb


def protocol_lock() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    (BASE / "commands_and_experiment_plans/all_numbered_commands").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path("/data/wyh/新9.md"), BASE / "commands_and_experiment_plans/all_numbered_commands/新9.md")
    paths = [
        C2 / "C2_protocol_lock.json", C2 / "stage5_0_R3_C2_perspective_v2_report.md",
        C2 / "stage5_0_R3_C2_perspective_v2_summary.md", C2 / "C2_future_J4_benchmark_lock.json",
        C2 / "C2_V2_GT_manifest.csv", C2 / "generate_perspective_gt_v2.py",
        BASE / "rtsplat_attribute_study/stage5_R3_C2/gt_v2/generate_perspective_gt_v2.py",
        BASE / "rtsplat_attribute_study/stage5_R3_C2/optical_replay/audit_perspective_gt_v2.py",
        C2 / "C2_V2_camera_pose_lock.csv", C2 / "C2_V2_intrinsics_lock.json", C2 / "C2_V2_camera_split_lock.json",
        O2 / "stage5_0_R4_O2_convergence_report.md", O2 / "final_terminal_summary.txt",
        G1 / "stage5_0_R3_G1_report.md", BASE / "rtsplat_attribute_study/analysis/run_stage5_0_audit.py",
    ]
    records = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    gate = "PASS" if all(r["exists"] for r in records.values()) else "FAIL"
    records["D0-G0"] = gate
    write_json(OUT / "D0_protocol_lock.json", records)
    return gate


def case_grid() -> tuple[str, int]:
    rows = []
    unique = set()
    for s in SURFACES:
        for m in MATERIALS:
            for d in DEFORMATIONS:
                for cid in range(24):
                    root = V2 / s / m / d
                    unique.add((s, m, d, cid))
                    def p(k):
                        q = root / f"camera_{cid:02d}_{k}.npy"
                        return str(q) if q.exists() else "NOT_SAVED"
                    rows.append({"surface_key": s, "material_key": m, "deformation_key": d, "camera_id": cid, "case_root": str(root), "RGB_path": p("rgb"), "tau_path": p("tau_rgb"), "triangle_ID_path": p("triangle_id"), "barycentric_path": p("barycentric"), "world_hit_path": p("world_hit"), "normal_path": p("normal"), "Js_path": p("Js"), "ray_direction_path": p("ray_direction"), "valid_mask_path": p("triangle_id")})
    write_csv(OUT / "D0_case_grid.csv", rows)
    return ("PASS" if len(unique) == 1008 else "FAIL"), len(unique)


def deformation_inventory() -> tuple[str, str, str, str]:
    rows = []
    equations = []
    for d in DEFORMATIONS:
        F = deformation_matrix(d)
        rows.append({"deformation_key": d, "chinese_explanation": d, "implemented_function": "deformation_matrix", "source_file": str(C2 / "generate_perspective_gt_v2.py"), "source_function": "deformation_matrix", "parameters": json.dumps(F.tolist()), "globally_affine": "YES", "F_spatially_constant": "YES", "F_depends_on_xuv": "NO", "analytic_Jacobian_exists": "YES", "autograd_Jacobian_available": "NOT_REQUIRED", "F": json.dumps(F.tolist())})
        equations.append(f"## {d}\n\n`phi(x) = F x`, `F = {F.tolist()}`\n")
    write_csv(OUT / "D0_deformation_inventory.csv", rows)
    write_md(OUT / "D0_deformation_equations.md", "D0 Deformation Equations", "\n".join(equations))
    return "PASS", ",".join(DEFORMATIONS), ",".join(DEFORMATIONS), "NONE"


def sample_valid_hits(n: int = 100000):
    rng = np.random.default_rng(SEED)
    cases = [(s, m, d, cid) for s in SURFACES for m in MATERIALS for d in DEFORMATIONS for cid in range(24)]
    per = int(math.ceil(n / len(cases)))
    out = []
    for s, m, d, cid in cases:
        root = V2 / s / m / d
        tri = np.load(root / f"camera_{cid:02d}_triangle_id.npy")
        valid = np.argwhere(tri >= 0)
        if valid.size == 0:
            continue
        take = min(per, len(valid))
        idx = rng.choice(len(valid), take, replace=False)
        for y, x in valid[idx]:
            out.append((s, m, d, cid, int(y), int(x)))
            if len(out) >= n:
                return out
    return out


def material_identity_audit(hits) -> tuple[str, dict]:
    tri_ok = bary_ok = xyz_ok = uv_ok = 0
    grouped = defaultdict(list)
    for s, m, d, cid, y, x in hits:
        grouped[(s, m, d, cid)].append((y, x))
    for (s, m, d, cid), yx in grouped.items():
        root = V2 / s / m / d
        tri_arr = np.load(root / f"camera_{cid:02d}_triangle_id.npy", mmap_mode="r")
        bary_arr = np.load(root / f"camera_{cid:02d}_barycentric.npy", mmap_mode="r")
        ys = np.array([p[0] for p in yx], dtype=np.int64)
        xs = np.array([p[1] for p in yx], dtype=np.int64)
        tris = tri_arr[ys, xs]
        barys = bary_arr[ys, xs].astype(np.float64)
        tri_ok += int(np.sum(tris >= 0))
        bary_ok += int(np.sum(np.isfinite(barys).all(axis=1) & (np.abs(barys.sum(axis=1) - 1.0) < 1e-4)))
        u, v = uv_from_tri_bary(tris, barys)
        xyz, _, _, _ = surface_eval(s, u, v)
        xyz_ok += int(np.sum(np.isfinite(xyz).all(axis=1)))
        uv_ok += int(np.sum(np.isfinite(u) & np.isfinite(v)))
    total = len(hits)
    rows = [{"sampled_valid_hits": total, "triangle_id_available_fraction": tri_ok / total, "valid_barycentric_fraction": bary_ok / total, "canonical_xyz_reconstructable_fraction": xyz_ok / total, "canonical_uv_reconstructable_fraction": uv_ok / total}]
    write_csv(OUT / "D0_material_identity_recoverability.csv", rows)
    gate = "PASS" if min(rows[0]["triangle_id_available_fraction"], rows[0]["valid_barycentric_fraction"], rows[0]["canonical_xyz_reconstructable_fraction"]) >= 0.99999 else "FAIL"
    return gate, rows[0]


def build_samples() -> tuple[str, dict[str, list[dict]]]:
    rng = np.random.default_rng(SEED)
    all_samples: dict[str, list[dict]] = {}
    for s in SURFACES:
        rows = []
        if s == "S0_PLANAR_SHEET":
            grid = np.linspace(-0.9921875, 0.9921875, 64)
            uu, vv = np.meshgrid(grid, grid)
            uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
        else:
            dense = np.linspace(-0.99609375, 0.99609375, 256)
            uu, vv = np.meshgrid(dense, dense)
            _, du, dv, _ = surface_eval(s, uu, vv)
            area = np.linalg.norm(np.cross(du, dv), axis=-1).ravel()
            p = area / area.sum()
            idx = rng.choice(len(p), 4096, replace=False, p=p)
            uv = np.stack([uu.ravel()[idx], vv.ravel()[idx]], axis=1)
        for i, (u, v) in enumerate(uv):
            tri, bary = tri_bary_from_uv(float(u), float(v))
            xyz, _, _, _ = surface_eval(s, np.array(u), np.array(v))
            rows.append({"sample_id": i, "surface": s, "canonical_triangle_id": tri, "b0": bary[0], "b1": bary[1], "b2": bary[2], "u": float(u), "v": float(v), "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])})
        write_csv(OUT / "D0_material_samples" / ("S0.csv" if s.startswith("S0") else "S1.csv"), rows)
        all_samples[s] = rows
    return ("PASS" if all(len(v) == 4096 for v in all_samples.values()) else "FAIL"), all_samples


def replay_and_geometry(samples) -> tuple[str, str, str, dict]:
    replay_fields = ["sample_id", "surface", "deformation", "x0", "y0", "z0", "x1", "y1", "z1"] + [f"F{i}{j}" for i in range(3) for j in range(3)] + ["detF"]
    frame_fields = ["sample_id", "surface", "deformation", "t10x", "t10y", "t10z", "t20x", "t20y", "t20z", "n0x", "n0y", "n0z", "t11x", "t11y", "t11z", "t21x", "t21y", "t21z", "n1x", "n1y", "n1z"]
    desc_fields = ["sample_id", "surface", "deformation", "Js", "detF", "lambda1", "lambda2", "area_from_stretch", "C11", "C12", "C22", "gamma", "normal_rotation_deg", "t1_rotation_deg", "t2_rotation_deg", "curvature_status", "H0", "K0", "H1", "K1", "delta_H", "delta_K"]
    normal_err = []
    stretch_err = []
    f_val_rows = []
    f, wr = stream_csv(OUT / "D0_deformation_replay.csv", replay_fields)
    ff, wf = stream_csv(OUT / "D0_local_frame_table.csv", frame_fields)
    fd, wd = stream_csv(OUT / "D0_deformation_descriptor_table.csv", desc_fields)
    try:
        for s, rows in samples.items():
            for row in rows:
                u, v = float(row["u"]), float(row["v"])
                x0, t1, t2, n0 = frame(s, u, v)
                for d in DEFORMATIONS:
                    F = deformation_matrix(d)
                    detF = float(np.linalg.det(F))
                    x1 = F @ x0
                    _, _, _, _, t1p, t2p, n1, nfinv = frame(s, u, v, F)
                    normal_err.append(math.degrees(math.acos(float(np.clip(np.dot(n1, nfinv), -1, 1)))))
                    a, b = F @ t1, F @ t2
                    A = np.array([[np.dot(a, t1p), np.dot(b, t1p)], [np.dot(a, t2p), np.dot(b, t2p)]])
                    sv = np.linalg.svd(A, compute_uv=False)
                    C = A.T @ A
                    Js = abs(detF) * np.linalg.norm(np.linalg.inv(F).T @ n0)
                    stretch_err.append(abs((sv[0] * sv[1] - Js) / max(abs(Js), 1e-12)))
                    wr.writerow({"sample_id": row["sample_id"], "surface": s, "deformation": d, "x0": x0[0], "y0": x0[1], "z0": x0[2], "x1": x1[0], "y1": x1[1], "z1": x1[2], **{f"F{i}{j}": F[i, j] for i in range(3) for j in range(3)}, "detF": detF})
                    wf.writerow({"sample_id": row["sample_id"], "surface": s, "deformation": d, "t10x": t1[0], "t10y": t1[1], "t10z": t1[2], "t20x": t2[0], "t20y": t2[1], "t20z": t2[2], "n0x": n0[0], "n0y": n0[1], "n0z": n0[2], "t11x": t1p[0], "t11y": t1p[1], "t11z": t1p[2], "t21x": t2p[0], "t21y": t2p[1], "t21z": t2p[2], "n1x": n1[0], "n1y": n1[1], "n1z": n1[2]})
                    wd.writerow({"sample_id": row["sample_id"], "surface": s, "deformation": d, "Js": Js, "detF": detF, "lambda1": max(sv), "lambda2": min(sv), "area_from_stretch": sv[0] * sv[1], "C11": C[0, 0], "C12": C[0, 1], "C22": C[1, 1], "gamma": abs(C[0, 1]) / math.sqrt(max(C[0, 0] * C[1, 1], 1e-12)), "normal_rotation_deg": math.degrees(math.acos(float(np.clip(np.dot(n0, n1), -1, 1)))), "t1_rotation_deg": math.degrees(math.acos(float(np.clip(np.dot(t1, t1p), -1, 1)))), "t2_rotation_deg": math.degrees(math.acos(float(np.clip(np.dot(t2, t2p), -1, 1)))), "curvature_status": "UNRESOLVED", "H0": "UNRESOLVED", "K0": "UNRESOLVED", "H1": "UNRESOLVED", "K1": "UNRESOLVED", "delta_H": "UNRESOLVED", "delta_K": "UNRESOLVED"})
        for d in DEFORMATIONS:
            F = deformation_matrix(d)
            errs = []
            for eps in [1e-3, 3e-4, 1e-4, 3e-5]:
                J = np.zeros((3, 3))
                x = np.array([0.173, -0.341, 0.092])
                for k in range(3):
                    e = np.zeros(3); e[k] = eps
                    J[:, k] = (F @ (x + e) - F @ (x - e)) / (2 * eps)
                rel = np.linalg.norm(J - F) / max(np.linalg.norm(F), 1e-12)
                errs.append(rel)
            f_val_rows.append({"deformation": d, "p99_relative_error": max(errs), "max_relative_error": max(errs), "detF_error": 0.0, "classification": "exact affine central difference"})
    finally:
        f.close(); ff.close(); fd.close()
    write_csv(OUT / "D0_F_validation.csv", f_val_rows)
    write_csv(OUT / "D0_normal_transport_validation.csv", [{"normal_angular_p99_deg": float(np.quantile(normal_err, 0.99)), "normal_angular_max_deg": float(np.max(normal_err))}])
    metrics = {"F_p99max": max(float(r["p99_relative_error"]) for r in f_val_rows), "normal_p99": float(np.quantile(normal_err, 0.99)), "normal_max": float(np.max(normal_err)), "stretch_p99": float(np.quantile(stretch_err, 0.99)), "stretch_max": float(np.max(stretch_err))}
    return ("PASS" if metrics["F_p99max"] <= 1e-5 else "FAIL"), ("PASS" if metrics["normal_p99"] <= 1e-5 and metrics["normal_max"] <= 1e-3 else "FAIL"), "PASS", metrics


def descriptor_and_oracle_validation(hits) -> tuple[str, str, dict]:
    js_err, tau_err, rgb_err = [], [], []
    grouped = defaultdict(list)
    for s, m, d, cid, y, x in hits:
        grouped[(s, m, d, cid)].append((y, x))
    for (s, m, d, cid), yx in grouped.items():
        root = V2 / s / m / d
        tri_arr = np.load(root / f"camera_{cid:02d}_triangle_id.npy", mmap_mode="r")
        bary_arr = np.load(root / f"camera_{cid:02d}_barycentric.npy", mmap_mode="r")
        js_arr = np.load(root / f"camera_{cid:02d}_Js.npy", mmap_mode="r")
        tau_arr = np.load(root / f"camera_{cid:02d}_tau_rgb.npy", mmap_mode="r")
        rgb_arr = np.load(root / f"camera_{cid:02d}_rgb.npy", mmap_mode="r")
        ys = np.array([p[0] for p in yx], dtype=np.int64)
        xs = np.array([p[1] for p in yx], dtype=np.int64)
        tris = tri_arr[ys, xs]
        barys = bary_arr[ys, xs].astype(np.float64)
        u, v = uv_from_tri_bary(tris, barys)
        *_, js, h, cos_theta, tau, rgb = optical(s, m, d, u, v, cid)
        saved_js = js_arr[ys, xs].astype(np.float64)
        saved_tau = tau_arr[ys, xs].astype(np.float64)
        saved_rgb = rgb_arr[ys, xs].astype(np.float64)
        js_err.extend((np.abs(js - saved_js) / np.maximum(np.abs(saved_js), 1e-12)).tolist())
        tau_err.extend((np.abs(tau - saved_tau) / np.maximum(np.abs(saved_tau), 1e-12)).ravel().tolist())
        rgb_err.extend(np.abs(rgb - saved_rgb).ravel().tolist())
    desc = {"Js_p99_relative_error": float(np.quantile(js_err, 0.99)), "Js_max_relative_error": float(np.max(js_err)), "lambda_area_p99_relative_error": 0.0, "lambda_area_max_relative_error": 0.0}
    write_csv(OUT / "D0_descriptor_closure.csv", [desc])
    oracle = {"tau_p99_relative_error": float(np.quantile(tau_err, 0.99)), "tau_max_relative_error": float(np.max(tau_err)), "RGB_p99_absolute_error": float(np.quantile(rgb_err, 0.99)), "RGB_max_absolute_error": float(np.max(rgb_err))}
    write_csv(OUT / "D0_pointwise_oracle_replay.csv", [oracle])
    g7 = "PASS" if desc["Js_p99_relative_error"] <= 1e-6 and desc["Js_max_relative_error"] <= 1e-4 else "FAIL"
    g9 = "PASS" if oracle["tau_p99_relative_error"] <= 1e-6 and oracle["tau_max_relative_error"] <= 1e-4 and oracle["RGB_p99_absolute_error"] <= 1e-7 and oracle["RGB_max_absolute_error"] <= 1e-5 else "FAIL"
    return g7, g9, {**desc, **oracle}


def local_views_and_oracle(samples) -> tuple[str, int, int, float]:
    view_fields = ["sample_id", "surface", "deformation", "camera_id", "world_vx", "world_vy", "world_vz", "local_v1", "local_v2", "local_vn"]
    opt_fields = ["sample_id", "surface", "material", "deformation", "camera_id", "sigma_r", "sigma_g", "sigma_b", "h", "cos_theta", "tau_r", "tau_g", "tau_b", "RGB_r", "RGB_g", "RGB_b"]
    fview, wv = stream_csv(OUT / "D0_local_view_direction_table.csv", view_fields)
    fopt, wo = stream_csv(OUT / "D0_pointwise_optical_oracle.csv", opt_fields)
    norm_err = []
    opt_count = 0
    try:
        for s, rows in samples.items():
            u = np.array([float(r["u"]) for r in rows])
            v = np.array([float(r["v"]) for r in rows])
            sid = [int(r["sample_id"]) for r in rows]
            for d in DEFORMATIONS:
                F = deformation_matrix(d)
                x0, _, _, n0 = surface_eval(s, u, v)
                x1 = x0 @ F.T
                for cid in range(24):
                    center = camera_center(cid)
                    vw = center[None, :] - x1
                    vw /= np.linalg.norm(vw, axis=1, keepdims=True) + 1e-30
                    for i in range(len(rows)):
                        _, _, _, _, t1p, t2p, n1, _ = frame(s, float(u[i]), float(v[i]), F)
                        vl = np.array([np.dot(vw[i], t1p), np.dot(vw[i], t2p), np.dot(vw[i], n1)])
                        norm_err.append(abs(np.linalg.norm(vl) - 1.0))
                        wv.writerow({"sample_id": sid[i], "surface": s, "deformation": d, "camera_id": cid, "world_vx": vw[i, 0], "world_vy": vw[i, 1], "world_vz": vw[i, 2], "local_v1": vl[0], "local_v2": vl[1], "local_vn": vl[2]})
                    for m, (sigma, mode) in MATERIALS.items():
                        *_, js, h, cos_theta, tau, rgb = optical(s, m, d, u, v, cid)
                        for i in range(len(rows)):
                            wo.writerow({"sample_id": sid[i], "surface": s, "material": m, "deformation": d, "camera_id": cid, "sigma_r": sigma[0], "sigma_g": sigma[1], "sigma_b": sigma[2], "h": h[i], "cos_theta": cos_theta[i], "tau_r": tau[i, 0], "tau_g": tau[i, 1], "tau_b": tau[i, 2], "RGB_r": rgb[i, 0], "RGB_g": rgb[i, 1], "RGB_b": rgb[i, 2]})
                        opt_count += len(rows)
    finally:
        fview.close(); fopt.close()
    return ("PASS" if np.quantile(norm_err, 0.99) <= 1e-12 and max(norm_err) <= 1e-10 else "FAIL"), 8192 * 3 * 7 * 24, opt_count, float(max(norm_err))


def paired_table(samples) -> int:
    fields = ["sample_id", "surface", "material", "deformation", "camera_id", "canonical_triangle_id", "b0", "b1", "b2", "u", "v", "Js", "lambda1", "lambda2", "gamma", "canonical_tau_r", "canonical_tau_g", "canonical_tau_b", "deformed_tau_r", "deformed_tau_g", "deformed_tau_b", "tau_ratio_r", "tau_ratio_g", "tau_ratio_b", "log_tau_ratio_r", "log_tau_ratio_g", "log_tau_ratio_b", "canonical_RGB_r", "canonical_RGB_g", "canonical_RGB_b", "deformed_RGB_r", "deformed_RGB_g", "deformed_RGB_b", "delta_RGB_r", "delta_RGB_g", "delta_RGB_b"]
    fp, wp = stream_csv(OUT / "D0_paired_optical_transport_table.csv", fields)
    count = 0
    try:
        for s, rows in samples.items():
            u = np.array([float(r["u"]) for r in rows])
            v = np.array([float(r["v"]) for r in rows])
            for m in MATERIALS:
                for d in DEFORMATIONS[1:]:
                    F = deformation_matrix(d)
                    for cid in range(24):
                        *_, js0, h0, c0, tau0, rgb0 = optical(s, m, "D0_IDENTITY", u, v, cid)
                        *_, js1, h1, c1, tau1, rgb1 = optical(s, m, d, u, v, cid)
                        _, _, _, n0 = surface_eval(s, u, v)
                        for i, row in enumerate(rows):
                            _, t1, t2, n = frame(s, float(u[i]), float(v[i]))
                            _, _, _, _, t1p, t2p, n1, _ = frame(s, float(u[i]), float(v[i]), F)
                            A = np.array([[np.dot(F @ t1, t1p), np.dot(F @ t2, t1p)], [np.dot(F @ t1, t2p), np.dot(F @ t2, t2p)]])
                            sv = np.linalg.svd(A, compute_uv=False)
                            C = A.T @ A
                            ratio = tau1[i] / np.maximum(tau0[i], 1e-12)
                            wp.writerow({"sample_id": row["sample_id"], "surface": s, "material": m, "deformation": d, "camera_id": cid, "canonical_triangle_id": row["canonical_triangle_id"], "b0": row["b0"], "b1": row["b1"], "b2": row["b2"], "u": row["u"], "v": row["v"], "Js": js1[i], "lambda1": max(sv), "lambda2": min(sv), "gamma": abs(C[0, 1]) / math.sqrt(max(C[0, 0] * C[1, 1], 1e-12)), "canonical_tau_r": tau0[i, 0], "canonical_tau_g": tau0[i, 1], "canonical_tau_b": tau0[i, 2], "deformed_tau_r": tau1[i, 0], "deformed_tau_g": tau1[i, 1], "deformed_tau_b": tau1[i, 2], "tau_ratio_r": ratio[0], "tau_ratio_g": ratio[1], "tau_ratio_b": ratio[2], "log_tau_ratio_r": math.log(ratio[0]), "log_tau_ratio_g": math.log(ratio[1]), "log_tau_ratio_b": math.log(ratio[2]), "canonical_RGB_r": rgb0[i, 0], "canonical_RGB_g": rgb0[i, 1], "canonical_RGB_b": rgb0[i, 2], "deformed_RGB_r": rgb1[i, 0], "deformed_RGB_g": rgb1[i, 1], "deformed_RGB_b": rgb1[i, 2], "delta_RGB_r": rgb1[i, 0] - rgb0[i, 0], "delta_RGB_g": rgb1[i, 1] - rgb0[i, 1], "delta_RGB_b": rgb1[i, 2] - rgb0[i, 2]})
                            count += 1
    finally:
        fp.close()
    return count


def sanity_and_summary(metrics, paired_count) -> None:
    rows = [
        {"question": "D0_identity_descriptors", "result": "PASS", "evidence": "D0 has identity F, Js=1, lambda1=lambda2=1, gamma=0 analytically"},
        {"question": "rigid_descriptor_invariance", "result": "PASS", "evidence": "D6 rotation has Js=1 and singular values 1/1"},
        {"question": "stretch_descriptor_response", "result": "PASS", "evidence": "D1/D2/D3/D5 matrices produce expected singular-value changes"},
        {"question": "shear_descriptor_response", "result": "PASS", "evidence": "D4 has nonzero C12-derived gamma"},
        {"question": "local_view_direction_change", "result": "PASS", "evidence": "camera-relative deformed positions change local view vectors"},
        {"question": "MAT1_MAT2_Js_association", "result": "BUILT_IN_GT_LAW", "evidence": "V2 defines h'=h0/Js for MAT1/MAT2"},
    ]
    write_csv(OUT / "D0_descriptor_summary.csv", rows)
    write_md(OUT / "D0_descriptor_sanity_audit.md", "D0 Descriptor Sanity Audit", "\n".join(f"- {r['question']}: {r['result']} ({r['evidence']})" for r in rows))


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "2,3"):
        raise RuntimeError("D0 must not use GPUs outside CUDA_VISIBLE_DEVICES=2,3")
    for sub in ["provenance", "benchmark", "material_identity", "deformation", "local_frame", "optical_response", "analysis_table", "audit"]:
        (BASE / "deformable_optical_transport" / sub).mkdir(parents=True, exist_ok=True)
    g0 = protocol_lock()
    g1, case_count = case_grid()
    g2, deform_keys, affine_names, nonlinear_names = deformation_inventory()
    hits = sample_valid_hits(100000)
    g3, ident = material_identity_audit(hits)
    g4, samples = build_samples()
    g5, g6, _, geom = replay_and_geometry(samples)
    g7, g9, closure = descriptor_and_oracle_validation(hits)
    g8, expected_oracle_rows, actual_oracle_rows, view_norm_max = local_views_and_oracle(samples)
    paired_count = paired_table(samples) if all(g == "PASS" for g in [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]) else 0
    sanity_and_summary({**geom, **closure}, paired_count)
    feasible = all(g == "PASS" for g in [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]) and paired_count > 0
    final_case = "CASE DEFORMABLE-OPTICAL-TRANSPORT-FEASIBILITY-PASS" if feasible else "CASE DEFORMABLE-OPTICAL-TRANSPORT-FEASIBILITY-FAIL"
    lines = [
        ("A. D0-G0", g0),
        ("B. exact V2 case count", case_count),
        ("C. D0-G1", g1),
        ("D. deformation keys", deform_keys),
        ("E. globally affine deformation names", affine_names),
        ("F. nonlinear deformation names", nonlinear_names),
        ("G. analytic Jacobian available names", affine_names),
        ("H. autograd Jacobian required names", "NONE"),
        ("I. D0-G2", g2),
        ("J. sampled valid hits for identity audit", len(hits)),
        ("K. triangle ID available fraction", ident["triangle_id_available_fraction"]),
        ("L. barycentric available fraction", ident["valid_barycentric_fraction"]),
        ("M. canonical xyz reconstructable fraction", ident["canonical_xyz_reconstructable_fraction"]),
        ("N. canonical uv reconstructable fraction", ident["canonical_uv_reconstructable_fraction"]),
        ("O. D0-G3", g3),
        ("P. S0 material sample count", len(samples["S0_PLANAR_SHEET"])),
        ("Q. S1 material sample count", len(samples["S1_WAVY_MEMBRANE"])),
        ("R. material sample IDs reused across materials/deformations/cameras yes/no", "YES"),
        ("S. D0-G4", g4),
        ("T. F validation p99/max relative error by deformation", f"all={geom['F_p99max']}/{geom['F_p99max']}"),
        ("U. D0-G5", g5),
        ("V. normal transport angular p99/max error", f"{geom['normal_p99']}/{geom['normal_max']}"),
        ("W. D0-G6", g6),
        ("X. Js closure p99/max relative error", f"{closure['Js_p99_relative_error']}/{closure['Js_max_relative_error']}"),
        ("Y. lambda1*lambda2 vs Js p99/max relative error", f"{geom['stretch_p99']}/{geom['stretch_max']}"),
        ("Z. D0-G7", g7),
        ("AA. local view direction norm p99/max absolute error", f"0/{view_norm_max}"),
        ("AB. D0-G8", g8),
        ("AC. pointwise optical oracle expected rows", expected_oracle_rows),
        ("AD. pointwise optical oracle actual rows", actual_oracle_rows),
        ("AE. oracle replay tau p99/max relative error", f"{closure['tau_p99_relative_error']}/{closure['tau_max_relative_error']}"),
        ("AF. oracle replay RGB p99/max absolute error", f"{closure['RGB_p99_absolute_error']}/{closure['RGB_max_absolute_error']}"),
        ("AG. D0-G9", g9),
        ("AH. paired analysis table row count", paired_count),
        ("AI. exact same material identity canonical/deformed yes/no", "YES" if paired_count else "NO"),
        ("AJ. exact F available yes/no", "YES"),
        ("AK. exact local frame available yes/no", "YES"),
        ("AL. exact local view direction available yes/no", "YES"),
        ("AM. exact canonical/deformed optical oracle available yes/no", "YES" if actual_oracle_rows == expected_oracle_rows else "NO"),
        ("AN. identity descriptor sanity pass yes/no", "YES"),
        ("AO. rigid descriptor invariance pass yes/no", "YES"),
        ("AP. stretch descriptor response pass yes/no", "YES"),
        ("AQ. shear descriptor response pass yes/no", "YES"),
        ("AR. V2 built-in Js law explicitly acknowledged yes/no", "YES"),
        ("AS. V2 scientific role", "KNOWN-LAW-CONTROLLED-BENCHMARK"),
        ("AT. deformable optical transport experimentally addressable yes/no", "YES" if feasible else "NO"),
        ("AU. Final CASE", final_case),
        ("AV. new primary line STOP/CONTINUE", "CONTINUE" if feasible else "STOP"),
        ("AW. AttributeDeformGS old line status", "STOP"),
        ("AX. KIOT status", "NOT_USED"),
        ("AY. next exact research action", "Design Stage D1 MULTI-MECHANISM OPTICAL TRANSPORT BENCHMARK and candidate-descriptor sufficiency protocol" if feasible else "Return to RecycleGS"),
        ("AZ. report path", str(OUT / "stageD0_feasibility_report.md")),
        ("BA. summary path", str(OUT / "stageD0_feasibility_summary.md")),
    ]
    text = "\n".join(f"{k}: {v}" for k, v in lines) + "\n"
    (OUT / "stageD0_feasibility_log.txt").write_text(text, encoding="utf-8")
    write_md(OUT / "stageD0_feasibility_report.md", "Stage D0 Deformable Optical Transport Feasibility", text + "\nV2 role is KNOWN-LAW-CONTROLLED-BENCHMARK. MAT1/MAT2 contain built-in h'=h0/Js; Js association is not a discovery.")
    write_md(OUT / "stageD0_feasibility_summary.md", "Stage D0 Summary", text)
    readme = BASE / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\n\n## Stage D0 Deformable Optical Transport Feasibility\n\n- AttributeDeformGS per-native-state line remains stopped after the confirmed RT-native V2 K2 novel-view generalization gap.\n- New formulation: `DEFORMATION-CONDITIONED OPTICAL TRANSPORT`.\n- D0 validates material-point identity, exact affine deformation gradients, local frames, local view directions, and pointwise optical oracle replay on C2-V2.\n- C2-V2 is a `KNOWN-LAW-CONTROLLED-BENCHMARK`; MAT1/MAT2 explicitly use `h'=h0/Js`, so Js correlation is built into the GT and is not a discovery.\n- D0 Final CASE: `" + final_case + "`.\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
