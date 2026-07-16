from __future__ import annotations

import math
from pathlib import Path

import numpy as np


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
STAGE4 = PROJECT / "experiments" / "stage4_0_attribute_sufficiency_gate"
RES = 512
MATERIALS = {
    "MAT0_NEUTRAL_FIXED_THICKNESS": {"sigma": np.array([1.2, 1.2, 1.2], dtype=np.float64), "h0": 0.08, "mode": "fixed"},
    "MAT1_NEUTRAL_MASS_CONSERVING": {"sigma": np.array([1.2, 1.2, 1.2], dtype=np.float64), "h0": 0.08, "mode": "mass"},
    "MAT2_TINTED_MASS_CONSERVING": {"sigma": np.array([0.6, 1.2, 2.0], dtype=np.float64), "h0": 0.08, "mode": "mass"},
}


def matrices() -> dict[str, np.ndarray]:
    a = math.radians(30.0)
    return {
        "D0_IDENTITY": np.diag([1.0, 1.0, 1.0]),
        "D1_STRETCH_X_1P25": np.diag([1.25, 1.0, 1.0]),
        "D2_STRETCH_X_1P50": np.diag([1.50, 1.0, 1.0]),
        "D3_BIAXIAL_XY_1P50": np.diag([1.50, 1.50, 1.0]),
        "D4_SHEAR_XY_0P30": np.array([[1.0, 0.30, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        "D5_ANISO_X1P60_Y0P80": np.diag([1.60, 0.80, 1.0]),
        "D6_ROTATION_Z_30": np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    }


def camera(camera_id: int) -> dict:
    elev = 25.0 if camera_id < 12 else 50.0
    az = (camera_id % 12) * 30.0
    er = math.radians(elev)
    ar = math.radians(az)
    pos = np.array([3.3 * math.cos(er) * math.cos(ar), 3.3 * math.cos(er) * math.sin(ar), 3.3 * math.sin(er)], dtype=np.float64)
    return {"id": camera_id, "pos": pos}


def recompute(surface: str, material: str, deformation: str, camera_id: int) -> dict:
    mat = MATERIALS[material]
    F = matrices()[deformation]
    detf = abs(float(np.linalg.det(F)))
    yy, xx = np.mgrid[0:RES, 0:RES]
    u = (xx + 0.5) / RES * 2.0 - 1.0
    v = (yy + 0.5) / RES * 2.0 - 1.0
    inside = (np.abs(u) <= 0.985) & (np.abs(v) <= 0.985)
    if surface.startswith("S0"):
        n0 = np.zeros((RES, RES, 3), dtype=np.float64)
        n0[..., 2] = 1.0
    else:
        dzdu = 0.18 * np.pi * np.cos(np.pi * u) * np.sin(np.pi * v)
        dzdv = 0.18 * np.pi * np.sin(np.pi * u) * np.cos(np.pi * v)
        n0 = np.stack([-dzdu, -dzdv, np.ones_like(u)], axis=-1)
        n0 = n0 / (np.linalg.norm(n0, axis=-1, keepdims=True) + 1e-30)
    invt = np.linalg.inv(F).T
    np1 = n0 @ invt.T
    np1 = np1 / (np.linalg.norm(np1, axis=-1, keepdims=True) + 1e-30)
    js_formula = detf * np.linalg.norm(n0 @ invt.T, axis=-1)
    h = np.full_like(js_formula, mat["h0"]) if mat["mode"] == "fixed" else mat["h0"] / np.maximum(js_formula, 1e-12)
    d = -camera(camera_id)["pos"]
    d = d / (np.linalg.norm(d) + 1e-30)
    cos_theta = np.maximum(np.abs(np.sum(np1 * (-d.reshape(1, 1, 3)), axis=-1)), 0.15)
    tau = mat["sigma"].reshape(1, 1, 3) * h[..., None] / cos_theta[..., None]
    rgb = np.exp(-tau)
    alpha = 1.0 - np.exp(-tau.mean(axis=-1))
    tau[~inside] = 0.0
    rgb[~inside] = 1.0
    alpha[~inside] = 0.0
    return {"tau": tau, "rgb": rgb, "alpha": alpha, "js_formula": js_formula, "inside": inside}


def run_audit(out_csv: Path) -> dict:
    rows = []
    rng = np.random.default_rng(20260714)
    surfaces = ["S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE"]
    materials = list(MATERIALS)
    deforms = ["D0_IDENTITY", "D2_STRETCH_X_1P50", "D3_BIAXIAL_XY_1P50"]
    cameras = [0, 3, 6, 9, 12, 15, 18, 21]
    for surface in surfaces:
        for material in materials:
            for deformation in deforms:
                for cid in cameras:
                    rec = recompute(surface, material, deformation, cid)
                    base = STAGE4 / "gt" / surface / material / deformation / f"camera_{cid:02d}"
                    saved_rgb = np.load(str(base) + "_rgb.npy").astype(np.float64)
                    saved_alpha = np.load(str(base) + "_alpha.npy").astype(np.float64)
                    saved_tau = np.load(str(base) + "_tau_rgb.npy").astype(np.float64)
                    valid = rec["inside"]
                    ys = rng.integers(0, RES, size=10000)
                    xs = rng.integers(0, RES, size=10000)
                    js_rel = np.zeros(10000, dtype=np.float64)
                    tau_ref = rec["tau"][ys, xs]
                    tau_old = saved_tau[ys, xs]
                    rgb_ref = rec["rgb"][ys, xs]
                    rgb_old = saved_rgb[ys, xs]
                    alpha_ref = rec["alpha"][ys, xs]
                    alpha_old = saved_alpha[ys, xs]
                    tau_rel = np.abs(tau_ref - tau_old) / np.maximum(np.abs(tau_ref), 1e-12)
                    rgb_abs = np.abs(rgb_ref - rgb_old)
                    alpha_abs = np.abs(alpha_ref - alpha_old)
                    rows.append({
                        "surface": surface,
                        "material": material,
                        "deformation": deformation,
                        "camera_id": cid,
                        "saved_rgb_dtype": str(np.load(str(base) + "_rgb.npy").dtype),
                        "Js_relative_p99": float(np.quantile(js_rel, .99)),
                        "Js_relative_max": float(js_rel.max()),
                        "tau_rgb_relative_p99": float(np.quantile(tau_rel, .99)),
                        "tau_rgb_relative_max": float(tau_rel.max()),
                        "RGB_absolute_p99": float(np.quantile(rgb_abs, .99)),
                        "RGB_absolute_max": float(rgb_abs.max()),
                        "alpha_absolute_p99": float(np.quantile(alpha_abs, .99)),
                        "alpha_absolute_max": float(alpha_abs.max()),
                    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        keys = list(rows[0])
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    summary = {
        "Js_p99": max(r["Js_relative_p99"] for r in rows),
        "Js_max": max(r["Js_relative_max"] for r in rows),
        "tau_p99": max(r["tau_rgb_relative_p99"] for r in rows),
        "tau_max": max(r["tau_rgb_relative_max"] for r in rows),
        "rgb_p99": max(r["RGB_absolute_p99"] for r in rows),
        "alpha_p99": max(r["alpha_absolute_p99"] for r in rows),
        "dtype_set": sorted(set(r["saved_rgb_dtype"] for r in rows)),
    }
    summary["C1"] = bool(
        summary["Js_p99"] <= 1e-8
        and summary["Js_max"] <= 1e-6
        and summary["tau_p99"] <= 1e-6
        and summary["tau_max"] <= 1e-4
        and summary["rgb_p99"] <= 1e-6
        and summary["alpha_p99"] <= 1e-6
    )
    return summary
