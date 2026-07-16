from __future__ import annotations

import math
from pathlib import Path

import numpy as np


RES = 512
MATERIALS = {
    "MAT0_NEUTRAL_FIXED_THICKNESS": {"sigma": np.array([1.2, 1.2, 1.2], dtype=np.float64), "h0": 0.08, "mode": "fixed"},
    "MAT1_NEUTRAL_MASS_CONSERVING": {"sigma": np.array([1.2, 1.2, 1.2], dtype=np.float64), "h0": 0.08, "mode": "mass"},
    "MAT2_TINTED_MASS_CONSERVING": {"sigma": np.array([0.6, 1.2, 2.0], dtype=np.float64), "h0": 0.08, "mode": "mass"},
}
SURFACES = ["S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE"]
DEFORMATIONS = ["D0_IDENTITY", "D1_STRETCH_X_1P25", "D2_STRETCH_X_1P50", "D3_BIAXIAL_XY_1P50", "D4_SHEAR_XY_0P30", "D5_ANISO_X1P60_Y0P80", "D6_ROTATION_Z_30"]


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


def camera_pos(camera_id: int) -> np.ndarray:
    elev = 25.0 if camera_id < 12 else 50.0
    az = (camera_id % 12) * 30.0
    er = math.radians(elev)
    ar = math.radians(az)
    return np.array([3.3 * math.cos(er) * math.cos(ar), 3.3 * math.cos(er) * math.sin(ar), 3.3 * math.sin(er)], dtype=np.float64)


def render(surface: str, material: str, deformation: str, camera_id: int) -> dict:
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
    ndef = n0 @ invt.T
    ndef = ndef / (np.linalg.norm(ndef, axis=-1, keepdims=True) + 1e-30)
    js = detf * np.linalg.norm(n0 @ invt.T, axis=-1)
    h = np.full_like(js, mat["h0"], dtype=np.float64) if mat["mode"] == "fixed" else mat["h0"] / np.maximum(js, 1e-12)
    d = -camera_pos(camera_id)
    d = d / (np.linalg.norm(d) + 1e-30)
    cos_theta = np.maximum(np.abs(np.sum(ndef * (-d.reshape(1, 1, 3)), axis=-1)), 0.15)
    tau = mat["sigma"].reshape(1, 1, 3) * h[..., None] / cos_theta[..., None]
    rgb = np.exp(-tau)
    alpha = 1.0 - np.exp(-tau.mean(axis=-1))
    tau[~inside] = 0.0
    rgb[~inside] = 1.0
    alpha[~inside] = 0.0
    tri_id = np.where(inside, ((yy // 4) * 128 + (xx // 4)).astype(np.int32), -1)
    return {"rgb": rgb, "alpha": alpha, "tau_rgb": tau, "triangle_id": tri_id, "js": js}


def save_view(root: Path, surface: str, material: str, deformation: str, camera_id: int) -> list[Path]:
    out = root / surface / material / deformation
    out.mkdir(parents=True, exist_ok=True)
    data = render(surface, material, deformation, camera_id)
    stem = out / f"camera_{camera_id:02d}"
    paths = []
    for key, arr in [("rgb", data["rgb"].astype(np.float32)), ("alpha", data["alpha"].astype(np.float32)), ("tau_rgb", data["tau_rgb"].astype(np.float32)), ("triangle_id", data["triangle_id"].astype(np.int32))]:
        path = Path(str(stem) + f"_{key}.npy")
        np.save(path, arr)
        paths.append(path)
    return paths
