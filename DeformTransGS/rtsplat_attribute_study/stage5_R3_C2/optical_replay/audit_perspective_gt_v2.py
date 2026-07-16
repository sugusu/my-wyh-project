from __future__ import annotations

import math
from pathlib import Path

import numpy as np


RES = 512
H0 = 0.08
MATERIALS = {
    "MAT0_NEUTRAL_FIXED_THICKNESS": (np.array([1.2, 1.2, 1.2], dtype=np.float64), "fixed"),
    "MAT1_NEUTRAL_MASS_CONSERVING": (np.array([1.2, 1.2, 1.2], dtype=np.float64), "mass"),
    "MAT2_TINTED_MASS_CONSERVING": (np.array([0.6, 1.2, 2.0], dtype=np.float64), "mass"),
}


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


def normals_and_js(surface: str, deformation: str, hit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = deformation_matrix(deformation)
    q = hit @ np.linalg.inv(F).T
    u, v = q[..., 0], q[..., 1]
    if surface == "S0_PLANAR_SHEET":
        n0 = np.zeros(hit.shape, dtype=np.float64)
        n0[..., 2] = 1.0
    else:
        dzdu = 0.18 * np.pi * np.cos(np.pi * u) * np.sin(np.pi * v)
        dzdv = 0.18 * np.pi * np.sin(np.pi * u) * np.cos(np.pi * v)
        n0 = np.stack([-dzdu, -dzdv, np.ones_like(u)], axis=-1)
        n0 /= np.linalg.norm(n0, axis=-1, keepdims=True) + 1e-30
    invt = np.linalg.inv(F).T
    ndef = n0 @ invt.T
    ndef /= np.linalg.norm(ndef, axis=-1, keepdims=True) + 1e-30
    js = abs(float(np.linalg.det(F))) * np.linalg.norm(n0 @ invt.T, axis=-1)
    return ndef, js


def replay_view(root: Path, surface: str, material: str, deformation: str, camera_id: int) -> dict[str, float]:
    d = root / surface / material / deformation
    stem = d / f"camera_{camera_id:02d}"
    rgb = np.load(str(stem) + "_rgb.npy").astype(np.float64)
    tau = np.load(str(stem) + "_tau_rgb.npy").astype(np.float64)
    alpha = np.load(str(stem) + "_alpha.npy").astype(np.float64)
    hit = np.load(str(stem) + "_world_hit.npy").astype(np.float64)
    ray = np.load(str(stem) + "_ray_direction.npy").astype(np.float64)
    saved_js = np.load(str(stem) + "_Js.npy").astype(np.float64)
    tri = np.load(str(stem) + "_triangle_id.npy")
    valid = tri >= 0
    ndef, js = normals_and_js(surface, deformation, hit)
    sigma, mode = MATERIALS[material]
    h = np.full((RES, RES), H0, dtype=np.float64) if mode == "fixed" else H0 / np.maximum(js, 1e-12)
    cos_theta = np.maximum(np.abs(np.sum(ndef * (-ray), axis=-1)), 0.15)
    rtau = sigma.reshape(1, 1, 3) * h[..., None] / cos_theta[..., None]
    rrgb = np.exp(-rtau)
    ralpha = 1.0 - np.exp(-rtau.mean(axis=-1))
    rtau = np.where(valid[..., None], rtau, 0.0)
    rrgb = np.where(valid[..., None], rrgb, 1.0)
    ralpha = np.where(valid, ralpha, 0.0)
    rjs = np.where(valid, js, 0.0)
    rel_js = np.abs(rjs[valid] - saved_js[valid]) / np.maximum(np.abs(rjs[valid]), 1e-12)
    rel_tau = np.abs(rtau[valid] - tau[valid]) / np.maximum(np.abs(rtau[valid]), 1e-12)
    abs_rgb = np.abs(rrgb[valid] - rgb[valid])
    abs_alpha = np.abs(ralpha[valid] - alpha[valid])
    return {
        "valid_pixels": int(valid.sum()),
        "Js_rel_p99": float(np.quantile(rel_js, 0.99)) if rel_js.size else 0.0,
        "Js_rel_max": float(np.max(rel_js)) if rel_js.size else 0.0,
        "tau_rel_p99": float(np.quantile(rel_tau, 0.99)) if rel_tau.size else 0.0,
        "tau_rel_max": float(np.max(rel_tau)) if rel_tau.size else 0.0,
        "rgb_abs_p99": float(np.quantile(abs_rgb, 0.99)) if abs_rgb.size else 0.0,
        "rgb_abs_max": float(np.max(abs_rgb)) if abs_rgb.size else 0.0,
        "alpha_abs_p99": float(np.quantile(abs_alpha, 0.99)) if abs_alpha.size else 0.0,
        "alpha_abs_max": float(np.max(abs_alpha)) if abs_alpha.size else 0.0,
    }
