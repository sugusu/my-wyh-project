from __future__ import annotations

import math
from pathlib import Path

import numpy as np


RES = 512
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


def deformation_matrix(name: str) -> np.ndarray:
    a = math.radians(30.0)
    mats = {
        "D0_IDENTITY": np.diag([1.0, 1.0, 1.0]),
        "D1_STRETCH_X_1P25": np.diag([1.25, 1.0, 1.0]),
        "D2_STRETCH_X_1P50": np.diag([1.50, 1.0, 1.0]),
        "D3_BIAXIAL_XY_1P50": np.diag([1.50, 1.50, 1.0]),
        "D4_SHEAR_XY_0P30": np.array([[1.0, 0.30, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        "D5_ANISO_X1P60_Y0P80": np.diag([1.60, 0.80, 1.0]),
        "D6_ROTATION_Z_30": np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    }
    return mats[name]


def camera_pos(camera_id: int) -> np.ndarray:
    elev = 25.0 if camera_id < 12 else 50.0
    az = (camera_id % 12) * 30.0
    er = math.radians(elev)
    ar = math.radians(az)
    return np.array([3.3 * math.cos(er) * math.cos(ar), 3.3 * math.cos(er) * math.sin(ar), 3.3 * math.sin(er)], dtype=np.float64)


def camera_basis(camera_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = camera_pos(camera_id)
    forward = -center / (np.linalg.norm(center) + 1e-30)
    up_candidate = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_candidate))) >= 0.99:
        up_candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up_candidate)
    right /= np.linalg.norm(right) + 1e-30
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up) + 1e-30
    return center, right, true_up, forward


def pixel_rays(camera_id: int, fovy_deg: float) -> tuple[np.ndarray, np.ndarray]:
    center, right, up, forward = camera_basis(camera_id)
    f = RES / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    yy, xx = np.mgrid[0:RES, 0:RES]
    xcam = (xx.astype(np.float64) + 0.5 - RES / 2.0) / f
    ycam = -(yy.astype(np.float64) + 0.5 - RES / 2.0) / f
    dirs = forward.reshape(1, 1, 3) + xcam[..., None] * right.reshape(1, 1, 3) + ycam[..., None] * up.reshape(1, 1, 3)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-30
    return center, dirs


def surface_eval(surface: str, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if surface == "S0_PLANAR_SHEET":
        z = np.zeros_like(u)
        n0 = np.zeros(u.shape + (3,), dtype=np.float64)
        n0[..., 2] = 1.0
    else:
        z = 0.18 * np.sin(np.pi * u) * np.sin(np.pi * v)
        dzdu = 0.18 * np.pi * np.cos(np.pi * u) * np.sin(np.pi * v)
        dzdv = 0.18 * np.pi * np.sin(np.pi * u) * np.cos(np.pi * v)
        n0 = np.stack([-dzdu, -dzdv, np.ones_like(u)], axis=-1)
        n0 /= np.linalg.norm(n0, axis=-1, keepdims=True) + 1e-30
    return np.stack([u, v, z], axis=-1), n0


def intersect_surface(surface: str, deformation: str, camera_id: int, fovy_deg: float) -> dict[str, np.ndarray]:
    F = deformation_matrix(deformation)
    Finv = np.linalg.inv(F)
    center, dirs = pixel_rays(camera_id, fovy_deg)
    c0 = center.reshape(1, 1, 3)
    if surface == "S0_PLANAR_SHEET":
        t = -c0[..., 2] / (dirs[..., 2] + 1e-30)
        p = c0 + t[..., None] * dirs
        q = p @ Finv.T
        u, v = q[..., 0], q[..., 1]
    else:
        # The deformations used here have no z mixing. Reduce the ray/surface
        # intersection to one scalar t and solve z(t)=wave(u(t),v(t)).
        t = -c0[..., 2] / (dirs[..., 2] + 1e-30)
        for _ in range(10):
            p = c0 + t[..., None] * dirs
            q = p @ Finv.T
            u, v = q[..., 0], q[..., 1]
            z = 0.18 * np.sin(np.pi * u) * np.sin(np.pi * v)
            dzdu = 0.18 * np.pi * np.cos(np.pi * u) * np.sin(np.pi * v)
            dzdv = 0.18 * np.pi * np.sin(np.pi * u) * np.cos(np.pi * v)
            g = q[..., 2] - z
            dqdt = dirs @ Finv.T
            gp = dqdt[..., 2] - dzdu * dqdt[..., 0] - dzdv * dqdt[..., 1]
            t = t - g / (gp + 1e-30)
        p = c0 + t[..., None] * dirs
        q = p @ Finv.T
        u, v = q[..., 0], q[..., 1]
    x0, n0 = surface_eval(surface, u, v)
    p = x0 @ F.T
    invt = np.linalg.inv(F).T
    ndef = n0 @ invt.T
    ndef /= np.linalg.norm(ndef, axis=-1, keepdims=True) + 1e-30
    js = abs(float(np.linalg.det(F))) * np.linalg.norm(n0 @ invt.T, axis=-1)
    valid = (t > 0.0) & np.isfinite(t) & (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
    cell_u = np.clip(np.floor((u + 1.0) * 64.0).astype(np.int32), 0, 127)
    cell_v = np.clip(np.floor((v + 1.0) * 64.0).astype(np.int32), 0, 127)
    fu = (u + 1.0) * 64.0 - cell_u
    fv = (v + 1.0) * 64.0 - cell_v
    upper = (fu + fv) > 1.0
    tri = (cell_v * 128 + cell_u) * 2 + upper.astype(np.int32)
    tri = np.where(valid, tri, -1).astype(np.int32)
    b0 = np.where(upper, fu + fv - 1.0, 1.0 - fu - fv)
    b1 = np.where(upper, 1.0 - fv, fu)
    b2 = np.where(upper, 1.0 - fu, fv)
    bary = np.stack([b0, b1, b2], axis=-1)
    bary = np.where(valid[..., None], bary, 0.0)
    return {
        "valid": valid,
        "u": u,
        "v": v,
        "triangle_id": tri,
        "world_hit": np.where(valid[..., None], p, 0.0),
        "barycentric": bary,
        "ray_direction": dirs,
        "normal": ndef,
        "Js": js,
    }


def render_view(surface: str, material: str, deformation: str, camera_id: int, fovy_deg: float) -> dict[str, np.ndarray]:
    geom = intersect_surface(surface, deformation, camera_id, fovy_deg)
    sigma, mode = MATERIALS[material]
    h = np.full((RES, RES), H0, dtype=np.float64) if mode == "fixed" else H0 / np.maximum(geom["Js"], 1e-12)
    cos_theta = np.maximum(np.abs(np.sum(geom["normal"] * (-geom["ray_direction"]), axis=-1)), 0.15)
    tau = sigma.reshape(1, 1, 3) * h[..., None] / cos_theta[..., None]
    rgb = np.exp(-tau)
    alpha = 1.0 - np.exp(-tau.mean(axis=-1))
    valid = geom["valid"]
    tau = np.where(valid[..., None], tau, 0.0)
    rgb = np.where(valid[..., None], rgb, 1.0)
    alpha = np.where(valid, alpha, 0.0)
    return {
        "rgb": rgb.astype(np.float32),
        "tau_rgb": tau.astype(np.float32),
        "alpha": alpha.astype(np.float32),
        "triangle_id": geom["triangle_id"].astype(np.int32),
        "world_hit": geom["world_hit"].astype(np.float32),
        "barycentric": geom["barycentric"].astype(np.float32),
        "ray_direction": geom["ray_direction"].astype(np.float32),
        "Js": np.where(valid, geom["Js"], 0.0).astype(np.float32),
    }


def save_view(root: Path, surface: str, material: str, deformation: str, camera_id: int, fovy_deg: float) -> list[Path]:
    out = root / surface / material / deformation
    out.mkdir(parents=True, exist_ok=True)
    data = render_view(surface, material, deformation, camera_id, fovy_deg)
    stem = out / f"camera_{camera_id:02d}"
    paths: list[Path] = []
    for key, arr in data.items():
        path = Path(str(stem) + f"_{key}.npy")
        np.save(path, arr)
        paths.append(path)
    return paths
