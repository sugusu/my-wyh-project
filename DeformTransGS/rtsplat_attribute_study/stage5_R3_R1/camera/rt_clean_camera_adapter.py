from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


RES = 512
FOV_DEG = 60.0
ZNEAR = 0.01
ZFAR = 100.0


@dataclass(frozen=True)
class RTCleanCameraSpec:
    camera_id: int
    width: int
    height: int
    fov_x_rad: float
    fov_y_rad: float
    origin: np.ndarray
    R: np.ndarray
    T: np.ndarray
    world_view_transform: np.ndarray
    projection_matrix: np.ndarray
    full_proj_transform: np.ndarray


def clean_camera_pos(camera_id: int) -> np.ndarray:
    elev = 25.0 if camera_id < 12 else 50.0
    az = (camera_id % 12) * 30.0
    er = math.radians(elev)
    ar = math.radians(az)
    return np.array(
        [
            3.3 * math.cos(er) * math.cos(ar),
            3.3 * math.cos(er) * math.sin(ar),
            3.3 * math.sin(er),
        ],
        dtype=np.float64,
    )


def clean_material_grid_project(points_uvz: np.ndarray) -> np.ndarray:
    """Exact clean GT pixel convention from clean_gt_renderer.py.

    The clean generator maps material coordinates directly to pixel centers:
    u=(x+0.5)/RES*2-1 and v=(y+0.5)/RES*2-1. camera_id changes optical
    path length through camera_pos(), but does not define a perspective image
    projection.
    """

    pts = np.asarray(points_uvz, dtype=np.float64)
    x = (pts[:, 0] + 1.0) * 0.5 * RES - 0.5
    y = (pts[:, 1] + 1.0) * 0.5 * RES - 0.5
    return np.stack([x, y, pts[:, 2]], axis=1)


def get_world2view2(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.zeros((4, 4), dtype=np.float64)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    return np.linalg.inv(C2W).astype(np.float64)


def get_projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> np.ndarray:
    tan_half_y = math.tan(fov_y / 2.0)
    tan_half_x = math.tan(fov_x / 2.0)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def make_rt_clean_camera_spec(camera_id: int) -> RTCleanCameraSpec:
    fov = math.radians(FOV_DEG)
    R = np.eye(3, dtype=np.float64)
    T = clean_camera_pos(camera_id)
    world = get_world2view2(R, T).T
    proj = get_projection_matrix(ZNEAR, ZFAR, fov, fov).T
    full = world @ proj
    return RTCleanCameraSpec(
        camera_id=camera_id,
        width=RES,
        height=RES,
        fov_x_rad=fov,
        fov_y_rad=fov,
        origin=T.copy(),
        R=R,
        T=T,
        world_view_transform=world,
        projection_matrix=proj,
        full_proj_transform=full,
    )


def rt_matrix_project(points_xyz: np.ndarray, spec: RTCleanCameraSpec) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    clip = ph @ spec.full_proj_transform
    denom = clip[:, 3:4]
    finite = np.isfinite(denom[:, 0]) & (np.abs(denom[:, 0]) > 1e-12)
    ndc = np.full((pts.shape[0], 3), np.nan, dtype=np.float64)
    ndc[finite] = clip[finite, :3] / denom[finite]
    x = (ndc[:, 0] + 1.0) * 0.5 * spec.width - 0.5
    y = (ndc[:, 1] + 1.0) * 0.5 * spec.height - 0.5
    return np.stack([x, y, ndc[:, 2]], axis=1)


CAMERA_ADAPTER_TRACE = {
    "clean_projection_convention": "material_grid_orthographic_pixel_centers",
    "clean_pixel_formula": "x=(u+1)*0.5*512-0.5; y=(v+1)*0.5*512-0.5",
    "clean_camera_role": "camera_pos affects cos_theta/path length only, not pixel projection",
    "rt_projection_convention": "RT-Splatting Camera full_proj_transform using getWorld2View2 and perspective getProjectionMatrix",
    "rt_fov_degrees": FOV_DEG,
    "world_to_camera_convention": "R=I, T=clean_camera_pos, RT getWorld2View2 semantics",
    "pixel_center_convention": "0.5-centered clean grid",
}
