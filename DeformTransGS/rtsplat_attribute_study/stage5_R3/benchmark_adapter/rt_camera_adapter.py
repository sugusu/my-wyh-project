from __future__ import annotations

import math
import numpy as np
import torch

from scene.cameras import Camera


def clean_camera_pos(camera_id: int) -> np.ndarray:
    elev = 25.0 if camera_id < 12 else 50.0
    az = (camera_id % 12) * 30.0
    er = math.radians(elev)
    ar = math.radians(az)
    return np.array([3.3 * math.cos(er) * math.cos(ar), 3.3 * math.cos(er) * math.sin(ar), 3.3 * math.sin(er)], dtype=np.float32)


def make_rt_camera(camera_id: int, width: int = 512, height: int = 512, fov_deg: float = 60.0) -> Camera:
    image = torch.zeros(3, height, width, device="cuda")
    mask = torch.zeros(1, height, width, device="cuda")
    return Camera(
        camera_id,
        np.eye(3, dtype=np.float32),
        clean_camera_pos(camera_id),
        math.radians(fov_deg),
        math.radians(fov_deg),
        image,
        None,
        mask,
        f"clean_{camera_id:02d}",
        camera_id,
    )


def clean_material_grid_project(points_uvz: np.ndarray, width: int = 512, height: int = 512) -> np.ndarray:
    u = points_uvz[:, 0]
    v = points_uvz[:, 1]
    x = (u + 1.0) * 0.5 * width - 0.5
    y = (v + 1.0) * 0.5 * height - 0.5
    return np.stack([x, y, points_uvz[:, 2]], axis=1)


def rt_matrix_project(points_xyz: np.ndarray, camera: Camera) -> np.ndarray:
    pts = torch.tensor(points_xyz, dtype=torch.float32, device="cuda")
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device="cuda")
    ph = torch.cat([pts, ones], dim=1)
    clip = ph @ camera.full_proj_transform
    ndc = clip[:, :3] / torch.clamp(clip[:, 3:4], min=1e-7)
    x = (ndc[:, 0] + 1.0) * 0.5 * float(camera.image_width) - 0.5
    y = (ndc[:, 1] + 1.0) * 0.5 * float(camera.image_height) - 0.5
    return torch.stack([x, y, ndc[:, 2]], dim=1).detach().cpu().numpy()
