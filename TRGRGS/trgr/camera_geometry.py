from dataclasses import dataclass
from pathlib import Path
import sys
import importlib.util
import numpy as np


@dataclass(frozen=True)
class Camera:
    name: str
    width: int
    height: int
    K: np.ndarray
    world_to_camera: np.ndarray


def _load_colmap_api(tsgs_root: Path):
    # Loading scene.colmap_loader normally executes scene/__init__.py and pulls in
    # renderer-only dependencies. Load the official file directly without copying it.
    source = tsgs_root.resolve() / "scene/colmap_loader.py"
    spec = importlib.util.spec_from_file_location("tsgs_official_colmap_loader", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (module.read_extrinsics_binary, module.read_extrinsics_text,
            module.read_intrinsics_binary, module.read_intrinsics_text,
            module.qvec2rotmat)


def load_colmap_cameras(scene_path, tsgs_root):
    scene_path, tsgs_root = Path(scene_path), Path(tsgs_root)
    rb_e, rt_e, rb_i, rt_i, q2r = _load_colmap_api(tsgs_root)
    sparse = scene_path / "sparse"
    try:
        ex, intr = rt_e(str(sparse / "images.txt")), rt_i(str(sparse / "cameras.txt"))
    except Exception:
        ex, intr = rb_e(str(sparse / "images.bin")), rb_i(str(sparse / "cameras.bin"))
    cameras = []
    for item in sorted(ex.values(), key=lambda x: x.name):
        ci = intr[item.camera_id]
        if ci.model == "SIMPLE_PINHOLE":
            f, cx, cy = map(float, ci.params); fx = fy = f
        elif ci.model == "PINHOLE":
            fx, fy, cx, cy = map(float, ci.params)
        else:
            raise ValueError(f"Unsupported COLMAP model: {ci.model}")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        # COLMAP explicitly defines x_cam = R(qvec) x_world + tvec.
        w2c[:3, :3], w2c[:3, 3] = q2r(item.qvec), item.tvec
        cameras.append(Camera(item.name, ci.width, ci.height, K, w2c))
    return cameras


def world_to_camera(points, w2c):
    points = np.asarray(points, dtype=np.float64)
    return points @ w2c[:3, :3].T + w2c[:3, 3]


def camera_to_world(points, w2c):
    points = np.asarray(points, dtype=np.float64)
    return (points - w2c[:3, 3]) @ w2c[:3, :3]


def project(points_camera, K):
    points_camera = np.asarray(points_camera, dtype=np.float64)
    uvw = points_camera @ K.T
    return uvw[..., :2] / uvw[..., 2:3], points_camera[..., 2]


def unproject(uv, depth, K):
    uv, depth = np.asarray(uv, dtype=np.float64), np.asarray(depth, dtype=np.float64)
    pix = np.concatenate([uv, np.ones((*uv.shape[:-1], 1))], axis=-1)
    return (pix @ np.linalg.inv(K).T) * depth[..., None]


def roundtrip(points_world, camera):
    cam = world_to_camera(points_world, camera.world_to_camera)
    uv, depth = project(cam, camera.K)
    recovered = camera_to_world(unproject(uv, depth, camera.K), camera.world_to_camera)
    return recovered
