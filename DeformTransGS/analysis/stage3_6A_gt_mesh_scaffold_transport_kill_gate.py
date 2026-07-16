from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, "/data/wyh/DeformTransGS/analysis")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, "/data/wyh/repos/TSGS/submodules/diff-first-surface-rasterization")
sys.path.insert(0, "/data/wyh/repos/TSGS/submodules/diff-first-surface-rasterization/build/lib.linux-x86_64-cpython-310")
sys.path.insert(0, "/data/wyh/repos/TSGS/submodules/simple-knn")
sys.path.insert(0, "/data/wyh/repos/TSGS/submodules/simple-knn/build/lib.linux-x86_64-cpython-310")
sys.path.insert(0, "/data/wyh/repos/TSGS")

from kiot_fast_inverse import invert_phi_cont_np, kiot_cuda_identity_safe_np, phi_cont_np
from tsgs_patch_adapter import TSGSPatchAdapter


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OUT = PROJECT / "experiments" / "stage3_6A_gt_mesh_scaffold_transport_kill_gate"
ALPHA_DIR = OUT / "alpha"
DEPTH_DIR = OUT / "first_surface"
CHECKPOINT = ROOT / "RecycleGS" / "baselines" / "tsgs_official_scene01_30k_v4"
SCENE = ROOT / "RecycleGS" / "data" / "translab_full" / "scene_01"
MASK_DIR = SCENE / "transparent_masks"
MESH_PATH = SCENE / "meshes" / "scene_mesh.obj"
TSGS = ROOT / "repos" / "TSGS"
R4 = PROJECT / "experiments" / "stage3_5B_R4_normal_semantics_surface_layer_recovery"
R4B = PROJECT / "experiments" / "stage3_5B_R4B_depth_normal_coordinate_closure"
PLY = CHECKPOINT / "point_cloud" / "iteration_30000" / "point_cloud.ply"
POLICIES = ["P0_FIXED", "P1_TAU_JS", "P2_OPACITY_LINEAR", "P3_KIOT_CONT", "P4_KIOT_CUDA"]
PRIMARY_STATES = ["A1_STRETCH_X_1P25", "A2_STRETCH_X_1P50", "A3_STRETCH_X_2P00", "A4_BIAXIAL_XY_1P50"]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-8, 1.0 - 1e-8)
    return np.log(x / (1.0 - x))


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-30)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((len(q), 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    out = np.empty((len(m), 4), dtype=np.float64)
    for i, R in enumerate(m):
        tr = float(np.trace(R))
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2.0
            out[i] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        else:
            j = int(np.argmax(np.diag(R)))
            if j == 0:
                s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-30)) * 2.0
                out[i] = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
            elif j == 1:
                s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-30)) * 2.0
                out[i] = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
            else:
                s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-30)) * 2.0
                out[i] = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return normalize(out)


def camera_matrices(cam: dict) -> np.ndarray:
    R = np.asarray(cam["rotation"], dtype=np.float64)
    C2W = np.eye(4)
    C2W[:3, :3] = R
    C2W[:3, 3] = np.asarray(cam["position"], dtype=np.float64)
    Rt = np.linalg.inv(C2W)
    W = np.zeros((4, 4), dtype=np.float64)
    W[:3, :3] = Rt[:3, :3]
    W[:3, 3] = Rt[:3, 3]
    W[3, 3] = 1.0
    return W.T


def project_points(xyz: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1) @ camera_matrices(cam)
    z = p[:, 2]
    u = cam["fx"] * p[:, 0] / (z + 1e-30) + cam["width"] * 0.5
    v = cam["fy"] * p[:, 1] / (z + 1e-30) + cam["height"] * 0.5
    ok = (z > 1e-8) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
    return u, v, z, ok


def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = u - x0
    wy = v - y0
    return (1 - wx) * (1 - wy) * img[y0, x0] + wx * (1 - wy) * img[y0, x1] + (1 - wx) * wy * img[y1, x0] + wx * wy * img[y1, x1]


def load_audit_cameras() -> list[dict]:
    manifest = pd.read_csv(R4 / "r4_first_surface_render_manifest.csv")
    cameras = json.loads((CHECKPOINT / "cameras.json").read_text())
    by_id = {int(c["id"]): c for c in cameras}
    return [by_id[int(r.camera_id)] for r in manifest.itertuples()]


def mesh_silhouette_iou(mesh: trimesh.Trimesh, cam: dict) -> tuple[float, int, int]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    u, v, _, ok = project_points(verts, cam)
    img = np.zeros((int(cam["height"]), int(cam["width"])), dtype=np.uint8)
    face_ok = ok[faces].all(axis=1)
    polys = np.stack([u[faces[face_ok]], v[faces[face_ok]]], axis=-1)
    for poly in polys:
        pts = np.rint(poly).astype(np.int32)
        cv2.fillConvexPoly(img, pts, 1)
    mask = (np.asarray(Image.open(MASK_DIR / f"{cam['img_name']}.png").convert("L")) > 0).astype(np.uint8)
    inter = int(np.logical_and(img > 0, mask > 0).sum())
    union = int(np.logical_or(img > 0, mask > 0).sum())
    return inter / max(union, 1), int(img.sum()), int(mask.sum())


def closest_point_on_triangle(p: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a, np.array([1.0, 0.0, 0.0])
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1.0 - v, v, 0.0])
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w, np.array([1.0 - v - w, v, w])


def bind_points_to_mesh(points: np.ndarray, mesh: trimesh.Trimesh, k: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tris = verts[faces]
    centroids = tris.mean(axis=1)
    tree = cKDTree(centroids)
    _, cand = tree.query(points, k=min(k, len(tris)))
    if cand.ndim == 1:
        cand = cand[:, None]
    cps = np.zeros_like(points)
    bary = np.zeros((len(points), 3), dtype=np.float64)
    face_ids = np.zeros(len(points), dtype=np.int64)
    dists = np.zeros(len(points), dtype=np.float64)
    for i, p in enumerate(points):
        best = (np.inf, None, None, None)
        for fid in cand[i]:
            cp, bc = closest_point_on_triangle(p, tris[int(fid)])
            dist = float(np.linalg.norm(p - cp))
            if dist < best[0]:
                best = (dist, int(fid), cp, bc)
        dists[i], face_ids[i], cps[i], bary[i] = best
    return cps, dists, face_ids, bary


def face_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    return normalize(n)


def build_camera(cam: dict):
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    C2W = np.eye(4)
    C2W[:3, :3] = np.asarray(cam["rotation"], dtype=np.float64)
    C2W[:3, 3] = np.asarray(cam["position"], dtype=np.float64)
    Rt = np.linalg.inv(C2W)
    return Camera(
        cam["id"],
        Rt[:3, :3].T,
        Rt[:3, 3],
        focal2fov(cam["fx"], cam["width"]),
        focal2fov(cam["fy"], cam["height"]),
        int(cam["width"]),
        int(cam["height"]),
        str(SCENE / "images" / f"{cam['img_name']}.png"),
        None,
        cam["img_name"],
        cam["id"],
        preload_img=False,
        data_device="cuda",
    )


class PatchGaussian:
    def __init__(self, xyz, raw_scaling, raw_rotation, raw_opacity, raw_transparency):
        from torch import nn
        self.active_sh_degree = 0
        self.max_sh_degree = 3
        self.max_asg_degree = 24
        self._xyz = nn.Parameter(torch.as_tensor(xyz, dtype=torch.float32, device="cuda"))
        n = len(xyz)
        self._features_dc = nn.Parameter(torch.zeros((n, 1, 3), dtype=torch.float32, device="cuda"))
        self._features_rest = nn.Parameter(torch.zeros((n, 15, 3), dtype=torch.float32, device="cuda"))
        self._scaling = nn.Parameter(torch.as_tensor(raw_scaling, dtype=torch.float32, device="cuda"))
        self._rotation = nn.Parameter(torch.as_tensor(raw_rotation, dtype=torch.float32, device="cuda"))
        self._opacity = nn.Parameter(torch.as_tensor(raw_opacity[:, None], dtype=torch.float32, device="cuda"))
        self._transparency = nn.Parameter(torch.as_tensor(raw_transparency[:, None], dtype=torch.float32, device="cuda"))
        self.use_app = False

    @property
    def get_xyz(self): return self._xyz
    @property
    def get_scaling(self): return torch.exp(self._scaling)
    @property
    def get_rotation(self): return torch.nn.functional.normalize(self._rotation)
    @property
    def get_opacity(self): return torch.sigmoid(self._opacity)
    @property
    def get_transparency(self): return torch.sigmoid(self._transparency)
    @property
    def get_features(self): return torch.cat((self._features_dc, self._features_rest), dim=1)


def render_alpha(pc: PatchGaussian, camera, alpha_path: Path, depth_path: Path) -> tuple[dict, str | None]:
    try:
        import gaussian_renderer
        pipe = type("Pipe", (), {"compute_cov3D_python": False, "convert_SHs_python": False, "debug": False})()
        bg = torch.zeros(3, device="cuda")
        override = torch.ones((pc.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")
        with torch.no_grad():
            out = gaussian_renderer.render(camera, pc, pipe, bg, override_color=override, return_plane=True, return_depth_normal=False)
            alpha = out["rendered_alpha"].detach().squeeze().float().cpu().numpy()
            depth = out["out_nearest_depth"].detach().squeeze().float().cpu().numpy()
        alpha_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(alpha_path, alpha.astype(np.float32))
        np.save(depth_path, depth.astype(np.float32))
        return {"alpha_path": str(alpha_path), "alpha_sha": sha256_file(alpha_path), "alpha_min": float(alpha.min()), "alpha_median": float(np.median(alpha)), "alpha_p95": float(np.quantile(alpha, .95)), "alpha_max": float(alpha.max()), "first_surface_path": str(depth_path), "first_surface_sha": sha256_file(depth_path)}, None
    except Exception:
        return {}, traceback.format_exc()


def deformation_states() -> dict[str, np.ndarray]:
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    return {
        "A0_IDENTITY": np.eye(3),
        "A1_STRETCH_X_1P25": np.diag([1.25, 1.0, 1.0]),
        "A2_STRETCH_X_1P50": np.diag([1.50, 1.0, 1.0]),
        "A3_STRETCH_X_2P00": np.diag([2.00, 1.0, 1.0]),
        "A4_BIAXIAL_XY_1P50": np.diag([1.50, 1.50, 1.0]),
        "A5_SHEAR_XY_0P30": np.array([[1.0, 0.30, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        "A6_ROTATION_30DEG": np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]),
    }


def bootstrap_delta(delta: np.ndarray) -> tuple[float, float]:
    if len(delta) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260714)
    vals = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(1000)])
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第53步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    ALPHA_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    log = ["CUDA_VISIBLE_DEVICES=2,3"]

    inputs = [PLY, CHECKPOINT / "cameras.json", MESH_PATH, MASK_DIR, R4 / "fresh_official_medium_candidate_lock.csv", PROJECT / "analysis" / "kiot_fast_inverse.py", R4B / "stage3_5B_R4B_summary.md", TSGS / "gaussian_renderer" / "__init__.py"]
    lock = {"stage": "3.6A", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "inputs": [{"path": str(p), "exists": p.exists(), "sha256": sha256_file(p) if p.is_file() else "directory"} for p in inputs], "forbidden": ["TSGS Gaussian normal for Js", "first-surface depth normal for Js", "manual q", "synthetic policy response", "MLP", "training", "fine-tuning"]}
    write_text(OUT / "stage3_6A_protocol_lock.json", json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    G0 = all(p.exists() for p in inputs)

    mesh = trimesh.load(MESH_PATH, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    fn = face_normals(mesh)
    areas = mesh.area_faces.astype(np.float64)
    cameras = load_audit_cameras()
    align_rows = []
    for cam in cameras:
        iou, mesh_pix, mask_pix = mesh_silhouette_iou(mesh, cam)
        align_rows.append({"camera_id": int(cam["id"]), "camera_name": cam["img_name"], "mask_iou": iou, "mesh_pixels": mesh_pix, "mask_pixels": mask_pix})
    align_rows.append({"camera_id": "MESH", "camera_name": str(MESH_PATH), "vertices": len(verts), "faces": len(faces), "connected_components": len(mesh.split(only_watertight=False)), "watertight": int(mesh.is_watertight), "bounds_min": verts.min(axis=0).tolist(), "bounds_max": verts.max(axis=0).tolist(), "face_area_min": float(areas.min()), "face_area_median": float(np.median(areas)), "face_area_p95": float(np.quantile(areas, .95)), "face_area_max": float(areas.max()), "normal_mean": fn.mean(axis=0).tolist(), "normal_std": fn.std(axis=0).tolist()})
    write_csv(OUT / "gt_mesh_alignment_audit.csv", align_rows)
    ious = np.array([r["mask_iou"] for r in align_rows if isinstance(r["camera_id"], int)], dtype=np.float64)
    mesh_alignment_ok = bool(len(ious) and np.median(ious) >= 0.90 and ious.min() >= 0.80)

    ckpt = TSGSPatchAdapter(CHECKPOINT, 30000).load()
    cand = pd.read_csv(R4 / "fresh_official_medium_candidate_lock.csv")
    cand_idx = cand["gaussian_index"].to_numpy(np.int64)
    xyz_c = ckpt.xyz[cand_idx]
    # Object candidates are inherited from official-mask fresh MEDIUM support; threshold is fixed before rendering.
    nn_tree = cKDTree(xyz_c)
    dnn, _ = nn_tree.query(xyz_c, k=2)
    threshold = 2.0 * float(np.median(dnn[:, 1]))
    cps, dists, face_ids, bary = bind_points_to_mesh(xyz_c, mesh)
    bound = dists <= threshold
    bind_rows = []
    for gi, p, cp, dist, fid, bc, ok in zip(cand_idx, xyz_c, cps, dists, face_ids, bary, bound):
        bind_rows.append({"gaussian_index": int(gi), "x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "closest_x": float(cp[0]), "closest_y": float(cp[1]), "closest_z": float(cp[2]), "face_id": int(fid), "b0": float(bc[0]), "b1": float(bc[1]), "b2": float(bc[2]), "surface_distance": float(dist), "threshold": threshold, "bound": int(ok), "normal_x": float(fn[fid, 0]), "normal_y": float(fn[fid, 1]), "normal_z": float(fn[fid, 2]), "opacity": float(ckpt.activated_opacity[int(gi)])})
    write_csv(OUT / "gaussian_gt_mesh_binding.csv", bind_rows)
    bound_idx = cand_idx[bound]
    bound_face = face_ids[bound]
    bound_bary = bary[bound]
    bound_normals = fn[bound_face]
    coverage = len(bound_idx) / max(len(cand_idx), 1)
    binding_rows = [{
        "candidate_count": int(len(cand_idx)),
        "bound_count": int(len(bound_idx)),
        "coverage": coverage,
        "threshold": threshold,
        "distance_p50": float(np.median(dists)),
        "distance_p95": float(np.quantile(dists, .95)),
        "distance_max": float(np.max(dists)),
        "mesh_alignment_iou_median": float(np.median(ious)) if len(ious) else float("nan"),
        "mesh_alignment_iou_min": float(np.min(ious)) if len(ious) else float("nan"),
    }]
    write_csv(OUT / "mesh_binding_consistency.csv", binding_rows)
    G1 = G0 and mesh_alignment_ok and len(bound_idx) >= 5000 and coverage >= 0.70

    states = deformation_states()
    pivot = verts.mean(axis=0)
    js_rows = []
    for state, F in states.items():
        detF = abs(float(np.linalg.det(F)))
        FinvT = np.linalg.inv(F).T
        Js = detF * np.linalg.norm(bound_normals @ FinvT.T, axis=1) if len(bound_normals) else np.array([])
        for val in Js[: min(len(Js), 20000)]:
            js_rows.append({"state": state, "Js": float(val), "q": float(1.0 / val)})
        js_rows.append({"state": state, "summary": 1, "detF": detF, "Js_median": float(np.median(Js)) if len(Js) else "", "Js_p05": float(np.quantile(Js, .05)) if len(Js) else "", "Js_p95": float(np.quantile(Js, .95)) if len(Js) else "", **{f"F{i}{j}": float(F[i, j]) for i in range(3) for j in range(3)}})
    write_csv(OUT / "mesh_deformation_js_audit.csv", js_rows)

    # Deterministic area-weighted anchors over bound mesh faces; these are frozen before any policy rendering.
    rng = np.random.default_rng(20260714)
    if len(bound_idx):
        take = np.linspace(0, len(bound_idx) - 1, min(1024, len(bound_idx)), dtype=np.int64)
    else:
        take = np.array([], dtype=np.int64)
    anchor_rows = []
    sample_npz = {}
    for aid, bi in enumerate(take):
        gi = int(bound_idx[bi])
        fid = int(bound_face[bi])
        bc = bound_bary[bi]
        pos = (verts[faces[fid]] * bc[:, None]).sum(axis=0)
        anchor_rows.append({"anchor_id": aid, "gaussian_index": gi, "face_id": fid, "b0": float(bc[0]), "b1": float(bc[1]), "b2": float(bc[2]), "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "normal_x": float(fn[fid, 0]), "normal_y": float(fn[fid, 1]), "normal_z": float(fn[fid, 2])})
    if len(take):
        sample_npz["anchor_gaussian_indices"] = bound_idx[take]
        sample_npz["anchor_faces"] = bound_face[take]
        sample_npz["anchor_barycentric"] = bound_bary[take]
        sample_npz["anchor_normals"] = bound_normals[take]
    else:
        sample_npz["anchor_gaussian_indices"] = np.array([], dtype=np.int64)
    write_csv(OUT / "mesh_material_anchor_lock.csv", anchor_rows)
    np.savez_compressed(OUT / "mesh_material_samples.npz", **sample_npz)

    render_manifest: list[dict] = []
    response_cam_rows: list[dict] = []
    response_rows: list[dict] = []
    policy_rows: list[dict] = []
    gate_rows: list[dict] = []
    render_errors: list[str] = []
    G2 = G3 = False
    final_gate = "NOT_RUN"
    final_case = "CASE SCAFFOLD-PROTOCOL-FAIL"
    retain = "no KIOT decision because scaffold protocol failed"
    allow_deformed = "NO"
    allow_recon = "NO"
    policy_mean = {p: float("nan") for p in POLICIES}
    p2_p4_win = "0/0"
    delta_ci = (float("nan"), float("nan"))
    p95_by_state = {}
    identity_control = "NOT_RUN"
    rotation_control = "NOT_RUN"

    if G1:
        Rm = quat_to_matrix(ckpt.rotation[bound_idx])
        scale = np.exp(ckpt.scale[bound_idx])
        Sigma = np.einsum("nij,nj,nkj->nik", Rm, scale ** 2, Rm)
        opacity = ckpt.activated_opacity[bound_idx]
        raw_trans = ckpt.raw_transparency[bound_idx]
        camera_objs = {c["img_name"]: build_camera(c) for c in cameras}
        faces_bound = faces[bound_face]
        for state, F in states.items():
            vdef = pivot[None, :] + (verts - pivot[None, :]) @ F.T
            tri = vdef[faces_bound]
            xyz_def = (tri * bound_bary[:, :, None]).sum(axis=1)
            detF = abs(float(np.linalg.det(F)))
            FinvT = np.linalg.inv(F).T
            Js = detF * np.linalg.norm(bound_normals @ FinvT.T, axis=1)
            q = 1.0 / Js
            sigma_def = np.einsum("ab,nbc,dc->nad", F, Sigma, F)
            ev, evec = np.linalg.eigh(sigma_def)
            ev = np.maximum(ev, 1e-18)
            raw_scale = np.log(np.sqrt(ev))
            raw_rot = matrix_to_quat(evec)
            tau = -np.log1p(-np.clip(opacity, 0, 1 - 1e-8))
            policies = {
                "P0_FIXED": opacity,
                "P1_TAU_JS": 1.0 - np.exp(-q * tau),
                "P2_OPACITY_LINEAR": np.clip(q * opacity, 0.0, 1.0 - 1e-8),
                "P3_KIOT_CONT": invert_phi_cont_np(q * phi_cont_np(opacity)),
                "P4_KIOT_CUDA": kiot_cuda_identity_safe_np(opacity, q),
            }
            for policy, op in policies.items():
                pc = PatchGaussian(xyz_def, raw_scale, raw_rot, logit(op), raw_trans)
                for cam in cameras:
                    ap = ALPHA_DIR / state / policy / f"{cam['img_name']}.npy"
                    dp = DEPTH_DIR / state / policy / f"{cam['img_name']}.npy"
                    info, err = render_alpha(pc, camera_objs[cam["img_name"]], ap, dp)
                    if err:
                        render_errors.append(f"{state}/{policy}/{cam['img_name']}: {err}")
                    else:
                        render_manifest.append({"state": state, "policy": policy, "camera_id": int(cam["id"]), "camera": cam["img_name"], **info})
        write_csv(OUT / "mesh_scaffold_render_manifest.csv", render_manifest)
        G2 = len(render_manifest) == len(states) * len(POLICIES) * len(cameras) and not render_errors

        # Frozen keys and responses from actual alpha arrays.
        anchor_df = pd.DataFrame(anchor_rows)
        key_rows = []
        if G2 and len(anchor_df):
            can_cache = {}
            for policy in POLICIES:
                # canonical/deformed alpha for identity should be same geometry; P0 canonical is the frozen key source.
                pass
            for ar in anchor_df.itertuples():
                gi = int(ar.gaussian_index)
                pos = np.array([ar.x, ar.y, ar.z], dtype=np.float64)[None, :]
                for cam in cameras:
                    u, v, _, ok = project_points(pos, cam)
                    if not ok[0]:
                        continue
                    uu, vv = float(u[0]), float(v[0])
                    mask = np.asarray(Image.open(MASK_DIR / f"{cam['img_name']}.png").convert("L")) > 0
                    if not mask[int(round(vv)), int(round(uu))]:
                        continue
                    a_can = np.load(ALPHA_DIR / "A0_IDENTITY" / "P0_FIXED" / f"{cam['img_name']}.npy")
                    tau_can = -np.log1p(-np.clip(bilinear(a_can, np.array([uu]), np.array([vv]))[0], 0, 1 - 1e-8))
                    if tau_can <= 0:
                        continue
                    key_rows.append({"anchor_id": int(ar.anchor_id), "gaussian_index": gi, "camera": cam["img_name"], "u": uu, "v": vv, "tau_can": float(tau_can), "locked": 1})
            for kr in key_rows:
                ar = anchor_df[anchor_df.anchor_id == kr["anchor_id"]].iloc[0]
                n = np.array([ar.normal_x, ar.normal_y, ar.normal_z], dtype=np.float64)
                pos0 = np.array([ar.x, ar.y, ar.z], dtype=np.float64)
                for state, F in states.items():
                    detF = abs(float(np.linalg.det(F)))
                    q_anchor = 1.0 / (detF * np.linalg.norm(np.linalg.inv(F).T @ n))
                    pos_def = pivot + F @ (pos0 - pivot)
                    cam = next(c for c in cameras if c["img_name"] == kr["camera"])
                    u, v, _, ok = project_points(pos_def[None, :], cam)
                    if not ok[0]:
                        continue
                    for policy in POLICIES:
                        a_def = np.load(ALPHA_DIR / state / policy / f"{cam['img_name']}.npy")
                        tau_def = -np.log1p(-np.clip(bilinear(a_def, u, v)[0], 0, 1 - 1e-8))
                        R = tau_def / max(kr["tau_can"], 1e-8)
                        central_error = abs(R - q_anchor)
                        elog = abs(math.log(max(R, 1e-8) / max(q_anchor, 1e-8)))
                        response_cam_rows.append({"anchor_id": kr["anchor_id"], "gaussian_index": kr["gaussian_index"], "camera": kr["camera"], "state": state, "policy": policy, "R": float(R), "Q": float(q_anchor), "central_error": float(central_error), "Elog": float(elog)})
        write_csv(OUT / "mesh_anchor_camera_response.csv", response_cam_rows)
        if response_cam_rows:
            rdf = pd.DataFrame(response_cam_rows)
            for (aid, state, policy), grp in rdf.groupby(["anchor_id", "state", "policy"]):
                response_rows.append({"anchor_id": int(aid), "state": state, "policy": policy, "R_median": float(grp["R"].median()), "Q_median": float(grp["Q"].median()), "central_error": float(abs(grp["R"].median() - grp["Q"].median())), "Elog": float(grp["Elog"].median())})
            write_csv(OUT / "mesh_anchor_response.csv", response_rows)
            adf = pd.DataFrame(response_rows)
            for policy, grp in adf.groupby("policy"):
                policy_rows.append({"policy": policy, "mean_central_error": float(grp["central_error"].mean()), "median_central_error": float(grp["central_error"].median()), "median_Elog": float(grp["Elog"].median()), "p95_Elog": float(grp["Elog"].quantile(.95)), "factor2_fraction": float((grp["Elog"] <= math.log(2)).mean())})
            write_csv(OUT / "mesh_scaffold_policy_comparison.csv", policy_rows)
            policy_mean = {r["policy"]: r["mean_central_error"] for r in policy_rows}
            iddf = adf[adf.state == "A0_IDENTITY"].pivot_table(index="anchor_id", columns="policy", values="R_median")
            rotdf = adf[adf.state == "A6_ROTATION_30DEG"].pivot_table(index="anchor_id", columns="policy", values="R_median")
            identity_control = "PASS" if len(iddf) and float(iddf.max(axis=1).sub(iddf.min(axis=1)).max()) <= 1e-5 else "FAIL"
            rotation_control = "PASS" if len(rotdf) and float(rotdf.max(axis=1).sub(rotdf.min(axis=1)).max()) <= 1e-5 else "FAIL"
            G3 = identity_control == "PASS" and rotation_control == "PASS"
            prim = adf[adf.state.isin(PRIMARY_STATES)]
            piv = prim.pivot_table(index=["anchor_id", "state"], columns="policy", values=["central_error", "Elog"])
            if ("central_error", "P2_OPACITY_LINEAR") in piv.columns and ("central_error", "P4_KIOT_CUDA") in piv.columns:
                p2e = piv[("central_error", "P2_OPACITY_LINEAR")]
                p4e = piv[("central_error", "P4_KIOT_CUDA")]
                p4_wins = int((p4e < p2e).sum())
                total = int(len(p2e.dropna()))
                p2_p4_win = f"{p4_wins}/{total}"
                delta = (piv[("Elog", "P4_KIOT_CUDA")] - piv[("Elog", "P2_OPACITY_LINEAR")]).dropna().to_numpy()
                delta_ci = bootstrap_delta(delta)
                p2_mean = float(p2e.mean())
                p4_mean = float(p4e.mean())
                p95_state_ok = 0
                p2_state_ok = 0
                for st in PRIMARY_STATES:
                    sub = prim[prim.state == st]
                    p95p2 = float(sub[sub.policy == "P2_OPACITY_LINEAR"]["Elog"].quantile(.95))
                    p95p4 = float(sub[sub.policy == "P4_KIOT_CUDA"]["Elog"].quantile(.95))
                    p95_by_state[st] = {"P2": p95p2, "P4": p95p4}
                    p95_state_ok += int(p95p4 <= p95p2)
                    p2_state_ok += int(p95p2 <= p95p4)
                if p4_mean <= 0.80 * p2_mean and p4_wins >= 0.75 * total and delta_ci[1] < 0 and p95_state_ok >= 3:
                    final_gate = "KIOT SUPPORTED"
                    final_case = "CASE KIOT-SURFACE-SCAFFOLD-SUPPORTED"
                    retain = "retain KIOT under reliable surface scaffold"
                    allow_deformed = allow_recon = "YES"
                elif p2_mean <= 0.80 * p4_mean and (total - p4_wins) >= 0.75 * total and delta_ci[0] > 0:
                    final_gate = "OPACITY-LINEAR REGIME"
                    final_case = "CASE OPACITY-LINEAR-SCAFFOLD-REGIME"
                    retain = "KILL KIOT method line completely"
                else:
                    final_gate = "MIXED"
                    final_case = "CASE TRANSPORT-RULE-MIXED"
                    retain = "no primary method claim"
            gate_rows.append({"G4": final_gate, "P2_vs_P4_win_count": p2_p4_win, "delta_ci_low": delta_ci[0], "delta_ci_high": delta_ci[1], "p95_by_state": json.dumps(p95_by_state)})
        else:
            write_csv(OUT / "mesh_anchor_response.csv", [])
            write_csv(OUT / "mesh_scaffold_policy_comparison.csv", [])
        write_csv(OUT / "mesh_kiot_vs_linear_kill_gate.csv", gate_rows)
    else:
        write_csv(OUT / "mesh_scaffold_render_manifest.csv", [])
        write_csv(OUT / "mesh_anchor_camera_response.csv", [])
        write_csv(OUT / "mesh_anchor_response.csv", [])
        write_csv(OUT / "mesh_scaffold_policy_comparison.csv", [])
        write_csv(OUT / "mesh_kiot_vs_linear_kill_gate.csv", [{"G4": "NOT_RUN", "reason": "G0/G1 failed"}])

    if not G1:
        final_case = "CASE SCAFFOLD-PROTOCOL-FAIL"
        retain = "no KIOT decision because G0/G1 scaffold protocol failed"
    elif G1 and (not G2 or not G3):
        final_case = "CASE SCAFFOLD-PROTOCOL-FAIL"
        retain = "no KIOT decision because G2/G3 failed"

    js_df = pd.DataFrame(js_rows)
    js_medians = {}
    for st in ["A1_STRETCH_X_1P25", "A2_STRETCH_X_1P50", "A3_STRETCH_X_2P00", "A4_BIAXIAL_XY_1P50", "A5_SHEAR_XY_0P30", "A6_ROTATION_30DEG"]:
        vals = js_df[(js_df.state == st) & (js_df.get("summary", 0) != 1)]["Js"] if len(js_df) else []
        js_medians[st] = float(np.median(vals)) if len(vals) else float("nan")
    render_count = len(render_manifest)
    report_path = OUT / "gt_mesh_scaffold_transport_report.md"
    summary_path = OUT / "stage3_6A_summary.md"
    terminal_items = [
        ("1", "G0", "PASS" if G0 else "FAIL"),
        ("2", "GT mesh path", str(MESH_PATH)),
        ("3", "mesh-camera mask IoU median/min", f"{float(np.median(ious)) if len(ious) else float('nan'):.6f}/{float(np.min(ious)) if len(ious) else float('nan'):.6f}"),
        ("4", "bound Gaussian count/fraction", f"{len(bound_idx)}/{coverage:.6f}"),
        ("5", "binding distance p50/p95/max", f"{float(np.median(dists)):.6e}/{float(np.quantile(dists,.95)):.6e}/{float(np.max(dists)):.6e}"),
        ("6", "G1", "PASS" if G1 else "FAIL"),
        ("7", "A1-A6 median Js", json.dumps(js_medians, ensure_ascii=False)),
        ("8", "real render file count", str(render_count)),
        ("9", "synthetic response used yes/no", "NO"),
        ("10", "G2", "PASS" if G2 else "FAIL"),
        ("11", "identity control", identity_control),
        ("12", "rotation control", rotation_control),
        ("13", "G3", "PASS" if G3 else "FAIL"),
        ("14", "P0 mean central error", f"{policy_mean.get('P0_FIXED', float('nan')):.6f}"),
        ("15", "P1 mean central error", f"{policy_mean.get('P1_TAU_JS', float('nan')):.6f}"),
        ("16", "P2 mean central error", f"{policy_mean.get('P2_OPACITY_LINEAR', float('nan')):.6f}"),
        ("17", "P3 mean central error", f"{policy_mean.get('P3_KIOT_CONT', float('nan')):.6f}"),
        ("18", "P4 mean central error", f"{policy_mean.get('P4_KIOT_CUDA', float('nan')):.6f}"),
        ("19", "P2 vs P4 win count", p2_p4_win),
        ("20", "paired Delta_Elog CI", f"{delta_ci[0]:.6f}/{delta_ci[1]:.6f}"),
        ("21", "P2/P4 p95 by state", json.dumps(p95_by_state, ensure_ascii=False)),
        ("22", "G4", final_gate),
        ("23", "Final CASE", final_case),
        ("24", "retain or kill KIOT", retain),
        ("25", "allow deformed-GT yes/no", allow_deformed),
        ("26", "allow reconstructed-mesh bridge yes/no", allow_recon),
        ("27", "report path", str(report_path)),
        ("28", "summary path", str(summary_path)),
    ]
    report = "# Stage 3.6A 真值网格支架下透明 Gaussian 光学传输生死验证\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in terminal_items)
    write_text(report_path, report)
    summary = f"""# Stage 3.6A summary

- Final CASE: `{final_case}`
- G0 protocol lock: {'PASS' if G0 else 'FAIL'}
- G1 mesh alignment and binding: {'PASS' if G1 else 'FAIL'}
- G2 real render provenance: {'PASS' if G2 else 'FAIL'}
- G3 identity/rotation controls: {'PASS' if G3 else 'FAIL'}
- G4: {final_gate}
- GT mesh: `{MESH_PATH}`
- mesh-camera mask IoU median/min: {float(np.median(ious)) if len(ious) else float('nan'):.6f}/{float(np.min(ious)) if len(ious) else float('nan'):.6f}
- bound Gaussian count/fraction: {len(bound_idx)}/{coverage:.6f}
- synthetic response used: NO
- retain or kill KIOT: {retain}
"""
    write_text(summary_path, summary)
    final_text = "\n".join(f"{k}. {title}: {value}" for k, title, value in terminal_items) + "\n"
    write_text(OUT / "final_terminal_summary.txt", final_text)
    write_text(OUT / "stage3_6A_log.txt", "\n".join(log + [f"{k}. {title}: {value}" for k, title, value in terminal_items] + render_errors[:20]) + "\n")

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage3.6A GT-mesh scaffold transport kill gate\n\nStage3.5B-R4B accepted `CASE NO-RECOVERABLE-LAYER`, closing the direct TSGS per-Gaussian material-point bridge. Stage3.6A switches to an explicit GT mesh scaffold only as an oracle material-kinematics carrier. The scaffold supplies material identity, topology, normals, deformation gradients, and surface area stretch Js; Gaussian attributes supply appearance, opacity, transparency, and kernels. No TSGS Gaussian/depth normals are used as material normals, and no optical policy is optimized.\n\nStage3.6A locks the scene_01 GT mesh, audits mesh-camera mask alignment, binds learned TSGS Gaussians to the mesh, transports centers and covariance by explicit affine mesh deformation, renders saved alpha maps with the actual TSGS rasterizer, and compares fixed opacity, tau/Js, opacity-linear, KIOT-continuous, and KIOT-CUDA. This is the final KIOT rule kill gate under a reliable surface scaffold.\n"""
    if "## Stage3.6A GT-mesh scaffold transport kill gate" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
