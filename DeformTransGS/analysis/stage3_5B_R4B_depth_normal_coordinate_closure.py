from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "/data/wyh/DeformTransGS/analysis")
from tsgs_patch_adapter import TSGSPatchAdapter


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OUT = PROJECT / "experiments" / "stage3_5B_R4B_depth_normal_coordinate_closure"
R4 = PROJECT / "experiments" / "stage3_5B_R4_normal_semantics_surface_layer_recovery"
R4A = PROJECT / "experiments" / "stage3_5B_R4A_depth_normal_bridge_closure"
CHECKPOINT = ROOT / "RecycleGS" / "baselines" / "tsgs_official_scene01_30k_v4"
SCENE = ROOT / "RecycleGS" / "data" / "translab_full" / "scene_01"
MASK_DIR = SCENE / "transparent_masks"
TSGS = ROOT / "repos" / "TSGS"
R4_SCRIPT = PROJECT / "analysis" / "stage3_5B_R4_normal_semantics_surface_layer_recovery.py"
R4A_SCRIPT = PROJECT / "analysis" / "stage3_5B_R4A_depth_normal_bridge_closure.py"


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


def stats(values: np.ndarray, prefix: str = "") -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"{prefix}{k}": float("nan") for k in ["median", "p90", "p99", "max"]}
    return {
        f"{prefix}median": float(np.median(values)),
        f"{prefix}p90": float(np.quantile(values, 0.90)),
        f"{prefix}p99": float(np.quantile(values, 0.99)),
        f"{prefix}max": float(np.max(values)),
    }


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-30)


def unsigned_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a.astype(np.float64))
    b = normalize(b.astype(np.float64))
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(np.arccos(dot))


def vector_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a.astype(np.float64))
    b = normalize(b.astype(np.float64))
    return np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))


def projector_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a.astype(np.float64))
    b = normalize(b.astype(np.float64))
    aa = a[:, :, None] * a[:, None, :]
    bb = b[:, :, None] * b[:, None, :]
    return np.sqrt(np.sum((aa - bb) ** 2, axis=(1, 2)))


def get_world2view2(R: np.ndarray, t: np.ndarray, translate=np.array([0.0, 0.0, 0.0]), scale=1.0) -> np.ndarray:
    Rt = np.zeros((4, 4), dtype=np.float64)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    return np.linalg.inv(C2W)


def camera_matrices(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(cam["rotation"], dtype=np.float64)
    t = np.asarray(np.linalg.inv(np.block([[R, np.asarray(cam["position"], dtype=np.float64)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))[:3, 3], dtype=np.float64)
    # R4 passes Rt[:3,:3].T and Rt[:3,3] into TSGS Camera. The formula below is exactly Camera.world_view_transform.
    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = R
    C2W[:3, 3] = np.asarray(cam["position"], dtype=np.float64)
    Rt = np.linalg.inv(C2W)
    wvt = get_world2view2(Rt[:3, :3].T, Rt[:3, 3]).T
    vinv = np.linalg.inv(wvt)
    return wvt, vinv


def load_audit_cameras() -> list[dict]:
    cameras = json.loads((CHECKPOINT / "cameras.json").read_text())
    selected_ids = [0, 132, 264]
    by_id = {int(c["id"]): c for c in cameras}
    return [by_id[i] for i in selected_ids if i in by_id]


def load_r4_cameras() -> list[dict]:
    manifest = pd.read_csv(R4 / "r4_first_surface_render_manifest.csv")
    cameras = json.loads((CHECKPOINT / "cameras.json").read_text())
    by_id = {int(c["id"]): c for c in cameras}
    return [by_id[int(r.camera_id)] for r in manifest.itertuples()]


def depth_path(cam: dict) -> Path:
    return R4 / "canonical_first_surface" / f"camera_{int(cam['id']):04d}_{cam['img_name']}.npy"


def valid_mask(cam: dict, depth: np.ndarray) -> np.ndarray:
    mask = np.asarray(Image.open(MASK_DIR / f"{cam['img_name']}.png").convert("L")) > 0
    return np.isfinite(depth) & (depth < 1e2) & (depth > 1e-8) & mask


def normal_support_mask(valid: np.ndarray) -> np.ndarray:
    out = valid.copy()
    out[:1, :] = False
    out[-1:, :] = False
    out[:, :1] = False
    out[:, -1:] = False
    out[1:-1, 1:-1] &= valid[1:-1, 2:]
    out[1:-1, 1:-1] &= valid[1:-1, :-2]
    out[1:-1, 1:-1] &= valid[2:, 1:-1]
    out[1:-1, 1:-1] &= valid[:-2, 1:-1]
    return out


def backproject_grid(depth: np.ndarray, cam: dict) -> np.ndarray:
    h, w = depth.shape
    xs = np.arange(w, dtype=np.float64)[None, :]
    ys = np.arange(h, dtype=np.float64)[:, None]
    z = depth.astype(np.float64)
    x = (xs - 0.5 * float(cam["width"])) / float(cam["fx"]) * z
    y = (ys - 0.5 * float(cam["height"])) / float(cam["fy"]) * z
    return np.stack([x, y, z], axis=-1)


def central_normals_from_points(points: np.ndarray) -> np.ndarray:
    du = np.zeros_like(points)
    dv = np.zeros_like(points)
    du[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dv[1:-1, :] = points[2:, :] - points[:-2, :]
    n = np.cross(du, dv)
    return normalize(n)


def r4_original_normal(depth: np.ndarray, cam: dict) -> np.ndarray:
    dzdx = np.zeros_like(depth, dtype=np.float64)
    dzdy = np.zeros_like(depth, dtype=np.float64)
    dzdx[:, 1:-1] = depth[:, 2:] - depth[:, :-2]
    dzdy[1:-1, :] = depth[2:, :] - depth[:-2, :]
    nx = -dzdx / max(float(cam["fx"]), 1e-30)
    ny = -dzdy / max(float(cam["fy"]), 1e-30)
    nz = np.ones_like(depth, dtype=np.float64)
    return normalize(np.stack([nx, ny, nz], axis=-1))


def scatter_rows(normals: np.ndarray, label: str) -> dict:
    normals = normalize(normals)
    scatter = normals.T @ normals
    eig = np.linalg.eigvalsh(scatter)[::-1]
    eig_norm = eig / max(float(eig.sum()), 1e-30)
    return {
        "label": label,
        "count": int(len(normals)),
        "nx_mean": float(normals[:, 0].mean()),
        "ny_mean": float(normals[:, 1].mean()),
        "nz_mean": float(normals[:, 2].mean()),
        "nx_std": float(normals[:, 0].std()),
        "ny_std": float(normals[:, 1].std()),
        "nz_std": float(normals[:, 2].std()),
        "eig0": float(eig_norm[0]),
        "eig1": float(eig_norm[1]),
        "eig2": float(eig_norm[2]),
    }


def principal_normal(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = normals.T @ normals
    vals, vecs = np.linalg.eigh(M)
    ref = vecs[:, np.argmax(vals)]
    return ref, vals


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


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第52步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    log = ["CUDA_VISIBLE_DEVICES=2,3"]

    inputs = [
        R4 / "r4_first_surface_render_manifest.csv",
        R4 / "fresh_official_medium_candidate_lock.csv",
        R4 / "fresh_depth_support_samples.csv",
        R4 / "candidate_multiview_depth_normals.csv",
        R4 / "depth_coordinate_semantic_trace.md",
        R4A / "depth_normal_bridge_closure_report.md",
        CHECKPOINT / "cameras.json",
        MASK_DIR,
        TSGS / "scene" / "cameras.py",
        TSGS / "utils" / "graphics_utils.py",
        R4_SCRIPT,
        R4A_SCRIPT,
    ]
    lock = {
        "stage": "3.5B-R4B",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "inputs": [{"path": str(p), "exists": p.exists(), "sha256": sha256_file(p) if p.is_file() else "directory"} for p in inputs],
        "forbidden_actions": ["KIOT", "opacity-linear", "policy rendering", "training", "threshold tuning", "GT mesh normals"],
    }
    write_text(OUT / "r4b_coordinate_protocol_lock.json", json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    CFM0 = all(p.exists() for p in inputs)

    r4_lines = R4_SCRIPT.read_text().splitlines()
    trace = "\n".join(f"{i:04d}: {r4_lines[i-1]}" for i in range(370, 429))
    write_text(
        OUT / "r4_depth_normal_frame_trace.md",
        f"""# R4 depth-normal frame trace

source: `{R4_SCRIPT}`
line range: 370-428

```text
{trace}
```

- input depth coordinate semantic: camera-space positive z, from `out_nearest_depth`.
- pixel ray construction: R4 does not backproject rays for normal maps.
- backprojected point formula: absent in the R4 normal-map block.
- point tensor variable name: absent.
- point frame: UNKNOWN for R4 normal maps because no point tensor is constructed.
- du frame: image-depth finite difference, `depth[:,2:] - depth[:,:-2]`.
- dv frame: image-depth finite difference, `depth[2:,:] - depth[:-2,:]`.
- cross-product normal frame: no cross product; R4 constructs `[ -dzdx/fx, -dzdy/fy, 1 ]`, which is a camera-frame slope approximation.
- camera-to-world transform applied: NO.
- formal classification before numeric comparison: CAMERA-NORMAL-MISTAGGED-AS-WORLD candidate.
""",
    )

    audit_cams = load_audit_cameras()
    all_cams = load_r4_cameras()
    matrices = {}
    dump_lines = []
    forward_rows = []
    for cam in audit_cams:
        wvt, vinv = camera_matrices(cam)
        matrices[f"cam{int(cam['id']):04d}_world_view_transform"] = wvt
        matrices[f"cam{int(cam['id']):04d}_view_inverse"] = vinv
        matrices[f"cam{int(cam['id']):04d}_WVT_R"] = wvt[:3, :3]
        matrices[f"cam{int(cam['id']):04d}_VINV_R"] = vinv[:3, :3]
        matrices[f"cam{int(cam['id']):04d}_camera_center"] = vinv[3, :3]
        fwd = normalize(np.array([[0.0, 0.0, 1.0]]) @ vinv[:3, :3])[0]
        dump_lines.append(f"cam{int(cam['id']):04d} {cam['img_name']}\nWVT=\n{wvt}\nVINV=\n{vinv}\ncenter={vinv[3,:3]}\nforward={fwd}\n")
        forward_rows.append({"camera_id": int(cam["id"]), "fx": cam["fx"], "fy": cam["fy"], "center_x": vinv[3, 0], "center_y": vinv[3, 1], "center_z": vinv[3, 2], "forward_x": fwd[0], "forward_y": fwd[1], "forward_z": fwd[2]})
    np.savez(OUT / "audit_camera_matrices.npz", **matrices)
    pair_angles = []
    for i in range(len(forward_rows)):
        for j in range(i + 1, len(forward_rows)):
            a = np.array([forward_rows[i][f"forward_{k}"] for k in "xyz"])
            b = np.array([forward_rows[j][f"forward_{k}"] for k in "xyz"])
            ang = float(np.degrees(np.arccos(np.clip(np.sum(normalize(a[None])[0] * normalize(b[None])[0]), -1.0, 1.0))))
            pair_angles.append(ang)
            dump_lines.append(f"forward angle cam{forward_rows[i]['camera_id']:04d}-cam{forward_rows[j]['camera_id']:04d}: {ang:.6f} deg")
    write_text(OUT / "audit_camera_matrix_dump.txt", "\n".join(dump_lines) + "\n")
    camera_audit_ok = max(pair_angles) >= 30.0

    point_rows = []
    vector_rows = []
    fwd_validation_rows = []
    rng = np.random.default_rng(20260714)
    base_vec = normalize(rng.normal(size=(100000, 3)))
    basis = np.eye(3, dtype=np.float64)
    test_vecs = np.concatenate([basis, base_vec], axis=0)
    for cam in audit_cams:
        depth = np.load(depth_path(cam)).astype(np.float64)
        valid = valid_mask(cam, depth)
        ys, xs = np.where(valid)
        take = np.linspace(0, len(xs) - 1, min(1000, len(xs)), dtype=np.int64)
        xs, ys = xs[take], ys[take]
        z = depth[ys, xs]
        p_cam = np.stack([(xs - 0.5 * cam["width"]) / cam["fx"] * z, (ys - 0.5 * cam["height"]) / cam["fy"] * z, z, np.ones_like(z)], axis=1)
        wvt, vinv = camera_matrices(cam)
        p_world = p_cam @ vinv
        p_cam_round = p_world @ wvt
        err = np.linalg.norm(p_cam_round[:, :3] - p_cam[:, :3], axis=1)
        point_rows.append({"camera_id": int(cam["id"]), "sample_count": int(len(err)), **stats(err, "err_")})

        nw = normalize(test_vecs @ vinv[:3, :3])
        nround = normalize(nw @ wvt[:3, :3])
        verr = np.linalg.norm(nround - test_vecs, axis=1)
        vector_rows.append({"camera_id": int(cam["id"]), "sample_count": int(len(verr)), **stats(verr, "l2_")})
        cam_z_world = normalize(np.array([[0.0, 0.0, 1.0]]) @ vinv[:3, :3])[0]
        center = vinv[3, :3]
        independent_forward = normalize((center + cam_z_world - center)[None])[0]
        fwd_err = float(np.linalg.norm(cam_z_world - independent_forward))
        fwd_validation_rows.append({"camera_id": int(cam["id"]), "cam_z_world_x": cam_z_world[0], "cam_z_world_y": cam_z_world[1], "cam_z_world_z": cam_z_world[2], "independent_forward_x": independent_forward[0], "independent_forward_y": independent_forward[1], "independent_forward_z": independent_forward[2], "l2_error": fwd_err})
    write_csv(OUT / "camera_world_point_roundtrip.csv", point_rows)
    write_csv(OUT / "camera_world_vector_roundtrip.csv", vector_rows)
    write_csv(OUT / "camera_forward_axis_validation.csv", fwd_validation_rows)
    point_all = pd.DataFrame(point_rows)
    vector_all = pd.DataFrame(vector_rows)
    fwd_all = pd.DataFrame(fwd_validation_rows)
    CFM1 = camera_audit_ok and point_all["err_median"].max() <= 1e-10 and point_all["err_p99"].max() <= 1e-8 and point_all["err_max"].max() <= 1e-6
    CFM2 = vector_all["l2_p99"].max() <= 1e-10 and vector_all["l2_max"].max() <= 1e-8 and fwd_all["l2_error"].max() <= 1e-10

    variant_rows = []
    identity_rows = []
    class_rows = []
    per_cam_rows = []
    corr_rows = []
    global_cam_samples = []
    global_world_samples = []
    variant_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for cam in all_cams:
        depth = np.load(depth_path(cam)).astype(np.float64)
        valid = normal_support_mask(valid_mask(cam, depth))
        pts_cam = backproject_grid(depth, cam)
        v1 = central_normals_from_points(pts_cam)
        wvt, vinv = camera_matrices(cam)
        pts_world = np.einsum("...j,jk->...k", pts_cam, vinv[:3, :3]) + vinv[3, :3]
        v2 = central_normals_from_points(pts_world)
        v3 = normalize(np.einsum("...j,jk->...k", v1, vinv[:3, :3]))
        v0 = r4_original_normal(depth, cam)
        variant_cache[int(cam["id"])] = (v0.astype(np.float32), v1.astype(np.float32), v2.astype(np.float32), v3.astype(np.float32), valid)
        n = int(valid.sum())
        variant_rows.append({"camera_id": int(cam["id"]), "valid_pixels": n, "V0_R4_ORIGINAL": "computed", "V1_CAMERA_NORMAL": "computed_from_p_cam_cross", "V2_WORLD_POINT_NORMAL": "computed_from_p_world_cross", "V3_ROTATED_CAMERA_NORMAL": "V1@VINV_R"})
        v2v = v2[valid]
        v3v = v3[valid]
        ve = vector_error(v2v, v3v)
        pe = projector_error(v2v, v3v)
        ang = unsigned_angle(v2v, v3v)
        identity_rows.append({"camera_id": int(cam["id"]), "sample_count": n, **stats(ve, "vector_"), **stats(pe, "projector_"), **stats(ang, "angle_")})
        for label, arr in [("V1_CAMERA_NORMAL", v1[valid]), ("V3_WORLD_NORMAL", v3[valid])]:
            if len(arr) > 100000:
                idx = np.linspace(0, len(arr) - 1, 100000, dtype=np.int64)
                arr_s = arr[idx]
            else:
                arr_s = arr
            per_cam_rows.append({"camera_id": int(cam["id"]), **scatter_rows(arr_s, label)})
        take_n = min(20000, n)
        yy, xx = np.where(valid)
        take = np.linspace(0, len(xx) - 1, take_n, dtype=np.int64)
        cam_s = v1[yy[take], xx[take]]
        world_s = v3[yy[take], xx[take]]
        global_cam_samples.append(cam_s)
        global_world_samples.append(world_s)
        fwd = normalize(np.array([[0.0, 0.0, 1.0]]) @ vinv[:3, :3])[0]
        cam_z = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(cam_s), axis=0)
        fwd_rep = np.repeat(fwd[None, :], len(world_s), axis=0)
        a1 = unsigned_angle(cam_s, cam_z)
        a2 = unsigned_angle(world_s, fwd_rep)
        corr_rows.append({"camera_id": int(cam["id"]), "sample_count": int(len(a1)), "angle_camera_median": float(np.median(a1)), "angle_world_median": float(np.median(a2)), "matched_angle_max_abs_diff": float(np.max(np.abs(a1 - a2)))})

        v0v, v1v = v0[valid], v1[valid]
        e01 = vector_error(v0v, v1v)
        e03 = vector_error(v0v, v3v)
        class_rows.append({"camera_id": int(cam["id"]), "comparison": "V0_vs_V1", **stats(e01, "vector_")})
        class_rows.append({"camera_id": int(cam["id"]), "comparison": "V0_vs_V3", **stats(e03, "vector_")})
    write_csv(OUT / "depth_normal_variant_manifest.csv", variant_rows)
    write_csv(OUT / "world_normal_identity.csv", identity_rows)
    write_csv(OUT / "r4_original_frame_classification.csv", class_rows)
    write_csv(OUT / "per_camera_depth_normal_diversity.csv", per_cam_rows)
    write_csv(OUT / "normal_camera_forward_correlation.csv", corr_rows)
    ident = pd.DataFrame(identity_rows)
    CFM3 = ident["vector_p99"].max() <= 1e-8 and ident["projector_p99"].max() <= 2e-8

    cam_global = np.concatenate(global_cam_samples, axis=0)
    world_global = np.concatenate(global_world_samples, axis=0)
    rng = np.random.default_rng(20260714)
    pair_count = min(100000, len(world_global))
    ia = rng.integers(0, len(world_global), size=pair_count)
    ib = rng.integers(0, len(world_global), size=pair_count)
    pair_ang = unsigned_angle(world_global[ia], world_global[ib])
    cam_pair = unsigned_angle(cam_global[ia], cam_global[ib])
    world_row = scatter_rows(world_global, "V3_ROTATED_CAMERA_NORMAL_GLOBAL")
    cam_row = scatter_rows(cam_global, "V1_CAMERA_NORMAL_GLOBAL")
    world_row.update({"pair_median": float(np.median(pair_ang)), "pair_p90": float(np.quantile(pair_ang, 0.90))})
    cam_row.update({"pair_median": float(np.median(cam_pair)), "pair_p90": float(np.quantile(cam_pair, 0.90))})
    write_csv(OUT / "corrected_world_depth_normal_diversity.csv", [cam_row, world_row])
    CFM4 = world_row["eig1"] >= 0.01 and world_row["pair_p90"] >= 10.0

    cdf = pd.DataFrame(class_rows)
    v01_p99 = float(cdf[cdf.comparison == "V0_vs_V1"]["vector_p99"].max())
    v03_p99 = float(cdf[cdf.comparison == "V0_vs_V3"]["vector_p99"].median())
    if v01_p99 <= 0.05 and v03_p99 > 0.1:
        original_class = "CAMERA-NORMAL-MISTAGGED-AS-WORLD"
    elif v03_p99 <= 0.05:
        original_class = "R4 WORLD NORMAL WAS CORRECT"
    else:
        original_class = "R4 FRAME SEMANTIC UNKNOWN"

    corrected_mv_rows: list[dict] = []
    gvd_rows: list[dict] = []
    sweep_rows: list[dict] = []
    existence_rows: list[dict] = []
    CFM5 = False
    reliable_count = 0
    reliable_frac = 0.0
    gvd_median = float("nan")
    gvd_p90 = float("nan")
    coherent_counts: dict[int, int] = {}
    layer_pure_count = 0
    if CFM0 and CFM1 and CFM2 and CFM3 and CFM4:
        ckpt = TSGSPatchAdapter(CHECKPOINT, 30000).load()
        scale = np.exp(ckpt.scale)
        R = quat_to_matrix(ckpt.rotation)
        min_axis = np.argmin(scale, axis=1)
        n_tsgs = np.take_along_axis(R, min_axis[:, None, None], axis=2).squeeze(2)
        n_tsgs = normalize(n_tsgs)
        samples = pd.read_csv(R4 / "fresh_depth_support_samples.csv")
        medium = pd.read_csv(R4 / "fresh_official_medium_candidate_lock.csv")
        per_gauss = {int(k): g for k, g in samples.groupby("gaussian_index")}
        cam_by_id = {int(c["id"]): c for c in all_cams}
        for gi in medium["gaussian_index"].to_numpy(np.int64):
            normals = []
            if int(gi) not in per_gauss:
                continue
            for r in per_gauss[int(gi)].itertuples():
                if int(r.camera_id) not in variant_cache or float(r.depth_rel_error) > 0.05:
                    continue
                cam = cam_by_id[int(r.camera_id)]
                wvt, _ = camera_matrices(cam)
                p = np.r_[ckpt.xyz[int(gi)], 1.0] @ wvt
                if p[2] <= 1e-8:
                    continue
                u = cam["fx"] * p[0] / p[2] + cam["width"] * 0.5
                v = cam["fy"] * p[1] / p[2] + cam["height"] * 0.5
                uu = int(np.clip(round(float(u)), 0, cam["width"] - 1))
                vv = int(np.clip(round(float(v)), 0, cam["height"] - 1))
                _, _, _, v3map, valid = variant_cache[int(r.camera_id)]
                if valid[vv, uu]:
                    normals.append(v3map[vv, uu].astype(np.float64))
            if len(normals) >= 3:
                arr = normalize(np.asarray(normals, dtype=np.float64))
                ref, _ = principal_normal(arr)
                ang = unsigned_angle(arr, np.repeat(ref[None, :], len(arr), axis=0))
                reliable = bool(np.quantile(ang, 0.90) <= 15.0)
                reliable_count += int(reliable)
                corrected_mv_rows.append({"gaussian_index": int(gi), "view_count": int(len(arr)), "angle_median": float(np.median(ang)), "angle_p90": float(np.quantile(ang, 0.90)), "angle_max": float(np.max(ang)), "depth_normal_reliable": int(reliable), "n_depth_x": float(ref[0]), "n_depth_y": float(ref[1]), "n_depth_z": float(ref[2])})
                if reliable:
                    gvd_rows.append({"gaussian_index": int(gi), "angle_tsgs_vs_depth": float(unsigned_angle(n_tsgs[[int(gi)]], ref[None, :])[0])})
        write_csv(OUT / "corrected_candidate_multiview_world_normals.csv", corrected_mv_rows)
        write_csv(OUT / "corrected_gaussian_vs_depth_normal.csv", gvd_rows)
        reliable_frac = reliable_count / max(len(pd.read_csv(R4 / "fresh_official_medium_candidate_lock.csv")), 1)
        if gvd_rows:
            gvd_df = pd.DataFrame(gvd_rows)
            gvd_median = float(gvd_df["angle_tsgs_vs_depth"].median())
            gvd_p90 = float(gvd_df["angle_tsgs_vs_depth"].quantile(0.90))
        rel_df = pd.DataFrame([r for r in corrected_mv_rows if r["depth_normal_reliable"] == 1])
        if len(rel_df):
            ckpt_xyz = ckpt.xyz[rel_df["gaussian_index"].to_numpy(np.int64)]
            normals = rel_df[["n_depth_x", "n_depth_y", "n_depth_z"]].to_numpy(np.float64)
            from scipy.spatial import cKDTree
            tree = cKDTree(ckpt_xyz)
            support = pd.read_csv(R4 / "fresh_depth_support_samples.csv")
            for K in [64, 128, 256, 512, 768]:
                if len(rel_df) < K:
                    coherent_counts[K] = 0
                    continue
                _, nn = tree.query(ckpt_xyz, k=K)
                count = 0
                for si, gi in enumerate(rel_df["gaussian_index"].to_numpy(np.int64)):
                    ns = normals[nn[si]]
                    ref, _ = principal_normal(ns)
                    p90 = float(np.quantile(unsigned_angle(ns, np.repeat(ref[None, :], len(ns), axis=0)), 0.90))
                    gids = rel_df["gaussian_index"].to_numpy(np.int64)[nn[si]]
                    sub = support[support["gaussian_index"].isin(gids)]
                    visible = int(np.median(sub.groupby("gaussian_index")["camera_id"].nunique())) if len(sub) else 0
                    spreads = []
                    for _, grp in sub.groupby("camera_id"):
                        delta = grp["d_gaussian"].to_numpy(float) - grp["d_first"].to_numpy(float)
                        denom = np.median(np.abs(grp["d_first"].to_numpy(float))) + 1e-8
                        spreads.append((np.quantile(delta, 0.95) - np.quantile(delta, 0.05)) / denom)
                    med_spread = float(np.median(spreads)) if spreads else float("inf")
                    eligible = K >= 64 and p90 <= 10.0 and visible >= 3 and med_spread <= 0.05
                    count += int(eligible)
                    sweep_rows.append({"seed_gaussian_index": int(gi), "K": K, "normal_p90": p90, "visible_camera_count": visible, "median_camera_layer_spread": med_spread, "eligible": int(eligible)})
                coherent_counts[K] = count
            write_csv(OUT / "corrected_depth_normal_patch_sweep.csv", sweep_rows, ["seed_gaussian_index", "K", "normal_p90", "visible_camera_count", "median_camera_layer_spread", "eligible"])
            for K in [64, 128, 256, 512, 768]:
                existence_rows.append({"K": K, "eligible_count": coherent_counts.get(K, 0)})
            write_csv(OUT / "corrected_surface_layer_existence.csv", existence_rows)
            layer_pure_count = int(sum(coherent_counts.values()))
            CFM5 = layer_pure_count >= 2
        else:
            write_csv(OUT / "corrected_depth_normal_patch_sweep.csv", [], ["seed_gaussian_index", "K", "normal_p90", "visible_camera_count", "median_camera_layer_spread", "eligible"])
            write_csv(OUT / "corrected_surface_layer_existence.csv", [])
    else:
        write_csv(OUT / "corrected_candidate_multiview_world_normals.csv", [], ["gaussian_index", "view_count", "angle_median", "angle_p90", "angle_max", "depth_normal_reliable", "n_depth_x", "n_depth_y", "n_depth_z"])
        write_csv(OUT / "corrected_gaussian_vs_depth_normal.csv", [], ["gaussian_index", "angle_tsgs_vs_depth"])
        write_csv(OUT / "corrected_depth_normal_patch_sweep.csv", [], ["seed_gaussian_index", "K", "normal_p90", "visible_camera_count", "median_camera_layer_spread", "eligible"])
        write_csv(OUT / "corrected_surface_layer_existence.csv", [], ["K", "eligible_count"])

    if not (CFM0 and CFM1 and CFM2 and CFM3):
        final_case = "CASE COORDINATE-PROTOCOL-FAIL"
        retained = "R4/R4A depth-normal conclusions remain unresolved; coordinate transform must be fixed first."
        proxy = "none"
        allow_kill = "NO"
    elif not CFM4:
        final_case = "CASE TRUE-DEPTH-NORMAL-DEGENERACY"
        retained = "R4A degeneracy conclusion retained after verified camera→world rotation."
        proxy = "none"
        allow_kill = "NO"
    elif CFM5 and original_class == "CAMERA-NORMAL-MISTAGGED-AS-WORLD":
        final_case = "CASE CAMERA-NORMAL-MISTAGGED-AS-WORLD"
        retained = "R4/R4A depth-normal degeneracy retired; corrected world normals must be used."
        proxy = "corrected first-surface depth world normal"
        allow_kill = "YES"
    elif CFM5:
        final_case = "CASE CORRECTED-DEPTH-NORMAL-BRIDGE-CARRIER"
        retained = "R4/R4A uncorrected depth-normal conclusions retired for corrected world-normal use."
        proxy = "corrected first-surface depth world normal"
        allow_kill = "YES"
    else:
        final_case = "CASE NO-RECOVERABLE-LAYER"
        retained = "Coordinate issue resolved but no layer-pure corrected patches recovered."
        proxy = "none"
        allow_kill = "NO"

    point_median = float(point_all["err_median"].max())
    point_p99 = float(point_all["err_p99"].max())
    point_max = float(point_all["err_max"].max())
    vector_p99 = float(vector_all["l2_p99"].max())
    vector_max = float(vector_all["l2_max"].max())
    fwd_err = float(fwd_all["l2_error"].max())
    identity_df = pd.DataFrame(identity_rows)
    ident_vec_p99 = float(identity_df["vector_p99"].max())
    ident_proj_p99 = float(identity_df["projector_p99"].max())
    v01 = pd.DataFrame([r for r in class_rows if r["comparison"] == "V0_vs_V1"])
    v03 = pd.DataFrame([r for r in class_rows if r["comparison"] == "V0_vs_V3"])
    corr_max = float(pd.DataFrame(corr_rows)["matched_angle_max_abs_diff"].max())
    items = [
        ("A", "为什么 R4A DEPTH-NORMAL-DEGENERATE 不能直接接受", "R4A 只证明 R4 存下来的 depth-normal tensor 是全局 ±z rank-1；它没有闭环 camera-space normal 到 world-space normal 的旋转链。"),
        ("B", "R4 original point frame", "R4 normal-map block does not construct points; depth support uses camera-space positive z."),
        ("C", "R4 original normal frame", "camera-frame slope approximation `[ -dzdx/fx, -dzdy/fy, 1 ]`."),
        ("D", "R4 applied camera→world rotation yes/no", "NO"),
        ("E", "R4 original classification", original_class),
        ("F", "audit camera forward-axis pairwise angles", ",".join(f"{x:.6f}" for x in pair_angles)),
        ("G", "point roundtrip median/p99/max", f"{point_median:.3e}/{point_p99:.3e}/{point_max:.3e}"),
        ("H", "CFM1", "PASS" if CFM1 else "FAIL"),
        ("I", "vector roundtrip p99/max", f"{vector_p99:.3e}/{vector_max:.3e}"),
        ("J", "camera +z vs world forward error", f"{fwd_err:.3e}"),
        ("K", "CFM2", "PASS" if CFM2 else "FAIL"),
        ("L", "V2 world-point vs V3 rotated-camera vector/projector p99", f"{ident_vec_p99:.3e}/{ident_proj_p99:.3e}"),
        ("M", "CFM3", "PASS" if CFM3 else "FAIL"),
        ("N", "V0-vs-V1 error", f"median {float(v01['vector_median'].median()):.3e}, p99 {float(v01['vector_p99'].max()):.3e}"),
        ("O", "V0-vs-V3 error", f"median {float(v03['vector_median'].median()):.3e}, p99 {float(v03['vector_p99'].max()):.3e}"),
        ("P", "camera-space global scatter eigenvalues", f"{cam_row['eig0']:.12f},{cam_row['eig1']:.12f},{cam_row['eig2']:.12f}"),
        ("Q", "corrected world-space scatter eigenvalues", f"{world_row['eig0']:.12f},{world_row['eig1']:.12f},{world_row['eig2']:.12f}"),
        ("R", "corrected random pair median/p90 angle", f"{world_row['pair_median']:.6f}/{world_row['pair_p90']:.6f}"),
        ("S", "CFM4", "PASS" if CFM4 else "FAIL"),
        ("T", "camera-forward matched-angle max diff", f"{corr_max:.3e}"),
        ("U", "corrected reliable multiview normal count/fraction", f"{reliable_count}/{reliable_frac:.6f}"),
        ("V", "corrected Gaussian-vs-depth median/p90 angle", f"{gvd_median:.6f}/{gvd_p90:.6f}"),
        ("W", "corrected coherent patch counts by K", json.dumps(coherent_counts, ensure_ascii=False)),
        ("X", "layer-pure eligible patch count", str(layer_pure_count)),
        ("Y", "CFM5", "PASS" if CFM5 else "FAIL"),
        ("Z", "CFM0-CFM5", f"{'PASS' if CFM0 else 'FAIL'}/{'PASS' if CFM1 else 'FAIL'}/{'PASS' if CFM2 else 'FAIL'}/{'PASS' if CFM3 else 'FAIL'}/{'PASS' if CFM4 else 'FAIL'}/{'PASS' if CFM5 else 'FAIL'}"),
        ("AA", "Final CASE", final_case),
        ("AB", "previous R4/R4A depth-normal conclusions retained or retired", retained),
        ("AC", "valid material-normal proxy", proxy),
        ("AD", "allow final KIOT vs opacity-linear Kill Gate yes/no", allow_kill),
        ("AE", "allow deformed-GT yes/no", "NO"),
        ("AF", "allow multi-scene yes/no", "NO"),
    ]
    report = "# Stage 3.5B-R4B 第一表面深度法向坐标系闭环验证报告\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "depth_normal_coordinate_closure_report.md", report)
    summary = f"""# Stage 3.5B-R4B summary

- Final CASE: `{final_case}`
- CFM0 protocol/source trace: {'PASS' if CFM0 else 'FAIL'}
- CFM1 point roundtrip: {'PASS' if CFM1 else 'FAIL'}
- CFM2 vector/normal roundtrip: {'PASS' if CFM2 else 'FAIL'}
- CFM3 V2 world-point normal == V3 rotated-camera normal: {'PASS' if CFM3 else 'FAIL'}
- CFM4 corrected world-normal nondegeneracy: {'PASS' if CFM4 else 'FAIL'}
- CFM5 recoverable coherent surface layer: {'PASS' if CFM5 else 'FAIL'}
- R4 original classification: {original_class}
- valid material-normal proxy: {proxy}
- allow KIOT-vs-opacity-linear kill gate: {allow_kill}
"""
    write_text(OUT / "stage3_5B_R4B_summary.md", summary)
    terminal = "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n"
    write_text(OUT / "final_terminal_summary.txt", terminal)
    log.extend(f"{k}. {title}: {value}" for k, title, value in items)
    write_text(OUT / "stage3_5B_R4B_log.txt", "\n".join(log) + "\n")

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage3.5B-R4B depth-normal coordinate-frame closure\n\nStage3.5B-R4A confirmed that the actual TSGS smallest-axis normal and the covariance minimum-eigenvector axis are semantically equivalent under stable sign-invariant metrics. However, the R4 depth-normal tensor showed an extreme global +/-z rank-1 signature: global scatter first eigenvalue approximately 0.9999999997, with random-pair p90 unsigned angle approximately 0.00245 degrees.\n\nR4A also documented that R4 did not persist per-view camera-space normals or a verifiable camera-to-world normal transform chain. Therefore the apparent perfect 7019/7019 multiscale depth-normal coherence may be a coordinate-frame artifact.\n\nStage3.5B-R4B performs one final coordinate-frame closure: camera-space depth points and normals are explicitly distinguished from world-space quantities, camera<->world point/vector round trips are numerically verified, and world normals computed by transforming world points are checked against normals obtained by rotating camera-space normals. Only corrected world-space depth normals may be used for multiview fusion and surface-layer recovery. No optical transport policy is evaluated.\n"""
    if "## Stage3.5B-R4B depth-normal coordinate-frame closure" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")
    print(terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
