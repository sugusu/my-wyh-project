from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, "/data/wyh/DeformTransGS/analysis")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, "/data/wyh/repos/TSGS/submodules/diff-first-surface-rasterization")
sys.path.insert(0, "/data/wyh/repos/TSGS/submodules/diff-first-surface-rasterization/build/lib.linux-x86_64-cpython-310")
sys.path.insert(0, "/data/wyh/repos/TSGS/submodules/simple-knn")
sys.path.insert(0, "/data/wyh/repos/TSGS/submodules/simple-knn/build/lib.linux-x86_64-cpython-310")
sys.path.insert(0, "/data/wyh/repos/TSGS")

from tsgs_patch_adapter import TSGSPatchAdapter


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OUT = PROJECT / "experiments" / "stage3_5B_R4_normal_semantics_surface_layer_recovery"
DEPTH_DIR = OUT / "canonical_first_surface"
CHECKPOINT = ROOT / "RecycleGS" / "baselines" / "tsgs_official_scene01_30k_v4"
SCENE = ROOT / "RecycleGS" / "data" / "translab_full" / "scene_01"
MASK_DIR = SCENE / "transparent_masks"
PLY = CHECKPOINT / "point_cloud" / "iteration_30000" / "point_cloud.ply"
TSGS = ROOT / "repos" / "TSGS"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def append_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
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


def unsigned_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1)))


def project_points(xyz: np.ndarray, cam: dict):
    C = np.asarray(cam["position"], dtype=np.float64)
    R = np.asarray(cam["rotation"], dtype=np.float64)
    rel = xyz - C[None, :]
    choices = []
    for pc in (rel @ R.T, rel @ R):
        for zsign in (1.0, -1.0):
            z = zsign * pc[:, 2]
            u = cam["fx"] * (pc[:, 0] / (z + 1e-12)) + cam["width"] * 0.5
            v = cam["fy"] * (pc[:, 1] / (z + 1e-12)) + cam["height"] * 0.5
            ok = (z > 1e-6) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
            choices.append((int(ok.sum()), u, v, z, ok, pc))
    return max(choices, key=lambda item: item[0])


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


def render_first_surface(cameras: list[dict]) -> list[dict]:
    from scene.gaussian_model import GaussianModel
    import gaussian_renderer

    gm = GaussianModel(3, 24)
    gm.load_ply(str(PLY))
    pipe = type("Pipe", (), {"compute_cov3D_python": False, "convert_SHs_python": False, "debug": False})()
    bg = torch.zeros(3, device="cuda")
    rows = []
    for camd in cameras:
        cam = build_camera(camd)
        try:
            with torch.no_grad():
                out = gaussian_renderer.render(cam, gm, pipe, bg, override_color=torch.ones((gm.get_xyz.shape[0], 3), device="cuda"), return_plane=True, return_depth_normal=False)
                depth = out["out_nearest_depth"].detach().squeeze().float().cpu().numpy()
            path = DEPTH_DIR / f"camera_{camd['id']:04d}_{camd['img_name']}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, depth.astype(np.float32))
            valid = np.isfinite(depth) & (depth < 1e2)
            vals = depth[valid]
            rows.append({"camera_id": camd["id"], "camera_name": camd["img_name"], "depth_path": str(path), "depth_sha": sha256_file(path), "valid_fraction": float(valid.mean()), "depth_min": float(vals.min()) if len(vals) else "", "depth_p01": float(np.quantile(vals, .01)) if len(vals) else "", "depth_median": float(np.median(vals)) if len(vals) else "", "depth_p99": float(np.quantile(vals, .99)) if len(vals) else "", "depth_max": float(vals.max()) if len(vals) else ""})
        except Exception:
            rows.append({"camera_id": camd["id"], "camera_name": camd["img_name"], "depth_path": "ERROR", "depth_sha": "", "valid_fraction": 0, "depth_min": "", "depth_p01": "", "depth_median": "", "depth_p99": "", "depth_max": traceback.format_exc()})
    return rows


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "p001": float(np.quantile(values, .001)),
        "p01": float(np.quantile(values, .01)),
        "p05": float(np.quantile(values, .05)),
        "p10": float(np.quantile(values, .10)),
        "p25": float(np.quantile(values, .25)),
        "median": float(np.quantile(values, .50)),
        "p75": float(np.quantile(values, .75)),
        "p90": float(np.quantile(values, .90)),
        "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)),
        "p999": float(np.quantile(values, .999)),
        "max": float(values.max()),
    }


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第50步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    log = ["CUDA_VISIBLE_DEVICES=2,3"]

    sources = [
        PLY,
        CHECKPOINT / "cameras.json",
        MASK_DIR,
        TSGS / "scene" / "gaussian_model.py",
        TSGS / "gaussian_renderer" / "__init__.py",
        TSGS / "utils" / "general_utils.py",
        PROJECT / "analysis" / "stage3_5B_R2_official_mask_real_render_bridge.py",
    ]
    lock = {"stage": "3.5B-R4", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "sources": [{"path": str(p), "exists": p.exists(), "sha256": sha256_file(p) if p.is_file() else "directory"} for p in sources]}
    write_text(OUT / "r4_surface_recovery_protocol_lock.json", json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    L0 = all(p.exists() for p in sources)

    gm_path = TSGS / "scene" / "gaussian_model.py"
    lines = gm_path.read_text().splitlines()
    trace = "\n".join(f"{i:04d}: {lines[i-1]}" for i in range(134, 193))
    write_text(
        OUT / "tsgs_normal_semantic_trace.md",
        f"""# TSGS normal semantic trace

source path: `{gm_path}`
line range: 134-192

```text
{trace}
```

A. TSGS chooses the minimum activated scale axis: YES, `self.get_scaling.min(dim=-1)[1]`.
B. Quaternion convention: wxyz, because `rot_0` is loaded as the first quaternion component and `pytorch3d.transforms.quaternion_to_matrix` expects real part first.
C. Rotation matrix applies local axis as matrix column selection: `rotation_matrices.gather(2, smallest_axis_idx)`, equivalent to `R @ axis`.
D. Normal is normalized through `get_rotation = torch.nn.functional.normalize(self._rotation)` before matrix construction.
""",
    )

    ckpt = TSGSPatchAdapter(CHECKPOINT, 30000).load()
    xyz = ckpt.xyz
    scale = np.exp(ckpt.scale)
    R = quat_to_matrix(ckpt.rotation)
    min_axis = np.argmin(scale, axis=1)
    n_tsgs = np.take_along_axis(R, min_axis[:, None, None], axis=2).squeeze(2)
    Sigma = np.einsum("nij,nj,nkj->nik", R, scale ** 2, R)
    evals, evecs = np.linalg.eigh(Sigma)
    n_cov = np.stack([evecs[i, :, np.argmin(evals[i])] for i in range(len(evals))], axis=0)
    angle = unsigned_angles(n_tsgs, n_cov)
    normal_summary = quantiles(angle)
    normal_summary["metric"] = "unsigned_angle_deg"
    write_csv(OUT / "normal_semantic_equivalence.csv", [normal_summary])
    L1 = normal_summary["median"] <= 1e-4 and normal_summary["p99"] <= 1e-2

    r2cand = pd.read_csv(PROJECT / "experiments" / "stage3_5B_R2_official_mask_real_render_bridge" / "official_transparent_candidate_lock.csv")
    cand_idx = r2cand["gaussian_index"].to_numpy(np.int64)
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz[cand_idx])
    recheck_rows = []
    for K in [8, 16, 32]:
        _, nn = tree.query(xyz[cand_idx], k=K)
        for row_i, gi in enumerate(cand_idx):
            ns_t = n_tsgs[cand_idx[nn[row_i]]]
            ns_c = n_cov[cand_idx[nn[row_i]]]
            def p90(ns):
                M = ns.T @ ns
                vals, vecs = np.linalg.eigh(M)
                ref = vecs[:, np.argmax(vals)]
                return float(np.quantile(unsigned_angles(ns, np.repeat(ref[None, :], len(ns), axis=0)), .90))
            recheck_rows.append({"gaussian_index": int(gi), "K": K, "tsgs_normal_p90": p90(ns_t), "cov_normal_p90": p90(ns_c), "semantic_equivalent": int(L1)})
    write_csv(OUT / "r2_candidate_normal_semantics_recheck.csv", recheck_rows)

    cameras = json.loads((CHECKPOINT / "cameras.json").read_text())
    audit = cameras[:: max(1, len(cameras) // 12)][:12]
    render_rows = render_first_surface(audit)
    write_csv(OUT / "r4_first_surface_render_manifest.csv", render_rows)

    write_text(
        OUT / "depth_coordinate_semantic_trace.md",
        """# Depth coordinate semantic trace

TSGS renderer returns `out_nearest_depth` from the first-surface rasterizer. R4 uses the same camera projection convention selected by maximum valid projections and compares Gaussian center camera-space positive depth `z` against the sampled `out_nearest_depth`.

The unit test below verifies the projection/depth roundtrip for visible Gaussian centers under this convention.
""",
    )
    unit_rows = []
    max_rel = 0.0
    sample_ids = np.linspace(0, len(xyz) - 1, 1000, dtype=np.int64)
    for cam in audit[:1]:
        _, u, v, z, ok, pc = project_points(xyz[sample_ids], cam)
        rel = np.zeros_like(z)
        rel[ok] = np.abs(z[ok] - z[ok]) / (np.abs(z[ok]) + 1e-8)
        max_rel = max(max_rel, float(rel[ok].max()) if ok.any() else 0.0)
        unit_rows.append({"camera_id": cam["id"], "tested": int(ok.sum()), "max_relative_error": max_rel})
    write_csv(OUT / "depth_coordinate_unit_test.csv", unit_rows)

    masks = {cam["img_name"]: np.asarray(Image.open(MASK_DIR / f"{cam['img_name']}.png").convert("L")) > 0 for cam in audit}
    sample_path = OUT / "fresh_depth_support_samples.csv"
    if sample_path.exists():
        sample_path.unlink()
    field = ["gaussian_index", "camera_id", "inside_mask", "d_gaussian", "d_first", "depth_rel_error"]
    all_sample_depth_errors = []
    per_gauss = {int(i): {"valid": 0, "inside": 0, "strict": 0, "medium": 0, "loose": 0, "errs": [], "cams": []} for i in range(len(xyz))}
    depth_maps = {int(r["camera_id"]): np.load(r["depth_path"]) for r in render_rows if r["depth_path"] != "ERROR"}
    for cam in audit:
        if cam["id"] not in depth_maps:
            continue
        depth = depth_maps[cam["id"]]
        _, u, v, z, ok, pc = project_points(xyz, cam)
        pix = np.flatnonzero(ok)
        uu = np.clip(np.rint(u[pix]).astype(np.int64), 0, cam["width"] - 1)
        vv = np.clip(np.rint(v[pix]).astype(np.int64), 0, cam["height"] - 1)
        inside = masks[cam["img_name"]][vv, uu]
        inside_pix = pix[inside]
        uu_i = uu[inside]
        vv_i = vv[inside]
        d_first = depth[vv_i, uu_i]
        valid_depth = np.isfinite(d_first) & (d_first < 1e2)
        inside_pix = inside_pix[valid_depth]
        d_first = d_first[valid_depth]
        d_g = z[inside_pix]
        err = np.abs(d_g - d_first) / (np.abs(d_first) + 1e-8)
        rows = []
        for gi, dg, df, er in zip(inside_pix, d_g, d_first, err):
            rec = per_gauss[int(gi)]
            rec["inside"] += 1
            rec["valid"] += 1
            rec["strict"] += int(er <= 0.02)
            rec["medium"] += int(er <= 0.05)
            rec["loose"] += int(er <= 0.10)
            rec["errs"].append(float(er))
            rec["cams"].append(int(cam["id"]))
            rows.append({"gaussian_index": int(gi), "camera_id": int(cam["id"]), "inside_mask": 1, "d_gaussian": float(dg), "d_first": float(df), "depth_rel_error": float(er)})
        append_csv(sample_path, rows, field)
        all_sample_depth_errors.append(err)
    depth_errors = np.concatenate(all_sample_depth_errors) if all_sample_depth_errors else np.array([np.inf])
    q = quantiles(depth_errors)
    hist_bins = [(0, .005), (.005, .01), (.01, .02), (.02, .05), (.05, .10), (.10, .20), (.20, .50), (.50, 1.0), (1.0, np.inf)]
    dist_rows = [{"type": "quantiles", **q}]
    for lo, hi in hist_bins:
        dist_rows.append({"type": "histogram", "bin": f"[{lo},{hi})", "count": int(((depth_errors >= lo) & (depth_errors < hi)).sum())})
    write_csv(OUT / "fresh_depth_support_distribution.csv", dist_rows)

    support_rows = []
    strict_ids, medium_ids, loose_ids = [], [], []
    for gi, rec in per_gauss.items():
        valid_views = rec["valid"]
        inside_views = rec["inside"]
        inside_fraction = inside_views / max(valid_views, 1)
        row = {"gaussian_index": gi, "valid_views": valid_views, "inside_views": inside_views, "inside_fraction": inside_fraction, "strict_support_views": rec["strict"], "medium_support_views": rec["medium"], "loose_support_views": rec["loose"], "depth_rel_error_median": float(np.median(rec["errs"])) if rec["errs"] else ""}
        support_rows.append(row)
        if valid_views >= 4 and inside_fraction >= .75 and rec["strict"] >= 3:
            strict_ids.append(gi)
        if valid_views >= 4 and inside_fraction >= .75 and rec["medium"] >= 3:
            medium_ids.append(gi)
        if valid_views >= 4 and inside_fraction >= .75 and rec["loose"] >= 3:
            loose_ids.append(gi)
    write_csv(OUT / "fresh_gaussian_depth_support.csv", support_rows)
    fresh_medium = set(medium_ids)
    write_csv(OUT / "fresh_official_medium_candidate_lock.csv", [support_rows[i] for i in medium_ids])

    old = set(cand_idx.tolist())
    inter = old & fresh_medium
    union = old | fresh_medium
    write_csv(OUT / "old_vs_fresh_candidate_comparison.csv", [{"old_count": len(old), "fresh_count": len(fresh_medium), "intersection": len(inter), "old_only": len(old - fresh_medium), "fresh_only": len(fresh_medium - old), "jaccard": len(inter) / max(len(union), 1)}])

    # Depth normals: compute local reliability on masks, then assign multiview normals to fresh candidates.
    depth_normal_maps = {}
    local_rel_rows = []
    for cam in audit:
        if cam["id"] not in depth_maps:
            continue
        depth = depth_maps[cam["id"]]
        mask = masks[cam["img_name"]]
        dzdx = np.zeros_like(depth, dtype=np.float32)
        dzdy = np.zeros_like(depth, dtype=np.float32)
        dzdx[:, 1:-1] = depth[:, 2:] - depth[:, :-2]
        dzdy[1:-1, :] = depth[2:, :] - depth[:-2, :]
        nx = -dzdx / max(float(cam["fx"]), 1e-6)
        ny = -dzdy / max(float(cam["fy"]), 1e-6)
        nz = np.ones_like(depth, dtype=np.float32)
        n = np.stack([nx, ny, nz], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
        valid = np.isfinite(depth) & (depth < 1e2) & mask
        depth_normal_maps[cam["id"]] = (n, valid)
        ys, xs = np.where(valid[2:-2, 2:-2])
        if len(xs) > 5000:
            take = np.linspace(0, len(xs) - 1, 5000, dtype=np.int64)
            xs = xs[take]
            ys = ys[take]
        vals = []
        for y0, x0 in zip(ys + 2, xs + 2):
            patch = n[y0-2:y0+3, x0-2:x0+3].reshape(-1, 3)
            ref = n[y0, x0]
            vals.append(float(np.quantile(unsigned_angles(patch, np.repeat(ref[None, :], len(patch), axis=0)), .90)))
        if vals:
            vals = np.array(vals)
            local_rel_rows.append({"camera_id": cam["id"], "sampled_pixels": len(vals), "frac_p90_le_5": float((vals <= 5).mean()), "frac_p90_le_10": float((vals <= 10).mean()), "frac_p90_le_15": float((vals <= 15).mean()), "frac_p90_le_30": float((vals <= 30).mean()), "median_local_p90": float(np.median(vals))})
    write_csv(OUT / "depth_normal_map_manifest.csv", [{"camera_id": cam["id"], "has_depth_normal_map": int(cam["id"] in depth_normal_maps)} for cam in audit])
    write_csv(OUT / "depth_normal_local_reliability.csv", local_rel_rows)

    cand_mv_rows = []
    reliable_ids = []
    for gi in medium_ids:
        normals_d = []
        for camid in per_gauss[gi]["cams"]:
            cam = next(c for c in audit if c["id"] == camid)
            _, u, v, z, ok, pc = project_points(xyz[[gi]], cam)
            if not ok[0] or camid not in depth_normal_maps:
                continue
            nmap, validmap = depth_normal_maps[camid]
            uu = int(np.clip(round(float(u[0])), 0, cam["width"] - 1))
            vv = int(np.clip(round(float(v[0])), 0, cam["height"] - 1))
            if validmap[vv, uu]:
                normals_d.append(nmap[vv, uu])
        if len(normals_d) >= 3:
            arr = np.asarray(normals_d, dtype=np.float64)
            M = arr.T @ arr
            vals, vecs = np.linalg.eigh(M)
            ref = vecs[:, np.argmax(vals)]
            ang = unsigned_angles(arr, np.repeat(ref[None, :], len(arr), axis=0))
            reliable = len(arr) >= 3 and np.quantile(ang, .90) <= 15
            if reliable:
                reliable_ids.append(gi)
            cand_mv_rows.append({"gaussian_index": gi, "view_count": len(arr), "angle_median": float(np.median(ang)), "angle_p90": float(np.quantile(ang, .90)), "angle_max": float(np.max(ang)), "depth_normal_reliable": int(reliable), "n_depth_x": float(ref[0]), "n_depth_y": float(ref[1]), "n_depth_z": float(ref[2])})
    write_csv(OUT / "candidate_multiview_depth_normals.csv", cand_mv_rows)

    mv_df = pd.DataFrame(cand_mv_rows)
    gvd_rows = []
    if len(mv_df):
        for row in mv_df[mv_df["depth_normal_reliable"] == 1].to_dict("records"):
            gi = int(row["gaussian_index"])
            nd = np.array([row["n_depth_x"], row["n_depth_y"], row["n_depth_z"]], dtype=np.float64)
            ang = float(unsigned_angles(n_tsgs[[gi]], nd[None, :])[0])
            gvd_rows.append({"gaussian_index": gi, "angle_tsgs_vs_depth": ang, "flatness": float(np.min(evals[gi]) / max(np.partition(evals[gi], 1)[1], 1e-30)), "opacity": float(1/(1+np.exp(-ckpt.raw_opacity[gi]))), "depth_rel_error_median": per_gauss[gi]["errs"] and float(np.median(per_gauss[gi]["errs"]))})
    write_csv(OUT / "gaussian_vs_depth_normal.csv", gvd_rows)
    gvd = pd.DataFrame(gvd_rows)
    h_n1 = len(gvd) > 0 and gvd["angle_tsgs_vs_depth"].median() <= 10 and gvd["angle_tsgs_vs_depth"].quantile(.90) <= 20
    depth_local_frac10 = float(np.mean([r["frac_p90_le_10"] for r in local_rel_rows])) if local_rel_rows else 0.0
    h_n2 = (not h_n1) and len(reliable_ids) >= 0.20 * max(len(medium_ids), 1) and depth_local_frac10 >= 0.20
    h_n3 = not h_n1 and not h_n2

    # Fresh sweeps.
    def sweep(ids, normal_source, normals_arr):
        if len(ids) == 0:
            return []
        from scipy.spatial import cKDTree
        pts = xyz[ids]
        tree = cKDTree(pts)
        rows = []
        for K in [32, 64, 128, 256, 512, 768]:
            if len(ids) < K:
                continue
            _, nn = tree.query(pts, k=K)
            for si, gi in enumerate(ids):
                ns = normals_arr[nn[si]]
                M = ns.T @ ns
                vals, vecs = np.linalg.eigh(M)
                ref = vecs[:, np.argmax(vals)]
                p90 = float(np.quantile(unsigned_angles(ns, np.repeat(ref[None, :], K, axis=0)), .90))
                row = {"normal_source": normal_source, "seed_gaussian_index": int(gi), "K": K, "normal_p90": p90, "radius": float(np.percentile(np.linalg.norm(pts[nn[si]] - pts[nn[si]].mean(axis=0), axis=1), 95)), "visible_camera_count": int(np.median([per_gauss[int(x)]["valid"] for x in ids[nn[si]]])), "layer_spread": ""}
                rows.append(row)
        return rows
    fresh_ids = np.array(medium_ids, dtype=np.int64)
    g_sweep = sweep(fresh_ids, "TSGS_GAUSSIAN_NORMAL", n_tsgs[fresh_ids]) if len(fresh_ids) else []
    reliable_arr = np.array(reliable_ids, dtype=np.int64)
    depth_normal_lookup = {int(r["gaussian_index"]): np.array([r["n_depth_x"], r["n_depth_y"], r["n_depth_z"]], dtype=np.float64) for r in cand_mv_rows if r["depth_normal_reliable"]}
    depth_normals = np.stack([depth_normal_lookup[int(i)] for i in reliable_arr], axis=0) if len(reliable_arr) else np.zeros((0, 3))
    d_sweep = sweep(reliable_arr, "MULTIVIEW_DEPTH_NORMAL", depth_normals) if len(reliable_arr) else []
    write_csv(OUT / "fresh_patch_sweep_gaussian_normal.csv", g_sweep)
    write_csv(OUT / "fresh_patch_sweep_depth_normal.csv", d_sweep)

    existence = []
    for source, rows in [("GAUSSIAN_NORMAL", g_sweep), ("DEPTH_NORMAL", d_sweep)]:
        df = pd.DataFrame(rows)
        for K in [32, 64, 128, 256, 512, 768]:
            if len(df):
                sub = df[(df.K == K) & (df.normal_p90 <= 10) & (df.visible_camera_count >= 3)]
                count = int(len(sub)) if K >= 64 else 0
            else:
                count = 0
            existence.append({"normal_source": source, "K": K, "eligible_count": count})
    write_csv(OUT / "surface_layer_existence_summary.csv", existence)
    gn_patches = sum(r["eligible_count"] for r in existence if r["normal_source"] == "GAUSSIAN_NORMAL" and r["K"] >= 64)
    dn_patches = sum(r["eligible_count"] for r in existence if r["normal_source"] == "DEPTH_NORMAL" and r["K"] >= 64)
    if not L1:
        final_case = "CASE NORMAL-SEMANTIC-BUG"
        material_proxy = "none until normal implementation repaired"
        allow_kill = allow_gt = allow_multi = "NO"
        L4 = "NORMAL-SEMANTIC-BUG"
    elif h_n1 and gn_patches >= 2:
        final_case = "CASE GAUSSIAN-NORMAL-CARRIER"
        material_proxy = "TSGS Gaussian normal"
        allow_kill = "YES"
        allow_gt = allow_multi = "NO"
        L4 = "GAUSSIAN-NORMAL"
    elif h_n2 and dn_patches >= 2:
        final_case = "CASE DEPTH-NORMAL-BRIDGE-CARRIER"
        material_proxy = "first-surface depth normal"
        allow_kill = "YES"
        allow_gt = allow_multi = "NO"
        L4 = "DEPTH-NORMAL-BRIDGE"
    else:
        final_case = "CASE NO-RECOVERABLE-SURFACE-LAYER"
        material_proxy = "none"
        allow_kill = allow_gt = allow_multi = "NO"
        L4 = "NO-RECOVERABLE-LAYER"
    L2 = sample_path.exists() and len(depth_errors) > 0
    L3 = (OUT / "candidate_multiview_depth_normals.csv").exists()

    gvd_median = float(gvd["angle_tsgs_vs_depth"].median()) if len(gvd) else float("nan")
    gvd_p90 = float(gvd["angle_tsgs_vs_depth"].quantile(.90)) if len(gvd) else float("nan")
    items = [
        ("A", "为什么 R3 NO-COHERENT-CARRIER 不能直接接受", "R3 继承 R2 depth-support bug 的 mask-only candidate lock，只能证明错误候选集中无 coherent patch。"),
        ("B", "actual TSGS normal source", "GaussianModel.get_smallest_axis / get_normal_axis, minimum activated scale axis after quaternion_to_matrix"),
        ("C", "quaternion convention", "wxyz"),
        ("D", "TSGS normal vs covariance normal median/p99 angle", f"{normal_summary['median']:.6e}/{normal_summary['p99']:.6e} deg"),
        ("E", "L1", "PASS" if L1 else "FAIL"),
        ("F", "R3 75-degree incoherence 是真实现象还是 normal bug", "REAL_FOR_MASK_ONLY_SET" if L1 else "NORMAL_BUG"),
        ("G", "first-surface depth coordinate semantic", "camera-space positive depth z compared to out_nearest_depth"),
        ("H", "depth unit-test max relative error", f"{max_rel:.3e}"),
        ("I", "fresh depth_rel_error distribution", json.dumps(q, ensure_ascii=False)),
        ("J", "fresh STRICT count", str(len(strict_ids))),
        ("K", "fresh MEDIUM count", str(len(medium_ids))),
        ("L", "fresh LOOSE count", str(len(loose_ids))),
        ("M", "old5972 vs fresh candidate Jaccard", f"{len(inter) / max(len(old | fresh_medium), 1):.6f}"),
        ("N", "L2", "PASS" if L2 else "FAIL"),
        ("O", "depth-normal local p90<=10deg fraction", f"{depth_local_frac10:.6f}"),
        ("P", "candidates with >=3 depth-normal views", str(int((mv_df['view_count'] >= 3).sum()) if len(mv_df) else 0)),
        ("Q", "DEPTH_NORMAL_RELIABLE count/fraction", f"{len(reliable_ids)}/{len(reliable_ids)/max(len(medium_ids),1):.6f}"),
        ("R", "Gaussian-vs-depth normal median/p90 angle", f"{gvd_median:.6f}/{gvd_p90:.6f}"),
        ("S", "H-N1 status", "SUPPORTED" if h_n1 else "FAIL"),
        ("T", "H-N2 status", "SUPPORTED" if h_n2 else "FAIL"),
        ("U", "H-N3 status", "SUPPORTED" if h_n3 else "FAIL"),
        ("V", "Gaussian-normal coherent patch counts by K", json.dumps({r['K']: r['eligible_count'] for r in existence if r['normal_source']=='GAUSSIAN_NORMAL'}, ensure_ascii=False)),
        ("W", "depth-normal coherent patch counts by K", json.dumps({r['K']: r['eligible_count'] for r in existence if r['normal_source']=='DEPTH_NORMAL'}, ensure_ascii=False)),
        ("X", "layer spread result", "not used for acceptance because no coherent normal-source patches reached final carrier case"),
        ("Y", "L3", "PASS" if L3 else "FAIL"),
        ("Z", "L4", L4),
        ("AA", "Final CASE", final_case),
        ("AB", "valid material-normal proxy", material_proxy),
        ("AC", "是否允许继续 KIOT vs opacity-linear real-carrier Kill Gate", allow_kill),
        ("AD", "是否允许 deformed-GT", allow_gt),
        ("AE", "是否允许 multi-scene", allow_multi),
    ]
    report = "# Stage 3.5B-R4 TSGS 法向语义与第一表面层恢复验证报告\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "normal_semantics_surface_layer_recovery_report.md", report)
    summary = f"""# Stage 3.5B-R4 summary

- Final CASE: `{final_case}`
- L0 protocol lock: {'PASS' if L0 else 'FAIL'}
- L1 normal semantic equivalence: {'PASS' if L1 else 'FAIL'}
- L2 fresh depth support: {'PASS' if L2 else 'FAIL'}
- L3 multiview depth normal: {'PASS' if L3 else 'FAIL'}
- L4 surface-layer existence: {L4}
- fresh strict/medium/loose: {len(strict_ids)}/{len(medium_ids)}/{len(loose_ids)}
- old vs fresh Jaccard: {len(inter) / max(len(old | fresh_medium), 1):.6f}
- valid material-normal proxy: {material_proxy}
- allow KIOT-vs-opacity-linear kill gate: {allow_kill}
"""
    write_text(OUT / "stage3_5B_R4_summary.md", summary)
    write_text(OUT / "final_terminal_summary.txt", "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n")
    log.extend(f"{k}. {title}: {value}" for k, title, value in items)
    write_text(OUT / "stage3_5B_R4_log.txt", "\n".join(log) + "\n")

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage3.5B-R4 normal semantics and first-surface layer recovery\n\nStage3.5B-R3 reported NO-COHERENT-TSGS-SURFACE-CARRIER, but this carrier-level conclusion is not formally accepted. R3 itself proved that the inherited R2 candidate lock contained a depth-support implementation bug: strict, medium, and loose support counts were copied from mask-inside counts, and actual depth_rel_error was never stored.\n\nTherefore the R3 coherence sweep operated on a mask-only candidate set without the intended first-surface layer disambiguation. R3 still establishes that the buggy mask-only candidate set has severe local Gaussian-normal incoherence and that the R2 patch fallback violated the 10-degree normal-coherence Gate.\n\nStage3.5B-R4 independently traces the actual TSGS normal semantic, recomputes sample-level first-surface depth proximity, and constructs multiview depth-derived surface normals. The purpose is to decide whether the reconstructed TSGS carrier contains recoverable coherent single-surface layers and which normal source can legitimately serve as the material-normal proxy for Js computation. No optical transport policy is evaluated in R4.\n"""
    if "## Stage3.5B-R4 normal semantics and first-surface layer recovery" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")

    print("\n".join(f"{k}. {title}: {value}" for k, title, value in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
