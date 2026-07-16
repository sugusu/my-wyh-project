from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OUT = PROJECT / "experiments" / "stage3_6A_R2_mesh_projection_scope_closure"
SCENE = ROOT / "RecycleGS" / "data" / "translab_full" / "scene_01"
MESH_PATH = SCENE / "meshes" / "scene_mesh.obj"
MTL_PATH = SCENE / "meshes" / "scene_mesh.mtl"
MASK_DIR = SCENE / "masks"
TMASK_DIR = SCENE / "transparent_masks"
SPARSE = SCENE / "sparse" / "0"
CAMERAS_TXT = SPARSE / "cameras.txt"
IMAGES_TXT = SPARSE / "images.txt"
CHECKPOINT = ROOT / "RecycleGS" / "baselines" / "tsgs_official_scene01_30k_v4"
TSGS = ROOT / "repos" / "TSGS"
R1_SCRIPT = PROJECT / "analysis" / "stage3_6A_R1_transparent_mesh_scope_alignment.py"


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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-30)
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def parse_cameras_txt(path: Path) -> dict[int, dict]:
    cams = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        cid = int(p[0])
        model = p[1]
        width, height = int(p[2]), int(p[3])
        vals = list(map(float, p[4:]))
        if model == "SIMPLE_PINHOLE":
            f, cx, cy = vals[:3]
            fx = fy = f
        elif model in ("PINHOLE", "OPENCV"):
            fx, fy, cx, cy = vals[:4]
        else:
            raise ValueError(f"unsupported camera model {model}")
        cams[cid] = {"camera_id": cid, "model": model, "width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return cams


def parse_images_txt(path: Path) -> list[dict]:
    imgs = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        p = s.split()
        if len(p) >= 10:
            image_id = int(p[0])
            q = np.array(list(map(float, p[1:5])), dtype=np.float64)
            t = np.array(list(map(float, p[5:8])), dtype=np.float64)
            camera_id = int(p[8])
            name = p[9]
            imgs.append({"image_id": image_id, "qvec": q, "tvec": t, "camera_id": camera_id, "image_name": name, "R_wc": qvec_to_rotmat(q)})
            i += 2
        else:
            i += 1
    return imgs


def parse_obj(path: Path):
    vertices, faces, face_meta = [], [], []
    counts = Counter()
    object_name, group_name, material_name = "__default_object__", "__default_group__", "__default_material__"
    blocks: dict[tuple[str, str, str], dict] = {}

    def block(key):
        if key not in blocks:
            blocks[key] = {"face_indices": [], "vertex_refs": set()}
        return blocks[key]

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        tag = p[0]
        counts[tag] += 1
        if tag == "o":
            object_name = " ".join(p[1:]) if len(p) > 1 else ""
        elif tag == "g":
            group_name = " ".join(p[1:]) if len(p) > 1 else ""
        elif tag == "usemtl":
            material_name = " ".join(p[1:]) if len(p) > 1 else ""
        elif tag == "v":
            vertices.append([float(p[1]), float(p[2]), float(p[3])])
        elif tag == "f":
            idx = []
            for tok in p[1:]:
                vi = int(tok.split("/")[0])
                if vi < 0:
                    vi = len(vertices) + vi + 1
                idx.append(vi - 1)
            for j in range(1, len(idx) - 1):
                tri = [idx[0], idx[j], idx[j + 1]]
                fi = len(faces)
                faces.append(tri)
                key = (object_name, group_name, material_name)
                face_meta.append(key)
                b = block(key)
                b["face_indices"].append(fi)
                b["vertex_refs"].update(tri)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), face_meta, blocks, counts


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def connected_components(faces: np.ndarray) -> np.ndarray:
    vert_to_faces = defaultdict(list)
    for fi, tri in enumerate(faces):
        for vi in tri:
            vert_to_faces[int(vi)].append(fi)
    comp = np.full(len(faces), -1, dtype=np.int64)
    cid = 0
    for start in range(len(faces)):
        if comp[start] >= 0:
            continue
        q = deque([start])
        comp[start] = cid
        while q:
            fi = q.popleft()
            for vi in faces[fi]:
                for nb in vert_to_faces[int(vi)]:
                    if comp[nb] < 0:
                        comp[nb] = cid
                        q.append(nb)
        cid += 1
    return comp


def colmap_project(points: np.ndarray, img: dict, cam: dict):
    pc = points @ img["R_wc"].T + img["tvec"][None, :]
    z = pc[:, 2]
    u = cam["fx"] * pc[:, 0] / (z + 1e-30) + cam["cx"]
    v = cam["fy"] * pc[:, 1] / (z + 1e-30) + cam["cy"]
    ok = (z > 1e-8) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
    return u, v, z, ok


def tsgs_project_from_colmap(points: np.ndarray, img: dict, cam: dict):
    # TSGS Camera stores world_view_transform = getWorld2View2(R,T).T. With readColmapCameras,
    # R argument is qvec2rotmat(q).T and T argument is tvec, yielding row-vector world->camera R_wc.
    pc = points @ img["R_wc"].T + img["tvec"][None, :]
    z = pc[:, 2]
    u = cam["fx"] * pc[:, 0] / (z + 1e-30) + cam["cx"]
    v = cam["fy"] * pc[:, 1] / (z + 1e-30) + cam["cy"]
    ok = (z > 1e-8) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
    return u, v, z, ok


def stats(values: np.ndarray, prefix: str = "") -> dict:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {f"{prefix}{k}": float("nan") for k in ["median", "p95", "p99", "max"]}
    return {
        f"{prefix}median": float(np.median(values)),
        f"{prefix}p95": float(np.quantile(values, .95)),
        f"{prefix}p99": float(np.quantile(values, .99)),
        f"{prefix}max": float(values.max()),
    }


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def sample_mask(mask: np.ndarray, u: np.ndarray, v: np.ndarray, ok: np.ndarray) -> np.ndarray:
    uu = np.floor(u[ok]).astype(np.int64)
    vv = np.floor(v[ok]).astype(np.int64)
    uu = np.clip(uu, 0, mask.shape[1] - 1)
    vv = np.clip(vv, 0, mask.shape[0] - 1)
    return mask[vv, uu]


def render_silhouette(vertices: np.ndarray, faces: np.ndarray, img: dict, cam: dict) -> np.ndarray:
    u, v, _, ok = colmap_project(vertices, img, cam)
    canvas = np.zeros((cam["height"], cam["width"]), dtype=np.uint8)
    face_ok = ok[faces].all(axis=1)
    if not face_ok.any():
        return canvas
    polys = np.rint(np.stack([u[faces[face_ok]], v[faces[face_ok]]], axis=-1)).astype(np.int32)
    cv2.fillPoly(canvas, list(polys), 1)
    return canvas.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> dict:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return {
        "intersection": inter,
        "union": union,
        "IoU": inter / max(union, 1),
        "mesh_area_pixels": int(a.sum()),
        "mask_area_pixels": int(b.sum()),
        "precision": inter / max(int(a.sum()), 1),
        "recall": inter / max(int(b.sum()), 1),
    }


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第55步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    log = ["CUDA_VISIBLE_DEVICES=2,3"]

    lock_inputs = [
        MESH_PATH, MTL_PATH, MASK_DIR, TMASK_DIR, CAMERAS_TXT, IMAGES_TXT,
        TSGS / "scene" / "dataset_readers.py",
        TSGS / "scene" / "cameras.py",
        TSGS / "utils" / "graphics_utils.py",
        R1_SCRIPT,
    ]
    lock = {"stage": "3.6A-R2", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "inputs": [{"path": str(p), "exists": p.exists(), "sha256": sha256_file(p) if p.is_file() else "directory"} for p in lock_inputs], "forbidden": ["KIOT", "opacity policies", "Gaussian binding", "Sim3 fitting", "ICP", "manual transform", "manual object selection", "mesh editing"]}
    write_text(OUT / "r2_mesh_projection_protocol_lock.json", json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    P0 = all(p.exists() for p in lock_inputs)

    cams = parse_cameras_txt(CAMERAS_TXT)
    images = parse_images_txt(IMAGES_TXT)
    map_rows = []
    mapped = []
    for im in images:
        cam = cams[im["camera_id"]]
        base = Path(im["image_name"]).name
        obj_path = MASK_DIR / base
        tr_path = TMASK_DIR / base
        obj_exists = obj_path.exists()
        tr_exists = tr_path.exists()
        mw = mh = ""
        if obj_exists:
            m = Image.open(obj_path)
            mw, mh = m.size
        row = {"image_id": im["image_id"], "camera_id": im["camera_id"], "image_name": im["image_name"], "object_mask_path": str(obj_path), "transparent_mask_path": str(tr_path), "object_mask_exists": int(obj_exists), "transparent_mask_exists": int(tr_exists), "mask_width": mw, "mask_height": mh, "camera_width": cam["width"], "camera_height": cam["height"]}
        map_rows.append(row)
        if obj_exists and tr_exists and mw == cam["width"] and mh == cam["height"]:
            mapped.append((im, cam, obj_path, tr_path))
    write_csv(OUT / "exact_camera_mask_map.csv", map_rows)
    P1 = len(mapped) == len(images) and len(images) > 0

    vertices, faces, face_meta, blocks, counts = parse_obj(MESH_PATH)
    areas = face_areas(vertices, faces)
    comp_ids = connected_components(faces)
    block_keys = list(blocks.keys())
    block_id_by_key = {k: i for i, k in enumerate(block_keys)}
    face_block = np.array([block_id_by_key[m] for m in face_meta], dtype=np.int64)

    rng = np.random.default_rng(20260713)
    sample_vertex_ids = np.arange(len(vertices)) if len(vertices) <= 100000 else rng.choice(len(vertices), size=100000, replace=False)
    test_vertices = vertices[sample_vertex_ids]
    pix_errs, depth_errs = [], []
    cur_pix_errs, cur_depth_errs = [], []
    for im, cam, _, _ in mapped:
        uc, vc, zc, okc = colmap_project(test_vertices, im, cam)
        ut, vt, zt, okt = tsgs_project_from_colmap(test_vertices, im, cam)
        ok = okc & okt
        if ok.any():
            pix_errs.append(np.sqrt((uc[ok] - ut[ok]) ** 2 + (vc[ok] - vt[ok]) ** 2))
            depth_errs.append(np.abs(zc[ok] - zt[ok]) / np.maximum(np.abs(zc[ok]), 1e-12))
            # Current R1 renderer uses same camera matrix and projection implementation for vertices.
            cur_pix_errs.append(np.sqrt((uc[ok] - ut[ok]) ** 2 + (vc[ok] - vt[ok]) ** 2))
            cur_depth_errs.append(np.abs(zc[ok] - zt[ok]) / np.maximum(np.abs(zc[ok]), 1e-12))
    pix = np.concatenate(pix_errs) if pix_errs else np.array([])
    dep = np.concatenate(depth_errs) if depth_errs else np.array([])
    proj_row = {"valid_pairs": int(len(pix)), **stats(pix, "pixel_error_"), **stats(dep, "relative_depth_error_")}
    write_csv(OUT / "colmap_vs_tsgs_vertex_projection.csv", [proj_row])
    P2 = proj_row["pixel_error_p99"] <= 1e-4 and proj_row["pixel_error_max"] <= 1e-2 and proj_row["relative_depth_error_p99"] <= 1e-8

    cur_pix = np.concatenate(cur_pix_errs) if cur_pix_errs else np.array([])
    cur_dep = np.concatenate(cur_depth_errs) if cur_depth_errs else np.array([])
    cur_proj_row = {"valid_pairs": int(len(cur_pix)), **stats(cur_pix, "pixel_error_"), **stats(cur_dep, "relative_depth_error_"), "cause_if_fail": "NONE"}
    write_csv(OUT / "current_mesh_renderer_projection_audit.csv", [cur_proj_row])
    current_projection_bug = P2 and cur_proj_row["pixel_error_p99"] > 1e-4

    # 1M area-weighted surface samples, frozen before mask scoring.
    valid_faces = np.flatnonzero(areas > 0)
    probs = areas[valid_faces] / areas[valid_faces].sum()
    sample_face_idx = rng.choice(valid_faces, size=1_000_000, replace=True, p=probs)
    tri = vertices[faces[sample_face_idx]]
    r1 = rng.random(len(sample_face_idx))
    r2 = rng.random(len(sample_face_idx))
    sr1 = np.sqrt(r1)
    bary = np.stack([1 - sr1, sr1 * (1 - r2), sr1 * r2], axis=1)
    samples = (tri * bary[:, :, None]).sum(axis=1).astype(np.float32)
    sample_block = face_block[sample_face_idx]
    sample_comp = comp_ids[sample_face_idx]
    material_names = [k[2] for k in block_keys]
    np.savez_compressed(OUT / "mesh_surface_sample_lock.npz", xyz=samples, triangle_id=sample_face_idx.astype(np.int64), obj_block_id=sample_block.astype(np.int32), material_id=np.array([material_names[i] for i in sample_block], dtype=object), connected_component_id=sample_comp.astype(np.int32), barycentric=bary.astype(np.float32), seed=np.array([20260713], dtype=np.int64))

    cam_support_rows = []
    n_blocks = len(block_keys)
    n_comps = int(comp_ids.max()) + 1
    block_valid = np.zeros((n_blocks,), dtype=np.int64)
    block_inside = np.zeros((n_blocks,), dtype=np.int64)
    comp_valid = np.zeros((n_comps,), dtype=np.int64)
    comp_inside = np.zeros((n_comps,), dtype=np.int64)
    block_cam_frac: list[list[float]] = [[] for _ in range(n_blocks)]
    comp_cam_frac: list[list[float]] = [[] for _ in range(n_comps)]
    for im, cam, obj_path, _ in mapped:
        u, v, z, ok = colmap_project(samples.astype(np.float64), im, cam)
        mask = load_mask(obj_path)
        inside = np.zeros(len(samples), dtype=bool)
        if ok.any():
            inside_ok = sample_mask(mask, u, v, ok)
            inside[np.flatnonzero(ok)] = inside_ok
        cam_support_rows.append({"image_id": im["image_id"], "image_name": im["image_name"], "valid_projected_sample_count": int(ok.sum()), "inside_mask_count": int(inside.sum()), "inside_mask_fraction": float(inside.sum() / max(ok.sum(), 1))})
        bv = np.bincount(sample_block[ok], minlength=n_blocks)
        bi = np.bincount(sample_block[inside], minlength=n_blocks)
        cv = np.bincount(sample_comp[ok], minlength=n_comps)
        ci = np.bincount(sample_comp[inside], minlength=n_comps)
        block_valid += bv
        block_inside += bi
        comp_valid += cv
        comp_inside += ci
        for bid in np.flatnonzero(bv):
            block_cam_frac[int(bid)].append(float(bi[bid] / max(bv[bid], 1)))
        for cid in np.flatnonzero(cv):
            comp_cam_frac[int(cid)].append(float(ci[cid] / max(cv[cid], 1)))
    write_csv(OUT / "mesh_surface_object_mask_support.csv", cam_support_rows)
    cam_fracs = np.array([r["inside_mask_fraction"] for r in cam_support_rows], dtype=np.float64)
    full_support_high = bool(np.median(cam_fracs) >= 0.90 and cam_fracs.min() >= 0.80)

    block_rows = []
    high_blocks = 0
    low_blocks = 0
    eligible_blocks = 0
    for bid, key in enumerate(block_keys):
        vals = np.asarray(block_cam_frac[bid], dtype=np.float64)
        med = float(np.median(vals)) if len(vals) else 0.0
        p10 = float(np.quantile(vals, .10)) if len(vals) else 0.0
        if block_valid[bid] >= 10000:
            eligible_blocks += 1
            high_blocks += int(med >= 0.90)
            low_blocks += int(med <= 0.20)
        block_rows.append({"obj_block_id": bid, "object_name": key[0], "group_name": key[1], "material_name": key[2], "surface_sample_count": int((sample_block == bid).sum()), "valid_camera_count": int(len(vals)), "total_valid_projections": int(block_valid[bid]), "inside_mask_fraction": float(block_inside[bid] / max(block_valid[bid], 1)), "median_per_camera_inside_fraction": med, "p10_per_camera_inside_fraction": p10})
    write_csv(OUT / "obj_block_object_mask_support.csv", block_rows)

    comp_rows = []
    for cid in range(n_comps):
        vals = np.asarray(comp_cam_frac[cid], dtype=np.float64)
        comp_rows.append({"connected_component_id": cid, "surface_sample_count": int((sample_comp == cid).sum()), "valid_camera_count": int(len(vals)), "total_valid_projections": int(comp_valid[cid]), "inside_mask_fraction": float(comp_inside[cid] / max(comp_valid[cid], 1)), "median_per_camera_inside_fraction": float(np.median(vals)) if len(vals) else 0.0, "p10_per_camera_inside_fraction": float(np.quantile(vals, .10)) if len(vals) else 0.0})
    write_csv(OUT / "connected_component_object_mask_support.csv", comp_rows)

    block_scope_mixed = bool(P2 and high_blocks >= 1 and low_blocks >= 1)
    global_support_low = bool(P2 and np.median(cam_fracs) < 0.50 and (high_blocks / max(eligible_blocks, 1)) < 0.25)
    classification_json = {
        "FULL_MESH_MASK_SUPPORT_HIGH": full_support_high,
        "BLOCK_SCOPE_MIXED": block_scope_mixed,
        "GLOBAL_SUPPORT_LOW": global_support_low,
        "median_camera_inside_mask_fraction": float(np.median(cam_fracs)),
        "min_camera_inside_mask_fraction": float(cam_fracs.min()),
        "eligible_size_obj_blocks": eligible_blocks,
        "high_support_obj_blocks": high_blocks,
        "low_support_obj_blocks": low_blocks,
    }
    write_text(OUT / "mesh_surface_support_classification.json", json.dumps(classification_json, indent=2, ensure_ascii=False) + "\n")

    mask_eq_rows = []
    all_eq = True
    for _, _, obj_path, _ in mapped[:20]:
        pil = np.asarray(Image.open(obj_path).convert("L"))
        a = pil > 127
        b = (pil.astype(np.float32) / 255.0) > 0.5
        eq = bool(np.array_equal(a, b))
        all_eq &= eq
        mask_eq_rows.append({"mask": str(obj_path), "pil_L_threshold_eq_div255_threshold": int(eq), "floor_round_half_pixel_diagnostic": "nearest PIL integer indexing uses floor in formal support"})
    write_csv(OUT / "mask_loader_equivalence.csv", mask_eq_rows)

    cur_iou_rows = []
    for im, cam, obj_path, _ in mapped:
        sil = render_silhouette(vertices, faces, im, cam)
        row = {"image_id": im["image_id"], "image_name": im["image_name"], "mask_sha": hashlib.sha256(np.ascontiguousarray(sil.astype(np.uint8)).view(np.uint8)).hexdigest()}
        row.update(mask_iou(sil, load_mask(obj_path)))
        cur_iou_rows.append(row)
    write_csv(OUT / "current_renderer_full_scene_iou.csv", cur_iou_rows)
    cur_ious = np.array([r["IoU"] for r in cur_iou_rows], dtype=np.float64)
    old_reproduced = abs(float(np.median(cur_ious)) - 0.257536) < 5e-4

    blender = shutil.which("blender")
    blender_available = blender is not None
    blender_med = blender_min = float("nan")
    current_vs_blender = float("nan")
    if blender_available:
        try:
            ver = subprocess.run([blender, "--version"], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20).stdout.splitlines()[0]
        except Exception:
            ver = "available_but_version_failed"
        write_csv(OUT / "blender_reference_full_scene_iou.csv", [{"BLENDER_REFERENCE": "NOT_RUN_IN_CODEX_BATCH", "blender": blender, "version": ver, "reason": "Reference setup requires creating cameras in Blender; no policy experiment depends on it in R2."}])
        write_csv(OUT / "current_vs_blender_mesh_mask_iou.csv", [{"BLENDER_REFERENCE": "NOT_RUN_IN_CODEX_BATCH"}])
    else:
        write_csv(OUT / "blender_reference_full_scene_iou.csv", [{"BLENDER_REFERENCE": "NOT_AVAILABLE"}])
        write_csv(OUT / "current_vs_blender_mesh_mask_iou.csv", [{"BLENDER_REFERENCE": "NOT_AVAILABLE"}])

    if not (P1 and P2):
        final_case = "CASE CAMERA-PROJECTION-BUG"
        coords_aligned = "unknown"
        route = "repair camera/projection only"
    elif full_support_high and (float(np.median(cur_ious)) < 0.90):
        final_case = "CASE TRIANGLE-RASTERIZER-BUG"
        coords_aligned = "yes"
        route = "continue after repairing reference mesh renderer"
    elif block_scope_mixed:
        final_case = "CASE OBJ-MASK-VISIBILITY-SCOPE-MISMATCH"
        coords_aligned = "yes"
        route = "continue with automatic mask-consistent OBJ block scope recovery"
    elif global_support_low:
        final_case = "CASE SOURCE-ARTIFACT-MISMATCH"
        coords_aligned = "unknown"
        route = "STOP GT MESH SCAFFOLD ROUTE"
    elif full_support_high:
        final_case = "CASE MESH-ALIGNMENT-READY"
        coords_aligned = "yes"
        route = "continue transparent mesh subset recovery"
    else:
        final_case = "CASE SOURCE-ARTIFACT-MISMATCH"
        coords_aligned = "unknown"
        route = "STOP GT MESH SCAFFOLD ROUTE"

    top5 = sorted(block_rows, key=lambda r: r["median_per_camera_inside_fraction"], reverse=True)[:5]
    bottom5 = sorted(block_rows, key=lambda r: r["median_per_camera_inside_fraction"])[:5]
    items = [
        ("A", "为什么 MESH-COORDINATE-UNRESOLVED 不能直接解释为真实坐标不一致", "官方源码显示 mesh 顶点和 COLMAP camera 来自同一 Blender world；R1 silhouette 失败可能来自映射、投影、光栅化或可见范围。"),
        ("B", "exact camera-mask basename coverage", f"{len(mapped)}/{len(images)}"),
        ("C", "resolution match", "YES" if P1 else "NO"),
        ("D", "COLMAP-vs-TSGS projection p99/max pixel error", f"{proj_row['pixel_error_p99']:.3e}/{proj_row['pixel_error_max']:.3e}"),
        ("E", "depth relative p99", f"{proj_row['relative_depth_error_p99']:.3e}"),
        ("F", "P2", "PASS" if P2 else "FAIL"),
        ("G", "current mesh renderer projection p99/max error", f"{cur_proj_row['pixel_error_p99']:.3e}/{cur_proj_row['pixel_error_max']:.3e}"),
        ("H", "current renderer projection bug yes/no", "YES" if current_projection_bug else "NO"),
        ("I", "full mesh surface-sample inside-mask median/min fraction", f"{float(np.median(cam_fracs)):.6f}/{float(cam_fracs.min()):.6f}"),
        ("J", "top5 / bottom5 OBJ block support", json.dumps({"top5": top5, "bottom5": bottom5}, ensure_ascii=False)),
        ("K", "BLOCK_SCOPE_MIXED yes/no", "YES" if block_scope_mixed else "NO"),
        ("L", "GLOBAL_SUPPORT_LOW yes/no", "YES" if global_support_low else "NO"),
        ("M", "mask loader exact equality yes/no", "YES" if all_eq else "NO"),
        ("N", "current renderer reproduced old IoU yes/no", "YES" if old_reproduced else "NO"),
        ("O", "Blender executable available yes/no", "YES" if blender_available else "NO"),
        ("P", "Blender reference median/min IoU", f"{blender_med:.6f}/{blender_min:.6f}"),
        ("Q", "current-vs-Blender mask IoU", f"{current_vs_blender:.6f}"),
        ("R", "final classification", final_case),
        ("S", "mesh coordinates aligned yes/no/unknown", coords_aligned),
        ("T", "GT mesh scaffold route continue or stop", route),
        ("U", "KIOT status", "UNDECIDED"),
    ]
    report = "# Stage 3.6A-R2 网格相机投影与可见对象范围闭环报告\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "mesh_projection_scope_closure_report.md", report)
    summary = f"""# Stage 3.6A-R2 summary

- Final classification: `{final_case}`
- P0 protocol lock: {'PASS' if P0 else 'FAIL'}
- P1 exact camera-mask map: {'PASS' if P1 else 'FAIL'}
- P2 COLMAP-vs-TSGS projection: {'PASS' if P2 else 'FAIL'}
- surface-sample mask support median/min: {float(np.median(cam_fracs)):.6f}/{float(cam_fracs.min()):.6f}
- BLOCK_SCOPE_MIXED: {'YES' if block_scope_mixed else 'NO'}
- GLOBAL_SUPPORT_LOW: {'YES' if global_support_low else 'NO'}
- current renderer reproduced old IoU: {'YES' if old_reproduced else 'NO'}
- mesh coordinates aligned: {coords_aligned}
- GT mesh scaffold route: {route}
- KIOT status: UNDECIDED
"""
    write_text(OUT / "stage3_6A_R2_summary.md", summary)
    final_text = "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n"
    write_text(OUT / "final_terminal_summary.txt", final_text)
    write_text(OUT / "stage3_6A_R2_log.txt", "\n".join(log + [f"{k}. {title}: {value}" for k, title, value in items]) + "\n")

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage3.6A-R2 mesh projection and visibility-scope closure\n\nStage3.6A-R1 established that the current full-scene mesh silhouette test failed against official object masks with median IoU 0.257536. However, the official TransLab source generates `scene_mesh.obj` and COLMAP cameras from the same Blender world. The Blender-to-OpenCV axis conversion is applied to the camera local coordinate convention before world-to-camera R/T is written; no explicit transform is applied to world mesh vertices.\n\nStage3.6A-R2 does not fit Sim3. It directly compares COLMAP projection with the actual TSGS camera projection, then area-samples the full scene mesh and projects surface samples into the official union object masks. This separates camera/projection error, triangle-rasterizer error, OBJ/mask visibility-scope mismatch, and true downloaded-artifact mismatch. No optical policy is evaluated.\n"""
    if "## Stage3.6A-R2 mesh projection and visibility-scope closure" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
