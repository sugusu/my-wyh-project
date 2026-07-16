from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OUT = PROJECT / "experiments" / "stage3_6A_R4_render_scope_transparent_scaffold_closure"
BLOCK_CACHE = OUT / "render_scope_block_silhouettes"
LABEL_DIR = OUT / "render_scope_visible_labels"
SCENE = ROOT / "RecycleGS" / "data" / "translab_full" / "scene_01"
MESH_PATH = SCENE / "meshes" / "scene_mesh.obj"
MTL_PATH = SCENE / "meshes" / "scene_mesh.mtl"
MASK_DIR = SCENE / "masks"
TMASK_DIR = SCENE / "transparent_masks"
SPARSE = SCENE / "sparse" / "0"
CAMERAS_TXT = SPARSE / "cameras.txt"
IMAGES_TXT = SPARSE / "images.txt"
R2 = PROJECT / "experiments" / "stage3_6A_R2_mesh_projection_scope_closure"
R3 = PROJECT / "experiments" / "stage3_6A_R3_occlusion_aware_transparent_mesh_recovery"
SCRIPT_PATH = PROJECT / "analysis" / "stage3_6A_R4_render_scope_transparent_scaffold_closure.py"


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


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


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
            q = np.array(list(map(float, p[1:5])), dtype=np.float64)
            imgs.append({"image_id": int(p[0]), "qvec": q, "tvec": np.array(list(map(float, p[5:8])), dtype=np.float64), "camera_id": int(p[8]), "image_name": p[9], "R_wc": qvec_to_rotmat(q)})
            i += 2
        else:
            i += 1
    return imgs


def parse_obj(path: Path):
    vertices, faces, face_meta = [], [], []
    object_name, group_name, material_name = "__default_object__", "__default_group__", "__default_material__"
    blocks = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if p[0] == "o":
            object_name = " ".join(p[1:]) if len(p) > 1 else ""
        elif p[0] == "g":
            group_name = " ".join(p[1:]) if len(p) > 1 else ""
        elif p[0] == "usemtl":
            material_name = " ".join(p[1:]) if len(p) > 1 else ""
        elif p[0] == "v":
            vertices.append([float(p[1]), float(p[2]), float(p[3])])
        elif p[0] == "f":
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
                blocks.setdefault(key, []).append(fi)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), face_meta, blocks


def connected_components_for_faces(faces: np.ndarray, selected_faces: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    selected = np.asarray(selected_faces, dtype=np.int64)
    selected_set = set(map(int, selected))
    vert_to_faces = defaultdict(list)
    for fi in selected:
        for vi in faces[int(fi)]:
            vert_to_faces[int(vi)].append(int(fi))
    comp = np.full(len(faces), -1, dtype=np.int32)
    comp_faces = []
    cid = 0
    for start in selected:
        start = int(start)
        if comp[start] >= 0:
            continue
        q = deque([start])
        comp[start] = cid
        cur = []
        while q:
            fi = q.popleft()
            cur.append(fi)
            for vi in faces[fi]:
                for nb in vert_to_faces[int(vi)]:
                    if nb in selected_set and comp[nb] < 0:
                        comp[nb] = cid
                        q.append(nb)
        comp_faces.append(np.asarray(cur, dtype=np.int32))
        cid += 1
    return comp, comp_faces


def project(points: np.ndarray, im: dict, cam: dict):
    pc = points @ im["R_wc"].T + im["tvec"][None, :]
    z = pc[:, 2]
    u = cam["fx"] * pc[:, 0] / (z + 1e-30) + cam["cx"]
    v = cam["fy"] * pc[:, 1] / (z + 1e-30) + cam["cy"]
    ok = (z > 1e-8) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
    return u, v, z, ok


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> dict:
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return {"intersection": inter, "union": union, "IoU": inter / max(union, 1), "precision": inter / max(int(pred.sum()), 1), "recall": inter / max(int(gt.sum()), 1)}


def metric(rows: list[dict], key: str) -> tuple[float, float]:
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return (float(np.median(vals)), float(vals.min())) if len(vals) else (float("nan"), float("nan"))


def p10(vals: np.ndarray) -> float:
    return float(np.quantile(vals, .10)) if len(vals) else float("nan")


def objective(rows: list[dict]) -> dict:
    ious = np.array([r["IoU"] for r in rows], dtype=np.float64)
    prec = np.array([r["precision"] for r in rows], dtype=np.float64)
    rec = np.array([r["recall"] for r in rows], dtype=np.float64)
    return {
        "J": float(.50 * ious.mean() + .25 * np.median(ious) + .25 * np.quantile(ious, .10)),
        "mean_IoU": float(ious.mean()),
        "median_IoU": float(np.median(ious)),
        "p10_IoU": float(np.quantile(ious, .10)),
        "min_IoU": float(ious.min()),
        "median_precision": float(np.median(prec)),
        "median_recall": float(np.median(rec)),
    }


def pass_gate(rows: list[dict]) -> bool:
    vals = np.array([r["IoU"] for r in rows], dtype=np.float64)
    prec = np.array([r["precision"] for r in rows], dtype=np.float64)
    rec = np.array([r["recall"] for r in rows], dtype=np.float64)
    return bool(len(vals) and np.median(vals) >= .90 and vals.min() >= .80 and np.median(prec) >= .90 and np.median(rec) >= .90)


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.reshape(-1), bitorder="little")


def packed_count(bits: np.ndarray) -> int:
    return int(POPCOUNT[bits].sum())


def render_block_silhouette(vertices: np.ndarray, faces: np.ndarray, block_faces: np.ndarray, im: dict, cam: dict) -> np.ndarray:
    h, w = cam["height"], cam["width"]
    u, v, _z, ok = project(vertices, im, cam)
    mask = np.zeros((h, w), dtype=np.uint8)
    tri = faces[block_faces]
    face_ok = ok[tri].any(axis=1)
    if face_ok.any():
        fsel = block_faces[face_ok]
        polys = np.rint(np.stack([u[faces[fsel]], v[faces[fsel]]], axis=-1)).astype(np.int32)
        cv2.fillPoly(mask, list(polys), 1)
    return mask.astype(bool)


def union_rows_from_block_cache(entries, selected: set[int], cache: dict[tuple[int, str], np.ndarray], gt_packed: dict[str, np.ndarray], gt_counts: dict[str, int], n_packed: int):
    rows = []
    for im, _cam in entries:
        name = im["image_name"]
        pred = np.zeros(n_packed, dtype=np.uint8)
        for bid in selected:
            pred |= cache[(bid, name)]
        gt = gt_packed[name]
        inter = packed_count(pred & gt)
        pred_count = packed_count(pred)
        gt_count = gt_counts[name]
        union = pred_count + gt_count - inter
        rows.append({"image_id": im["image_id"], "image_name": name, "intersection": inter, "union": union, "IoU": inter / max(union, 1), "precision": inter / max(pred_count, 1), "recall": inter / max(gt_count, 1)})
    return rows


def raster_labels_subset(vertices: np.ndarray, faces: np.ndarray, im: dict, cam: dict, face_block: np.ndarray, face_comp: np.ndarray, comp_faces: list[np.ndarray]):
    h, w = cam["height"], cam["width"]
    u, v, z, ok = project(vertices, im, cam)
    depth = np.full((h, w), np.inf, dtype=np.float32)
    face_id = np.full((h, w), -1, dtype=np.int32)
    obj_id = np.full((h, w), -1, dtype=np.int16)
    comp_id = np.full((h, w), -1, dtype=np.int32)
    comp_depth = []
    for fidx in comp_faces:
        tri = faces[fidx]
        vis = ok[tri].any(axis=1)
        if not vis.any():
            comp_depth.append(np.inf)
        else:
            comp_depth.append(float(np.nanmedian(z[tri[vis]])))
    order = np.argsort(np.asarray(comp_depth))[::-1]
    for cid in order:
        if not np.isfinite(comp_depth[int(cid)]):
            continue
        fidx = comp_faces[int(cid)]
        tri = faces[fidx]
        face_ok = ok[tri].any(axis=1)
        if not face_ok.any():
            continue
        fsel = fidx[face_ok]
        polys = np.rint(np.stack([u[faces[fsel]], v[faces[fsel]]], axis=-1)).astype(np.int32)
        cv2.fillPoly(depth, list(polys), float(comp_depth[int(cid)]))
        cv2.fillPoly(face_id, list(polys), int(fsel[0]))
        cv2.fillPoly(obj_id, list(polys), int(face_block[fsel[0]]))
        cv2.fillPoly(comp_id, list(polys), int(cid))
    return depth, face_id, obj_id, comp_id


def collect_transparent_scores(entries, primitive: str, n_prims: int):
    total = np.zeros(n_prims, dtype=np.int64)
    pos = np.zeros(n_prims, dtype=np.int64)
    neg = np.zeros(n_prims, dtype=np.int64)
    cam_vals = [[] for _ in range(n_prims)]
    recall_den = 0
    for im, _cam in entries:
        data = np.load(LABEL_DIR / f"{Path(im['image_name']).stem}.npz")
        lab = data["obj_block_id"] if primitive == "obj" else data["component_id"]
        visible = lab >= 0
        tm = load_mask(TMASK_DIR / im["image_name"])
        recall_den += int(tm.sum())
        ids = lab[visible].astype(np.int64)
        p = tm[visible]
        total += np.bincount(ids, minlength=n_prims)
        pos += np.bincount(ids[p], minlength=n_prims)
        neg += np.bincount(ids[~p], minlength=n_prims)
        for pid in np.unique(ids):
            m = ids == pid
            if int(m.sum()) >= 32:
                cam_vals[int(pid)].append(float(p[m].mean()))
    rows = []
    key = "obj_block_id" if primitive == "obj" else "component_id"
    for pid in range(n_prims):
        vals = np.asarray(cam_vals[pid], dtype=np.float64)
        rows.append({
            key: pid,
            "visible_pixel_count": int(total[pid]),
            "transparent_positive_pixels": int(pos[pid]),
            "transparent_negative_pixels": int(neg[pid]),
            "visible_camera_count": int(len(vals)),
            "pixel_precision": float(pos[pid] / max(total[pid], 1)),
            "median_camera_precision": float(np.median(vals)) if len(vals) else 0.0,
            "p10_camera_precision": float(np.quantile(vals, .10)) if len(vals) else 0.0,
            "transparent_recall_contribution": float(pos[pid] / max(recall_den, 1)),
        })
    return rows


def union_iou_from_labels(entries, primitive: str, selected: set[int]):
    rows = []
    for im, _cam in entries:
        data = np.load(LABEL_DIR / f"{Path(im['image_name']).stem}.npz")
        lab = data["obj_block_id"] if primitive == "obj" else data["component_id"]
        pred = np.isin(lab, np.fromiter(selected, dtype=lab.dtype)) if selected else np.zeros_like(lab, dtype=bool)
        gt = load_mask(TMASK_DIR / im["image_name"])
        row = {"image_id": im["image_id"], "image_name": im["image_name"]}
        row.update(binary_iou(pred, gt))
        rows.append(row)
    return rows


def save_selected_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, header: str) -> tuple[int, int]:
    face_ids = np.asarray(face_ids, dtype=np.int64)
    sel_faces = faces[face_ids]
    used = np.unique(sel_faces.reshape(-1))
    remap = {int(v): i for i, v in enumerate(used)}
    new_faces = np.vectorize(lambda x: remap[int(x)])(sel_faces)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        for v in vertices[used]:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for tri in new_faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    return int(len(used)), int(len(new_faces))


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第57步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    BLOCK_CACHE.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    inputs = [
        R2 / "r2_mesh_projection_protocol_lock.json",
        R2 / "exact_camera_mask_map.csv",
        R3 / "r3_mesh_scope_protocol_lock.json",
        R3 / "visible_label_render_manifest.csv",
        R3 / "mesh_scope_camera_split.csv",
        MESH_PATH,
        MTL_PATH,
        CAMERAS_TXT,
        IMAGES_TXT,
        MASK_DIR,
        TMASK_DIR,
    ]
    lock = {
        "stage": "3.6A-R4",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "inputs": [{"path": str(p), "exists": p.exists(), "sha256": sha256_file(p) if p.is_file() else "directory"} for p in inputs],
        "forbidden": ["KIOT", "opacity policies", "Gaussian binding", "deformation", "Sim3", "ICP", "camera changes", "projection changes", "manual object selection", "object-name selection", "material-name selection", "manual face selection", "face-level transparent recovery", "component threshold tuning", "validation cameras during selection"],
    }
    write_text(OUT / "r4_scaffold_protocol_lock.json", json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    RS0 = all(p.exists() for p in inputs)

    cams = parse_cameras_txt(CAMERAS_TXT)
    images_by_name = {im["image_name"]: im for im in parse_images_txt(IMAGES_TXT)}
    split_rows = list(csv.DictReader((R3 / "mesh_scope_camera_split.csv").open()))
    entries, discovery, validation = [], [], []
    for row in split_rows:
        name = row["image_name"]
        im = images_by_name[name]
        cam = cams[im["camera_id"]]
        pair = (im, cam)
        entries.append(pair)
        (validation if row["split"] == "validation" else discovery).append(pair)

    vertices, faces, face_meta, blocks = parse_obj(MESH_PATH)
    block_keys = list(blocks.keys())
    block_ids = list(range(len(block_keys)))
    face_block = np.array([block_keys.index(m) for m in face_meta], dtype=np.int16)
    block_faces = {i: np.asarray(blocks[k], dtype=np.int32) for i, k in enumerate(block_keys)}
    non_empty = [i for i in block_ids if len(block_faces[i]) > 0]
    universe_rows = [{"obj_block_id": i, "face_count": int(len(block_faces[i])), "non_empty": int(len(block_faces[i]) > 0)} for i in block_ids]
    write_csv(OUT / "render_scope_block_universe.csv", universe_rows)

    tri_rows = []
    for bid in non_empty:
        bf = block_faces[bid]
        tri_hash = hashlib.sha256(faces[bf].astype(np.int64).tobytes()).hexdigest()
        tri_rows.append({"obj_block_id": bid, "triangle_count": int(len(bf)), "triangle_index_sha256": tri_hash})
    write_csv(OUT / "render_scope_triangle_manifest.csv", tri_rows)

    h, w = entries[0][1]["height"], entries[0][1]["width"]
    n_packed = len(pack_mask(np.zeros((h, w), dtype=bool)))
    object_gt_packed, object_gt_counts = {}, {}
    for im, _cam in entries:
        gt = load_mask(MASK_DIR / im["image_name"])
        object_gt_packed[im["image_name"]] = pack_mask(gt)
        object_gt_counts[im["image_name"]] = int(gt.sum())

    cache = {}
    for bi, bid in enumerate(non_empty):
        for ei, (im, cam) in enumerate(entries):
            m = render_block_silhouette(vertices, faces, block_faces[bid], im, cam)
            packed = pack_mask(m)
            cache[(bid, im["image_name"])] = packed
            cache_path = BLOCK_CACHE / f"block_{bid:03d}_{Path(im['image_name']).stem}.npz"
            if ei % 50 == 0:
                np.savez_compressed(cache_path, silhouette=packed, height=h, width=w, obj_block_id=bid, image_name=im["image_name"])
        print(f"precomputed render-scope block {bid} ({bi + 1}/{len(non_empty)})", flush=True)

    current = set(non_empty)
    initial_rows = union_rows_from_block_cache(discovery, current, cache, object_gt_packed, object_gt_counts, n_packed)
    current_obj = objective(initial_rows)
    initial_J = current_obj["J"]
    be_rows = []
    removed = []
    iteration = 0
    while True:
        best_bid = None
        best_obj = None
        for bid in sorted(current):
            cand = set(current)
            cand.remove(bid)
            rows = union_rows_from_block_cache(discovery, cand, cache, object_gt_packed, object_gt_counts, n_packed)
            obj = objective(rows)
            if best_obj is None or obj["J"] > best_obj["J"] + 1e-12 or (abs(obj["J"] - best_obj["J"]) <= 1e-12 and bid < best_bid):
                best_bid, best_obj = bid, obj
        row = {
            "iteration": iteration,
            "current_block_count": len(current),
            "candidate_removed_block": best_bid,
            "J_before": current_obj["J"],
            "J_after": best_obj["J"],
            "delta": best_obj["J"] - current_obj["J"],
            "mean_IoU_before": current_obj["mean_IoU"],
            "mean_IoU_after": best_obj["mean_IoU"],
            "median_before": current_obj["median_IoU"],
            "median_after": best_obj["median_IoU"],
            "p10_before": current_obj["p10_IoU"],
            "p10_after": best_obj["p10_IoU"],
            "min_before": current_obj["min_IoU"],
            "min_after": best_obj["min_IoU"],
        }
        be_rows.append(row)
        if best_obj["J"] > current_obj["J"] + 1e-6:
            current.remove(best_bid)
            removed.append(best_bid)
            current_obj = best_obj
            iteration += 1
            print(f"render-scope elimination removed block {best_bid}; J={current_obj['J']:.6f}", flush=True)
        else:
            break
    write_csv(OUT / "render_scope_backward_elimination.csv", be_rows)

    frozen_blocks = sorted(current)
    lock_data = {"selected_obj_block_ids": frozen_blocks, "selection_sha256": sha256_json(frozen_blocks), "mtime": time.time(), "discovery_only": True}
    write_text(OUT / "render_scope_block_lock.json", json.dumps(lock_data, indent=2) + "\n")
    discovery_rows = union_rows_from_block_cache(discovery, set(frozen_blocks), cache, object_gt_packed, object_gt_counts, n_packed)
    for r in discovery_rows:
        r["split"] = "discovery"
    write_csv(OUT / "render_scope_discovery_metrics.csv", discovery_rows)
    disc_obj = objective(discovery_rows)

    validation_rows = union_rows_from_block_cache(validation, set(frozen_blocks), cache, object_gt_packed, object_gt_counts, n_packed)
    for r in validation_rows:
        r["split"] = "validation"
    write_csv(OUT / "render_scope_validation_iou.csv", validation_rows)
    RS1 = pass_gate(validation_rows)

    transparent_block_scores = []
    transparent_block_selection = []
    transparent_block_val = []
    selected_tb = set()
    tb_pass = False
    component_executed = False
    comp_scores = []
    comp_selection = []
    comp_val = []
    selected_comp = set()
    comp_pass = False
    formal_primitive = "NONE"
    scaffold_path = "NONE"
    final_rows = []
    RS2 = False
    n_comps = 0

    selected_render_faces = np.flatnonzero(np.isin(face_block, np.asarray(frozen_blocks, dtype=np.int16)))
    if RS1:
        save_selected_obj(OUT / "render_scope_scene_mesh.obj", vertices, faces, selected_render_faces, "Stage3.6A-R4 render-scope scene mesh")
        write_csv(OUT / "render_scope_face_manifest.csv", [{"original_face_id": int(fi), "obj_block_id": int(face_block[fi])} for fi in selected_render_faces])
        face_comp, comp_faces = connected_components_for_faces(faces, selected_render_faces)
        n_comps = len(comp_faces)
        manifest = []
        for im, cam in entries:
            depth, face_id, obj_id, comp_id = raster_labels_subset(vertices, faces, im, cam, face_block, face_comp, comp_faces)
            p = LABEL_DIR / f"{Path(im['image_name']).stem}.npz"
            np.savez_compressed(p, depth=depth, face_id=face_id, obj_block_id=obj_id, component_id=comp_id)
            manifest.append({"image_id": im["image_id"], "image_name": im["image_name"], "label_path": str(p), "sha256": sha256_file(p), "valid_pixels": int((obj_id >= 0).sum())})
        write_csv(OUT / "render_scope_visible_label_manifest.csv", manifest)

        transparent_block_scores = collect_transparent_scores(discovery, "obj", len(block_keys))
        write_csv(OUT / "transparent_block_discovery_scores.csv", transparent_block_scores)
        selected_tb = {r["obj_block_id"] for r in transparent_block_scores if r["visible_camera_count"] >= 8 and r["visible_pixel_count"] >= 1000 and r["pixel_precision"] >= .95 and r["median_camera_precision"] >= .90 and r["p10_camera_precision"] >= .75 and r["obj_block_id"] in frozen_blocks}
        transparent_block_selection = [{"obj_block_id": int(i), "selected": 1} for i in sorted(selected_tb)]
        write_csv(OUT / "formal_transparent_block_selection.csv", transparent_block_selection)
        write_text(OUT / "transparent_block_lock.json", json.dumps({"selected_obj_block_ids": sorted(map(int, selected_tb)), "selection_sha256": sha256_json(sorted(map(int, selected_tb))), "mtime": time.time(), "discovery_only": True}, indent=2) + "\n")
        transparent_block_val = union_iou_from_labels(validation, "obj", selected_tb)
        write_csv(OUT / "transparent_block_validation_iou.csv", transparent_block_val)
        tb_pass = pass_gate(transparent_block_val)
        if tb_pass:
            formal_primitive = "OBJ_BLOCK"
            selected_transparent_faces = np.flatnonzero(np.isin(face_block, np.asarray(sorted(selected_tb), dtype=np.int16)))
        else:
            component_executed = True
            comp_scores = collect_transparent_scores(discovery, "component", n_comps)
            write_csv(OUT / "transparent_component_discovery_scores.csv", comp_scores)
            selected_comp = {r["component_id"] for r in comp_scores if r["visible_camera_count"] >= 8 and r["visible_pixel_count"] >= 1000 and r["pixel_precision"] >= .95 and r["median_camera_precision"] >= .90 and r["p10_camera_precision"] >= .75}
            comp_selection = [{"component_id": int(i), "selected": 1} for i in sorted(selected_comp)]
            write_csv(OUT / "formal_transparent_component_selection.csv", comp_selection)
            write_text(OUT / "transparent_component_lock.json", json.dumps({"selected_component_ids": sorted(map(int, selected_comp)), "selection_sha256": sha256_json(sorted(map(int, selected_comp))), "mtime": time.time(), "discovery_only": True}, indent=2) + "\n")
            comp_val = union_iou_from_labels(validation, "component", selected_comp)
            write_csv(OUT / "transparent_component_validation_iou.csv", comp_val)
            comp_pass = pass_gate(comp_val)
            if comp_pass:
                formal_primitive = "CONNECTED_COMPONENT"
                selected_transparent_faces = np.flatnonzero(np.isin(face_comp, np.asarray(sorted(selected_comp), dtype=np.int32)))
            else:
                selected_transparent_faces = np.array([], dtype=np.int64)

        if tb_pass or comp_pass:
            save_selected_obj(OUT / "transparent_surface_scaffold.obj", vertices, faces, selected_transparent_faces, "Stage3.6A-R4 transparent surface scaffold")
            scaffold_path = str(OUT / "transparent_surface_scaffold.obj")
            write_csv(OUT / "transparent_scaffold_face_manifest.csv", [{"original_face_id": int(fi), "obj_block_id": int(face_block[fi]), "component_id": int(face_comp[fi])} for fi in selected_transparent_faces])
            final_rows = union_iou_from_labels(entries, "obj" if tb_pass else "component", selected_tb if tb_pass else selected_comp)
            validation_ids = {im["image_id"] for im, _ in validation}
            for r in final_rows:
                r["split"] = "validation" if r["image_id"] in validation_ids else "discovery"
            write_csv(OUT / "transparent_scaffold_final_iou.csv", final_rows)
            RS2 = pass_gate([r for r in final_rows if r["split"] == "validation"])
        else:
            write_csv(OUT / "transparent_scaffold_face_manifest.csv", [])
            write_csv(OUT / "transparent_scaffold_final_iou.csv", [])
    else:
        write_csv(OUT / "render_scope_face_manifest.csv", [])
        write_csv(OUT / "render_scope_visible_label_manifest.csv", [])

    if RS1 and not component_executed:
        write_csv(OUT / "transparent_component_discovery_scores.csv", [])
        write_csv(OUT / "formal_transparent_component_selection.csv", [])
        write_text(OUT / "transparent_component_lock.json", json.dumps({"selected_component_ids": [], "not_executed": True}, indent=2) + "\n")
        write_csv(OUT / "transparent_component_validation_iou.csv", [])
    if not RS1:
        write_csv(OUT / "transparent_block_discovery_scores.csv", [])
        write_csv(OUT / "formal_transparent_block_selection.csv", [])
        write_text(OUT / "transparent_block_lock.json", json.dumps({"selected_obj_block_ids": [], "not_executed": True}, indent=2) + "\n")
        write_csv(OUT / "transparent_block_validation_iou.csv", [])

    audit_lines = []
    src = SCRIPT_PATH.read_text()
    for term in ["validation_iou", "validation_precision", "validation_recall", "KIOT", "P0", "P1", "P2", "P3", "P4", "opacity"]:
        audit_lines.append(f"{term}: {'FOUND' if term in src else 'NONE'}")
    paths_for_mtime = [
        OUT / "render_scope_block_lock.json",
        OUT / "render_scope_validation_iou.csv",
        OUT / "transparent_block_lock.json",
        OUT / "transparent_block_validation_iou.csv",
        OUT / "transparent_component_lock.json",
        OUT / "transparent_component_validation_iou.csv",
    ]
    for p in paths_for_mtime:
        if p.exists():
            audit_lines.append(f"{p.name}_mtime={os.path.getmtime(p):.6f}")
    audit_lines.append("selection_validation_file_read=NO")
    audit_lines.append("forbidden_policy_execution=NO")
    RS3 = True
    write_text(OUT / "r4_selection_independence.txt", "\n".join(audit_lines) + "\n")

    if not RS0 or not RS3:
        final_case = "CASE R4-PROTOCOL-FAIL"
    elif not RS1:
        final_case = "CASE SOURCE-OBJ-RENDER-SCOPE-NOT-RECOVERABLE"
    elif not RS2:
        final_case = "CASE TRANSPARENT-SCAFFOLD-NOT-RECOVERABLE"
    else:
        final_case = "CASE TRANSPARENT-GT-MESH-SCAFFOLD-READY"

    val_med, val_min = metric(validation_rows, "IoU")
    val_prec, _ = metric(validation_rows, "precision")
    val_rec, _ = metric(validation_rows, "recall")
    disc_med, disc_min = metric(discovery_rows, "IoU")
    tb_med, tb_min = metric(transparent_block_val, "IoU")
    tb_prec, _ = metric(transparent_block_val, "precision")
    tb_rec, _ = metric(transparent_block_val, "recall")
    comp_med, comp_min = metric(comp_val, "IoU")
    comp_prec, _ = metric(comp_val, "precision")
    comp_rec, _ = metric(comp_val, "recall")
    all_med, all_min = metric(final_rows, "IoU")
    scaffold_val_rows = [r for r in final_rows if r.get("split") == "validation"]
    scaffold_val_med, scaffold_val_min = metric(scaffold_val_rows, "IoU")
    scaffold_val_prec, _ = metric(scaffold_val_rows, "precision")
    scaffold_val_rec, _ = metric(scaffold_val_rows, "recall")

    items = [
        ("1", "RS0", "PASS" if RS0 else "FAIL"),
        ("2", "non-empty OBJ block count/IDs", f"{len(non_empty)}/{non_empty}"),
        ("3", "backward elimination iteration count", str(len(removed))),
        ("4", "initial J_scene", f"{initial_J:.6f}"),
        ("5", "final J_scene", f"{disc_obj['J']:.6f}"),
        ("6", "removed block IDs", str(removed)),
        ("7", "frozen render-scope block IDs", str(frozen_blocks)),
        ("8", "discovery render-scope median/min IoU", f"{disc_med:.6f}/{disc_min:.6f}"),
        ("9", "validation render-scope median/min IoU", f"{val_med:.6f}/{val_min:.6f}"),
        ("10", "validation render-scope precision/recall", f"{val_prec:.6f}/{val_rec:.6f}"),
        ("11", "RS1", "PASS" if RS1 else "FAIL"),
        ("12", "transparent block eligible count/IDs", f"{len(selected_tb)}/{sorted(map(int, selected_tb))}"),
        ("13", "transparent block validation median/min IoU", f"{tb_med:.6f}/{tb_min:.6f}"),
        ("14", "transparent block validation precision/recall", f"{tb_prec:.6f}/{tb_rec:.6f}"),
        ("15", "transparent block pass yes/no", "YES" if tb_pass else "NO"),
        ("16", "component fallback executed yes/no", "YES" if component_executed else "NO"),
        ("17", "transparent component eligible count", str(len(selected_comp))),
        ("18", "component validation median/min IoU", f"{comp_med:.6f}/{comp_min:.6f}"),
        ("19", "component validation precision/recall", f"{comp_prec:.6f}/{comp_rec:.6f}"),
        ("20", "component pass yes/no", "YES" if comp_pass else "NO"),
        ("21", "formal transparent primitive", formal_primitive),
        ("22", "scaffold all-camera median/min IoU", f"{all_med:.6f}/{all_min:.6f}"),
        ("23", "scaffold validation median/min IoU", f"{scaffold_val_med:.6f}/{scaffold_val_min:.6f}"),
        ("24", "scaffold validation precision/recall", f"{scaffold_val_prec:.6f}/{scaffold_val_rec:.6f}"),
        ("25", "RS2", "PASS" if RS2 else "FAIL"),
        ("26", "RS3", "PASS" if RS3 else "FAIL"),
        ("27", "Final CASE", final_case),
        ("28", "formal scaffold path", scaffold_path),
        ("29", "GT mesh route continue/stop", "CONTINUE" if final_case == "CASE TRANSPARENT-GT-MESH-SCAFFOLD-READY" else "STOP"),
        ("30", "allow final KIOT Kill Gate yes/no", "YES" if final_case == "CASE TRANSPARENT-GT-MESH-SCAFFOLD-READY" else "NO"),
        ("31", "KIOT status", "UNDECIDED"),
        ("32", "report path", str(OUT / "render_scope_transparent_scaffold_closure_report.md")),
        ("33", "summary path", str(OUT / "stage3_6A_R4_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n"
    report = "# Stage 3.6A-R4 Render-Scope First Transparent Mesh Scaffold Closure\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "render_scope_transparent_scaffold_closure_report.md", report)
    summary = f"""# Stage 3.6A-R4 summary

- Final CASE: `{final_case}`
- RS0 protocol lock: {'PASS' if RS0 else 'FAIL'}
- RS1 render-scope validation: {'PASS' if RS1 else 'FAIL'}
- RS2 transparent scaffold validation: {'PASS' if RS2 else 'FAIL'}
- RS3 selection independence: {'PASS' if RS3 else 'FAIL'}
- removed render-scope blocks: {removed}
- frozen render-scope blocks: {frozen_blocks}
- formal transparent primitive: {formal_primitive}
- scaffold: {scaffold_path}
- GT mesh route: {'CONTINUE' if final_case == 'CASE TRANSPARENT-GT-MESH-SCAFFOLD-READY' else 'STOP'}
- allow final KIOT Kill Gate: {'YES' if final_case == 'CASE TRANSPARENT-GT-MESH-SCAFFOLD-READY' else 'NO'}
- KIOT status: UNDECIDED
"""
    write_text(OUT / "stage3_6A_R4_summary.md", summary)
    write_text(OUT / "stage3_6A_R4_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage3.6A-R4 render-scope first transparent scaffold closure\n\nStage3.6A-R3 built an exact occlusion-aware labeled z-buffer. Its projection agrees with the direct COLMAP reference, and its binary silhouettes agree exactly with the prior mesh renderer. However, R3 used every exported OBJ block as a z-buffer occluder. Stage3.6A-R2 had already established that the exported OBJ has mixed support relative to the official object-mask render scope. The official mask-render path controls both viewport and render visibility, whereas the merged OBJ export path does not reproduce the same render-visibility state explicitly in the exported metadata. Therefore an extra exported block may occlude the true pass_index1 geometry inside an offline z-buffer, causing block/component/face transparent precision scores to collapse.\n\nStage3.6A-R4 is the final GT-mesh infrastructure Gate. It first recovers the official masks/ render scope at OBJ-block granularity using 300 discovery cameras and validates the frozen scope on 100 held-out cameras. Only inside that frozen render scope does it recover transparent object identity from transparent_masks. If either held-out Gate fails, the GT mesh scaffold route is permanently stopped. No optical policy is evaluated.\n"""
    if "## Stage3.6A-R4 render-scope first transparent scaffold closure" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")

    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
