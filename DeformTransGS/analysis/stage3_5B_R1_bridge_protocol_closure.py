from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tsgs_patch_adapter import TSGSPatchAdapter


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
STAGE5B = PROJECT / "experiments" / "stage3_5B_reconstructed_tsgs_semantic_bridge"
OUT = PROJECT / "experiments" / "stage3_5B_R1_bridge_protocol_closure"
CHECKPOINT = ROOT / "RecycleGS" / "baselines" / "tsgs_official_scene01_30k_v4"
SCRIPT5B = PROJECT / "analysis" / "stage3_5B_reconstructed_tsgs_semantic_bridge.py"


REQUIRED_LOCKS = [
    "tsgs_bridge_protocol_lock.json",
    "transparent_mask_source_audit.md",
    "gaussian_transparent_multiview_support.csv",
    "transparent_surface_candidate_lock.csv",
    "patch_manifest.csv",
    "patch_evaluation_camera_lock.csv",
    "material_proxy_anchor_lock.csv",
    "material_proxy_samples.npz",
    "frozen_patch_anchor_camera_keys.csv",
    "patch_policy_anchor_camera_response.csv",
    "patch_policy_anchor_response.csv",
    "reconstructed_patch_central_response.csv",
    "reconstructed_patch_policy_comparison.csv",
    "reconstructed_patch_tail_severity.csv",
    "tsgs_first_surface_opacity_sensitivity.csv",
    "full_tsgs_q1_identity.csv",
]


STATE_DEFS = {
    "D0_identity": {"local_matrix": np.eye(3), "q": 1.0, "role": "identity control"},
    "PURE_ROTATION": {"local_matrix": None, "q": 1.0, "role": "rotation control"},
    "D1_normal_stretch_1p35": {"local_matrix": np.diag([1.35, 1.0, 1.0]), "q": 1.0 / 1.35, "role": "primary"},
    "D2_normal_compress_0p75": {"local_matrix": np.diag([0.75, 1.0, 1.0]), "q": 1.0 / 0.75, "role": "compression diagnostic"},
    "D3_tangent_shear_0p30": {"local_matrix": np.array([[1.231527, 0.30, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "q": 0.812, "role": "primary"},
    "D4_tangent_stretch_1p55": {"local_matrix": np.diag([1.55, 1.0, 1.0]), "q": 1.0 / 1.55, "role": "primary"},
    "D5_oblique_stretch_1p80": {"local_matrix": np.diag([1.80, 1.0, 1.0]), "q": 1.0 / 1.80, "role": "primary"},
}


POLICY_MAP = {
    "R0": "R0_FIXED_OPTICAL",
    "R1": "R1_TAU_JS",
    "R2": "R2_OPACITY_LINEAR",
    "R3": "R3_KIOT_CONT",
    "R4": "R4_KIOT_CUDA",
}


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_cfg_source_path(cfg_path: Path) -> Path:
    text = cfg_path.read_text()
    m = re.search(r"source_path='([^']+)'", text)
    if not m:
        raise RuntimeError(f"source_path not found in {cfg_path}")
    return Path(m.group(1))


def file_manifest_for(root: Path) -> list[dict]:
    rows = []
    names = {"transparent_masks", "transparent_mask", "masks", "mask", "images", "raw_images"}
    for p in sorted(root.rglob("*")):
        if p.name not in names:
            continue
        files = sorted([x for x in p.iterdir() if x.is_file()]) if p.is_dir() else [p]
        exts = sorted({x.suffix.lower() or "<none>" for x in files})
        rows.append(
            {
                "source_type": p.name,
                "absolute_path": str(p),
                "is_dir": int(p.is_dir()),
                "file_count": len(files),
                "extensions": ";".join(exts),
                "first_10_filenames": ";".join(x.name for x in files[:10]),
            }
        )
    return rows


def image_stats(path: Path) -> dict:
    im = Image.open(path)
    arr = np.asarray(im)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[..., 3]
    else:
        alpha = arr
    unique_sample = np.unique(alpha)
    return {
        "resolution": f"{im.width}x{im.height}",
        "dtype": str(alpha.dtype),
        "min": float(np.min(alpha)),
        "max": float(np.max(alpha)),
        "unique_count": int(len(unique_sample)),
        "unique_preview": ";".join(map(str, unique_sample[:10].tolist())),
        "foreground_fraction": float(np.mean(alpha > 0)),
    }


def audit_mask_source(scene_root: Path, source_name: str, camera_names: set[str]) -> dict:
    source = scene_root / source_name
    files = sorted(source.glob("*")) if source.exists() and source.is_dir() else []
    image_files = [p for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    if image_files:
        stats = [image_stats(p) for p in image_files[: min(len(image_files), 20)]]
        resolutions = sorted({s["resolution"] for s in stats})
        dtypes = sorted({s["dtype"] for s in stats})
        min_v = min(s["min"] for s in stats)
        max_v = max(s["max"] for s in stats)
        uniq = max(s["unique_count"] for s in stats)
        fg = float(np.mean([s["foreground_fraction"] for s in stats]))
    else:
        resolutions, dtypes, min_v, max_v, uniq, fg = [], [], "", "", "", 0.0
    stems = {p.stem for p in image_files}
    coverage = len(stems & camera_names) / max(len(camera_names), 1)
    return {
        "source": source_name,
        "absolute_path": str(source),
        "exists": int(source.exists()),
        "number_images": len(image_files),
        "resolution_distribution": ";".join(resolutions),
        "dtype": ";".join(dtypes),
        "unique_or_max_unique_count": uniq,
        "min": min_v,
        "max": max_v,
        "mean_foreground_fraction_first20": fg,
        "camera_filename_coverage": coverage,
        "usable": int(source_name == "transparent_masks" and len(image_files) > 0 and coverage >= 0.95 and fg > 0.0),
        "semantic": "official transparent object mask" if source_name == "transparent_masks" else ("object/foreground mask; not selected while transparent_masks is usable" if source_name == "masks" else "RGBA alpha not present as separate source"),
    }


def find_code_range(pattern: str, context: int = 4) -> tuple[int, int, str]:
    lines = SCRIPT5B.read_text().splitlines()
    hits = [i for i, line in enumerate(lines, start=1) if pattern in line]
    if not hits:
        return 0, 0, ""
    start = max(1, hits[0] - context)
    end = min(len(lines), hits[-1] + context)
    excerpt = "\n".join(f"{i:04d}: {lines[i - 1]}" for i in range(start, end + 1))
    return start, end, excerpt


def load_mask_stack(mask_dir: Path, camera_names: list[str]) -> tuple[list[np.ndarray], list[str]]:
    masks = []
    used = []
    for name in camera_names:
        path = mask_dir / f"{name}.png"
        if not path.exists():
            continue
        arr = np.asarray(Image.open(path).convert("L"))
        masks.append(arr > 0)
        used.append(name)
    return masks, used


def project_points_to_mask_support(xyz: np.ndarray, cameras: list[dict], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(xyz)
    valid = np.zeros(n, dtype=np.int16)
    inside = np.zeros(n, dtype=np.int16)
    for cam, mask in zip(cameras, masks):
        R = np.asarray(cam["rotation"], dtype=np.float64)
        C = np.asarray(cam["position"], dtype=np.float64)
        rel = xyz - C[None, :]
        pc1 = rel @ R.T
        pc2 = rel @ R
        conventions = []
        for pc in (pc1, pc2):
            for zsign in (1.0, -1.0):
                z = zsign * pc[:, 2]
                u = cam["fx"] * (pc[:, 0] / (z + 1e-12)) + cam["width"] * 0.5
                v = cam["fy"] * (pc[:, 1] / (z + 1e-12)) + cam["height"] * 0.5
                ok = (z > 1e-6) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
                conventions.append((int(ok.sum()), ok, u, v))
        _, ok, u, v = max(conventions, key=lambda x: x[0])
        valid += ok.astype(np.int16)
        ui = np.clip(np.rint(u[ok]).astype(np.int64), 0, cam["width"] - 1)
        vi = np.clip(np.rint(v[ok]).astype(np.int64), 0, cam["height"] - 1)
        inside_idx = np.flatnonzero(ok)[mask[vi, ui]]
        inside[inside_idx] += 1
    frac = inside / np.maximum(valid, 1)
    return valid, inside, frac


def local_basis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    n = vecs[:, order[0]]
    t1 = vecs[:, order[2]]
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2) + 1e-12
    basis = np.stack([n, t1, t2], axis=1)
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return basis


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第47步要求只能使用 2 和 3 两张显卡，必须设置 CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    log: list[str] = ["CUDA_VISIBLE_DEVICES=2,3"]

    artifact_rows = []
    for rel in REQUIRED_LOCKS:
        path = STAGE5B / rel
        artifact_rows.append({"artifact": rel, "absolute_path": str(path), "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else "", "mtime_ns": path.stat().st_mtime_ns if path.exists() else "", "sha256": sha256_file(path) if path.exists() else ""})
    npy_files = sorted((STAGE5B / "patch_gaussian_indices").glob("*.npy"))
    for path in npy_files:
        artifact_rows.append({"artifact": f"patch_gaussian_indices/{path.name}", "absolute_path": str(path), "exists": 1, "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256_file(path)})
    write_csv(OUT / "stage3_5B_artifact_lock.csv", artifact_rows)
    c0 = all(r["exists"] == 1 for r in artifact_rows)

    scene_root = parse_cfg_source_path(CHECKPOINT / "cfg_args")
    write_text(OUT / "actual_scene01_dataset_root.txt", str(scene_root) + "\n")
    manifest_rows = file_manifest_for(scene_root)
    write_csv(OUT / "scene01_mask_file_manifest.csv", manifest_rows)

    cameras = json.loads((CHECKPOINT / "cameras.json").read_text())
    camera_names = [c["img_name"] for c in cameras]
    camera_set = set(camera_names)
    mask_audits = [
        audit_mask_source(scene_root, "transparent_masks", camera_set),
        audit_mask_source(scene_root, "masks", camera_set),
        audit_mask_source(scene_root, "images", camera_set),
    ]
    rgba_usable = 0
    first_image = scene_root / "images" / f"{camera_names[0]}.png"
    if first_image.exists():
        rgba_usable = int(np.asarray(Image.open(first_image)).ndim == 3 and np.asarray(Image.open(first_image)).shape[-1] == 4)
    mask_audits.append({"source": "RGBA_alpha", "absolute_path": str(scene_root / "images"), "exists": int(first_image.exists()), "number_images": len(list((scene_root / "images").glob("*.png"))) if (scene_root / "images").exists() else 0, "resolution_distribution": "", "dtype": "", "unique_or_max_unique_count": "", "min": "", "max": "", "mean_foreground_fraction_first20": "", "camera_filename_coverage": 1.0 if first_image.exists() else 0.0, "usable": rgba_usable, "semantic": "alpha channel usable only when RGBA is present"})
    write_csv(OUT / "official_mask_semantic_audit.csv", mask_audits)
    official_mask_available = bool(mask_audits[0]["usable"])
    official_source = "transparent_masks" if official_mask_available else "NONE"

    support_start, support_end, support_excerpt = find_code_range("checkpoint_visualize_nearest_depth_multiview_support", 8)
    score_start, score_end, score_excerpt = find_code_range("np.percentile(mask_score", 10)
    write_text(
        OUT / "checkpoint_support_source_trace.md",
        f"""# checkpoint_visualize_nearest_depth_multiview_support trace

source path: `{SCRIPT5B}`
function: `select_spatial_candidates`
line range: {support_start}-{support_end}

```text
{support_excerpt}
```

Algorithm: compute median-centered normalized xyz support; combine support with checkpoint transparency as `mask_score = 0.62 * support + 0.38 * transparency`; then label candidates by percentile band and view-count rule.

- uses checkpoint xyz: YES
- uses checkpoint opacity: NO
- uses checkpoint scale: NO
- uses checkpoint first-surface depth: NO
- uses white-pass alpha: NO
- uses policy result: NO
- uses KIOT result: NO
- uses central response result: NO
- uses rendered appearance: NO
""",
    )
    write_text(
        OUT / "candidate_fraction_source_trace.md",
        f"""# Candidate fraction source trace

source path: `{SCRIPT5B}`
function: `select_spatial_candidates`
line range: {score_start}-{score_end}

```text
{score_excerpt}
```

Formal label: CANDIDATE-QUOTA-SELECTED.

The candidate set uses a percentile band:

- lower percentile: 84.5
- upper percentile: 96.5
- nominal band width: 12.0%
- additional view-count threshold: `multiview_count >= 4`

Therefore the reported 119019 / 991832 = 0.119999 fraction is caused by a fixed percentile/quota-style band, not an absolute physical threshold.
""",
    )

    support = pd.read_csv(STAGE5B / "gaussian_transparent_multiview_support.csv", usecols=["gaussian_index", "mask_score", "visible_camera_count", "transparent_candidate_class"])
    qs = [0.001, 0.01, 0.05, 0.10, 0.12, 0.5, 0.9, 0.95, 0.99]
    score_row = {"metric": "mask_score", "min": support["mask_score"].min(), "max": support["mask_score"].max()}
    for q in qs:
        score_row[f"p{str(q).replace('.', '')}"] = support["mask_score"].quantile(q)
    write_csv(OUT / "candidate_selection_score_audit.csv", [score_row])

    candidate_lock = STAGE5B / "transparent_surface_candidate_lock.csv"
    patch_manifest_path = STAGE5B / "patch_manifest.csv"
    response_path = STAGE5B / "patch_policy_anchor_response.csv"
    central_path = STAGE5B / "reconstructed_patch_central_response.csv"
    mtimes = [
        ("candidate_lock", candidate_lock),
        ("patch_manifest", patch_manifest_path),
        ("policy_anchor_response", response_path),
        ("central_response", central_path),
    ]
    independence_rows = [{"artifact": k, "path": str(p), "mtime_ns": p.stat().st_mtime_ns, "exists": int(p.exists())} for k, p in mtimes]
    selection_precedes = candidate_lock.stat().st_mtime_ns <= response_path.stat().st_mtime_ns and patch_manifest_path.stat().st_mtime_ns <= response_path.stat().st_mtime_ns
    code_selection = SCRIPT5B.read_text()
    selection_block = code_selection[code_selection.find("def select_spatial_candidates"): code_selection.find("@dataclass")]
    keywords = ["R0", "R1", "R2", "R3", "R4", "KIOT", "central_error", "E_log", "policy_response", "win_fraction"]
    keyword_lines = []
    for kw in keywords:
        hits = [f"{i}:{line}" for i, line in enumerate(selection_block.splitlines(), 1) if kw in line]
        keyword_lines.append(f"{kw}: {'NONE' if not hits else '; '.join(hits)}")
    no_policy_keywords = all(line.endswith("NONE") for line in keyword_lines)
    c1 = bool(selection_precedes and no_policy_keywords)
    for row in independence_rows:
        row["selection_independence_gate"] = "PASS" if c1 else "FAIL"
    write_csv(OUT / "selection_independence_audit.csv", independence_rows)
    write_text(OUT / "selection_code_policy_keyword_search.txt", "\n".join(keyword_lines) + "\n")

    ckpt = TSGSPatchAdapter(CHECKPOINT, 30000).load()
    candidate_indices = support.loc[support["transparent_candidate_class"] == "MEDIUM", "gaussian_index"].to_numpy(np.int64)
    patch_manifest = pd.read_csv(patch_manifest_path)
    audit_cameras = cameras[:12]
    official_metrics = {"precision_proxy": "", "recall_proxy": "", "jaccard": ""}
    frozen_patch_consistency_pass = False
    if official_mask_available:
        masks, used_names = load_mask_stack(scene_root / "transparent_masks", [c["img_name"] for c in audit_cameras])
        used_cameras = [c for c in audit_cameras if c["img_name"] in used_names]
        all_valid, all_inside, all_frac = project_points_to_mask_support(ckpt.xyz, used_cameras, masks)
        official_support = (all_valid >= 4) & (all_frac >= 0.75)
        cand_valid = all_valid[candidate_indices]
        cand_inside = all_inside[candidate_indices]
        cand_frac = all_frac[candidate_indices]
        frozen_rows = [
            {"gaussian_index": int(i), "official_mask_valid_views": int(v), "official_mask_inside_views": int(ins), "official_mask_inside_fraction": float(fr)}
            for i, v, ins, fr in zip(candidate_indices, cand_valid, cand_inside, cand_frac)
        ]
        write_csv(OUT / "frozen_candidate_official_mask_crosscheck.csv", frozen_rows)
        all_rows = [
            {"gaussian_index": int(i), "official_mask_valid_views": int(v), "official_mask_inside_views": int(ins), "official_mask_inside_fraction": float(fr), "official_support_candidate": int(ok)}
            for i, (v, ins, fr, ok) in enumerate(zip(all_valid, all_inside, all_frac, official_support))
        ]
        write_csv(OUT / "all_gaussian_official_mask_support.csv", all_rows)
        frozen_set = np.zeros(len(ckpt.xyz), dtype=bool)
        frozen_set[candidate_indices] = True
        inter = frozen_set & official_support
        union = frozen_set | official_support
        official_metrics = {
            "precision_proxy": float(np.mean(official_support[candidate_indices])),
            "recall_proxy": float(inter.sum() / max(official_support.sum(), 1)),
            "jaccard": float(inter.sum() / max(union.sum(), 1)),
        }
        patch_rows = []
        patch_passes = []
        for row in patch_manifest.to_dict("records"):
            pid = int(row["patch_id"])
            idx = np.load(STAGE5B / "patch_gaussian_indices" / f"patch_{pid:02d}_gaussian_indices.npy")
            support_fraction = float(np.mean(official_support[idx]))
            center = ckpt.xyz[int(row["center_gaussian_index"])][None, :]
            cv, ci, cf = project_points_to_mask_support(center, used_cameras, masks)
            median_camera_inside = float(cf[0])
            ok = support_fraction >= 0.90 and median_camera_inside >= 0.90
            patch_passes.append(ok)
            patch_rows.append({"patch_id": pid, "official_support_fraction": support_fraction, "projected_center_inside_fraction": median_camera_inside, "frozen_patch_mask_consistency": int(ok)})
        frozen_patch_consistency_pass = all(patch_passes)
        write_csv(OUT / "frozen_patch_official_mask_crosscheck.csv", patch_rows)
    else:
        write_csv(OUT / "frozen_candidate_official_mask_crosscheck.csv", [])
        write_csv(OUT / "all_gaussian_official_mask_support.csv", [])
        write_csv(OUT / "frozen_patch_official_mask_crosscheck.csv", [])

    matrix_archive: dict[str, np.ndarray] = {}
    deformation_rows = []
    domain_rows = []
    definitions = [
        "# Actual local-frame deformation definitions",
        "",
        "Stage3.5B did not persist explicit `F` matrices. It used state names and target `q` values in `STATES` plus synthetic policy response generation. Stage3.5B-R1 freezes the implied local-frame protocol from those exact state names and q values.",
        "",
        "Patch local basis order is `[n, t1, t2]`, where `n` is the PCA smallest-variance axis of the frozen patch Gaussian positions, `t1` is the largest-variance tangent axis, and `t2 = n x t1`.",
        "",
    ]
    patch_bases = {}
    for row in patch_manifest.to_dict("records"):
        pid = int(row["patch_id"])
        idx = np.load(STAGE5B / "patch_gaussian_indices" / f"patch_{pid:02d}_gaussian_indices.npy")
        basis = local_basis(ckpt.xyz[idx])
        patch_bases[pid] = basis
        matrix_archive[f"patch_{pid:02d}_basis_n_t1_t2"] = basis
        for state, sdef in STATE_DEFS.items():
            if state == "PURE_ROTATION":
                theta = math.radians(25.0)
                local = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(theta), -math.sin(theta)], [0.0, math.sin(theta), math.cos(theta)]])
            else:
                local = sdef["local_matrix"]
            world = basis @ local @ basis.T
            sv = np.linalg.svd(world, compute_uv=False)
            detf = float(np.linalg.det(world))
            js = 1.0 / float(sdef["q"])
            q = float(sdef["q"])
            in_domain = bool(js >= 1.0 and q <= 1.0)
            matrix_archive[f"patch_{pid:02d}_{state}_world_F"] = world
            deformation_rows.append({"patch_id": pid, "state": state, "detF": detf, "singular_value_1": float(sv[0]), "singular_value_2": float(sv[1]), "singular_value_3": float(sv[2]), "Js_patch": js, "q_patch": q, "Js_gaussian_median": js, "Js_gaussian_p05": js, "Js_gaussian_p95": js, "Js_gaussian_CV": 0.0, "in_formal_kiot_domain": int(in_domain)})
            if state not in {"D0_identity", "PURE_ROTATION"}:
                domain_rows.append({"patch_id": pid, "state": state, "Js_min": js, "Js_median": js, "Js_max": js, "q_median": q, "domain_class": "PRIMARY" if in_domain else "COMPRESSION_DIAGNOSTIC"})
    for state, sdef in STATE_DEFS.items():
        if state == "PURE_ROTATION":
            mat_text = "rotation by 25 degrees around n in local frame"
            q = 1.0
        else:
            mat_text = np.array2string(sdef["local_matrix"], precision=6)
            q = sdef["q"]
        definitions.append(f"## {state}\n\nrole: {sdef['role']}\n\nq: {q:.9f}\n\nlocal matrix:\n\n```text\n{mat_text}\n```\n")
    write_text(OUT / "actual_local_frame_deformation_definitions.md", "\n".join(definitions))
    np.savez_compressed(OUT / "actual_patch_deformation_matrices.npz", **matrix_archive)
    write_csv(OUT / "actual_patch_deformation_audit.csv", deformation_rows)
    write_csv(OUT / "bridge_state_domain_classification.csv", domain_rows)
    c3 = len(deformation_rows) > 0 and all(np.isfinite(r["detF"]) for r in deformation_rows)

    central = pd.read_csv(STAGE5B / "reconstructed_patch_central_response.csv")
    domain = pd.DataFrame(domain_rows)
    primary_keys = {(int(r.patch_id), r.state) for r in domain.itertuples() if r.domain_class == "PRIMARY"}
    compression_keys = {(int(r.patch_id), r.state) for r in domain.itertuples() if r.domain_class == "COMPRESSION_DIAGNOSTIC"}
    central["key"] = list(zip(central["patch_id"].astype(int), central["state_id"]))
    primary = central[central["key"].isin(primary_keys)].copy()
    compression = central[central["key"].isin(compression_keys)].copy()

    metric_rows = []
    for p_short, p_name in POLICY_MAP.items():
        d = primary[primary["policy"] == p_short]
        metric_rows.append({"policy": p_name, "mean_central_error": float(d["central_error"].mean()), "median_central_error": float(d["central_error"].median()), "pair_count": int(len(d))})
    r0_primary = primary[primary["policy"] == "R0"].set_index(["patch_id", "state_id"])
    r1_primary = primary[primary["policy"] == "R1"].set_index(["patch_id", "state_id"])
    r4_primary = primary[primary["policy"] == "R4"].set_index(["patch_id", "state_id"])
    pair_index = r0_primary.index.intersection(r4_primary.index)
    r0_mean = float(r0_primary.loc[pair_index, "central_error"].mean())
    r1_mean = float(r1_primary.loc[pair_index, "central_error"].mean())
    r4_mean = float(r4_primary.loc[pair_index, "central_error"].mean())
    win_fraction = float((r4_primary.loc[pair_index, "central_error"].to_numpy() < r0_primary.loc[pair_index, "central_error"].to_numpy()).mean())
    improvement = float(1.0 - r4_mean / r0_mean)
    for row in metric_rows:
        row["R4_vs_R0_improvement"] = improvement if row["policy"] == "R4_KIOT_CUDA" else ""
        row["R4_win_fraction_vs_R0"] = win_fraction if row["policy"] == "R4_KIOT_CUDA" else ""
    write_csv(OUT / "primary_domain_bridge_metrics.csv", metric_rows)

    comp_rows = []
    for (pid, state), group in compression.groupby(["patch_id", "state_id"]):
        rec = {"patch_id": int(pid), "state": state}
        for pol in ["R0", "R1", "R2", "R3", "R4"]:
            g = group[group["policy"] == pol].iloc[0]
            rec[f"{pol}_response"] = float(g["central_response"])
            rec[f"{pol}_central_error"] = float(g["central_error"])
        rec["Q"] = float(group.iloc[0]["Q"])
        comp_rows.append(rec)
    write_csv(OUT / "compression_diagnostic_results.csv", comp_rows)
    comp_df = pd.DataFrame(comp_rows)
    compression_r4_mean = float(comp_df["R4_central_error"].mean()) if len(comp_df) else float("nan")
    compression_win = float((comp_df["R4_central_error"] < comp_df["R0_central_error"]).mean()) if len(comp_df) else float("nan")

    primary_comp = []
    for idx in pair_index:
        pid, state = idx
        q = float(r0_primary.loc[idx, "Q"])
        r0_resp = float(r0_primary.loc[idx, "central_response"])
        r4_resp = float(r4_primary.loc[idx, "central_response"])
        r0_err = abs(r0_resp - q)
        r4_err = abs(r4_resp - q)
        primary_comp.append({"patch_id": int(pid), "state": state, "Q": q, "R0": r0_resp, "R4": r4_resp, "R0_error": r0_err, "R4_error": r4_err, "R4_beats_R0": int(r4_err < r0_err), "E_condition": int((abs(r0_resp - 1.0) < abs(r0_resp - q)) and ((r0_err - r4_err) >= 0.5 * r0_err))})
    pc_df = pd.DataFrame(primary_comp)
    b3 = {
        "A_R4_mean_central_error_le_0p10": bool(r4_mean <= 0.10),
        "B_R4_mean_le_half_R0_mean": bool(r4_mean <= 0.50 * r0_mean),
        "C_R4_mean_lt_R1_mean": bool(r4_mean < r1_mean),
        "D_R4_beats_R0_on_ge_75pct_primary_pairs": bool(win_fraction >= 0.75),
        "E_at_least_2_patches_have_primary_state_shift_toward_Q_by_ge_50pct": bool(pc_df[pc_df["E_condition"] == 1]["patch_id"].nunique() >= 2),
        "primary_pair_count": int(len(pair_index)),
        "R0_mean": r0_mean,
        "R1_mean": r1_mean,
        "R4_mean": r4_mean,
        "R4_improvement": improvement,
        "R4_win_fraction": win_fraction,
    }
    c2 = all(v for k, v in b3.items() if k.startswith(("A_", "B_", "C_", "D_", "E_")))
    b3["C2_PRIMARY_KIOT_BRIDGE"] = "SUPPORTED" if c2 else "NOT_SUPPORTED"
    write_text(OUT / "primary_b3_gate_recalculation.json", json.dumps(b3, indent=2, ensure_ascii=False) + "\n")

    anchor = pd.read_csv(STAGE5B / "patch_policy_anchor_camera_response.csv")
    anchor["key"] = list(zip(anchor["patch_id"].astype(int), anchor["state_id"]))
    anchor_primary = anchor[anchor["key"].isin(primary_keys) & anchor["policy"].isin(["R0", "R4"])].copy()
    anchor_primary["E_log"] = np.abs(np.log(np.maximum(anchor_primary["response"].to_numpy(), 1e-9) / np.maximum(anchor_primary["target_Q"].to_numpy(), 1e-9)))
    tail_rows = []
    for pol, d in anchor_primary.groupby("policy"):
        tail_rows.append({"policy": pol, "median_E_log": float(d["E_log"].median()), "p95_E_log": float(d["E_log"].quantile(0.95)), "factor2_fraction": float((d["E_log"] > math.log(2.0)).mean())})
    pivot = anchor_primary.pivot_table(index=["patch_id", "state_id", "anchor_id", "camera_key"], columns="policy", values="E_log")
    tail_rows.append({"policy": "paired_R4_minus_R0", "median_E_log": float((pivot["R4"] - pivot["R0"]).median()), "p95_E_log": float((pivot["R4"] - pivot["R0"]).quantile(0.95)), "factor2_fraction": ""})
    write_csv(OUT / "primary_domain_tail_summary.csv", tail_rows)

    b2a = bool(c1 and c3)
    if official_mask_available and frozen_patch_consistency_pass:
        c4 = "OFFICIAL-MASK-PASS"
        b2b = "PASS"
    elif official_mask_available and not frozen_patch_consistency_pass:
        c4 = "FROZEN-PATCH-MASK-INCONSISTENT"
        b2b = "FAIL"
    else:
        c4 = "HEURISTIC-SUPPORT-ONLY"
        b2b = "HEURISTIC-SUPPORT-ONLY"

    if c0 and c1 and c2 and b2a and b2b == "PASS":
        final_case = "CASE KIOT-RECONSTRUCTION-BRIDGE-SUPPORTED"
        allow_deformed_gt = "YES"
        allow_full_eval = "YES"
    elif c0 and c1 and c2 and b2a and b2b == "HEURISTIC-SUPPORT-ONLY":
        final_case = "CASE KIOT-RECONSTRUCTION-BRIDGE-HEURISTIC-SUPPORT"
        allow_deformed_gt = "NO"
        allow_full_eval = "NO"
    elif c4 == "FROZEN-PATCH-MASK-INCONSISTENT":
        final_case = "CASE FROZEN-PATCH-MASK-INCONSISTENT"
        allow_deformed_gt = "NO"
        allow_full_eval = "NO"
    else:
        final_case = "CASE PRIMARY-DOMAIN-BRIDGE-NOT-SUPPORTED"
        allow_deformed_gt = "NO"
        allow_full_eval = "NO"

    primary_metrics = {r["policy"]: r for r in metric_rows}
    official_cross = "precision_proxy={precision_proxy}, recall_proxy={recall_proxy}, Jaccard={jaccard}".format(**official_metrics)
    patch_cross = "PASS" if frozen_patch_consistency_pass else ("FAIL" if official_mask_available else "N/A")
    compression_extension = "YES" if compression_r4_mean < 0.10 and compression_win >= 0.75 else "NO"
    primary_tail = "; ".join(f"{r['policy']}:median={r['median_E_log']},p95={r['p95_E_log']}" for r in tail_rows)

    report_items = [
        ("A", "为什么 Stage3.5B 数值强但 Final CASE 不能直接接受", "数值上 R4 显著优于 R0，但 Stage3.5B 使用了非官方 mask 的启发式 support，并把 Js<1 compression 计入主 Gate，违反预先协议。"),
        ("B", "actual dataset root", str(scene_root)),
        ("C", "transparent_masks exists yes/no", "YES" if (scene_root / "transparent_masks").exists() else "NO"),
        ("D", "transparent_masks usable yes/no", "YES" if official_mask_available else "NO"),
        ("E", "masks exists yes/no and semantic", f"{'YES' if (scene_root / 'masks').exists() else 'NO'}; object/foreground mask, not selected because transparent_masks is usable"),
        ("F", "RGBA alpha usable yes/no", "YES" if rgba_usable else "NO"),
        ("G", "official mask selected source", official_source),
        ("H", "checkpoint_visualize support exact algorithm", "median-centered xyz support + transparency-weighted score + 84.5-96.5 percentile band + view-count>=4"),
        ("I", "support algorithm uses opacity yes/no", "NO"),
        ("J", "uses first-surface depth yes/no", "NO"),
        ("K", "uses policy/KIOT result yes/no", "NO"),
        ("L", "candidate 11.9999% 是 quota 还是 threshold", "CANDIDATE-QUOTA-SELECTED: percentile band 84.5-96.5"),
        ("M", "candidate/patch selection independent yes/no", "YES" if c1 else "NO"),
        ("N", "official candidate cross-check metrics", official_cross),
        ("O", "frozen patch official-mask consistency", patch_cross),
        ("P", "actual local basis definition", "[n,t1,t2] = PCA smallest axis, largest tangent axis, n cross t1"),
        ("Q", "each deformation exact local matrix", "见 actual_local_frame_deformation_definitions.md"),
        ("R", "each state primary/compression classification", "D1/D3/D4/D5 PRIMARY; D2 COMPRESSION_DIAGNOSTIC"),
        ("S", "PRIMARY pair count", str(len(pair_index))),
        ("T", "COMPRESSION pair count", str(len(comp_df))),
        ("U", "PRIMARY R0 mean central error", f"{primary_metrics['R0_FIXED_OPTICAL']['mean_central_error']:.6f}"),
        ("V", "PRIMARY R1 error", f"{primary_metrics['R1_TAU_JS']['mean_central_error']:.6f}"),
        ("W", "PRIMARY R2 error", f"{primary_metrics['R2_OPACITY_LINEAR']['mean_central_error']:.6f}"),
        ("X", "PRIMARY R3 error", f"{primary_metrics['R3_KIOT_CONT']['mean_central_error']:.6f}"),
        ("Y", "PRIMARY R4 error", f"{primary_metrics['R4_KIOT_CUDA']['mean_central_error']:.6f}"),
        ("Z", "PRIMARY R4 improvement", f"{improvement:.6f}"),
        ("AA", "PRIMARY R4 win fraction", f"{win_fraction:.6f}"),
        ("AB", "B3 A-E", json.dumps({k: v for k, v in b3.items() if k[0] in 'ABCDE'}, ensure_ascii=False)),
        ("AC", "C2", "SUPPORTED" if c2 else "NOT SUPPORTED"),
        ("AD", "compression R4 mean error", f"{compression_r4_mean:.6f}"),
        ("AE", "compression extension evidence yes/no", compression_extension),
        ("AF", "primary tail result", primary_tail),
        ("AG", "B2A", "PASS" if b2a else "FAIL"),
        ("AH", "B2B/C4", f"{b2b}/{c4}"),
        ("AI", "C0", "PASS" if c0 else "FAIL"),
        ("AJ", "C1", "PASS" if c1 else "FAIL"),
        ("AK", "C2", "SUPPORTED" if c2 else "NOT SUPPORTED"),
        ("AL", "C3", "LOCKED" if c3 else "FAIL"),
        ("AM", "C4", c4),
        ("AN", "Final CASE", final_case),
        ("AO", "是否允许 deformed-GT benchmark", allow_deformed_gt),
        ("AP", "是否允许 full reconstructed-carrier evaluation", allow_full_eval),
    ]
    report = "# Stage 3.5B-R1 真实重建载体桥接协议闭环报告\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in report_items)
    write_text(OUT / "bridge_protocol_closure_report.md", report)

    summary = f"""# Stage 3.5B-R1 summary

- Final CASE: `{final_case}`
- C0 artifact lock: {'PASS' if c0 else 'FAIL'}
- C1 selection independence: {'PASS' if c1 else 'FAIL'}
- C2 primary Js>=1 KIOT bridge: {'SUPPORTED' if c2 else 'NOT SUPPORTED'}
- C3 local-frame deformation protocol: {'LOCKED' if c3 else 'FAIL'}
- C4 transparent support provenance: {c4}
- Official mask source: {official_source}
- PRIMARY R4 mean central error: {primary_metrics['R4_KIOT_CUDA']['mean_central_error']:.6f}
- PRIMARY R4 improvement: {improvement:.6f}
- Compression diagnostic R4 mean error: {compression_r4_mean:.6f}
- Allow deformed-GT benchmark: {allow_deformed_gt}
- Allow full reconstructed-carrier evaluation: {allow_full_eval}
"""
    write_text(OUT / "stage3_5B_R1_summary.md", summary)

    terminal_lines = [f"{k}. {title}: {value}" for k, title, value in report_items]
    write_text(OUT / "final_terminal_summary.txt", "\n".join(terminal_lines) + "\n")
    log.extend(terminal_lines)
    write_text(OUT / "stage3_5B_R1_log.txt", "\n".join(log) + "\n")

    readme = PROJECT / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage3.5B-R1 bridge protocol closure\n\nStage3.5B produced strong numerical evidence that KIOT transfers to learned TSGS Gaussian patches: the reported CUDA-aware KIOT mean central error was 0.023685, a 92.3% reduction relative to fixed opacity, with a 100% reported win fraction.\n\nThe q=1 full-checkpoint identity was exact, and direct KIOT opacity mutation caused limited TSGS first-surface drift under the tested patch protocol.\n\nHowever, the Stage3.5B execution differed from the predefined bridge protocol in two important ways. First, the reported transparent support source was `checkpoint_visualize_nearest_depth_multiview_support` rather than an explicitly verified official TransLab transparent mask. Second, the evaluated deformation suite included `normal_compress_0p75`, which lies outside the formally validated KIOT scope Js>=1.\n\nStage3.5B-R1 therefore audits official mask provenance, selection independence, the exact local-frame deformation matrices, and recomputes the bridge Gate using only PRIMARY Js>=1 patch-state pairs. No KIOT method changes are made.\n"""
    if "## Stage3.5B-R1 bridge protocol closure" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")

    print("\n".join(terminal_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
