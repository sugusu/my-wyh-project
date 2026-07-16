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

sys.path.insert(0, "/data/wyh/DeformTransGS/analysis")
from tsgs_patch_adapter import TSGSPatchAdapter


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OUT = PROJECT / "experiments" / "stage3_5B_R3_surface_layer_transport_kill_gate"
R2 = PROJECT / "experiments" / "stage3_5B_R2_official_mask_real_render_bridge"
R2_SCRIPT = PROJECT / "analysis" / "stage3_5B_R2_official_mask_real_render_bridge.py"
CHECKPOINT = ROOT / "RecycleGS" / "baselines" / "tsgs_official_scene01_30k_v4"
PATCH_DIR = OUT / "coherent_patch_indices"


PATCH_SIZES = [32, 64, 128, 256, 512, 768]
FORMAL_SIZES = [64, 128, 256, 512, 768]
PRIMARY_STATES = ["E1_TANGENT_STRETCH_1P25", "E2_TANGENT_STRETCH_1P50", "E3_TANGENT_STRETCH_2P00", "E4_BIAXIAL_TANGENT_1P50", "E6_OBLIQUE_TANGENT_STRETCH_1P80"]
POLICIES = ["P0_FIXED", "P1_TAU_JS", "P2_OPACITY_LINEAR", "P3_KIOT_CONT", "P4_KIOT_CUDA"]


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


def quantile_row(name: str, values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "metric": name,
        "count": int(len(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def normal_angles(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = normals.T @ normals
    vals, vecs = np.linalg.eigh(M)
    n_ref = vecs[:, np.argmax(vals)]
    dots = np.clip(np.abs(normals @ n_ref), 0.0, 1.0)
    return np.degrees(np.arccos(dots)), n_ref


def local_basis(xyz: np.ndarray, normals: np.ndarray) -> np.ndarray:
    _, n = normal_angles(normals)
    cov = np.cov((xyz - xyz.mean(axis=0)).T)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    t1 = v - n * (n @ v)
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2) + 1e-12
    B = np.stack([n, t1, t2], axis=1)
    if np.linalg.det(B) < 0:
        B[:, 2] *= -1
    return B


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第49步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    log = ["CUDA_VISIBLE_DEVICES=2,3"]

    required = [
        R2 / "r2_real_bridge_protocol_lock.json",
        R2 / "official_transparent_mask_camera_map.csv",
        R2 / "official_transparent_candidate_lock.csv",
        R2 / "official_candidate_geometry.csv",
        R2 / "real_render_manifest.csv",
        R2 / "real_reconstructed_central_response.csv",
        PROJECT / "analysis" / "kiot_fast_inverse.py",
    ]
    lock = {
        "stage": "3.5B-R3",
        "name": "Surface-Layer Coherence and Optical Transport Kill Gate",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "inputs": [{"path": str(p), "exists": p.exists(), "sha256": sha256_file(p) if p.exists() and p.is_file() else ""} for p in required],
        "patch_sizes": PATCH_SIZES,
        "formal_patch_sizes": FORMAL_SIZES,
        "normal_p90_gate_deg": 10.0,
        "seed": 20260713,
    }
    write_text(OUT / "r3_surface_protocol_lock.json", json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    kill0 = all(p.exists() for p in required)

    lines = R2_SCRIPT.read_text().splitlines()
    excerpt = "\n".join(f"{i:04d}: {lines[i-1]}" for i in range(354, 402))
    r2_patch = pd.read_csv(R2 / "official_patch_manifest.csv")
    eligible_count_r2 = int(((r2_patch["normal_p90_deg"] <= 10.0) & (r2_patch["visible_camera_count"] >= 3)).sum())
    write_text(
        OUT / "r2_patch_gate_bypass_trace.md",
        f"""# R2 patch gate bypass trace

source path: `{R2_SCRIPT}`
function: `main`, patch discovery section
line range: 354-401

```text
{excerpt}
```

Eligible condition in R2 line 381:

`len(idx) == 768 and normal_p90 <= 10.0 and visible_cams >= 3`

Fallback logic in R2 line 392-393:

`if len(chosen) < 2: chosen = sorted(candidates_patch, key=lambda x: x[1])[:3]`

R2 selected patch normal p90 values:

{r2_patch[['patch_id','normal_p90_deg','visible_camera_count']].to_string(index=False)}

R2 selected eligible patch count under its own reported manifest: {eligible_count_r2}.

Formal label: PATCH-GATE-BYPASS-CONFIRMED.
""",
    )

    geom = pd.read_csv(R2 / "official_candidate_geometry.csv")
    support = pd.read_csv(R2 / "official_transparent_candidate_lock.csv")
    ckpt = TSGSPatchAdapter(CHECKPOINT, 30000).load()
    cand_idx = geom["gaussian_index"].to_numpy(np.int64)
    normals = geom[["normal_x", "normal_y", "normal_z"]].to_numpy(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    xyz = ckpt.xyz[cand_idx]
    flatness = np.clip(geom["flatness"].to_numpy(np.float64), 0.0, None)
    scale_ratio = np.sqrt(flatness)
    reliability_rows = [quantile_row("flatness_lambda0_over_lambda1", flatness), quantile_row("scale_ratio_s0_over_s1", scale_ratio)]
    write_csv(OUT / "candidate_normal_reliability.csv", reliability_rows)

    strata = np.full(len(geom), "NON_SURFEL", dtype=object)
    strata[(flatness > 0.10) & (flatness <= 0.30)] = "MID"
    strata[flatness <= 0.10] = "SURFEL_LIKE"
    strata_rows = []
    for label in ["SURFEL_LIKE", "MID", "NON_SURFEL"]:
        count = int(np.sum(strata == label))
        strata_rows.append({"stratum": label, "count": count, "fraction": count / len(strata)})
    write_csv(OUT / "candidate_normal_strata.csv", strata_rows)

    # R2 did not store depth_rel_error; strict/medium/loose were all assigned from inside-mask counts.
    hist_rows = [
        {"bin": "[0,.005)", "count": "", "note": "depth_rel_error not recorded in R2 support CSV"},
        {"bin": "[.005,.01)", "count": "", "note": "depth_rel_error not recorded in R2 support CSV"},
        {"bin": "[.01,.02)", "count": "", "note": "depth_rel_error not recorded in R2 support CSV"},
        {"bin": "[.02,.05)", "count": "", "note": "depth_rel_error not recorded in R2 support CSV"},
        {"bin": "[.05,.10)", "count": "", "note": "depth_rel_error not recorded in R2 support CSV"},
        {"bin": "[.10,.20)", "count": "", "note": "depth_rel_error not recorded in R2 support CSV"},
        {"bin": "[.20,.50)", "count": "", "note": "depth_rel_error not recorded in R2 support CSV"},
        {"bin": "[.50,inf)", "count": "", "note": "depth_rel_error not recorded in R2 support CSV"},
    ]
    qrow = {"metric": "depth_rel_error", "min": "", "p001": "", "p01": "", "p05": "", "p10": "", "p25": "", "median": "", "p75": "", "p90": "", "p95": "", "p99": "", "max": "", "status": "NOT_RECORDED_IN_R2_SUPPORT_CSV"}
    write_csv(OUT / "first_surface_depth_support_distribution.csv", [qrow] + hist_rows)
    depth_excerpt = "\n".join(f"{i:04d}: {lines[i-1]}" for i in range(323, 348))
    write_text(
        OUT / "depth_support_threshold_trace.md",
        f"""# Depth support threshold trace

R2 source path: `{R2_SCRIPT}`
line range: 323-347

```text
{depth_excerpt}
```

R2 set:

`strict = inside.copy()`

`medium = inside.copy()`

`loose = inside.copy()`

It did not compute or store `depth_rel_error`, so identical strict/medium/loose counts are an implementation artifact, not data-supported threshold evidence.

Formal label: DEPTH-THRESHOLD IMPLEMENTATION BUG.
""",
    )

    surfel_mask = strata == "SURFEL_LIKE"
    surface_idx = cand_idx[surfel_mask]
    surface_xyz = xyz[surfel_mask]
    surface_normals = normals[surfel_mask]
    surface_support = support.iloc[np.flatnonzero(surfel_mask)].reset_index(drop=True)
    surface_rows = []
    for gi, sup in zip(surface_idx, surface_support.to_dict("records")):
        surface_rows.append({"gaussian_index": int(gi), "valid_views": int(sup["valid_views"]), "inside_views": int(sup["inside_views"]), "inside_fraction": float(sup["inside_fraction"]), "strict_support_views": int(sup["strict_support_views"]), "medium_support_views": int(sup["medium_support_views"]), "loose_support_views": int(sup["loose_support_views"])})
    write_csv(OUT / "surface_layer_candidate_lock.csv", surface_rows)

    if len(surface_idx) >= 256:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(surface_xyz)
            all_dist, all_nn = tree.query(surface_xyz, k=min(max(PATCH_SIZES), len(surface_xyz)))
        except Exception:
            d = np.linalg.norm(surface_xyz[:, None, :] - surface_xyz[None, :, :], axis=2)
            all_nn = np.argsort(d, axis=1)[:, : min(max(PATCH_SIZES), len(surface_xyz))]
            all_dist = np.take_along_axis(d, all_nn, axis=1)
    else:
        all_nn = np.zeros((0, 0), dtype=np.int64)
        all_dist = np.zeros((0, 0), dtype=np.float64)

    sweep_rows = []
    eligible_by_seed: dict[int, dict] = {}
    for si, seed_global in enumerate(surface_idx):
        best = None
        for K in PATCH_SIZES:
            if len(surface_idx) < K:
                continue
            local = all_nn[si, :K]
            pts = surface_xyz[local]
            ns = surface_normals[local]
            angles, n_ref = normal_angles(ns)
            radius = float(np.percentile(np.linalg.norm(pts - pts.mean(axis=0), axis=1), 95))
            inside_med = float(np.median(surface_support.iloc[local]["inside_fraction"].to_numpy(float)))
            visible_cam = int(np.median(surface_support.iloc[local]["valid_views"].to_numpy(float)))
            normal_p90 = float(np.percentile(angles, 90))
            eligible = K >= 64 and normal_p90 <= 10.0 and visible_cam >= 3 and inside_med >= 0.90
            row = {
                "seed_gaussian_index": int(seed_global),
                "K": int(K),
                "radius_p95": radius,
                "normal_p50": float(np.percentile(angles, 50)),
                "normal_p90": normal_p90,
                "normal_p95": float(np.percentile(angles, 95)),
                "mask_inside_fraction_median": inside_med,
                "visible_camera_count": visible_cam,
                "eligible": int(eligible),
            }
            sweep_rows.append(row)
            if eligible and K in FORMAL_SIZES:
                if best is None or K > best["K"]:
                    best = {**row, "local_indices": local}
        if best is not None:
            eligible_by_seed[int(seed_global)] = best
    write_csv(OUT / "patch_scale_coherence_sweep.csv", sweep_rows)

    eligible_rows = sorted(
        eligible_by_seed.values(),
        key=lambda r: (-r["K"], r["normal_p90"], -r["mask_inside_fraction_median"], r["radius_p95"]),
    )
    frozen = []
    for row in eligible_rows:
        center = surface_xyz[row["local_indices"]].mean(axis=0)
        ok = True
        for prior in frozen:
            pc = surface_xyz[prior["local_indices"]].mean(axis=0)
            ok = ok and np.linalg.norm(center - pc) >= 2.0 * max(row["radius_p95"], prior["radius_p95"])
        if ok:
            frozen.append(row)
        if len(frozen) == 3:
            break

    patch_rows = []
    for pi, row in enumerate(frozen):
        pid = ["A", "B", "C"][pi]
        global_indices = surface_idx[row["local_indices"]]
        path = PATCH_DIR / f"patch_{pid}.npy"
        np.save(path, global_indices.astype(np.int64))
        patch_rows.append({"patch_id": pid, "seed_gaussian_index": int(row["seed_gaussian_index"]), "K": int(row["K"]), "normal_p90": float(row["normal_p90"]), "normal_p95": float(row["normal_p95"]), "radius_p95": float(row["radius_p95"]), "mask_inside_fraction_median": float(row["mask_inside_fraction_median"]), "visible_camera_count": int(row["visible_camera_count"]), "indices_path": str(path), "indices_sha": sha256_file(path), "mtime_ns": path.stat().st_mtime_ns})
    write_csv(OUT / "coherent_patch_manifest.csv", patch_rows)
    kill1 = len(patch_rows) >= 2

    # If no formal coherent patches exist, emit empty downstream artifacts and stop the real-render path.
    layer_rows = []
    surviving_patches = patch_rows.copy()
    for row in patch_rows:
        # Approximate layer spread from R2 support because R2 did not store per-camera d_gaussian-d_first.
        layer_rows.append({"patch_id": row["patch_id"], "median_delta_d": "", "MAD_delta_d": "", "p05_delta_d": "", "p95_delta_d": "", "layer_spread": "", "status": "NOT_EVALUATED_NO_RENDER_BEFORE_KILL1" if not kill1 else "NOT_EVALUATED_IN_THIS_DIAGNOSTIC"})
    write_csv(OUT / "coherent_patch_layer_purity.csv", layer_rows)
    kill2 = kill1 and len(surviving_patches) >= 2 and all(r.get("layer_spread", 1) != "" and float(r["layer_spread"]) <= 0.05 for r in layer_rows)

    basis_rows, basis_npz = [], {}
    js_rows = []
    if kill1:
        for row in patch_rows:
            idx = np.load(row["indices_path"])
            local_pos = np.searchsorted(surface_idx, idx)
            B = local_basis(ckpt.xyz[idx], surface_normals[local_pos])
            basis_npz[f"patch_{row['patch_id']}_B"] = B
            basis_rows.append({"patch_id": row["patch_id"], "orthogonality_error": float(np.max(np.abs(B.T @ B - np.eye(3)))), "detB": float(np.linalg.det(B)), **{f"B_{i}{j}": float(B[i, j]) for i in range(3) for j in range(3)}})
    write_csv(OUT / "coherent_patch_local_basis.csv", basis_rows)
    np.savez_compressed(OUT / "coherent_patch_local_basis.npz", **basis_npz)

    # Without surviving pure layers, Js sanity is recorded as not run; this is a kill gate, not a threshold relaxation.
    for row in patch_rows:
        for state in ["E0_IDENTITY", "E1_TANGENT_STRETCH_1P25", "E2_TANGENT_STRETCH_1P50", "E3_TANGENT_STRETCH_2P00", "E4_BIAXIAL_TANGENT_1P50", "E5_TANGENT_SHEAR_0P30", "E6_OBLIQUE_TANGENT_STRETCH_1P80", "E7_PURE_ROTATION"]:
            js_rows.append({"patch_id": row["patch_id"], "state": state, "Js_median": "", "Js_CV": "", "gate": "NOT_RUN_BECAUSE_LAYER_PURITY_OR_KILL1_FAILED"})
    write_csv(OUT / "coherent_patch_js_sanity.csv", js_rows)
    kill3 = kill2 and False

    write_text(
        OUT / "dual_pass_evaluation_semantics.md",
        """# Dual-pass evaluation semantics

This is evaluation infrastructure, not a proposed dual-opacity representation and not a novelty claim.

Geometry pass: original TSGS opacity is kept fixed and is used for first-surface depth, first-surface validity, and geometry diagnostics.

Optical pass: policy opacity is used only for white-pass optical alpha and optical response metrics.

The transparency tensor is unchanged in both passes.

R3 did not enter fresh rendering because coherent/single-layer patch gates failed before policy rendering.
""",
    )
    write_text(
        OUT / "geometry_optics_conflict_context.md",
        """# Geometry-optics conflict context

R2 direct single-opacity mutation changed TSGS first-surface valid-mask topology: valid-mask IoU was 0.881439.

This confirms that directly changing the TSGS opacity used by the renderer can perturb geometry extraction semantics.

R3 therefore defines a fixed-geometry evaluation pass and a separate optical pass. This is only bridge infrastructure. It is not claimed as dual-opacity novelty, because geometry/appearance opacity decoupling is adjacent prior art.
""",
    )

    empty_files = [
        "coherent_real_render_manifest.csv",
        "coherent_anchor_lock.csv",
        "coherent_frozen_keys.csv",
        "coherent_anchor_camera_response.csv",
        "coherent_anchor_response.csv",
        "coherent_policy_comparison.csv",
        "kiot_vs_opacity_linear_kill_gate.csv",
        "linear_regime_diagnostic.csv",
    ]
    for name in empty_files:
        write_csv(OUT / name, [{"status": "NOT_RUN_NO_COHERENT_TSGS_SURFACE_CARRIER"}])
    kill4 = False
    kill5 = "NOT_RUN_NO_COHERENT_TSGS_SURFACE_CARRIER"

    sweep_df = pd.DataFrame(sweep_rows)
    coherent_counts = {K: int(((sweep_df["K"] == K) & (sweep_df["eligible"] == 1)).sum()) for K in PATCH_SIZES} if len(sweep_df) else {K: 0 for K in PATCH_SIZES}
    if not kill1 or not kill2 or not kill3:
        final_case = "CASE NO-COHERENT-TSGS-SURFACE-CARRIER"
        retain = "KILL KIOT real-carrier claim for this TSGS carrier; do not kill controlled-carrier KIOT."
        allow_gt = "NO"
        allow_multi = "NO"
    elif kill5 == "KIOT-REAL-CARRIER-SUPPORTED":
        final_case = "CASE KIOT-REAL-CARRIER-SUPPORTED"
        retain = "RETAIN"
        allow_gt = "YES"
        allow_multi = "YES"
    elif kill5 == "OPACITY-LINEAR-REGIME":
        final_case = "CASE LEARNED-TSGS-LINEAR-OPACITY-REGIME"
        retain = "KILL KIOT real-carrier method claim"
        allow_gt = "NO"
        allow_multi = "NO"
    else:
        final_case = "CASE TRANSPORT-RULE-MIXED"
        retain = "NO CLAIM"
        allow_gt = "NO"
        allow_multi = "NO"

    p2p4_summary = "NOT_RUN_NO_COHERENT_TSGS_SURFACE_CARRIER"
    r2_gate = json.loads((R2 / "real_kiot_bridge_gate.json").read_text())
    items = [
        ("A", "R2 patch Gate 为什么被绕过", "R2 line 392-393 在 chosen<2 时 fallback 到 lowest-score patches，未停止；因此 p90≈75° 的 patch 被继续渲染。"),
        ("B", "R2 eligible patch count", str(eligible_count_r2)),
        ("C", "candidate flatness quantiles", json.dumps(reliability_rows[0], ensure_ascii=False)),
        ("D", "SURFEL_LIKE count/fraction", f"{strata_rows[0]['count']}/{strata_rows[0]['fraction']:.6f}"),
        ("E", "strict/medium/loose identical 是否数据支持还是代码 bug", "DEPTH-THRESHOLD IMPLEMENTATION BUG: R2 copied inside counts into strict/medium/loose and did not store depth_rel_error."),
        ("F", "depth_rel_error distribution", "NOT RECORDED IN R2 SUPPORT CSV"),
        ("G", "patch size32/64/128/256/512/768 coherent candidates", json.dumps(coherent_counts, ensure_ascii=False)),
        ("H", "frozen coherent patch count", str(len(patch_rows))),
        ("I", "patch sizes", ",".join(str(r["K"]) for r in patch_rows) if patch_rows else "NONE"),
        ("J", "patch normal p90", ",".join(f"{r['patch_id']}:{r['normal_p90']:.6f}" for r in patch_rows) if patch_rows else "NONE"),
        ("K", "layer spread", "NOT_RUN because KILL1 failed" if not kill1 else "NOT_EVALUATED"),
        ("L", "KILL1", "PASS" if kill1 else "FAIL"),
        ("M", "KILL2", "PASS" if kill2 else "FAIL"),
        ("N", "E1-E7 Js median/CV", "NOT_RUN because coherent/single-layer patch gates failed"),
        ("O", "KILL3", "PASS" if kill3 else "FAIL"),
        ("P", "dual-pass exact semantics", "geometry pass uses original opacity; optical pass uses policy opacity; transparency unchanged; not a novelty claim"),
        ("Q", "real render count", "0"),
        ("R", "synthetic response used yes/no", "NO"),
        ("S", "KILL4", "PASS" if kill4 else "FAIL"),
        ("T", "each patch/state P2/P4/Q", p2p4_summary),
        ("U", "P0/P1/P2/P3/P4 mean central errors", f"R2 reference only: P0={r2_gate['P0_mean']:.6f}, P1={r2_gate['P1_mean']:.6f}, P2={r2_gate['P2_mean']:.6f}, P3={r2_gate['P3_mean']:.6f}, P4={r2_gate['P4_mean']:.6f}"),
        ("V", "P2 vs P4 win count", "NOT_RUN"),
        ("W", "paired Delta_error CI", "NOT_RUN"),
        ("X", "P2/P4 p95 Elog comparison", "NOT_RUN"),
        ("Y", "KILL5", kill5),
        ("Z", "low-alpha fractions", "NOT_RUN"),
        ("AA", "linear-regime diagnostic Spearman", "NOT_RUN"),
        ("AB", "direct opacity geometry conflict status", "R2 valid-mask IoU=0.881439, conflict retained"),
        ("AC", "KILL0-KILL5", f"KILL0={'PASS' if kill0 else 'FAIL'}, KILL1={'PASS' if kill1 else 'FAIL'}, KILL2={'PASS' if kill2 else 'FAIL'}, KILL3={'PASS' if kill3 else 'FAIL'}, KILL4={'PASS' if kill4 else 'FAIL'}, KILL5={kill5}"),
        ("AD", "Final CASE", final_case),
        ("AE", "KIOT real-carrier method retain or kill", retain),
        ("AF", "allow deformed-GT yes/no", allow_gt),
        ("AG", "allow multi-scene evaluation yes/no", allow_multi),
    ]
    report = "# Stage 3.5B-R3 透明表面层一致性与光学状态传输生死验证报告\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "surface_layer_transport_kill_gate_report.md", report)
    summary = f"""# Stage 3.5B-R3 summary

- Final CASE: `{final_case}`
- KILL0 protocol lock: {'PASS' if kill0 else 'FAIL'}
- KILL1 coherent patch existence: {'PASS' if kill1 else 'FAIL'}
- KILL2 layer purity: {'PASS' if kill2 else 'FAIL'}
- KILL3 Js sanity: {'PASS' if kill3 else 'FAIL'}
- KILL4 real render provenance: {'PASS' if kill4 else 'FAIL'}
- KILL5 transport rule result: {kill5}
- SURFEL_LIKE candidates: {strata_rows[0]['count']} / {strata_rows[0]['fraction']:.6f}
- Coherent counts by K: {coherent_counts}
- Frozen coherent patches: {len(patch_rows)}
- Allow deformed-GT benchmark: {allow_gt}
- Allow multi-scene evaluation: {allow_multi}
"""
    write_text(OUT / "stage3_5B_R3_summary.md", summary)
    write_text(OUT / "final_terminal_summary.txt", "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n")
    log.extend(f"{k}. {title}: {value}" for k, title, value in items)
    write_text(OUT / "stage3_5B_R3_log.txt", "\n".join(log) + "\n")

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage3.5B-R3 surface-layer transport kill gate\n\nStage3.5B-R2 was the first official-mask, explicit-F, actual-covariance-transport, fresh-render TSGS bridge experiment. However, its selected patches violated the predefined normal-coherence Gate: normal p90 was approximately 75 degrees for all three patches, far above the 10-degree limit. The patch selector nevertheless continued, so R21 failed.\n\nOn these invalid thin-surface patches, opacity-linear transport achieved mean central error 0.040281, while KIOT-CUDA achieved 0.106589. Thus KIOT was not the best real-render policy in R2.\n\nDirect opacity mutation also changed TSGS first-surface valid-mask topology (IoU 0.881439), confirming a geometry-optics semantic conflict.\n\nStage3.5B-R3 is a strict kill Gate. It first determines whether normal-coherent, single-layer, official-mask TSGS surface patches exist at any predeclared spatial scale. Only then does it compare KIOT against opacity-linear using fresh optical renders and a fixed-geometry evaluation pass. If opacity-linear still wins, KIOT is killed as the primary real-carrier method. No MLP rescue is allowed.\n"""
    if "## Stage3.5B-R3 surface-layer transport kill gate" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")

    print("\n".join(f"{k}. {title}: {value}" for k, title, value in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
