from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
STAGE4 = PROJECT / "experiments" / "stage4_0_attribute_sufficiency_gate"
OUT = PROJECT / "experiments" / "stage4_0_R1_oracle_attribute_protocol_repair"
SOURCE = PROJECT / "attribute_study" / "run_stage4_0.py"
SCRIPT = PROJECT / "attribute_study" / "analysis" / "stage4_0_R1_oracle_attribute_protocol_repair.py"
SEED = 20260714
RELEASE_ORDER = [
    "R0_GEOMETRY_ONLY",
    "R1_O",
    "R2_C",
    "R3_V",
    "R4_O_C",
    "R5_O_V",
    "R6_C_V",
    "R7_O_C_V_FULL",
]
RELEASE_ATTRS = {
    "R0_GEOMETRY_ONLY": tuple(),
    "R1_O": ("O",),
    "R2_C": ("C",),
    "R3_V": ("V",),
    "R4_O_C": ("O", "C"),
    "R5_O_V": ("O", "V"),
    "R6_C_V": ("C", "V"),
    "R7_O_C_V_FULL": ("O", "C", "V"),
}
MATERIALS = [
    "MAT0_NEUTRAL_FIXED_THICKNESS",
    "MAT1_NEUTRAL_MASS_CONSERVING",
    "MAT2_TINTED_MASS_CONSERVING",
]
SURFACES = ["S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE"]
DEFORMATIONS = ["D1_STRETCH_X_1P25", "D2_STRETCH_X_1P50", "D3_BIAXIAL_XY_1P50", "D4_SHEAR_XY_0P30", "D5_ANISO_X1P60_Y0P80", "D6_ROTATION_Z_30"]


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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_row(path: Path) -> dict:
    exists = path.exists()
    st = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": int(exists),
        "size": st.st_size if st else 0,
        "mtime": st.st_mtime if st else 0,
        "sha256": sha256_file(path) if exists and path.is_file() else ("directory" if exists else "MISSING"),
    }


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(np.unique(x)) < 3 or len(np.unique(y)) < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if float(np.std(rx)) < 1e-12 or float(np.std(ry)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def matrices() -> dict[str, np.ndarray]:
    a = math.radians(30.0)
    return {
        "D1_STRETCH_X_1P25": np.diag([1.25, 1.0, 1.0]),
        "D2_STRETCH_X_1P50": np.diag([1.50, 1.0, 1.0]),
        "D3_BIAXIAL_XY_1P50": np.diag([1.50, 1.50, 1.0]),
        "D4_SHEAR_XY_0P30": np.array([[1.0, 0.30, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        "D5_ANISO_X1P60_Y0P80": np.diag([1.60, 0.80, 1.0]),
        "D6_ROTATION_Z_30": np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    }


def gt_change(material: str, deformation: str) -> float:
    F = matrices()[deformation]
    detf = abs(float(np.linalg.det(F)))
    js = detf * np.linalg.norm(np.linalg.inv(F).T @ np.array([0.0, 0.0, 1.0]))
    if material == "MAT0_NEUTRAL_FIXED_THICKNESS":
        return 0.0
    return abs(math.log(max(1.0 / js, 1e-12)))


def bootstrap_ci(vals: np.ndarray, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    if len(vals) == 0:
        return float("nan"), float("nan")
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        means[i] = rng.choice(vals, size=len(vals), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第2步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)

    key_paths = [
        STAGE4 / "stage4_0_protocol_lock.json",
        STAGE4 / "benchmark_camera_lock.npz",
        STAGE4 / "canonical_fit_metrics.csv",
        STAGE4 / "oracle_run_manifest.csv",
        STAGE4 / "attribute_release_metrics.csv",
        STAGE4 / "attribute_release_primary_error.csv",
        STAGE4 / "oracle_attribute_delta.csv",
        STAGE4 / "oracle_view_attribute_diagnostic.parquet",
        SOURCE,
    ]
    oracle_ckpts = sorted((STAGE4 / "oracle_checkpoints").glob("*.pt"))
    canonical_ckpts = sorted((STAGE4 / "canonical_models").glob("*.pt"))
    gt_files = sorted((STAGE4 / "gt").glob("*/*/*/*.npy"))
    lock = {
        "stage": "4.0-R1",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "stage4_key_artifacts": [lock_row(p) for p in key_paths],
        "canonical_checkpoint_count": len(canonical_ckpts),
        "oracle_checkpoint_count": len(oracle_ckpts),
        "gt_file_count": len(gt_files),
        "canonical_checkpoints": [lock_row(p) for p in canonical_ckpts],
        "oracle_manifest_sha256": sha256_file(STAGE4 / "oracle_run_manifest.csv"),
        "primary_error_sha256": sha256_file(STAGE4 / "attribute_release_primary_error.csv"),
    }
    write_text(OUT / "stage4_0_R1_protocol_lock.json", json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    B0 = all(p.exists() for p in key_paths) and len(oracle_ckpts) == 288 and len(canonical_ckpts) == 6

    src = SOURCE.read_text()
    terms = ["synthetic", "fake", "mock", "target_metric", "preset_metric", "expected_metric", "hardcoded_psnr", "hardcoded_elog", "surrogate_result"]
    search_lines = []
    for term in terms:
        matches = [(i + 1, line.strip()) for i, line in enumerate(src.splitlines()) if term.lower() in line.lower()]
        search_lines.append(f"## {term}\n")
        search_lines.extend(f"{ln}: {line}" for ln, line in matches)
        if not matches:
            search_lines.append("NO_MATCH")
        search_lines.append("")
    write_text(OUT / "oracle_source_search.txt", "\n".join(search_lines) + "\n")

    real_optimization = bool(re.search(r"\bAdam\b|torch\.optim", src)) and "synthetic_release_error" not in src
    declared_release = "RELEASES" in src and "released" in src
    test_leakage = "split'] == \"test\"" in src and "synthetic_release_error" in src
    fresh_render_metrics = False
    synthetic_metrics = "synthetic_release_error" in src or "psnr = max(18.0, 42.0 - 35.0 * eopt)" in src
    hardcoded = "36.0" in src and "34.5" in src and "synthetic_release_error" in src
    B1 = bool(real_optimization and declared_release and not test_leakage and fresh_render_metrics and not synthetic_metrics)
    provenance = f"""# Stage4.0 Oracle Provenance Trace

A. Were R1-R7 tensors actually optimized with autograd and Adam?

NO. The source does not instantiate `torch.optim.Adam` for oracle jobs and uses `synthetic_release_error(...)` to populate release metrics.

B. Did every release group expose only its declared tensors?

NO FORMAL OPTIMIZATION OCCURRED. Checkpoints record declared release metadata, but no tensor release optimizer path was executed.

C. Were TEST cameras excluded from optimization?

No optimizer used train or test cameras. This avoids leakage, but also means the intended train-camera optimization did not occur.

D. Were release jobs initialized from the SAME canonical checkpoint?

NO REAL RELEASE OPTIMIZATION OCCURRED. The manifest creates per-job checkpoints with metadata rather than optimized states initialized from canonical checkpoints.

E. Were metrics computed from fresh rendered TEST outputs?

NO. Metrics were generated from deterministic surrogate formulas in `synthetic_release_error(...)`, not fresh Gaussian renders.

F. Any synthetic metric generation?

YES.

G. Any hardcoded target E_OPT values?

YES. The Stage4.0 implementation contains fixed formula branches that determine E_OPT by material/deformation/release.

Formal B1: {'PASS' if B1 else 'FAIL'}
"""
    write_text(OUT / "oracle_provenance_trace.md", provenance)

    manifest = read_csv(STAGE4 / "oracle_run_manifest.csv")
    primary = read_csv(STAGE4 / "attribute_release_primary_error.csv")
    pkeys = [(r["surface"], r["material"], r["deformation"], r["release"]) for r in primary]
    mkeys = [(r["surface"], r["material"], r["deformation"], r["release"]) for r in manifest]
    primary_by_key = {k: r for k, r in zip(pkeys, primary)}
    manifest_by_key = {k: r for k, r in zip(mkeys, manifest)}
    expected_keys = [(s, m, d, r) for s in SURFACES for m in MATERIALS for d in DEFORMATIONS for r in RELEASE_ORDER]
    completeness_rows = []
    for k in expected_keys:
        mr = manifest_by_key.get(k)
        pr = primary_by_key.get(k)
        ckpt = Path(mr["checkpoint"]) if mr else Path("__missing__")
        eopt = float(pr["E_OPT"]) if pr else float("nan")
        completeness_rows.append({
            "surface": k[0], "material": k[1], "deformation": k[2], "release": k[3],
            "manifest_row": int(mr is not None),
            "checkpoint_exists": int(ckpt.exists()),
            "metric_row": int(pr is not None),
            "finite_E_OPT": int(np.isfinite(eopt)),
            "E_OPT": eopt,
        })
    duplicate_keys = len(pkeys) - len(set(pkeys))
    missing_keys = len([r for r in completeness_rows if not (r["manifest_row"] and r["checkpoint_exists"] and r["metric_row"] and r["finite_E_OPT"])])
    write_csv(OUT / "oracle_artifact_completeness.csv", completeness_rows)
    B2 = len(completeness_rows) == 288 and missing_keys == 0 and duplicate_keys == 0

    by_case: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in primary:
        by_case[(r["surface"], r["material"], r["deformation"])][r["release"]] = float(r["E_OPT"])
    case_rows = []
    valid_cases = {}
    for case, vals in sorted(by_case.items()):
        r0 = vals["R0_GEOMETRY_ONLY"]
        r7 = vals["R7_O_C_V_FULL"]
        valid = r7 <= 0.15
        valid_cases[case] = valid
        case_rows.append({
            "surface": case[0],
            "material": case[1],
            "deformation": case[2],
            "R0_E_OPT": r0,
            "R7_E_OPT": r7,
            "R7_R0_ratio": r7 / max(r0, 1e-12),
            "R7_le_0p15": int(valid),
            "R0_le_1p10_R7": int(r0 <= 1.10 * r7),
            "GT_CHANGE": gt_change(case[1], case[2]),
        })
    write_csv(OUT / "full_oracle_case_audit.csv", case_rows)
    write_text(OUT / "gt_change_metric_definition.md", "GT_CHANGE is a diagnostic material-coordinate proxy: for MAT0 fixed thickness it is 0; for mass-conserving regimes it is abs(log(1/Js)) under the locked affine deformation and canonical material normal. It uses material identity, not image-pixel correspondence.\n")
    valid_count = sum(valid_cases.values())
    regime_valid = {m: sum(v for k, v in valid_cases.items() if k[1] == m) for m in MATERIALS}
    A3R_initial = valid_count >= 29 and all(regime_valid[m] >= 10 for m in MATERIALS)
    A3R_final = A3R_initial
    write_text(OUT / "repaired_oracle_capacity_gate.json", json.dumps({
        "ORACLE_VALID_CASE_definition": "R7 E_OPT <= 0.15",
        "overall_valid_count": valid_count,
        "MAT0_valid_count": regime_valid[MATERIALS[0]],
        "MAT1_valid_count": regime_valid[MATERIALS[1]],
        "MAT2_valid_count": regime_valid[MATERIALS[2]],
        "A3R_initial": "PASS" if A3R_initial else "FAIL",
        "closure_executed": False,
        "A3R_final": "PASS" if A3R_final else "FAIL",
    }, indent=2) + "\n")
    write_csv(OUT / "r7_failed_case_convergence_audit.csv", [])
    write_csv(OUT / "r7_capacity_closure_manifest.csv", [])

    minimal_rows = []
    necessity_rows = []
    gap_rows = []
    for case, vals in sorted(by_case.items()):
        if not valid_cases[case]:
            continue
        r0 = vals["R0_GEOMETRY_ONLY"]
        r7 = vals["R7_O_C_V_FULL"]
        if r0 <= 1.10 * r7:
            chosen = "R0_NONE"
        else:
            sufficient = []
            for rel in RELEASE_ORDER[1:]:
                er = vals[rel]
                gap = (r0 - er) / (r0 - r7 + 1e-12)
                gap_rows.append({"surface": case[0], "material": case[1], "deformation": case[2], "release": rel, "gap_recovery": gap})
                if er <= 1.10 * r7 and gap >= 0.90:
                    sufficient.append(rel)
            chosen = min(sufficient, key=lambda rel: (len(RELEASE_ATTRS[rel]), RELEASE_ORDER.index(rel))) if sufficient else "NONE"
        minimal_rows.append({"surface": case[0], "material": case[1], "deformation": case[2], "minimal_sufficient_release": chosen, "families": "NONE" if chosen == "R0_NONE" else "+".join(RELEASE_ATTRS.get(chosen, tuple()))})
        for attr in ["O", "C", "V"]:
            if chosen == "R0_NONE":
                best_rel = "R0_GEOMETRY_ONLY"
                eb = r0
                delta = eb - r7
                nec = 0
            else:
                candidates = ["R0_GEOMETRY_ONLY"] + [rel for rel in RELEASE_ORDER[1:] if attr not in RELEASE_ATTRS[rel]]
                best_rel = min(candidates, key=lambda rel: vals[rel])
                eb = vals[best_rel]
                delta = eb - r7
                nec = int(eb >= 1.25 * r7 and delta > 0.02)
            necessity_rows.append({"surface": case[0], "material": case[1], "deformation": case[2], "attribute": attr, "best_without_attribute": best_rel, "E_BEST_WITHOUT_A": eb, "E_FULL": r7, "Delta_A": delta, "CASE_NECESSARY": nec})
    write_csv(OUT / "repaired_minimal_sufficient_attribute_by_case.csv", minimal_rows)
    write_csv(OUT / "repaired_attribute_necessity_by_case.csv", necessity_rows)

    boot_rows = []
    regime_nec = {}
    for material in MATERIALS:
        for attr in ["O", "C", "V"]:
            rows = [r for r in necessity_rows if r["material"] == material and r["attribute"] == attr]
            vals = np.array([float(r["Delta_A"]) for r in rows], dtype=np.float64)
            lo, hi = bootstrap_ci(vals)
            frac = sum(int(r["CASE_NECESSARY"]) for r in rows) / max(len(rows), 1)
            nec = len(rows) >= 10 and frac >= 0.75 and lo > 0
            regime_nec[(material, attr)] = nec
            boot_rows.append({"material": material, "attribute": attr, "valid_case_count": len(rows), "case_necessary_fraction": frac, "mean_Delta": float(vals.mean()) if len(vals) else float("nan"), "median_Delta": float(np.median(vals)) if len(vals) else float("nan"), "ci95_low": lo, "ci95_high": hi, "REGIME_NECESSARY": int(nec)})
    write_csv(OUT / "repaired_attribute_necessity_bootstrap.csv", boot_rows)

    geom_rows = []
    for material in MATERIALS:
        rows = [r for r in case_rows if r["material"] == material and int(r["R7_le_0p15"])]
        frac = sum(int(r["R0_le_1p10_R7"]) for r in rows) / max(len(rows), 1)
        geom_rows.append({"material": material, "oracle_valid_count": len(rows), "geometry_only_sufficient_count": sum(int(r["R0_le_1p10_R7"]) for r in rows), "geometry_only_sufficient_fraction": frac})
    write_csv(OUT / "repaired_geometry_only_sufficiency.csv", geom_rows)

    hist_rows = []
    hist_by_mat = {}
    all_states = ["R0_NONE"] + RELEASE_ORDER[1:]
    for material in MATERIALS:
        c = Counter(r["minimal_sufficient_release"] for r in minimal_rows if r["material"] == material)
        hist_by_mat[material] = c
        row = {"material": material}
        for state in all_states:
            row[state] = c.get(state, 0)
        hist_rows.append(row)
    write_csv(OUT / "repaired_minimal_state_histogram.csv", hist_rows, ["material"] + all_states)

    classifications = {}
    for material in MATERIALS:
        go = next(r["geometry_only_sufficient_fraction"] for r in geom_rows if r["material"] == material)
        nec_attrs = [a for a in ["O", "C", "V"] if regime_nec[(material, a)]]
        non_static = [r for r in minimal_rows if r["material"] == material and r["minimal_sufficient_release"] != "R0_NONE"]
        containing_o_not_v = sum(("O" in RELEASE_ATTRS.get(r["minimal_sufficient_release"], tuple()) and "V" not in RELEASE_ATTRS.get(r["minimal_sufficient_release"], tuple())) for r in non_static) / max(len(non_static), 1)
        if go >= 0.75:
            cls = "CASE STATIC-OPTICAL-STATE-SUFFICIENT"
        elif len(nec_attrs) >= 2:
            cls = "CASE MULTI-ATTRIBUTE-STATE-NECESSARY"
        elif regime_nec[(material, "O")] and not regime_nec[(material, "V")] and containing_o_not_v >= 0.75:
            cls = "CASE SCALAR-OPACITY-DYNAMIC-STATE-SUFFICIENT"
        elif regime_nec[(material, "V")]:
            cls = "CASE VIEW-DEPENDENT-OPTICAL-STATE-NECESSARY"
        elif regime_nec[(material, "C")]:
            cls = "CASE APPEARANCE-STATE-NECESSARY"
        else:
            cls = "CASE ATTRIBUTE-REGIME-MIXED"
        classifications[material] = cls
    write_text(OUT / "repaired_regime_classification.json", json.dumps(classifications, indent=2) + "\n")

    write_text(OUT / "stage4_association_bug_trace.md", "# Stage4.0 Association Bug Trace\n\nThe Stage4.0 source computes grouped association by material/surface/deformation. Within a fixed deformation, detF and singular values are constant; for the planar surface Js is also constant. The source then sets correlation to `0.0` when either side has near-zero standard deviation. Therefore undefined Spearman correlations were converted to zero.\n")
    undefined_to_zero = True

    delta = read_csv(STAGE4 / "oracle_attribute_delta.csv")
    pooled_rows = []
    feature_names = ["Js", "logJs", "detF", "sv1", "sv2", "sv3", "normal_change_angle"]
    target_names = ["Delta_logit_O", "Delta_C_norm", "Delta_V_norm"]
    for material in MATERIALS:
        for surface in SURFACES:
            rows = [r for r in delta if r["material"] == material and r["surface"] == surface]
            for feat in feature_names:
                x = np.array([math.log(float(r["Js"])) if feat == "logJs" else float(r[feat]) for r in rows], dtype=np.float64)
                for target in target_names:
                    y = np.array([float(r[target]) for r in rows], dtype=np.float64)
                    rho = spearman(x, y)
                    pooled_rows.append({"material": material, "surface": surface, "feature": feat, "target": target, "feature_unique_count": len(np.unique(x)), "feature_std": float(np.std(x)), "spearman": rho, "reason": "CONSTANT_FEATURE" if not np.isfinite(rho) else "OK"})
    write_csv(OUT / "pooled_attribute_association.csv", pooled_rows)

    per_rows = []
    by_g = defaultdict(list)
    for r in delta:
        by_g[(r["material"], r["surface"], r["gaussian_id"])].append(r)
    for material in MATERIALS:
        for surface in SURFACES:
            for feat in feature_names:
                for target in target_names:
                    vals = []
                    for (m, s, gid), rows in by_g.items():
                        if m != material or s != surface:
                            continue
                        x = np.array([math.log(float(r["Js"])) if feat == "logJs" else float(r[feat]) for r in rows], dtype=np.float64)
                        y = np.array([float(r[target]) for r in rows], dtype=np.float64)
                        rho = spearman(x, y)
                        if np.isfinite(rho):
                            vals.append(rho)
                    arr = np.array(vals, dtype=np.float64)
                    per_rows.append({"material": material, "surface": surface, "feature": feat, "target": target, "valid_gaussian_count": len(arr), "median": float(np.median(arr)) if len(arr) else float("nan"), "p25": float(np.quantile(arr, .25)) if len(arr) else float("nan"), "p75": float(np.quantile(arr, .75)) if len(arr) else float("nan"), "p90": float(np.quantile(arr, .90)) if len(arr) else float("nan")})
    write_csv(OUT / "per_gaussian_deformation_association.csv", per_rows)

    view = read_csv(STAGE4 / "oracle_view_attribute_diagnostic.parquet")
    view_rows = []
    for material in MATERIALS:
        for surface in SURFACES:
            rows = [r for r in view if r["material_regime"] == material]
            for feat in ["abs_Delta_n_dot_v", "deformed_abs_n_dot_v", "Js", "logJs"]:
                if feat == "abs_Delta_n_dot_v":
                    x = np.abs(np.array([float(r["Delta_n_dot_v"]) for r in rows], dtype=np.float64))
                elif feat == "deformed_abs_n_dot_v":
                    x = np.abs(np.array([float(r["n_dot_v_deformed"]) for r in rows], dtype=np.float64))
                elif feat == "logJs":
                    x = np.log(np.array([float(r["Js"]) for r in rows], dtype=np.float64))
                else:
                    x = np.array([float(r[feat]) for r in rows], dtype=np.float64)
                y = np.array([float(r["oracle_Delta_V_norm"]) for r in rows], dtype=np.float64)
                rho = spearman(x, y)
                view_rows.append({"material": material, "surface": surface, "feature": feat, "target": "Delta_V_norm", "feature_unique_count": len(np.unique(x)), "feature_std": float(np.std(x)), "spearman": rho, "reason": "CONSTANT_FEATURE" if not np.isfinite(rho) else "OK"})
    write_csv(OUT / "view_dependent_attribute_association.csv", view_rows)

    if not (B0 and B1 and B2):
        final_case = "FINAL CASE STAGE4-PROVENANCE-FAIL"
    elif not A3R_final:
        final_case = "FINAL CASE ATTRIBUTE-ORACLE-CAPACITY-UNRESOLVED"
    elif all(v == "CASE STATIC-OPTICAL-STATE-SUFFICIENT" for v in classifications.values()):
        final_case = "FINAL CASE STATIC-STATE-SUFFICIENT"
    else:
        final_case = "FINAL CASE ATTRIBUTE-DYNAMICS-SUPPORTED"
    allow_stage41 = final_case == "FINAL CASE ATTRIBUTE-DYNAMICS-SUPPORTED" and any(hist_by_mat[m].get("R0_NONE", 0) < sum(hist_by_mat[m].values()) * 0.25 for m in MATERIALS)

    def strongest(rows, target):
        ok = [r for r in rows if r.get("target") == target and np.isfinite(float(r["spearman"]))]
        ok.sort(key=lambda r: abs(float(r["spearman"])), reverse=True)
        return "; ".join(f"{r['material']}/{r.get('surface','')} {r['feature']}={float(r['spearman']):.3f}" for r in ok[:3]) if ok else "NONE"

    nec_status = {m: {a: regime_nec[(m, a)] for a in ["O", "C", "V"]} for m in MATERIALS}
    items = [
        ("A", "为什么原 A3 从逻辑上与 MAT0 控制冲突", "MAT0 的成功预期是 R0≈R7；原 A3 又要求 R7<=0.5*R0，导致静态控制成功时必然失败。"),
        ("B", "原 A3 理论最大通过 case 数 under MAT0 expected behavior", "24/36，低于原 29/36 Gate。"),
        ("C", "B0", "PASS" if B0 else "FAIL"),
        ("D", "R1-R7 real optimization yes/no", "YES" if real_optimization else "NO"),
        ("E", "TEST camera leakage yes/no", "YES" if test_leakage else "NO"),
        ("F", "synthetic/hardcoded metric yes/no", "YES" if synthetic_metrics or hardcoded else "NO"),
        ("G", "B1", "PASS" if B1 else "FAIL"),
        ("H", "288 artifact complete count", str(sum(1 for r in completeness_rows if r["manifest_row"] and r["checkpoint_exists"] and r["metric_row"]))),
        ("I", "finite E_OPT count", str(sum(1 for r in completeness_rows if r["finite_E_OPT"]))),
        ("J", "duplicate/missing keys", f"duplicate={duplicate_keys}, missing={missing_keys}"),
        ("K", "B2", "PASS" if B2 else "FAIL"),
        ("L", "36 cases R7<=0.15 count", str(valid_count)),
        ("M", "MAT0/MAT1/MAT2 oracle-valid count", f"{regime_valid[MATERIALS[0]]}/{regime_valid[MATERIALS[1]]}/{regime_valid[MATERIALS[2]]}"),
        ("N", "initial A3R", "PASS" if A3R_initial else "FAIL"),
        ("O", "R7 closure executed yes/no", "NO"),
        ("P", "closure case count", "0"),
        ("Q", "final A3R", "PASS" if A3R_final else "FAIL"),
        ("R", "MAT0 minimal state histogram including R0_NONE", str(dict(hist_by_mat[MATERIALS[0]]))),
        ("S", "MAT1 minimal state histogram", str(dict(hist_by_mat[MATERIALS[1]]))),
        ("T", "MAT2 minimal state histogram", str(dict(hist_by_mat[MATERIALS[2]]))),
        ("U", "MAT0 O/C/V necessary", str(nec_status[MATERIALS[0]])),
        ("V", "MAT1 O/C/V necessary", str(nec_status[MATERIALS[1]])),
        ("W", "MAT2 O/C/V necessary", str(nec_status[MATERIALS[2]])),
        ("X", "MAT0 geometry-only sufficient fraction", str(next(r["geometry_only_sufficient_fraction"] for r in geom_rows if r["material"] == MATERIALS[0]))),
        ("Y", "MAT1 fraction", str(next(r["geometry_only_sufficient_fraction"] for r in geom_rows if r["material"] == MATERIALS[1]))),
        ("Z", "MAT2 fraction", str(next(r["geometry_only_sufficient_fraction"] for r in geom_rows if r["material"] == MATERIALS[2]))),
        ("AA", "MAT0 classification", classifications[MATERIALS[0]]),
        ("AB", "MAT1 classification", classifications[MATERIALS[1]]),
        ("AC", "MAT2 classification", classifications[MATERIALS[2]]),
        ("AD", "undefined Spearman previously converted to zero yes/no", "YES" if undefined_to_zero else "NO"),
        ("AE", "strongest valid pooled associations for Delta O", strongest(pooled_rows, "Delta_logit_O")),
        ("AF", "strongest valid pooled associations for Delta C", strongest(pooled_rows, "Delta_C_norm")),
        ("AG", "strongest valid pooled associations for Delta V", strongest(pooled_rows, "Delta_V_norm")),
        ("AH", "strongest valid view-level association for Delta V", strongest(view_rows, "Delta_V_norm")),
        ("AI", "Final CASE", final_case),
        ("AJ", "new main hypothesis supported yes/no", "YES" if final_case == "FINAL CASE ATTRIBUTE-DYNAMICS-SUPPORTED" else "NO"),
        ("AK", "exact discovered dynamic-state pattern", "PROVENANCE-INVALID: diagnostic pattern is MAT0->R0_NONE, MAT1->R1_O, MAT2->R7_O_C_V_FULL, but it cannot be claimed scientifically because B1 failed."),
        ("AL", "allow Stage4.1 yes/no", "YES" if allow_stage41 else "NO"),
        ("AM", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("AN", "report path", str(OUT / "stage4_0_R1_attribute_protocol_repair_report.md")),
        ("AO", "summary path", str(OUT / "stage4_0_R1_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n"
    report = "# Stage 4.0-R1 Oracle 上限与动态属性分类协议修复报告\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "stage4_0_R1_attribute_protocol_repair_report.md", report)
    write_text(OUT / "stage4_0_R1_summary.md", f"# Stage 4.0-R1 summary\n\n- Final CASE: `{final_case}`\n- B0: {'PASS' if B0 else 'FAIL'}\n- B1: {'PASS' if B1 else 'FAIL'}\n- B2: {'PASS' if B2 else 'FAIL'}\n- A3R: {'PASS' if A3R_final else 'FAIL'}\n- Stage4.1 allowed: {'YES' if allow_stage41 else 'NO'}\n- KIOT status: CONTROLLED-CARRIER-ONLY\n")
    write_text(OUT / "stage4_0_R1_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage4.0-R1 Oracle Protocol Repair\n\nStage4.0 originally reported `ATTRIBUTE-ORACLE-INSUFFICIENT`. Stage4.0-R1 identifies that the original A3 improvement clause was logically incompatible with the MAT0 fixed-thickness static-control regime: if MAT0 behaves correctly, geometry-only R0 is expected to approximately match FULL R7, so requiring R7 to reduce R0 error by 50% fails by design for all 12 MAT0 cases. Even perfect success on all 24 MAT1/MAT2 cases would reach only 24/36, below the original 29/36 requirement.\n\nStage4.0 also omitted `R0_NONE` from the minimal dynamic-state search, forcing static cases to select a dynamic release group. The original per-deformation Spearman analysis grouped constant global-affine features, making several correlations undefined; those undefined values must not be reported as zero correlation. Stage4.0-R1 repairs these protocol calculations without changing the GT benchmark, O/C/V attribute definitions, E_OPT definition, or frozen numerical sufficiency/necessity thresholds.\n\nThe Stage4.0-R1 provenance audit found that the current Stage4.0 implementation generated oracle metrics from deterministic surrogate formulas rather than real autograd/Adam oracle optimization and fresh TEST renders. Therefore the repaired diagnostic pattern is not accepted as a scientific Stage4 conclusion until real oracle provenance is restored.\n"""
    if "## Stage4.0-R1 Oracle Protocol Repair" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
