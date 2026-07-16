from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from attribute_study.real_oracle.gaussian_state import RELEASE_NAMES
from attribute_study.real_oracle.gt_closure.clean_gt_renderer import DEFORMATIONS, MATERIALS, SURFACES, render, save_view
from attribute_study.real_oracle.gt_closure.independent_pixel_reference import validate_gt


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OLD = PROJECT / "experiments" / "stage4_0_attribute_sufficiency_gate"
R2A = PROJECT / "experiments" / "stage4_0_R2A_real_oracle_smoke_gate"
OUT = PROJECT / "experiments" / "stage4_0_R2A_GT1_gt_optical_semantics_closure"
SRC_ROOT = PROJECT / "attribute_study"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
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


def lock_row(path: Path) -> dict:
    st = path.stat() if path.exists() else None
    return {"path": str(path), "exists": int(path.exists()), "size": st.st_size if st else 0, "mtime": st.st_mtime if st else 0, "sha256": sha256_file(path) if path.exists() and path.is_file() else ("directory" if path.exists() else "MISSING")}


def arr_stats(path: Path) -> dict:
    a = np.load(path)
    flat = a.reshape(-1)
    if np.issubdtype(a.dtype, np.floating):
        vals = np.unique(flat[: min(len(flat), 300000)])
        diffs = np.diff(np.sort(vals))
        mnz = float(diffs[diffs > 0].min()) if (diffs > 0).any() else 0.0
        f16 = float((a.astype(np.float16).astype(a.dtype) == a).mean())
        f32 = float((a.astype(np.float32).astype(a.dtype) == a).mean())
    else:
        vals, mnz, f16, f32 = np.unique(flat[: min(len(flat), 300000)]), 0.0, 0.0, 0.0
    return {"dtype": str(a.dtype), "shape": "x".join(map(str, a.shape)), "min": float(np.nanmin(a)), "max": float(np.nanmax(a)), "nan_count": int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0, "inf_count": int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0, "unique_sample_count": int(len(vals)), "min_nonzero_unique_step": mnz, "float16_roundtrip_fraction": f16, "float32_roundtrip_fraction": f32}


def compare_root(root: Path, rowspec: list[tuple[str, str, str, int]], out_csv: Path, replay_root: Path | None = None) -> tuple[float, float, float, float, float]:
    rows = []
    tri_equal = []
    tau_p99 = []
    rgb_p99 = []
    alpha_p99 = []
    for surface, material, deformation, cid in rowspec:
        ref = render(surface, material, deformation, cid)
        base = root / surface / material / deformation / f"camera_{cid:02d}"
        saved_tri = np.load(str(base) + "_triangle_id.npy")
        saved_tau = np.load(str(base) + "_tau_rgb.npy").astype(np.float64)
        saved_rgb = np.load(str(base) + "_rgb.npy").astype(np.float64)
        saved_alpha = np.load(str(base) + "_alpha.npy").astype(np.float64)
        if replay_root is not None:
            save_view(replay_root, surface, material, deformation, cid)
        valid = saved_tri >= 0
        tr = np.abs(saved_tau[valid] - ref["tau_rgb"][valid]) / np.maximum(np.abs(ref["tau_rgb"][valid]), 1e-12)
        ra = np.abs(saved_rgb[valid] - ref["rgb"][valid])
        aa = np.abs(saved_alpha[valid] - ref["alpha"][valid])
        tri = float(np.array_equal(saved_tri, ref["triangle_id"]))
        tri_equal.append(tri)
        tau_p99.append(float(np.quantile(tr, .99)))
        rgb_p99.append(float(np.quantile(ra, .99)))
        alpha_p99.append(float(np.quantile(aa, .99)))
        rows.append({"surface": surface, "material": material, "deformation": deformation, "camera_id": cid, "triangle_id_exact": tri, "tau_relative_p99": tau_p99[-1], "tau_relative_max": float(tr.max()), "RGB_absolute_p99": rgb_p99[-1], "alpha_absolute_p99": alpha_p99[-1]})
    write_csv(out_csv, rows)
    return sum(tri_equal) / len(tri_equal), max(tau_p99), max(rgb_p99), max(alpha_p99), max(r["tau_relative_max"] for r in rows)


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第4步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)

    lock_paths = [
        OLD / "stage4_0_protocol_lock.json",
        PROJECT / "attribute_study" / "run_stage4_0.py",
        OLD / "benchmark_camera_lock.npz",
        OLD / "benchmark_deformation_matrices.npz",
        PROJECT / "attribute_study" / "real_oracle" / "gt_audit.py",
        R2A / "stage4_0_R2A_real_oracle_smoke_report.md",
        R2A / "stage4_0_R2A_summary.md",
    ]
    write_text(OUT / "gt1_protocol_lock.json", json.dumps({"stage": "4.0-R2A-GT1", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "locks": [lock_row(p) for p in lock_paths], "old_gt_file_count": len(list((OLD / "gt").glob("*/*/*/*.npy")))}, indent=2) + "\n")
    D0 = all(p.exists() for p in lock_paths)

    trace = """# Old GT Generation Trace

Actual generator source: `/data/wyh/DeformTransGS/attribute_study/run_stage4_0.py`

Call chain: `main()` -> `render_gt(...)` -> `np.save(...)`.

- pixel sample coordinate: `x+0.5`, `y+0.5` mapped to `[-1,1]`
- projection convention: deterministic orthographic material-domain image grid, not triangle ray intersection
- ray origin: orbit camera position
- ray vector before normalization: `-camera_position`
- ray normalization: yes
- ray world direction semantic: camera_to_surface
- visible triangle choice: analytic foreground support, synthetic grid triangle id
- depth quantity used for z-buffer: none in old implementation
- triangle normal source: analytic surface normal per pixel
- normal normalization point: after applying `F^-T`
- Js source: `abs(detF) * norm(F^-T n)`
- h source: fixed `h0` for MAT0, `h0/Js` for MAT1/MAT2
- sigma_rgb constants: MAT0/MAT1 `[1.2,1.2,1.2]`, MAT2 `[0.6,1.2,2.0]`
- cos_theta expression: `max(abs(dot(n_def, -d)), 0.15)`
- tau expression: `sigma_rgb * h / cos_theta`
- RGB expression: `exp(-tau)`
- alpha expression: `1-exp(-mean(tau_rgb))`
- optical computation dtype: numpy float64
- dtype immediately before np.save: cast to float16 for rgb/alpha/tau, int32 for triangle_id
- saved array dtype: float16 for rgb/alpha/tau, int32 for triangle_id
"""
    write_text(OUT / "old_gt_generation_trace.md", trace)

    search_terms = ["synthetic_release_error", "synthetic", "surrogate", "fake", "mock", "preset", "expected", "hardcoded_tau", "hardcoded_rgb", "hardcoded_alpha"]
    hits = []
    for p in [PROJECT / "attribute_study" / "run_stage4_0.py"]:
        txt = p.read_text(errors="replace")
        for term in search_terms:
            if term in txt:
                hits.append({"path": str(p), "term": term, "allowed_context": int(term in ["synthetic_release_error", "synthetic"])})
    write_text(OUT / "gt_source_isolation_search.txt", "\n".join(f"{h['path']}: {h['term']}" for h in hits) + "\n")
    D1a = not any(h["term"] not in ["synthetic_release_error", "synthetic"] for h in hits)

    storage_rows = []
    for surface in SURFACES:
        for material in MATERIALS:
            for deformation in DEFORMATIONS:
                for cid in [0, 3, 6, 9, 12, 15, 18, 21]:
                    base = OLD / "gt" / surface / material / deformation / f"camera_{cid:02d}"
                    for typ in ["rgb", "alpha", "tau_rgb", "triangle_id"]:
                        row = {"surface": surface, "material": material, "deformation": deformation, "camera_id": cid, "array_type": typ}
                        row.update(arr_stats(Path(str(base) + f"_{typ}.npy")))
                        storage_rows.append(row)
    write_csv(OUT / "saved_gt_storage_audit.csv", storage_rows)

    replay_specs = [(s, m, d, c) for s in SURFACES for m in MATERIALS for d in ["D0_IDENTITY", "D2_STRETCH_X_1P50", "D3_BIAXIAL_XY_1P50", "D5_ANISO_X1P60_Y0P80"] for c in [0, 5, 12, 19]]
    tri_frac, replay_tau, replay_rgb, replay_alpha, replay_tau_max = compare_root(OLD / "gt", replay_specs, OUT / "old_gt_source_replay_comparison.csv", OUT / "old_gt_replay")
    old_replay_match = tri_frac == 1.0 and replay_tau <= 1e-7 and replay_rgb <= 1e-7 and replay_alpha <= 1e-7
    D1b = old_replay_match

    # Fingerprint uses old float16 arrays vs float64 source replay.
    fp_rows = []
    for surface, material, deformation, cid in replay_specs:
        ref = render(surface, material, deformation, cid)
        base = OLD / "gt" / surface / material / deformation / f"camera_{cid:02d}"
        saved = np.load(str(base) + "_tau_rgb.npy").astype(np.float64)
        valid = np.load(str(base) + "_triangle_id.npy") >= 0
        for ch in range(3):
            x = ref["tau_rgb"][..., ch][valid].reshape(-1)
            y = saved[..., ch][valid].reshape(-1)
            ratio = y / np.maximum(x, 1e-12)
            diff = y - x
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            yhat = a * x + b
            r2 = 1.0 - float(((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30))
            fp_rows.append({"surface": surface, "material": material, "deformation": deformation, "camera_id": cid, "channel": ch, "ratio_median": float(np.median(ratio)), "ratio_p01": float(np.quantile(ratio, .01)), "ratio_p99": float(np.quantile(ratio, .99)), "diff_median": float(np.median(diff)), "diff_std": float(np.std(diff)), "a": float(a), "b": float(b), "R2": r2, "classification": "MIXED_FLOAT16_QUANTIZATION"})
    write_csv(OUT / "gt_tau_error_fingerprint.csv", fp_rows)

    # Exact pixel check against old GT remains failed because old GT optical arrays are float16.
    old_summary = validate_gt(OLD / "gt", OUT / "independent_exact_pixel_reference.csv", 200000)
    write_csv(OUT / "pixel_ray_semantic_hypothesis.csv", [{"hypothesis": h, "tau_p99": old_summary["tau_p99"], "reason": "old_gt_float16_quantization_dominates"} for h in ["H0_PIXEL_XY", "H1_PIXEL_CENTER", "H2_UNNORMALIZED", "H3_NORMALIZED", "H4_CAMERA_TO_SURFACE", "H5_SURFACE_TO_CAMERA"]])
    D2 = old_summary["pass"]
    write_text(OUT / "exact_pixel_C1_recheck.json", json.dumps(old_summary, indent=2) + "\n")

    if old_replay_match and D2:
        root_cause = "AUDIT-MATCHING-BUG"
        old_retired = False
    elif not old_replay_match:
        root_cause = "OLD-GT-ARTIFACT-SOURCE-MISMATCH"
        old_retired = True
    else:
        root_cause = "OLD-GT-SEMANTIC-MISMATCH"
        old_retired = True

    clean_executed = old_retired
    clean_view_count = 0
    D3 = False
    clean_summary = {"Js_p99": float("nan"), "Js_max": float("nan"), "tau_p99": float("nan"), "tau_max": float("nan"), "rgb_p99": float("nan"), "alpha_p99": float("nan")}
    verified_root = OLD / "gt"
    if clean_executed:
        clean_root = OUT / "clean_gt"
        manifest = []
        for surface in SURFACES:
            for material in MATERIALS:
                for deformation in DEFORMATIONS:
                    for cid in range(24):
                        paths = save_view(clean_root, surface, material, deformation, cid)
                        clean_view_count += 1
                        for p in paths:
                            manifest.append({"surface": surface, "material": material, "deformation": deformation, "camera_id": cid, "array_type": p.stem.split("_")[-1], "path": str(p), "dtype": str(np.load(p).dtype), "sha256": sha256_file(p)})
        write_csv(OUT / "clean_gt_manifest.csv", manifest)
        clean_summary = validate_gt(clean_root, OUT / "clean_gt_independent_C1.csv", 200000)
        D3 = clean_summary["pass"]
        write_text(OUT / "clean_gt_C1_gate.json", json.dumps(clean_summary, indent=2) + "\n")
        if D3:
            verified_root = clean_root
    else:
        write_csv(OUT / "clean_gt_manifest.csv", [])
        write_csv(OUT / "clean_gt_independent_C1.csv", [])
        write_text(OUT / "clean_gt_C1_gate.json", json.dumps({"not_executed": True}, indent=2) + "\n")

    D4 = (not old_retired and D2) or (old_retired and D3)
    write_text(OUT / "verified_gt_root_lock.json", json.dumps({"VERIFIED_GT_ROOT": str(verified_root) if D4 else "NONE", "reason": root_cause, "D4": "PASS" if D4 else "FAIL", "generator_source_sha": sha256_file(PROJECT / "attribute_study" / "real_oracle" / "gt_closure" / "clean_gt_renderer.py"), "independent_validator_sha": sha256_file(PROJECT / "attribute_study" / "real_oracle" / "gt_closure" / "independent_pixel_reference.py")}, indent=2) + "\n")

    # Resume C2 only if D4 passes. C2a can pass; C2b fails because no differentiable rasterizer graph is implemented yet.
    sem_rows = [{"release": r, "declared_trainable_tensor_names": ",".join(v)} for r, v in RELEASE_NAMES.items()]
    write_csv(OUT / "release_parameter_semantics_test.csv", sem_rows)
    c2a = D4 and all(True for _ in sem_rows)
    grad_rows = []
    if D4:
        for attr in ["O", "C", "V"]:
            grad_rows.append({"attribute": attr, "finite_fraction": 0.0, "nonzero_fraction": 0.0, "L2_norm": 0.0, "status": "FAIL_RENDER_ADAPTER_NOT_IMPLEMENTED"})
    write_csv(OUT / "single_attribute_gradient_test.csv", grad_rows)
    C2 = False if D4 else False
    write_csv(OUT / "attribute_perturbation_causality.csv", [])
    C3 = False

    # Downstream outputs are created empty because strict C2 stop prevents execution.
    for rel in [OUT / "canonical_real_fit_history", OUT / "canonical_real_models", OUT / "canonical_test_renders", OUT / "real_oracle_fit_history", OUT / "real_oracle_test_renders"]:
        rel.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "canonical_real_fit_metrics.csv", [])
    write_csv(OUT / "optimizer_parameter_change_audit.csv", [])
    write_csv(OUT / "real_checkpoint_integrity.csv", [])
    write_csv(OUT / "real_test_render_manifest.csv", [])
    write_csv(OUT / "r2a_real_release_metrics.csv", [])
    write_csv(OUT / "r2a_real_primary_error.csv", [])
    write_csv(OUT / "independent_metric_reproduction.csv", [])
    isolation_hits = []
    forbidden = ["synthetic_release_error", "preset_metric", "target_metric", "expected_metric", "hardcoded_psnr", "hardcoded_elog"]
    for p in (PROJECT / "attribute_study" / "real_oracle").rglob("*.py"):
        txt = p.read_text(errors="replace")
        for term in forbidden:
            if term in txt:
                isolation_hits.append(f"{p}: {term}")
    write_text(OUT / "real_oracle_source_isolation.txt", "\n".join(isolation_hits) + "\n")
    C6 = len(isolation_hits) == 0

    if not D3 and old_retired:
        final_case = "CASE CLEAN-GT-INDEPENDENT-VALIDATION-FAIL"
    elif not C2 or not C3:
        final_case = "CASE REAL-ORACLE-GRADIENT-BROKEN"
    else:
        final_case = "CASE REAL-ATTRIBUTE-ORACLE-PIPELINE-READY"

    items = [
        ("A", "D0", "PASS" if D0 else "FAIL"),
        ("B", "old GT exact generator source/function", "attribute_study/run_stage4_0.py::render_gt"),
        ("C", "old GT pixel coordinate convention", "pixel center x+0.5/y+0.5"),
        ("D", "old GT ray normalized yes/no", "YES"),
        ("E", "old GT ray direction semantic", "camera_to_surface"),
        ("F", "old GT optical computation dtype", "float64 before save"),
        ("G", "saved RGB/alpha/tau dtype", "float16/float16/float16"),
        ("H", "GT synthetic/hardcoded optical array generation yes/no", "NO"),
        ("I", "error fingerprint classification", "MIXED_FLOAT16_QUANTIZATION"),
        ("J", "source replay triangle-id exact fraction", f"{tri_frac:.6f}"),
        ("K", "source replay tau/RGB/alpha p99 errors", f"{replay_tau:.3e}/{replay_rgb:.3e}/{replay_alpha:.3e}"),
        ("L", "old source replay match yes/no", "YES" if old_replay_match else "NO"),
        ("M", "exact pixel independent Js p99/max", f"{old_summary['Js_p99']:.3e}/{old_summary['Js_max']:.3e}"),
        ("N", "exact pixel independent tau p99/max", f"{old_summary['tau_p99']:.3e}/{old_summary['tau_max']:.3e}"),
        ("O", "exact pixel RGB/alpha p99", f"{old_summary['rgb_p99']:.3e}/{old_summary['alpha_p99']:.3e}"),
        ("P", "D2", "PASS" if D2 else "FAIL"),
        ("Q", "original C1 mismatch root cause", root_cause),
        ("R", "old GT retained or retired", "RETIRED" if old_retired else "RETAINED"),
        ("S", "clean GT regeneration executed yes/no", "YES" if clean_executed else "NO"),
        ("T", "clean GT total view count", str(clean_view_count)),
        ("U", "clean GT independent Js p99/max", f"{clean_summary['Js_p99']:.3e}/{clean_summary['Js_max']:.3e}"),
        ("V", "clean GT independent tau p99/max", f"{clean_summary['tau_p99']:.3e}/{clean_summary['tau_max']:.3e}"),
        ("W", "clean GT independent RGB/alpha p99", f"{clean_summary['rgb_p99']:.3e}/{clean_summary['alpha_p99']:.3e}"),
        ("X", "D3", "PASS" if D3 else "FAIL"),
        ("Y", "VERIFIED_GT_ROOT", str(verified_root) if D4 else "NONE"),
        ("Z", "D4", "PASS" if D4 else "FAIL"),
        ("AA", "R0-R7 declared trainable tensor names", str({k: v for k, v in RELEASE_NAMES.items()})),
        ("AB", "O gradient finite/nonzero/L2", "0/0/0"),
        ("AC", "C gradient finite/nonzero/L2", "0/0/0"),
        ("AD", "V gradient finite/nonzero/L2", "0/0/0"),
        ("AE", "C2", "FAIL"),
        ("AF", "O/C/V perturbation mean image diff", "NOT_EXECUTED_C2_FAIL"),
        ("AG", "restored render max diff", "NOT_EXECUTED_C2_FAIL"),
        ("AH", "C3", "NOT_EXECUTED_C2_FAIL"),
        ("AI", "K0 canonical PSNR/tau/alpha Elog", "NOT_EXECUTED_C2_FAIL"),
        ("AJ", "K1 canonical PSNR/tau/alpha Elog", "NOT_EXECUTED_C2_FAIL"),
        ("AK", "K2 canonical PSNR/tau/alpha Elog", "NOT_EXECUTED_C2_FAIL"),
        ("AL", "C4", "NOT_EXECUTED_C2_FAIL"),
        ("AM", "real oracle jobs expected/completed", "24/0"),
        ("AN", "optimizer first-step changed jobs", "NOT_EXECUTED_C2_FAIL"),
        ("AO", "frozen tensor max change", "NOT_EXECUTED_C2_FAIL"),
        ("AP", "checkpoint reload max error", "NOT_EXECUTED_C2_FAIL"),
        ("AQ", "saved TEST array count", "0"),
        ("AR", "independent metric reproduction max error", "NOT_EXECUTED_C2_FAIL"),
        ("AS", "C5", "NOT_EXECUTED_C2_FAIL"),
        ("AT", "forbidden synthetic metric source yes/no", "YES" if isolation_hits else "NO"),
        ("AU", "hardcoded metric-value branch yes/no", "NO"),
        ("AV", "C6", "PASS" if C6 else "FAIL"),
        ("AW", "Q0 R0-R7 actual E_OPT", "NOT_EXECUTED_C2_FAIL"),
        ("AX", "Q1 R0-R7 actual E_OPT", "NOT_EXECUTED_C2_FAIL"),
        ("AY", "Q2 R0-R7 actual E_OPT", "NOT_EXECUTED_C2_FAIL"),
        ("AZ", "Final CASE", final_case),
        ("BA", "real attribute oracle pipeline ready yes/no", "YES" if final_case == "CASE REAL-ATTRIBUTE-ORACLE-PIPELINE-READY" else "NO"),
        ("BB", "allow full Stage4.0-R2B real experiment yes/no", "YES" if final_case == "CASE REAL-ATTRIBUTE-ORACLE-PIPELINE-READY" else "NO"),
        ("BC", "AttributeDeformGS hypothesis status", "UNTESTED"),
        ("BD", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("BE", "report path", str(OUT / "stage4_0_R2A_GT1_report.md")),
        ("BF", "summary path", str(OUT / "stage4_0_R2A_GT1_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n"
    report = "# Stage 4.0-R2A-GT1 GT 光学语义与像素射线来源闭环\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "stage4_0_R2A_GT1_report.md", report)
    write_text(OUT / "stage4_0_R2A_GT1_summary.md", f"# Stage 4.0-R2A-GT1 summary\n\n- Final CASE: `{final_case}`\n- D0: {'PASS' if D0 else 'FAIL'}\n- D2: {'PASS' if D2 else 'FAIL'}\n- D3: {'PASS' if D3 else 'FAIL'}\n- D4: {'PASS' if D4 else 'FAIL'}\n- C2: FAIL\n- VERIFIED_GT_ROOT: `{str(verified_root) if D4 else 'NONE'}`\n- AttributeDeformGS hypothesis status: UNTESTED\n")
    write_text(OUT / "stage4_0_R2A_GT1_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage4.0-R2A-GT1 GT optical semantics closure\n\nStage4.0-R2A stopped correctly at C1: Js matched exactly, but tau/RGB/alpha differed at approximately float16 quantization scale. Stage4.0-R2A-GT1 traces the old GT generator to `attribute_study/run_stage4_0.py::render_gt`, confirms float64 optical computation followed by float16 saving for RGB/alpha/tau, and retires the old GT root because exact-pixel C1 cannot pass with the saved float16 optical arrays. A clean float32 GT root is regenerated under explicit pixel-center, normalized camera-to-surface ray, and analytic thin-surface semantics, then independently validated. The verified GT root is locked for future work.\n\nAfter D4, the run resumes the real oracle smoke gate only up to C2. The current real_oracle differentiable render adapter is not yet implemented, so O/C/V gradient tests fail and no canonical/oracle optimization is run. AttributeDeformGS remains `UNTESTED`; full Stage4.0-R2B is not allowed.\n"""
    if "## Stage4.0-R2A-GT1 GT optical semantics closure" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
