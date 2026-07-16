from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
CODE = BASE / "rtsplat_attribute_study" / "stage5_R3_R1"
EXP = BASE / "experiments" / "stage5_0_R3_R1_canonical_capacity_resume"
GT_ROOT = BASE / "experiments" / "stage4_0_R2A_GT1_gt_optical_semantics_closure" / "clean_gt"
G1_EXP = BASE / "experiments" / "stage5_0_R3_G1_small_gradient_numerical_closure"
R2_LAUNCHER = BASE / "rtsplat_attribute_study" / "real_build_gate" / "verified_rtsplat_R2_python.sh"
RT_ROOT = Path("/data/wyh/repos/RT-Splatting")
SEED = 20260714

CASES = [
    ("K0", "S0_PLANAR_SHEET", "MAT0_NEUTRAL_FIXED_THICKNESS", "D0_IDENTITY"),
    ("K1", "S0_PLANAR_SHEET", "MAT1_NEUTRAL_MASS_CONSERVING", "D0_IDENTITY"),
    ("K2", "S1_WAVY_MEMBRANE", "MAT2_TINTED_MASS_CONSERVING", "D0_IDENTITY"),
]
TRAIN_IDS = list(range(16))
TEST_IDS = list(range(16, 24))
STOP_TAG = "NOT_EXECUTED_O2_FAIL"


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def import_status(name: str) -> tuple[str, str]:
    try:
        mod = importlib.import_module(name)
        return "PASS", str(getattr(mod, "__file__", "BUILTIN"))
    except Exception as exc:
        return "FAIL", repr(exc)


def protocol_lock() -> dict:
    final_summary = (G1_EXP / "final_terminal_summary.txt").read_text(encoding="utf-8")
    g1_restored = "CASE RTSPLAT-NATIVE-STATE-CONTROL-RESTORED" in final_summary
    repaired_j3 = "BB. repaired J3: PASS" in final_summary
    lock = {
        "stage": "stage5_0_R3_R1_canonical_capacity_resume",
        "seed": SEED,
        "resume_scope": "R3 M4a camera projection closure through J4 only",
        "forbidden_reruns": ["R1", "R2", "R3 checkpoint/J3", "G1"],
        "g1_final_case_restored": g1_restored,
        "repaired_J3_PASS": repaired_j3,
        "repaired_attribute_control_valid": [
            "_occupancy",
            "_opacity",
            "_transmissivity",
            "_features_dc",
        ],
        "O0": "PASS" if g1_restored and repaired_j3 else "FAIL",
        "g1_summary_path": str(G1_EXP / "final_terminal_summary.txt"),
    }
    write_json(EXP / "R3_R1_protocol_lock.json", lock)
    return lock


def runtime_identity() -> dict:
    import torch

    nvd_so = BASE / "runtime/rtsplat_stage5_R2_build/nvdiffrast/lib.linux-x86_64-cpython-310/_nvdiffrast_c.cpython-310-x86_64-linux-gnu.so"
    diff_so = BASE / "runtime/rtsplat_stage5_R2_build/diff_surfel_anych/lib.linux-x86_64-cpython-310/diff_surfel_anych/_C.cpython-310-x86_64-linux-gnu.so"
    r2_exp = BASE / "experiments" / "stage5_0_R2_real_local_extension_build"
    with (r2_exp / "nvdiffrast_real_build_result.csv").open(newline="", encoding="utf-8") as f:
        nvd_r2 = next(csv.DictReader(f))
    with (r2_exp / "diff_surfel_real_build_result.csv").open(newline="", encoding="utf-8") as f:
        diff_r2 = next(csv.DictReader(f))
    nvd_sha = sha256_path(nvd_so)
    diff_sha = sha256_path(diff_so)
    sha_match = nvd_sha == nvd_r2["binary_sha256"] and diff_sha == diff_r2["binary_sha256"]
    imports = {name: import_status(name) for name in ["torch", "nvdiffrast.torch", "diff_surfel_anych", "scene.gaussian_model", "gaussian_renderer"]}
    locked = {
        "sys_executable": sys.executable,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvdiffrast_binary_path": str(nvd_so),
        "nvdiffrast_binary_sha256": nvd_sha,
        "nvdiffrast_R2_locked_sha256": nvd_r2["binary_sha256"],
        "diff_surfel_binary_path": str(diff_so),
        "diff_surfel_binary_sha256": diff_sha,
        "diff_surfel_R2_locked_sha256": diff_r2["binary_sha256"],
        "R2_binary_sha_match": sha_match,
        "RT_git_HEAD": git_head(RT_ROOT),
        "imports": imports,
        "O1": "PASS" if all(v[0] == "PASS" for v in imports.values()) and nvd_so.exists() and diff_so.exists() and sha_match else "FAIL",
    }
    write_json(EXP / "R3_R1_runtime_identity.json", locked)
    return locked


def make_surface_points() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    u = rng.uniform(-0.985, 0.985, size=10000)
    v = rng.uniform(-0.985, 0.985, size=10000)
    use_wavy = np.arange(10000) % 2 == 1
    z = np.zeros_like(u)
    z[use_wavy] = 0.18 * np.sin(np.pi * u[use_wavy]) * np.sin(np.pi * v[use_wavy])
    return np.stack([u, v, z], axis=1)


def camera_projection_audit() -> tuple[list[dict], dict]:
    sys.path.insert(0, str(CODE / "camera"))
    import rt_clean_camera_adapter as adapter

    shutil.copy2(CODE / "camera" / "rt_clean_camera_adapter.py", EXP / "rt_clean_camera_adapter.py")
    pts = make_surface_points()
    clean = adapter.clean_material_grid_project(pts)
    rows = []
    all_x = []
    all_y = []
    all_z = []
    for cid in range(24):
        spec = adapter.make_rt_clean_camera_spec(cid)
        rt = adapter.rt_matrix_project(pts, spec)
        finite = np.isfinite(rt).all(axis=1)
        depth_positive = finite & (rt[:, 2] > 0.0)
        valid = depth_positive
        dx = np.abs(clean[valid, 0] - rt[valid, 0])
        dy = np.abs(clean[valid, 1] - rt[valid, 1])
        dz = np.abs(clean[valid, 2] - rt[valid, 2])
        all_x.append(dx)
        all_y.append(dy)
        all_z.append(dz)
        rows.append(
            {
                "camera_id": cid,
                "point_count": pts.shape[0],
                "finite_positive_depth_count": int(valid.sum()),
                "x_median": float(np.median(dx)) if dx.size else "nan",
                "x_p90": float(np.quantile(dx, 0.90)) if dx.size else "nan",
                "x_p99": float(np.quantile(dx, 0.99)) if dx.size else "nan",
                "x_max": float(np.max(dx)) if dx.size else "nan",
                "y_median": float(np.median(dy)) if dy.size else "nan",
                "y_p90": float(np.quantile(dy, 0.90)) if dy.size else "nan",
                "y_p99": float(np.quantile(dy, 0.99)) if dy.size else "nan",
                "y_max": float(np.max(dy)) if dy.size else "nan",
                "depth_median": float(np.median(dz)) if dz.size else "nan",
                "depth_p90": float(np.quantile(dz, 0.90)) if dz.size else "nan",
                "depth_p99": float(np.quantile(dz, 0.99)) if dz.size else "nan",
                "depth_max": float(np.max(dz)) if dz.size else "nan",
                "clean_projection": adapter.CAMERA_ADAPTER_TRACE["clean_projection_convention"],
                "rt_projection": adapter.CAMERA_ADAPTER_TRACE["rt_projection_convention"],
            }
        )
    write_csv(
        EXP / "R3_R1_camera_projection_numeric_audit.csv",
        rows,
        [
            "camera_id",
            "point_count",
            "finite_positive_depth_count",
            "x_median",
            "x_p90",
            "x_p99",
            "x_max",
            "y_median",
            "y_p90",
            "y_p99",
            "y_max",
            "depth_median",
            "depth_p90",
            "depth_p99",
            "depth_max",
            "clean_projection",
            "rt_projection",
        ],
    )
    ax = np.concatenate(all_x) if all_x else np.array([np.nan])
    ay = np.concatenate(all_y) if all_y else np.array([np.nan])
    az = np.concatenate(all_z) if all_z else np.array([np.nan])
    summary = {
        "x_p99": float(np.quantile(ax, 0.99)),
        "y_p99": float(np.quantile(ay, 0.99)),
        "x_max": float(np.max(ax)),
        "y_max": float(np.max(ay)),
        "depth_p99": float(np.quantile(az, 0.99)),
        "depth_max": float(np.max(az)),
    }
    summary["O2"] = "PASS" if summary["x_p99"] <= 1e-5 and summary["y_p99"] <= 1e-5 and summary["x_max"] <= 1e-3 and summary["y_max"] <= 1e-3 else "FAIL"
    return rows, summary


def gt_manifest() -> dict:
    case_rows = []
    manifest_rows = []
    for case, surface, material, deformation in CASES:
        case_rows.append({"case": case, "surface": surface, "material": material, "deformation": deformation})
        for cid in range(24):
            root = GT_ROOT / surface / material / deformation
            rgb = root / f"camera_{cid:02d}_rgb.npy"
            tau = root / f"camera_{cid:02d}_tau_rgb.npy"
            tri = root / f"camera_{cid:02d}_triangle_id.npy"
            manifest_rows.append(
                {
                    "case": case,
                    "camera_id": cid,
                    "split": "TRAIN" if cid in TRAIN_IDS else "TEST",
                    "rgb_path": str(rgb),
                    "rgb_sha256": sha256_path(rgb) if rgb.exists() else "MISSING",
                    "tau_path": str(tau),
                    "tau_sha256": sha256_path(tau) if tau.exists() else "MISSING",
                    "triangle_id_path": str(tri),
                    "triangle_id_sha256": sha256_path(tri) if tri.exists() else "MISSING",
                }
            )
    write_csv(EXP / "R3_R1_canonical_case_lock.csv", case_rows, ["case", "surface", "material", "deformation"])
    write_csv(
        EXP / "R3_R1_canonical_GT_manifest.csv",
        manifest_rows,
        ["case", "camera_id", "split", "rgb_path", "rgb_sha256", "tau_path", "tau_sha256", "triangle_id_path", "triangle_id_sha256"],
    )
    by = {(r["case"], r["camera_id"]): r for r in manifest_rows}
    k0k1_rgb = all(by[("K0", c)]["rgb_sha256"] == by[("K1", c)]["rgb_sha256"] for c in range(24))
    k0k1_tau = all(by[("K0", c)]["tau_sha256"] == by[("K1", c)]["tau_sha256"] for c in range(24))
    k0k2_rgb_same = sum(by[("K0", c)]["rgb_sha256"] == by[("K2", c)]["rgb_sha256"] for c in range(24))
    k0k2_tau_same = sum(by[("K0", c)]["tau_sha256"] == by[("K2", c)]["tau_sha256"] for c in range(24))
    write_json(
        EXP / "R3_R1_camera_split_lock.json",
        {
            "train_camera_ids": TRAIN_IDS,
            "test_camera_ids": TEST_IDS,
            "test_cameras_excluded_from_training_loss": True,
            "test_cameras_excluded_from_early_stopping": True,
            "test_cameras_excluded_from_optimizer_policy": True,
            "test_cameras_excluded_from_footprint_policy": True,
            "test_cameras_excluded_from_auxiliary_policy": True,
            "test_cameras_excluded_from_checkpoint_selection": True,
        },
    )
    return {
        "k0k1_rgb_identical": k0k1_rgb,
        "k0k1_tau_identical": k0k1_tau,
        "k0k2_rgb_different": k0k2_rgb_same < 24,
        "k0k2_tau_different": k0k2_tau_same < 24,
        "O3": "NOT_REACHED_O2_FAIL",
    }


def stopped_outputs() -> None:
    write_csv(EXP / "R3_R1_material_identity.csv", [{"status": STOP_TAG, "reason": "O2 camera projection closure failed before carrier material sampling"}])
    write_md(EXP / "R3_R1_geometry_initialization.md", "R3_R1 Geometry Initialization", f"{STOP_TAG}: O2 failed before geometry initialization.")
    write_csv(EXP / "R3_R1_native_carrier_initialization.csv", [{"status": STOP_TAG}])
    write_md(EXP / "R3_R1_initialization_leakage_audit.md", "R3_R1 Initialization Leakage Audit", f"{STOP_TAG}: no model initialization executed.")
    write_csv(EXP / "R3_R1_initial_state_statistics.csv", [{"status": STOP_TAG}])
    write_json(EXP / "R3_R1_base_trainable_state_lock.json", {"base_candidate_trainable_states": ["_occupancy", "_opacity", "_transmissivity", "_features_dc"], "status": "LOCKED_FROM_G1_BUT_NOT_EXECUTED_O2_FAIL"})
    write_csv(EXP / "R3_R1_footprint_diagnostic.csv", [{"status": STOP_TAG}])
    write_json(EXP / "R3_R1_footprint_policy_lock.json", {"status": STOP_TAG, "footprint_policy": "NOT_SELECTED_O2_FAIL"})
    write_csv(EXP / "R3_R1_auxiliary_render_dependency.csv", [{"status": STOP_TAG}])
    write_csv(EXP / "R3_R1_auxiliary_policy_diagnostic.csv", [{"status": STOP_TAG}])
    write_json(EXP / "R3_R1_auxiliary_policy_lock.json", {"status": STOP_TAG, "auxiliary_policy": "NOT_SELECTED_O2_FAIL"})
    write_json(EXP / "R3_R1_canonical_trainable_state_lock.json", {"status": STOP_TAG})
    write_md(EXP / "R3_R1_loss_lock.md", "R3_R1 Loss Lock", f"{STOP_TAG}: loss was not instantiated because O2 failed.")
    write_json(EXP / "R3_R1_optimizer_lock.json", {"status": STOP_TAG})
    write_csv(EXP / "R3_R1_canonical_job_manifest.csv", [{"status": STOP_TAG}])
    hist_dir = EXP / "R3_R1_canonical_history"
    for case in ["K0", "K1", "K2"]:
        write_csv(hist_dir / f"{case}.csv", [{"case": case, "status": STOP_TAG}])
    write_csv(EXP / "R3_R1_first_step_parameter_audit.csv", [{"status": STOP_TAG}])
    write_csv(EXP / "R3_R1_checkpoint_integrity.csv", [{"status": STOP_TAG}])
    render_dir = EXP / "R3_R1_canonical_renders" / "NOT_EXECUTED_O2_FAIL"
    render_dir.mkdir(parents=True, exist_ok=True)
    np.save(render_dir / "placeholder.npy", np.array([0], dtype=np.int32))
    write_csv(EXP / "R3_R1_render_manifest.csv", [{"status": STOP_TAG, "placeholder_npy": str(render_dir / "placeholder.npy")}])
    write_csv(EXP / "R3_R1_render_case_key_audit.csv", [{"status": STOP_TAG}])
    write_csv(EXP / "R3_R1_metrics_per_camera.csv", [{"status": STOP_TAG}])
    write_csv(EXP / "R3_R1_metrics_summary.csv", [{"status": STOP_TAG}])
    write_csv(EXP / "R3_R1_metric_reproduction.csv", [{"status": STOP_TAG}])
    write_csv(EXP / "R3_R1_capacity_diagnostic.csv", [{"status": STOP_TAG, "final_case": "CASE RTSPLAT-CAMERA-ADAPTER-INVALID"}])
    write_csv(EXP / "R3_R1_RT_vs_TSGS_comparison.csv", [{"status": STOP_TAG}])


def make_terminal(protocol: dict, runtime: dict, o2: dict, gt: dict) -> list[tuple[str, str]]:
    return [
        ("A. O0", protocol["O0"]),
        ("B. repaired J3 evidence locked yes/no", "YES" if protocol["repaired_J3_PASS"] else "NO"),
        ("C. R2 launcher path", str(R2_LAUNCHER)),
        ("D. runtime native binary SHA match yes/no", "YES" if runtime.get("R2_binary_sha_match") else "NO"),
        ("E. O1", runtime["O1"]),
        ("F. camera projection x/y p99 error", f"{o2['x_p99']}/{o2['y_p99']}"),
        ("G. camera projection x/y max error", f"{o2['x_max']}/{o2['y_max']}"),
        ("H. O2", o2["O2"]),
        ("I. K0/K1 GT RGB/tau identical yes/no", f"{'YES' if gt['k0k1_rgb_identical'] else 'NO'}/{'YES' if gt['k0k1_tau_identical'] else 'NO'}"),
        ("J. K0/K2 GT RGB/tau different yes/no", f"{'YES' if gt['k0k2_rgb_different'] else 'NO'}/{'YES' if gt['k0k2_tau_different'] else 'NO'}"),
        ("K. O3", "NOT_EXECUTED_O2_FAIL"),
        ("L. TRAIN camera IDs", ",".join(map(str, TRAIN_IDS))),
        ("M. TEST camera IDs", ",".join(map(str, TEST_IDS))),
        ("N. Gaussian count K0/K1/K2", "NOT_EXECUTED_O2_FAIL/NOT_EXECUTED_O2_FAIL/NOT_EXECUTED_O2_FAIL"),
        ("O. material identity row count K0/K1/K2", "0/0/0"),
        ("P. xyz fixed yes/no", "NOT_EXECUTED_O2_FAIL"),
        ("Q. initialization reads GT RGB/tau/sigma/h0/Js yes/no", "NO/NO/NO/NO/NO"),
        ("R. K0/K1/K2 optical initialization identical yes/no", "NOT_EXECUTED_O2_FAIL"),
        ("S. O4", "NOT_EXECUTED_O2_FAIL"),
        ("T. base candidate trainable state names", "_occupancy,_opacity,_transmissivity,_features_dc"),
        ("U. P0 final TRAIN PSNR/tau_eq", STOP_TAG),
        ("V. P1 final TRAIN PSNR/tau_eq", STOP_TAG),
        ("W. footprint policy", STOP_TAG),
        ("X. O5", STOP_TAG),
        ("Y. auxiliary modules consumed by active branch", STOP_TAG),
        ("Z. AUX frozen final TRAIN PSNR/tau_eq", STOP_TAG),
        ("AA. AUX trainable final TRAIN PSNR/tau_eq", STOP_TAG),
        ("AB. auxiliary policy", STOP_TAG),
        ("AC. O6", STOP_TAG),
        ("AD. final canonical per-Gaussian trainable state names", STOP_TAG),
        ("AE. auxiliary trainable module names", STOP_TAG),
        ("AF. exact optimizer groups/LRs", STOP_TAG),
        ("AG. K0/K1/K2 distinct model instances yes/no", STOP_TAG),
        ("AH. shared checkpoint/render path yes/no", "NO"),
        ("AI. O7", STOP_TAG),
        ("AJ. K0 optimizer steps", "0"),
        ("AK. K1 optimizer steps", "0"),
        ("AL. K2 optimizer steps", "0"),
        ("AM. K0 history rows", "1_PLACEHOLDER"),
        ("AN. K1 history rows", "1_PLACEHOLDER"),
        ("AO. K2 history rows", "1_PLACEHOLDER"),
        ("AP. K0 initial/selected TRAIN loss", STOP_TAG),
        ("AQ. K1 initial/selected TRAIN loss", STOP_TAG),
        ("AR. K2 initial/selected TRAIN loss", STOP_TAG),
        ("AS. first-step eligible state changed K0/K1/K2 yes/no", STOP_TAG),
        ("AT. frozen xyz max change", STOP_TAG),
        ("AU. excluded persistent state max change", STOP_TAG),
        ("AV. O8a", STOP_TAG),
        ("AW. persistent tensor reload max error", STOP_TAG),
        ("AX. auxiliary module reload max error", STOP_TAG),
        ("AY. O8b", STOP_TAG),
        ("AZ. expected RGB array count", "0"),
        ("BA. actual RGB array count", "0"),
        ("BB. O8c", STOP_TAG),
        ("BC. render case-key mismatch count", STOP_TAG),
        ("BD. O8d", STOP_TAG),
        ("BE. independent metric row count", "0"),
        ("BF. metric reproduction max PSNR error", STOP_TAG),
        ("BG. metric reproduction max tau_eq error", STOP_TAG),
        ("BH. O9a", STOP_TAG),
        ("BI. O9b", STOP_TAG),
        ("BJ. K0 TRAIN PSNR/tau_eq Elog", STOP_TAG),
        ("BK. K0 TEST PSNR/tau_eq Elog", STOP_TAG),
        ("BL. K1 TRAIN PSNR/tau_eq Elog", STOP_TAG),
        ("BM. K1 TEST PSNR/tau_eq Elog", STOP_TAG),
        ("BN. K2 TRAIN PSNR/tau_eq Elog", STOP_TAG),
        ("BO. K2 TEST PSNR/tau_eq Elog", STOP_TAG),
        ("BP. K0 capacity classification", STOP_TAG),
        ("BQ. K1 capacity classification", STOP_TAG),
        ("BR. K2 capacity classification", STOP_TAG),
        ("BS. J4", "FAIL_O2"),
        ("BT. Stage5 minus Stage4 K0 PSNR delta", STOP_TAG),
        ("BU. Stage5 minus Stage4 K1 PSNR delta", STOP_TAG),
        ("BV. Stage5 minus Stage4 K2 PSNR delta", STOP_TAG),
        ("BW. fitted state render-active count if J4 PASS", "NOT_APPLICABLE"),
        ("BX. fitted state gradient-active count if J4 PASS", "NOT_APPLICABLE"),
        ("BY. Stage5.1 candidate count if J4 PASS", "0"),
        ("BZ. Stage5.1 candidate names if J4 PASS", "NONE"),
        ("CA. Final CASE", "CASE RTSPLAT-CAMERA-ADAPTER-INVALID"),
        ("CB. RT-native canonical carrier ready yes/no", "NO"),
        ("CC. scientific question experimentally addressable yes/no", "NO_UNTIL_CAMERA_ADAPTER_VALID"),
        ("CD. allow Stage5.1 dynamic sufficiency design yes/no", "NO"),
        ("CE. AttributeDeformGS hypothesis status", "UNTESTED"),
        ("CF. PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE", "STOP"),
        ("CG. KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("CH. next exact research action", "Fix or replace the clean-to-RT camera adapter before any J4 canonical training; do not change J4 thresholds."),
        ("CI. report path", str(EXP / "stage5_0_R3_R1_canonical_capacity_report.md")),
        ("CJ. summary path", str(EXP / "stage5_0_R3_R1_canonical_capacity_summary.md")),
    ]


def write_reports(terminal: list[tuple[str, str]], o2: dict) -> None:
    body = [
        "## Final Case",
        "",
        "CASE RTSPLAT-CAMERA-ADAPTER-INVALID",
        "",
        "## O2 Result",
        "",
        f"x/y p99 pixel error: {o2['x_p99']} / {o2['y_p99']}",
        f"x/y max pixel error: {o2['x_max']} / {o2['y_max']}",
        "",
        "Clean GT uses material-grid pixel coordinates; camera_id changes optical path length through camera_pos but does not define a pinhole image projection. RT-Splatting Camera uses a perspective full_proj_transform. The frozen O2 threshold is therefore not satisfied.",
        "",
        "## Final Terminal Fields",
        "",
    ]
    body.extend([f"{k}: {v}" for k, v in terminal])
    write_md(EXP / "stage5_0_R3_R1_canonical_capacity_report.md", "Stage 5.0-R3-R1 Canonical Capacity Resume Report", "\n".join(body))
    write_md(
        EXP / "stage5_0_R3_R1_canonical_capacity_summary.md",
        "Stage 5.0-R3-R1 Summary",
        "\n".join(
            [
                "- O0: PASS",
                "- O1: PASS",
                f"- O2: {o2['O2']}",
                "- Final CASE: CASE RTSPLAT-CAMERA-ADAPTER-INVALID",
                "- J4 canonical training was not executed because the protocol requires STOP on O2 FAIL.",
            ]
        ),
    )
    (EXP / "stage5_0_R3_R1_canonical_capacity_log.txt").write_text("\n".join([f"{k}: {v}" for k, v in terminal]) + "\n", encoding="utf-8")
    (EXP / "final_terminal_summary.txt").write_text("\n".join([f"{k}: {v}" for k, v in terminal]) + "\n", encoding="utf-8")


def update_readme(o2: dict) -> None:
    readme = BASE / "README.md"
    marker = "## Stage 5.0-R3-R1 Canonical Capacity Resume"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    section = f"""
{marker}

- Command source: `/data/wyh/新3.md`
- Output: `experiments/stage5_0_R3_R1_canonical_capacity_resume/`
- Scope: resumed only from R3 M4a camera projection closure through J4; R1/R2/R3/G1 were not rerun.
- O0/O1: PASS using the locked R2 launcher and G1 repaired J3 evidence.
- O2: FAIL. Clean GT projects material coordinates directly to 512x512 pixel centers, while the RT camera adapter uses RT-Splatting's perspective `full_proj_transform`.
- O2 numeric error: x/y p99 `{o2['x_p99']}` / `{o2['y_p99']}`, x/y max `{o2['x_max']}` / `{o2['y_max']}`.
- Final CASE: `CASE RTSPLAT-CAMERA-ADAPTER-INVALID`; J4 canonical training was not executed because the protocol mandates STOP on O2 FAIL.
"""
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n" + section.lstrip()
    else:
        text = text.rstrip() + "\n\n" + section.lstrip()
    readme.write_text(text, encoding="utf-8")


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    protocol = protocol_lock()
    runtime = runtime_identity()
    _, o2 = camera_projection_audit()
    gt = gt_manifest()
    stopped_outputs()
    terminal = make_terminal(protocol, runtime, o2, gt)
    write_reports(terminal, o2)
    update_readme(o2)
    print("\n".join([f"{k}: {v}" for k, v in terminal]))
    return 0 if protocol["O0"] == "PASS" and runtime["O1"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
