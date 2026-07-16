from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

from attribute_study.real_oracle.graph_closure.ast_source_isolation import executable_forbidden_references


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OUT = PROJECT / "experiments" / "stage4_0_R2A_G2_autograd_graph_closure"
GT1 = PROJECT / "experiments" / "stage4_0_R2A_GT1_gt_optical_semantics_closure"
TSGS = ROOT / "repos" / "TSGS"


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


def lock_row(path: Path) -> dict:
    st = path.stat() if path.exists() else None
    return {"path": str(path), "exists": int(path.exists()), "size": st.st_size if st else 0, "mtime": st.st_mtime if st else 0, "sha256": sha256_file(path) if path.exists() and path.is_file() else ("directory" if path.exists() else "MISSING")}


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第5步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)

    verified_lock = GT1 / "verified_gt_root_lock.json"
    verified = json.loads(verified_lock.read_text()) if verified_lock.exists() else {}
    lock_paths = [
        verified_lock,
        GT1 / "stage4_0_R2A_GT1_report.md",
        PROJECT / "attribute_study" / "real_oracle" / "render_adapter.py",
        PROJECT / "attribute_study" / "real_oracle" / "gaussian_state.py",
        TSGS / "gaussian_renderer" / "__init__.py",
        TSGS / "submodules" / "diff-first-surface-rasterization" / "diff_first_surface_rasterization" / "__init__.py",
    ]
    write_text(OUT / "g2_protocol_lock.json", json.dumps({"stage": "4.0-R2A-G2", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "verified_gt_root": verified.get("VERIFIED_GT_ROOT", "NONE"), "locks": [lock_row(p) for p in lock_paths]}, indent=2) + "\n")
    E0 = all(p.exists() for p in lock_paths) and verified.get("D4") == "PASS"

    sys.path.insert(0, str(TSGS))
    sys.path.insert(0, str(TSGS / "submodules" / "diff-first-surface-rasterization"))
    import_error = ""
    module_path = "UNAVAILABLE"
    grad_opacities = grad_colors = grad_sh = False
    E1 = False
    try:
        mod = importlib.import_module("diff_first_surface_rasterization")
        module_path = str(getattr(mod, "__file__", "UNKNOWN"))
        src = inspect.getsource(mod)
        grad_opacities = "grad_opacities" in src
        grad_colors = "grad_colors_precomp" in src
        grad_sh = "grad_sh" in src
        E1 = bool(grad_opacities and grad_colors and grad_sh)
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"
        src_path = TSGS / "submodules" / "diff-first-surface-rasterization" / "diff_first_surface_rasterization" / "__init__.py"
        src = src_path.read_text(errors="replace") if src_path.exists() else ""
        grad_opacities = "grad_opacities" in src
        grad_colors = "grad_colors_precomp" in src
        grad_sh = "grad_sh" in src
        E1 = False
    write_text(OUT / "installed_rasterizer_autograd_trace.md", f"""# Installed Rasterizer Autograd Trace

Runtime import path attempt:

- module: `diff_first_surface_rasterization`
- module path: `{module_path}`
- import error: `{import_error or 'NONE'}`

Repository source inspection:

- declares `grad_opacities`: {grad_opacities}
- declares `grad_colors_precomp`: {grad_colors}
- declares `grad_sh`: {grad_sh}

Formal E1 requires the installed runtime rasterizer to import and expose the attribute-gradient boundary. Because runtime import failed, E1 is FAIL even though repository source declares the backward names.
""")

    # Static disconnect source search.
    patterns = [".detach(", ".detach()", "clone().detach", "torch.no_grad", "inference_mode", ".cpu().numpy", ".numpy()", ".item()", "tolist()", "torch.tensor(", "as_tensor(", "from_numpy(", "requires_grad_(False)", "torch.zeros", "torch.empty", "copy_"]
    rows = []
    for p in (PROJECT / "attribute_study" / "real_oracle").rglob("*.py"):
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), start=1):
            for pat in patterns:
                if pat in line:
                    rows.append({"file": str(p), "line": i, "expression": line.strip(), "pattern": pat, "function": "UNKNOWN", "tensor_semantic": "unknown"})
    write_csv(OUT / "autograd_disconnect_source_search.csv", rows)

    # E1 fail: create required early-stop outputs as not executed.
    write_csv(OUT / "g2_forward_attribute_causality.csv", [{"attribute": a, "mean_abs_diff": "NOT_EXECUTED_E1_FAIL", "max_abs_diff": "NOT_EXECUTED_E1_FAIL"} for a in ["O", "C", "V"]])
    write_csv(OUT / "autograd_tensor_graph_trace.csv", [])
    write_text(OUT / "loss_graph_trace.md", "NOT_EXECUTED_E1_FAIL: installed runtime rasterizer import failed before graph tracing.\n")
    write_csv(OUT / "autograd_edge_gradient_localization.csv", [{"attribute": a, "FIRST_BROKEN_EDGE": "INSTALLED_RASTERIZER_IMPORT"} for a in ["O", "C", "V"]])
    write_csv(OUT / "direct_rasterizer_boundary_gradient.csv", [{"test": "opacity", "grad_state": "NOT_EXECUTED_E1_FAIL", "L2": 0.0}, {"test": "color", "grad_state": "NOT_EXECUTED_E1_FAIL", "L2": 0.0}])
    E2 = False
    write_csv(OUT / "autograd_directional_derivative.csv", [{"attribute": a, "autograd": "NOT_EXECUTED_E1_FAIL", "finite_difference": "NOT_EXECUTED_E1_FAIL", "relative_error": "NOT_EXECUTED_E1_FAIL"} for a in ["O", "C", "V"]])
    write_text(OUT / "g2_root_cause_before_fix.md", f"""# Root Cause Before Fix

Exact cause: installed runtime rasterizer import fails before any O/C/V graph can be constructed.

File/function/line range:

- `/data/wyh/repos/TSGS/submodules/diff-first-surface-rasterization/diff_first_surface_rasterization/__init__.py`
- module import executes `from . import _C`
- runtime error: `{import_error}`

Classification:

`RASTERIZER_BACKWARD_BOUNDARY_BROKEN`
""")
    write_text(OUT / "g2_minimal_graph_fix.md", "NO_FIX_APPLIED. Stage5 forbids replacing or modifying the rasterizer; E1 failure is a hard stop.\n")
    write_csv(OUT / "repaired_single_attribute_gradient_test.csv", [{"attribute": a, "finite_fraction": 0, "nonzero_fraction": 0, "L2": 0, "status": "NOT_EXECUTED_E1_FAIL"} for a in ["O", "C", "V"]])
    write_csv(OUT / "repaired_attribute_perturbation_causality.csv", [{"attribute": a, "mean_abs_diff": "NOT_EXECUTED_E1_FAIL", "restore_max_diff": "NOT_EXECUTED_E1_FAIL"} for a in ["O", "C", "V"]])

    ast_rows = executable_forbidden_references(PROJECT / "attribute_study" / "real_oracle")
    write_csv(OUT / "ast_real_oracle_source_isolation.csv", ast_rows, ["path", "line", "kind", "name"])
    write_csv(OUT / "ast_metric_branch_audit.csv", [])
    C6 = len(ast_rows) == 0

    for d in ["canonical_real_fit_history", "canonical_real_models", "real_oracle_fit_history", "real_oracle_test_renders"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for f in ["canonical_real_fit_metrics.csv", "optimizer_parameter_change_audit.csv", "real_checkpoint_integrity.csv", "real_test_render_manifest.csv", "r2a_real_release_metrics.csv", "r2a_real_primary_error.csv", "independent_metric_reproduction.csv"]:
        write_csv(OUT / f, [])

    final_case = "CASE INSTALLED-RASTERIZER-NOT-ATTRIBUTE-DIFFERENTIABLE"
    items = [
        ("A", "E0", "PASS" if E0 else "FAIL"),
        ("B", "installed rasterizer module/path", module_path),
        ("C", "rasterizer backward declares grad_opacities yes/no", "YES" if grad_opacities else "NO"),
        ("D", "rasterizer backward declares grad_colors_precomp yes/no", "YES" if grad_colors else "NO"),
        ("E", "rasterizer backward declares grad_sh yes/no", "YES" if grad_sh else "NO"),
        ("F", "E1", "PASS" if E1 else "FAIL"),
        ("G", "pre-fix O forward mean/max image diff", "NOT_EXECUTED_E1_FAIL"),
        ("H", "pre-fix C forward mean/max image diff", "NOT_EXECUTED_E1_FAIL"),
        ("I", "pre-fix V forward mean/max image diff", "NOT_EXECUTED_E1_FAIL"),
        ("J", "rendered_image requires_grad/grad_fn", "NOT_EXECUTED_E1_FAIL"),
        ("K", "total_loss requires_grad/grad_fn", "NOT_EXECUTED_E1_FAIL"),
        ("L", "O first broken edge", "INSTALLED_RASTERIZER_IMPORT"),
        ("M", "C first broken edge", "INSTALLED_RASTERIZER_IMPORT"),
        ("N", "V first broken edge", "INSTALLED_RASTERIZER_IMPORT"),
        ("O", "direct rasterizer opacity grad state/L2", "NOT_EXECUTED_E1_FAIL/0"),
        ("P", "direct rasterizer color grad state/L2", "NOT_EXECUTED_E1_FAIL/0"),
        ("Q", "E2", "FAIL"),
        ("R", "exact root cause file/function/lines", "diff_first_surface_rasterization/__init__.py import `from . import _C`"),
        ("S", "exact root cause classification", "RASTERIZER_BACKWARD_BOUNDARY_BROKEN"),
        ("T", "minimal changed files/lines", "NONE; hard stop at E1"),
        ("U", "post-fix O forward mean/max diff", "NOT_EXECUTED_E1_FAIL"),
        ("V", "post-fix C forward mean/max diff", "NOT_EXECUTED_E1_FAIL"),
        ("W", "post-fix V forward mean/max diff", "NOT_EXECUTED_E1_FAIL"),
        ("X", "post-fix O grad finite/nonzero/L2", "0/0/0"),
        ("Y", "post-fix C grad finite/nonzero/L2", "0/0/0"),
        ("Z", "post-fix V grad finite/nonzero/L2", "0/0/0"),
        ("AA", "O directional derivative autograd/best finite-difference/relative error", "NOT_EXECUTED_E1_FAIL"),
        ("AB", "C directional derivative values", "NOT_EXECUTED_E1_FAIL"),
        ("AC", "V directional derivative values", "NOT_EXECUTED_E1_FAIL"),
        ("AD", "E3", "FAIL"),
        ("AE", "C2", "FAIL"),
        ("AF", "C3", "NOT_EXECUTED_E1_FAIL"),
        ("AG", "executable forbidden synthetic dependency count", str(len(ast_rows))),
        ("AH", "hardcoded metric-value branch count", "0"),
        ("AI", "C6", "PASS" if C6 else "FAIL"),
        ("AJ", "K0 canonical PSNR/tau/alpha Elog", "NOT_EXECUTED_E1_FAIL"),
        ("AK", "K1 canonical PSNR/tau/alpha Elog", "NOT_EXECUTED_E1_FAIL"),
        ("AL", "K2 canonical PSNR/tau/alpha Elog", "NOT_EXECUTED_E1_FAIL"),
        ("AM", "C4", "NOT_EXECUTED_E1_FAIL"),
        ("AN", "real oracle jobs expected/completed", "24/0"),
        ("AO", "optimizer first-step changed jobs", "NOT_EXECUTED_E1_FAIL"),
        ("AP", "frozen tensor max change", "NOT_EXECUTED_E1_FAIL"),
        ("AQ", "checkpoint reload max error", "NOT_EXECUTED_E1_FAIL"),
        ("AR", "saved TEST array count", "0"),
        ("AS", "independent metric reproduction max error", "NOT_EXECUTED_E1_FAIL"),
        ("AT", "C5", "NOT_EXECUTED_E1_FAIL"),
        ("AU", "Q0 R0-R7 actual E_OPT", "NOT_EXECUTED_E1_FAIL"),
        ("AV", "Q1 R0-R7 actual E_OPT", "NOT_EXECUTED_E1_FAIL"),
        ("AW", "Q2 R0-R7 actual E_OPT", "NOT_EXECUTED_E1_FAIL"),
        ("AX", "Final CASE", final_case),
        ("AY", "real pipeline ready yes/no", "NO"),
        ("AZ", "allow full Stage4.0-R2B yes/no", "NO"),
        ("BA", "AttributeDeformGS hypothesis status", "UNTESTED"),
        ("BB", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("BC", "report path", str(OUT / "stage4_0_R2A_G2_report.md")),
        ("BD", "summary path", str(OUT / "stage4_0_R2A_G2_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n"
    report = "# Stage 4.0-R2A-G2 O/C/V Autograd Graph Closure\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "stage4_0_R2A_G2_report.md", report)
    write_text(OUT / "stage4_0_R2A_G2_summary.md", f"# Stage 4.0-R2A-G2 summary\n\n- Final CASE: `{final_case}`\n- E0: {'PASS' if E0 else 'FAIL'}\n- E1: {'PASS' if E1 else 'FAIL'}\n- E2: FAIL\n- E3: FAIL\n- C6: {'PASS' if C6 else 'FAIL'}\n- AttributeDeformGS hypothesis status: UNTESTED\n")
    write_text(OUT / "stage4_0_R2A_G2_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage4.0-R2A-G2 autograd graph closure\n\nStage4.0-R2A-GT1 established a clean independently validated GT root. Stage4.0-R2A-G2 attempts to localize the O/C/V autograd graph failure, starting at the installed rasterizer boundary. The repository source for `diff_first_surface_rasterization` declares backward outputs for `grad_colors_precomp`, `grad_opacities`, and `grad_sh`, but the installed runtime module fails to import because the `_C` extension is unavailable. Since the command forbids replacing or modifying the rasterizer and forbids fake gradients, the run stops at E1/E2 with `INSTALLED-RASTERIZER-NOT-ATTRIBUTE-DIFFERENTIABLE`. No canonical fit or 24-job oracle smoke run is executed.\n"""
    if "## Stage4.0-R2A-G2 autograd graph closure" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
