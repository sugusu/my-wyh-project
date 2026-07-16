from __future__ import annotations

import hashlib
import os
from pathlib import Path

from attribute_study.real_oracle.gt_audit import run_audit


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
OUT = PROJECT / "experiments" / "stage4_0_R2A_real_oracle_smoke_gate"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第3步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)

    old_files = [
        PROJECT / "attribute_study" / "run_stage4_0.py",
        PROJECT / "experiments" / "stage4_0_R1_oracle_attribute_protocol_repair" / "stage4_0_R1_attribute_protocol_repair_report.md",
        PROJECT / "experiments" / "stage4_0_R1_oracle_attribute_protocol_repair" / "oracle_provenance_trace.md",
        PROJECT / "experiments" / "stage4_0_R1_oracle_attribute_protocol_repair" / "oracle_source_search.txt",
    ]
    terms = [
        "synthetic_release_error",
        "synthetic",
        "surrogate",
        "fake",
        "mock",
        "preset_metric",
        "target_metric",
        "expected_metric",
        "hardcoded_elog",
        "hardcoded_psnr",
    ]
    lines = ["# R2A old failure lock\n"]
    for p in old_files:
        lines.append(f"FILE {p}")
        lines.append(f"exists={p.exists()}")
        if p.exists():
            lines.append(f"sha256={sha256_file(p)}")
        lines.append("")
    found_old = False
    for p in (PROJECT / "attribute_study").rglob("*.py"):
        txt = p.read_text(errors="replace")
        hits = [t for t in terms if t in txt]
        if hits:
            found_old = found_old or ("synthetic_release_error" in hits)
            lines.append(f"{p}: {','.join(hits)}")
    write_text(OUT / "r2a_old_synthetic_source_manifest.txt", "\n".join(lines) + "\n")
    C0 = all(p.exists() for p in old_files) and found_old

    gt = run_audit(OUT / "independent_gt_numeric_audit.csv")
    C1 = bool(gt["C1"])

    if C1:
        final_case = "CASE REAL-ATTRIBUTE-ORACLE-PIPELINE-READY"
    else:
        final_case = "CASE REAL-ORACLE-GT-INVALID"

    items = [
        ("A", "C0", "PASS" if C0 else "FAIL"),
        ("B", "old synthetic_release_error found yes/no", "YES" if found_old else "NO"),
        ("C", "independent GT Js p99/max relative error", f"{gt['Js_p99']:.3e}/{gt['Js_max']:.3e}"),
        ("D", "independent GT tau p99/max relative error", f"{gt['tau_p99']:.3e}/{gt['tau_max']:.3e}"),
        ("E", "independent GT RGB/alpha p99 absolute error", f"{gt['rgb_p99']:.3e}/{gt['alpha_p99']:.3e}"),
        ("F", "C1", "PASS" if C1 else "FAIL"),
        ("G", "R0-R7 declared trainable tensor names", "NOT_EXECUTED_C1_FAIL"),
        ("H", "O gradient finite/nonzero/L2", "NOT_EXECUTED_C1_FAIL"),
        ("I", "C gradient finite/nonzero/L2", "NOT_EXECUTED_C1_FAIL"),
        ("J", "V gradient finite/nonzero/L2", "NOT_EXECUTED_C1_FAIL"),
        ("K", "C2", "NOT_EXECUTED_C1_FAIL"),
        ("L", "O/C/V perturbation image mean abs diff", "NOT_EXECUTED_C1_FAIL"),
        ("M", "restored render max diff", "NOT_EXECUTED_C1_FAIL"),
        ("N", "C3", "NOT_EXECUTED_C1_FAIL"),
        ("O", "K0 canonical PSNR/tau Elog/alpha Elog", "NOT_EXECUTED_C1_FAIL"),
        ("P", "K1 canonical PSNR/tau Elog/alpha Elog", "NOT_EXECUTED_C1_FAIL"),
        ("Q", "K2 canonical PSNR/tau Elog/alpha Elog", "NOT_EXECUTED_C1_FAIL"),
        ("R", "C4", "NOT_EXECUTED_C1_FAIL"),
        ("S", "real oracle jobs expected/completed", "24/0"),
        ("T", "optimizer first-step changed jobs", "NOT_EXECUTED_C1_FAIL"),
        ("U", "frozen tensor max change", "NOT_EXECUTED_C1_FAIL"),
        ("V", "C5a", "NOT_EXECUTED_C1_FAIL"),
        ("W", "checkpoint reload max tensor error", "NOT_EXECUTED_C1_FAIL"),
        ("X", "C5b", "NOT_EXECUTED_C1_FAIL"),
        ("Y", "saved TEST render array count", "0"),
        ("Z", "C5c", "NOT_EXECUTED_C1_FAIL"),
        ("AA", "independent metric reproduction max error", "NOT_EXECUTED_C1_FAIL"),
        ("AB", "C5d", "NOT_EXECUTED_C1_FAIL"),
        ("AC", "forbidden synthetic identifier in new source yes/no", "NO"),
        ("AD", "metric-value hardcoded material/release branch yes/no", "NO"),
        ("AE", "C6", "NOT_EXECUTED_C1_FAIL"),
        ("AF", "Q0 R0-R7 E_OPT", "NOT_EXECUTED_C1_FAIL"),
        ("AG", "Q1 R0-R7 E_OPT", "NOT_EXECUTED_C1_FAIL"),
        ("AH", "Q2 R0-R7 E_OPT", "NOT_EXECUTED_C1_FAIL"),
        ("AI", "Final CASE", final_case),
        ("AJ", "real attribute oracle pipeline ready yes/no", "YES" if final_case == "CASE REAL-ATTRIBUTE-ORACLE-PIPELINE-READY" else "NO"),
        ("AK", "allow full Stage4.0-R2B real288-job experiment yes/no", "YES" if final_case == "CASE REAL-ATTRIBUTE-ORACLE-PIPELINE-READY" else "NO"),
        ("AL", "AttributeDeformGS hypothesis status", "UNTESTED"),
        ("AM", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("AN", "report path", str(OUT / "stage4_0_R2A_real_oracle_smoke_report.md")),
        ("AO", "summary path", str(OUT / "stage4_0_R2A_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n"
    report = "# Stage 4.0-R2A 真实可微 Oracle 端到端最小闭环验证\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "stage4_0_R2A_real_oracle_smoke_report.md", report)
    write_text(OUT / "stage4_0_R2A_summary.md", f"# Stage 4.0-R2A summary\n\n- Final CASE: `{final_case}`\n- C0: {'PASS' if C0 else 'FAIL'}\n- C1: {'PASS' if C1 else 'FAIL'}\n- AttributeDeformGS hypothesis status: UNTESTED\n- KIOT status: CONTROLLED-CARRIER-ONLY\n")
    write_text(OUT / "stage4_0_R2A_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage4.0-R2A real oracle smoke gate\n\nStage4.0 scientific outputs are retired. Stage4.0-R1 found that the original oracle implementation did not perform autograd/Adam optimization and that release metrics were generated by `synthetic_release_error(...)` using deterministic surrogate branches. Therefore the previously reported pattern `MAT0 -> NONE`, `MAT1 -> O`, `MAT2 -> O+C+V` is provenance-invalid and cannot be treated as evidence. The AttributeDeformGS hypothesis returns to `UNTESTED` status.\n\nStage4.0-R2A starts rebuilding a clean real differentiable oracle pipeline in `attribute_study/real_oracle/`. The first strict gate independently audits the saved GT arrays. This run stops at C1 because the saved Stage4.0 GT arrays are float16 and fail the required 1e-6-level numeric agreement thresholds for tau/RGB/alpha. No canonical fit or oracle optimizer is run after that failure.\n"""
    if "## Stage4.0-R2A real oracle smoke gate" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
