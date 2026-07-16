import json
from pathlib import Path

from stage0r_utils import OUT_STAGE0RC, write_md


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    comp = load(OUT_STAGE0RC / "failed_vs_exact_wrapper.json")
    oom = load(OUT_STAGE0RC / "oom_root_cause.json")
    growth = load(OUT_STAGE0RC / "gaussian_growth_summary.json")
    sparse = load(OUT_STAGE0RC / "sparse_layout_audit.json")
    deleted = load(OUT_STAGE0RC / "deleted_evidence.json")
    if comp and comp.get("FAILED_RUN_NOT_EXACT_OFFICIAL_WRAPPER"):
        state = "FAILED_RUN_CONFIG_INEXACT"
    else:
        state = "OFFICIAL_BASELINE_REPRODUCTION_FAILED"
    write_md(OUT_STAGE0RC / "stage0rc_report.md", [
        "# Stage 0R-C Report",
        "",
        f"Final state: `{state}`",
        "",
        "## Summary",
        "",
        f"- Deleted original Stage 0R evidence status: `{deleted.get('status') if deleted else 'missing'}`",
        f"- Fallback comparison model: `{deleted.get('fallback_model') if deleted else 'missing'}`",
        f"- Failed run exact wrapper mismatch: `{comp.get('FAILED_RUN_NOT_EXACT_OFFICIAL_WRAPPER') if comp else 'missing'}`",
        f"- Missing/different fields: `{', '.join(comp.get('missing_or_different_fields', [])) if comp else 'missing'}`",
        f"- OOM operation: `{oom.get('last_operation') if oom else 'missing'}`",
        f"- Last iteration: `{oom.get('last_iteration') if oom else 'missing'}`",
        f"- Last logged points: `{oom.get('last_logged_points') if oom else 'missing'}`",
        f"- POINT_COUNT_EXPLOSION: `{growth.get('POINT_COUNT_EXPLOSION') if growth else 'missing'}`",
        f"- Sparse layout OK: `{sparse.get('ok') if sparse else 'missing'}`",
        "",
        "Formal Gate 1, mask-risk, prune, recovery, reseed, and Stage 3 remain forbidden.",
    ])
    print(state)


if __name__ == "__main__":
    main()
