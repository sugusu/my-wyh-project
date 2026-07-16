import json
from pathlib import Path

from stage0r_utils import OUT_BASELINE, OUT_DEBUG, ensure_dirs, write_md


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    ensure_dirs()
    completion = load_json(OUT_DEBUG / "official_training_completion.json")
    cfg = load_json(OUT_DEBUG / "official_cfg_audit.json")
    trajectory = load_json(OUT_BASELINE / "official_scene01_training_trajectory.json")
    geometry_report = OUT_BASELINE / "official_scene01_geometry_report.md"
    training_ok = completion and completion.get("conclusion") == "TRAINING_COMPLETE_VALID"
    cfg_ok = cfg and cfg.get("conclusion") == "OFFICIAL_CFG_MATCH"
    render_ok = trajectory and trajectory.get("complete") is True
    geometry_ok = geometry_report.exists() and "FORMAL_GEOMETRY_PENDING" not in geometry_report.read_text(errors="replace")
    if training_ok and cfg_ok and render_ok and geometry_ok:
        state = "OFFICIAL_BASELINE_READY"
    elif training_ok and cfg_ok and render_ok and not geometry_ok:
        state = "BASELINE_RENDER_READY_GEOMETRY_PENDING"
    elif completion and completion.get("conclusion") == "TRAINING_INCOMPLETE":
        state = "TRAINING_IN_PROGRESS"
    else:
        state = "OFFICIAL_BASELINE_REPRODUCTION_FAILED"
    lines = [
        "# Stage 0R Rebase Report",
        "",
        f"Final state: `{state}`",
        "",
        "## Old Baseline Correction",
        "",
        "The old baseline was a simplified 15k TSGS pilot without the full TransLab-specific training configuration.",
        "",
        "The old pilot did not enable nofix_position, nofix_scaling, or nofix_rotation.",
        "",
        "Under TSGS selective post-15k control, the absence of these nofix flags would allow position, scaling, and rotation to be fixed after the stage transition.",
        "",
        "However, the old pilot terminated at iteration 15000 and therefore did not execute the full post-15k training stage.",
        "",
        "## Gate Policy",
        "",
        "Formal Gate 1, prune, recovery, and reseed remain forbidden unless Stage 0R reaches `OFFICIAL_BASELINE_READY`.",
        "",
        "## Inputs",
        "",
        f"- Training completion: `{completion.get('conclusion') if completion else 'missing'}`",
        f"- cfg audit: `{cfg.get('conclusion') if cfg else 'missing'}`",
        f"- render trajectory complete: `{render_ok}`",
        f"- formal geometry complete: `{geometry_ok}`",
    ]
    write_md(OUT_DEBUG / "stage0r_rebase_report.md", lines)
    print(state)


if __name__ == "__main__":
    main()
