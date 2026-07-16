import argparse
import csv
import json
from pathlib import Path

from stage0r_utils import OUT_DEBUG, OUT_STAGE0RC, ensure_dirs, parse_cfg_args, write_md


BASE_LRS = {
    "xyz": "position scheduler",
    "knn_f": 0.01,
    "f_dc": "feature_lr",
    "f_rest": "feature_lr / 20",
    "opacity": "opacity_lr",
    "transparency": "opacity_lr",
    "scaling": "scaling_lr",
    "rotation": "rotation_lr",
    "f_asg": "feature_lr",
}


def nofix_list(cfg):
    keep = ["transparency", "f_dc", "f_rest", "f_asg"]
    if cfg.get("nofix_position"):
        keep.append("xyz")
    if cfg.get("nofix_opacity"):
        keep.append("opacity")
    if cfg.get("nofix_scaling"):
        keep.append("scaling")
    if cfg.get("nofix_rotation"):
        keep.append("rotation")
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--provenance", default="/data/wyh/RecycleGS/outputs/debug/stage0rc/training_config_provenance.json")
    args = parser.parse_args()
    ensure_dirs()
    if Path(args.provenance).exists():
        prov = json.loads(Path(args.provenance).read_text())
        cfg = {r["field"]: r["resolved_value"] for r in prov.get("rows", [])}
    else:
        cfg = parse_cfg_args(Path(args.model_dir) / "cfg_args")
    freeze_iter = int(cfg.get("delight_iterations", 15000))
    keep = nofix_list(cfg)
    groups = ["xyz", "knn_f", "f_dc", "f_rest", "opacity", "transparency", "scaling", "rotation", "f_asg"]
    rows = []
    for it in [14999, 15000, 15001, 20000]:
        for name in groups:
            frozen = (not cfg.get("nofix_param", False)) and it > freeze_iter and name not in keep
            rows.append({
                "iteration": it,
                "name": name,
                "lr": 0.0 if frozen else BASE_LRS[name],
                "trainable": not frozen,
                "freeze_reason": "not in nofix_param_list after freeze_iter" if frozen else "",
                "nofix_flag": {
                    "xyz": "nofix_position",
                    "opacity": "nofix_opacity",
                    "scaling": "nofix_scaling",
                    "rotation": "nofix_rotation",
                }.get(name, "default-kept" if name in ["transparency", "f_dc", "f_rest", "f_asg"] else ""),
            })
    out_dir = OUT_STAGE0RC if Path(args.provenance).exists() else OUT_DEBUG
    with (out_dir / "official_stage_transition.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    kept_15001 = [r["name"] for r in rows if r["iteration"] == 15001 and r["trainable"]]
    frozen_15001 = [r["name"] for r in rows if r["iteration"] == 15001 and not r["trainable"]]
    write_md(out_dir / "official_stage_transition_report.md", [
        "# Official Stage Transition Audit",
        "",
        f"- freeze_iter: `{freeze_iter}`",
        f"- nofix_param: `{cfg.get('nofix_param', '<missing>')}`",
        f"- Derived from training config provenance plus `/data/wyh/repos/TSGS/scene/gaussian_model.py::selective_learning_rate_control`.",
        "",
        f"Parameters continuing at 15001: `{', '.join(kept_15001)}`",
        "",
        f"Parameters frozen at 15001: `{', '.join(frozen_15001)}`",
        "",
        f"Full table: `{out_dir / 'official_stage_transition.csv'}`",
    ])


if __name__ == "__main__":
    main()
