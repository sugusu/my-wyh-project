import argparse
import csv
from pathlib import Path

from stage0r_utils import (
    OUT_STAGE0RC,
    command_has_flag,
    command_value,
    ensure_dirs,
    parse_cfg_args,
    parse_command_line_from_logs,
    parse_saved_defaults,
    parse_typed_value,
    write_json,
    write_md,
)


FIELDS = {
    "sh_degree": 3,
    "asg_degree": 24,
    "source_path": None,
    "model_path": None,
    "images": "images",
    "resolution": 2,
    "white_background": False,
    "eval": True,
    "delight": True,
    "normal": True,
    "normal_folder": "normals",
    "mask_background": True,
    "use_transparencies_map": True,
    "iterations": 30000,
    "use_asg": True,
    "sd_normal_until_iter": 30000,
    "delight_iterations": 15000,
    "normal_cos_threshold_iter": 3000,
    "ncc_loss_from_iter": 7000,
    "nofix_position": True,
    "nofix_opacity": False,
    "nofix_param": False,
    "nofix_scaling": True,
    "nofix_rotation": True,
    "seed": 42,
}


def resolve(field, expected, cfg, command, defaults):
    if field in cfg:
        return cfg[field], "CFG_ARGS", "cfg_args", "HIGH"
    flag = "--" + field
    default = defaults.get(field)
    if isinstance(expected, bool):
        if command_has_flag(command, flag):
            return True, "COMMAND_LINE", flag, "HIGH"
        if field in defaults:
            return default, "SAVED_SOURCE_DEFAULT", "arguments/__init__.py", "MEDIUM"
    else:
        raw = command_value(command, flag)
        if raw is not None:
            return parse_typed_value(raw, expected if expected is not None else default), "COMMAND_LINE", flag, "HIGH"
        if field in defaults:
            return default, "SAVED_SOURCE_DEFAULT", "arguments/__init__.py", "MEDIUM"
    return None, "UNRESOLVED", "", "LOW"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    ensure_dirs()
    model_dir = Path(args.model_dir)
    cfg = parse_cfg_args(model_dir / "cfg_args")
    command = parse_command_line_from_logs(model_dir / "training_log.log", args.log)
    defaults = parse_saved_defaults(model_dir / "arguments/__init__.py")
    rows = []
    for field, expected in FIELDS.items():
        value, source, detail, confidence = resolve(field, expected, cfg, command, defaults)
        if expected is None:
            status = "RESOLVED" if source != "UNRESOLVED" else "UNRESOLVED"
        elif source == "UNRESOLVED":
            status = "UNRESOLVED"
        else:
            status = "MATCH" if value == expected else "MISMATCH"
        rows.append({
            "field": field,
            "resolved_value": value,
            "expected_value": expected,
            "source": source,
            "source_detail": detail,
            "confidence": confidence,
            "status": status,
        })
    core = [r for r in rows if r["expected_value"] is not None]
    conclusion = "OFFICIAL_CONFIG_MATCH" if all(r["status"] == "MATCH" for r in core) else "OFFICIAL_CONFIG_MISMATCH"
    with (OUT_STAGE0RC / "training_config_provenance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUT_STAGE0RC / "training_config_provenance.json", {
        "conclusion": conclusion,
        "command": command,
        "rows": rows,
    })
    lines = ["# Training Configuration Provenance", "", f"Conclusion: `{conclusion}`", "", f"Command: `{command}`", "", "| Field | Resolved | Expected | Source | Confidence | Status |", "|---|---:|---:|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['field']}` | `{r['resolved_value']}` | `{r['expected_value']}` | `{r['source']}` | `{r['confidence']}` | `{r['status']}` |")
    write_md(OUT_STAGE0RC / "training_config_provenance_report.md", lines)
    print(conclusion)


if __name__ == "__main__":
    main()
