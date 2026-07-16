import argparse
from pathlib import Path

from stage0r_utils import (
    OUT_DEBUG,
    ensure_dirs,
    parse_cfg_args,
    parse_command_line_from_logs,
    parse_saved_defaults,
    parse_typed_value,
    command_has_flag,
    command_value,
    write_json,
    write_md,
)


MODEL_EXPECTED = {
    "sh_degree": 3,
    "asg_degree": 24,
    "resolution": 2,
    "white_background": False,
    "eval": True,
    "delight": True,
    "normal": True,
    "normal_folder": "normals",
    "mask_background": True,
    "use_transparencies_map": True,
}
OPT_EXPECTED = {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--log", default="outputs/debug/stage0r/training_30k.log")
    args = parser.parse_args()
    ensure_dirs()
    model_dir = Path(args.model_dir)
    cfg_path = Path(args.model_dir) / "cfg_args"
    cfg = parse_cfg_args(cfg_path)
    command = parse_command_line_from_logs(model_dir / "training_log.log", args.log)
    defaults = parse_saved_defaults(model_dir / "arguments/__init__.py")
    rows = []
    mismatch = False
    for key, expected in MODEL_EXPECTED.items():
        actual = cfg.get(key, "<missing>")
        if actual == "<missing>":
            status = "UNVERIFIABLE_FROM_CFG_ARGS"
            ok = False
        else:
            ok = actual == expected
            status = "MATCH" if ok else "MISMATCH"
        mismatch = mismatch or not ok
        rows.append({"field": key, "actual": actual, "expected": expected, "source": "CFG_ARGS", "status": status})
    for key, expected in OPT_EXPECTED.items():
        flag = "--" + key
        default = defaults.get(key)
        if isinstance(expected, bool):
            if command_has_flag(command, flag):
                actual = True
                source = "COMMAND_LINE"
            else:
                actual = default if key in defaults else "<unresolved>"
                source = "SAVED_SOURCE_DEFAULT" if key in defaults else "UNRESOLVED"
        else:
            raw = command_value(command, flag)
            if raw is not None:
                actual = parse_typed_value(raw, default)
                source = "COMMAND_LINE"
            elif key in defaults:
                actual = default
                source = "SAVED_SOURCE_DEFAULT"
            else:
                actual = "<unresolved>"
                source = "UNRESOLVED"
        ok = actual == expected
        mismatch = mismatch or not ok
        rows.append({"field": key, "actual": actual, "expected": expected, "source": source, "status": "MATCH" if ok else "MISMATCH"})
    conclusion = "OFFICIAL_CFG_MISMATCH" if mismatch else "OFFICIAL_CFG_MATCH"
    data = {"conclusion": conclusion, "cfg_args": str(cfg_path), "command": command, "core": rows}
    write_json(OUT_DEBUG / "official_cfg_audit.json", data)
    lines = ["# Official Configuration Audit", "", f"Conclusion: `{conclusion}`", "", "Missing OptimizationParams in `cfg_args` are not treated as mismatches; command line and saved source defaults are used instead.", "", "| Field | Actual | Expected | Source | Status |", "|---|---:|---:|---|---|"]
    lines += [f"| `{r['field']}` | `{r['actual']}` | `{r['expected']}` | `{r['source']}` | `{r['status']}` |" for r in rows]
    write_md(OUT_DEBUG / "official_cfg_audit_report.md", lines)
    print(conclusion)


if __name__ == "__main__":
    main()
