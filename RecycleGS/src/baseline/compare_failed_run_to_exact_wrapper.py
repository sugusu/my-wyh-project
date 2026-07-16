import argparse
import json
from pathlib import Path

from stage0r_utils import OUT_STAGE0RC, command_has_flag, command_value, ensure_dirs, parse_command_line_from_logs, parse_saved_defaults, write_json, write_md


FIELDS = ["mask_background", "use_transparencies_map", "delight", "normal", "eval", "use_asg", "resolution", "iterations", "seed"]


def resolve_failed(command, defaults, field, exact=None):
    flag = "--" + field
    default = defaults.get(field)
    if isinstance(default, bool):
        if command_has_flag(command, flag):
            return True
        return default
    raw = command_value(command, flag)
    if raw is None:
        return default
    if isinstance(exact, int) and not isinstance(exact, bool):
        return int(raw)
    return int(raw) if isinstance(default, int) and not isinstance(default, bool) else raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--log", default="outputs/debug/stage0r/training_30k.log")
    args = parser.parse_args()
    ensure_dirs()
    model = Path(args.model_dir)
    wrapper = json.loads((OUT_STAGE0RC / "official_wrapper_resolved_config.json").read_text())
    command = parse_command_line_from_logs(model / "training_log.log", args.log)
    defaults = parse_saved_defaults(model / "arguments/__init__.py")
    rows = []
    for field in FIELDS:
        exact = wrapper.get(field)
        failed = resolve_failed(command, defaults, field, exact)
        rows.append({"field": field, "failed_run": failed, "exact_wrapper": exact, "match": failed == exact})
    inex = any(not r["match"] for r in rows)
    missing_flags = [r["field"] for r in rows if not r["match"]]
    data = {"FAILED_RUN_NOT_EXACT_OFFICIAL_WRAPPER": inex, "missing_or_different_fields": missing_flags, "rows": rows}
    write_json(OUT_STAGE0RC / "failed_vs_exact_wrapper.json", data)
    lines = ["# Failed Run vs Exact Wrapper", "", f"FAILED_RUN_NOT_EXACT_OFFICIAL_WRAPPER = `{str(inex).lower()}`", "", "| Field | Failed run | Exact wrapper | Match |", "|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| `{r['field']}` | `{r['failed_run']}` | `{r['exact_wrapper']}` | `{r['match']}` |")
    write_md(OUT_STAGE0RC / "failed_vs_exact_wrapper_report.md", lines)


if __name__ == "__main__":
    main()
