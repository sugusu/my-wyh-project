import argparse
import sys
from pathlib import Path

from stage0r_utils import OUT_DEBUG, ensure_dirs, extract_last_iteration_and_loss, log_error_counts, read_text, write_json, write_md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    ensure_dirs()
    model_dir = Path(args.model_dir)
    log_text = read_text(args.log) if Path(args.log).exists() else ""
    last_iter, last_loss = extract_last_iteration_and_loss(log_text)
    counts = log_error_counts(log_text)
    ply_30000 = model_dir / "point_cloud/iteration_30000/point_cloud.ply"
    cfg = model_dir / "cfg_args"
    checks = {
        "log_reaches_iteration_30000": bool(last_iter and last_iter >= 30000),
        "no_unhandled_traceback": counts["traceback_count"] == 0,
        "no_cuda_oom": counts["cuda_oom_count"] == 0,
        "no_nan_loss": counts["nan_count"] == 0,
        "no_inf_loss": counts["inf_count"] == 0,
        "iteration_30000_ply_exists": ply_30000.exists(),
        "cfg_args_exists": cfg.exists(),
    }
    if all(checks.values()):
        conclusion = "TRAINING_COMPLETE_VALID"
    elif counts["traceback_count"] or counts["cuda_oom_count"] or counts["nan_count"] or counts["inf_count"]:
        conclusion = "TRAINING_FAILED"
    else:
        conclusion = "TRAINING_INCOMPLETE"
    data = {"conclusion": conclusion, "last_logged_iteration": last_iter, "last_finite_loss": last_loss, "checks": checks, "counts": counts}
    write_json(OUT_DEBUG / "official_training_completion.json", data)
    lines = ["# Official Training Completion Audit", "", f"Conclusion: `{conclusion}`", "", "| Check | Pass |", "|---|---|"]
    lines += [f"| {k} | `{v}` |" for k, v in checks.items()]
    lines += ["", f"- Last logged iteration: `{last_iter}`", f"- Last finite loss: `{last_loss}`"]
    write_md(OUT_DEBUG / "official_training_completion_report.md", lines)
    print(conclusion)
    if conclusion != "TRAINING_COMPLETE_VALID":
        sys.exit(2)


if __name__ == "__main__":
    main()
