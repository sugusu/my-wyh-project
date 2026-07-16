import argparse
import os
import subprocess
from pathlib import Path

from stage0r_utils import OUT_DEBUG, ensure_dirs, extract_last_iteration_and_loss, log_error_counts, ply_vertex_count, read_text, write_json, write_md


def latest_iter_dir(model_dir, prefix):
    root = Path(model_dir) / prefix
    vals = []
    if root.exists():
        for p in root.glob("iteration_*"):
            try:
                vals.append(int(p.name.split("_")[-1]))
            except ValueError:
                pass
    return max(vals) if vals else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, default=1987496)
    parser.add_argument("--model-dir", default="/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k")
    parser.add_argument("--log", default="outputs/debug/stage0r/training_30k.log")
    args = parser.parse_args()
    ensure_dirs()

    proc = subprocess.run(["ps", "-p", str(args.pid), "-o", "pid=,etime=,%cpu=,%mem=,stat=,cmd="], text=True, capture_output=True)
    ps_line = proc.stdout.strip()
    exists = proc.returncode == 0 and bool(ps_line)
    log_text = read_text(args.log) if Path(args.log).exists() else ""
    last_iter, last_loss = extract_last_iteration_and_loss(log_text)
    counts = log_error_counts(log_text)
    latest_ply = latest_iter_dir(args.model_dir, "point_cloud")
    latest_checkpoint = None
    for p in Path(args.model_dir).glob("chkpnt*.pth"):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            latest_checkpoint = max(latest_checkpoint or 0, int(digits))

    data = {
        "pid": args.pid,
        "pid_exists": exists,
        "process_state": ps_line,
        "last_logged_iteration": last_iter,
        "last_finite_loss": last_loss,
        "latest_saved_ply_iteration": latest_ply,
        "latest_saved_checkpoint_iteration": latest_checkpoint,
        **counts,
    }
    write_json(OUT_DEBUG / "training_runtime_status.json", data)
    write_md(OUT_DEBUG / "training_runtime_status_report.md", [
        "# Training Runtime Status",
        "",
        f"- PID exists: `{exists}`",
        f"- Process state: `{ps_line or 'not running'}`",
        f"- Last logged iteration: `{last_iter}`",
        f"- Last finite loss: `{last_loss}`",
        f"- Latest saved PLY iteration: `{latest_ply}`",
        f"- Latest saved checkpoint iteration: `{latest_checkpoint}`",
        f"- Error count: `{counts['error_count']}`",
        f"- NaN count: `{counts['nan_count']}`",
        f"- Inf count: `{counts['inf_count']}`",
    ])


if __name__ == "__main__":
    main()
