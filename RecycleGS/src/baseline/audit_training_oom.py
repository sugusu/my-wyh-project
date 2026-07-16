import argparse
import re
from pathlib import Path

from stage0r_utils import OUT_STAGE0RC, ensure_dirs, extract_last_iteration_and_loss, read_text, write_json, write_md


def classify(traceback):
    tb = traceback.lower()
    if "densify_and_prune" in tb or "densification_postfix" in tb or "cat_tensors_to_optimizer" in tb:
        return "DENSIFICATION_OOM"
    if "render(" in tb or "gaussian_renderer" in tb:
        return "RENDER_OOM"
    if "multi_view" in tb or "ncc" in tb:
        return "MULTIVIEW_NCC_OOM"
    if "asg" in tb:
        return "ASG_OOM"
    if "evaluating" in tb or "validation" in tb:
        return "VALIDATION_OOM"
    if "save" in tb or "checkpoint" in tb:
        return "SAVE_CHECKPOINT_OOM"
    return "UNKNOWN_OOM"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    ensure_dirs()
    texts = []
    for path in [args.log, Path(args.model_dir) / "training_log.log"]:
        if Path(path).exists():
            texts.append(read_text(path))
    text = "\n".join(texts)
    last_iter, _ = extract_last_iteration_and_loss(text)
    points = None
    for m in re.finditer(r"Iteration\s+\d+:.*?Points=(\d+)", text):
        points = int(m.group(1))
    trace = ""
    idx = text.rfind("Traceback (most recent call last):")
    if idx >= 0:
        lines = []
        for line in text[idx:].splitlines():
            lines.append(line)
            if "OutOfMemoryError:" in line or line.startswith("RuntimeError: CUDA out of memory"):
                break
        trace = "\n".join(lines).strip()
    exc_type = None
    exc_msg = None
    if trace:
        last = trace.splitlines()[-1].strip()
        if ":" in last:
            exc_type, exc_msg = last.split(":", 1)
            exc_type = exc_type.strip()
            exc_msg = exc_msg.strip()
    mem = {}
    if exc_msg:
        for name, pat in {
            "cuda_allocated_if_logged": r"allocated memory ([0-9.]+)\s*GiB",
            "cuda_reserved_if_logged": r"reserved by PyTorch.*?([0-9.]+)\s*GiB",
        }.items():
            mm = re.search(pat, exc_msg)
            mem[name] = float(mm.group(1)) if mm else None
    data = {
        "exception_type": exc_type,
        "exception_message": exc_msg,
        "last_iteration": last_iter,
        "last_logged_points": points,
        "last_operation": classify(trace),
        **mem,
        "traceback": trace,
    }
    write_json(OUT_STAGE0RC / "oom_root_cause.json", data)
    write_md(OUT_STAGE0RC / "oom_root_cause_report.md", [
        "# OOM Root Cause Audit",
        "",
        f"- exception_type: `{exc_type}`",
        f"- exception_message: `{exc_msg}`",
        f"- last_iteration: `{last_iter}`",
        f"- last_logged_points: `{points}`",
        f"- last_operation: `{data['last_operation']}`",
        f"- cuda_allocated_if_logged: `{mem.get('cuda_allocated_if_logged')}`",
        f"- cuda_reserved_if_logged: `{mem.get('cuda_reserved_if_logged')}`",
        "",
        "## Traceback",
        "",
        "```text",
        trace,
        "```",
    ])


if __name__ == "__main__":
    main()
