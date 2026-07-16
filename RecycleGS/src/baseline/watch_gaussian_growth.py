import argparse
import csv
import re
import time
from pathlib import Path


def latest_rows(log_path):
    rows = []
    text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
    for m in re.finditer(r"Iteration\s+(\d+):.*?Points=(\d+)", text):
        it = int(m.group(1))
        n = int(m.group(2))
        if n > 3_000_000:
            warning = "WARNING_EXTREME_POINT_COUNT"
        elif n > 2_000_000:
            warning = "WARNING_VERY_HIGH_POINT_COUNT"
        elif n > 1_500_000:
            warning = "WARNING_HIGH_POINT_COUNT"
        else:
            warning = ""
        rows.append({"iteration": it, "gaussian_count": n, "warning": warning})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="/data/wyh/RecycleGS/outputs/debug/stage0rc/training_exact_scene01.log")
    parser.add_argument("--out", default="/data/wyh/RecycleGS/outputs/debug/stage0rc/gaussian_growth_live.csv")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        rows = latest_rows(args.log)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "gaussian_count", "warning"])
            writer.writeheader()
            writer.writerows(rows)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
