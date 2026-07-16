import argparse
import csv
import re
from pathlib import Path

from stage0r_utils import OUT_STAGE0RC, ensure_dirs, read_text, write_json, write_md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    ensure_dirs()
    text = read_text(Path(args.model_dir) / "training_log.log") if (Path(args.model_dir) / "training_log.log").exists() else ""
    if not text:
        text = read_text("/data/wyh/RecycleGS/outputs/debug/stage0r/training_30k.log")
    rows = []
    prev = None
    for m in re.finditer(r"Iteration\s+(\d+):.*?Points=(\d+)", text):
        it = int(m.group(1))
        n = int(m.group(2))
        delta = None if prev is None else n - prev[1]
        rate = None if prev is None or prev[1] == 0 else n / prev[1]
        rows.append({"iteration": it, "gaussian_count": n, "delta_count": delta, "growth_rate": rate})
        prev = (it, n)
    with (OUT_STAGE0RC / "gaussian_growth.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "gaussian_count", "delta_count", "growth_rate"])
        writer.writeheader()
        writer.writerows(rows)
    by = {r["iteration"]: r for r in rows}
    n7000 = by.get(7000, {}).get("gaussian_count")
    n10000 = by.get(10000, {}).get("gaussian_count")
    growth = (n10000 / n7000) if n7000 and n10000 else None
    max_delta = max((r["delta_count"] or 0 for r in rows), default=None)
    avg_added = None
    if n7000 and n10000:
        avg_added = (n10000 - n7000) / 30.0
    explosion = bool(growth and growth > 3)
    summary = {"N7000": n7000, "N10000": n10000, "growth_7000_to_10000": growth, "average_added_per_100_iterations": avg_added, "maximum_100_iteration_growth": max_delta, "POINT_COUNT_EXPLOSION": explosion}
    write_json(OUT_STAGE0RC / "gaussian_growth_summary.json", summary)
    lines = ["# Gaussian Growth Analysis", "", f"POINT_COUNT_EXPLOSION = `{str(explosion).lower()}`", "", "| Iteration | Gaussian count | Delta | Growth rate |", "|---:|---:|---:|---:|"]
    wanted = {500, 1000, 2000, 3000, 5000, 7000, 8000, 9000, 10000}
    if rows:
        wanted.add(rows[-1]["iteration"])
    for r in rows:
        if r["iteration"] in wanted:
            lines.append(f"| {r['iteration']} | {r['gaussian_count']} | {r['delta_count']} | {r['growth_rate']} |")
    lines += ["", f"- N(7000): `{n7000}`", f"- N(10000): `{n10000}`", f"- Growth 7000->10000: `{growth}`", f"- Average added per 100 iterations: `{avg_added}`", f"- Maximum 100-iteration growth: `{max_delta}`"]
    write_md(OUT_STAGE0RC / "gaussian_growth_report.md", lines)


if __name__ == "__main__":
    main()
