import argparse
import csv
import json
import sys
from pathlib import Path

from stage0r_utils import OUT_BASELINE, ensure_dirs, ply_vertex_count, sha256_file, write_json, write_md


def load_result(model_dir, iteration):
    path = Path(model_dir) / f"results/test/result_{iteration}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pick_metric(data, *names):
    if not data:
        return None
    for name in names:
        if name in data:
            return data[name]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    args = parser.parse_args()
    ensure_dirs()
    rows = []
    complete = True
    for it in args.iterations:
        data = load_result(args.model_dir, it)
        ply = Path(args.model_dir) / f"point_cloud/iteration_{it}/point_cloud.ply"
        if not data or not ply.exists():
            complete = False
        rows.append({
            "iteration": it,
            "psnr": pick_metric(data, "PSNR", "psnr", "psnr_mean"),
            "ssim": pick_metric(data, "SSIM", "ssim", "ssim_mean"),
            "lpips": pick_metric(data, "LPIPS", "lpips", "lpips_mean"),
            "gaussian_count": ply_vertex_count(ply),
            "ply_path": str(ply),
            "ply_sha256": sha256_file(ply),
        })
    csv_path = OUT_BASELINE / "official_scene01_training_trajectory.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUT_BASELINE / "official_scene01_training_trajectory.json", {"complete": complete, "rows": rows})
    by_it = {r["iteration"]: r for r in rows}
    def delta(a, b):
        if by_it.get(a, {}).get("psnr") is None or by_it.get(b, {}).get("psnr") is None:
            return "N/A"
        return float(by_it[b]["psnr"]) - float(by_it[a]["psnr"])
    lines = ["# Official Scene 01 Training Trajectory", "", f"Evaluation complete: `{complete}`", "", "| Iteration | PSNR | SSIM | LPIPS | N Gaussians |", "|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['iteration']} | {r['psnr']} | {r['ssim']} | {r['lpips']} | {r['gaussian_count']} |")
    lines += ["", f"- 15k -> 20k PSNR change: `{delta(15000, 20000)}`", f"- 20k -> 30k PSNR change: `{delta(20000, 30000)}`", "- 15k intervention point: `PENDING - missing complete official 15k/20k/30k trajectory`" if not complete else "- 15k intervention point: `available for formal review`"]
    write_md(OUT_BASELINE / "official_scene01_training_trajectory_report.md", lines)
    if not complete:
        sys.exit(2)


if __name__ == "__main__":
    main()
