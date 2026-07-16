import argparse
import csv
from pathlib import Path

from stage0r_utils import OUT_DEBUG, ensure_dirs, find_checkpoint, ply_vertex_count, sha256_file, write_md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    args = parser.parse_args()
    ensure_dirs()
    rows = []
    for it in args.iterations:
        ply = Path(args.model_dir) / f"point_cloud/iteration_{it}/point_cloud.ply"
        ckpt = find_checkpoint(args.model_dir, it)
        rows.append({
            "iteration": it,
            "ply_exists": ply.exists(),
            "ply_sha256": sha256_file(ply),
            "gaussian_count": ply_vertex_count(ply),
            "checkpoint_exists": bool(ckpt),
            "checkpoint_path": str(ckpt) if ckpt else "",
            "checkpoint_sha256": sha256_file(ckpt) if ckpt else None,
        })
    csv_path = OUT_DEBUG / "trajectory_artifact_audit.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Trajectory Artifact Audit", "", "| Iteration | PLY exists | PLY SHA256 | Gaussian count | Checkpoint exists | Checkpoint SHA256 |", "|---:|---|---|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['iteration']} | `{r['ply_exists']}` | `{r['ply_sha256']}` | `{r['gaussian_count']}` | `{r['checkpoint_exists']}` | `{r['checkpoint_sha256']}` |")
    write_md(OUT_DEBUG / "trajectory_artifact_audit_report.md", lines)


if __name__ == "__main__":
    main()
