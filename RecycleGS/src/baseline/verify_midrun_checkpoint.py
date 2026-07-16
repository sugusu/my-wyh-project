import argparse
import sys
from pathlib import Path

from stage0r_utils import OUT_DEBUG, ensure_dirs, find_checkpoint, parse_cfg_args, sha256_file, write_md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--iteration", type=int, default=15000)
    args = parser.parse_args()
    ensure_dirs()
    model = Path(args.model_dir)
    cfg = parse_cfg_args(model / "cfg_args")
    ckpt = find_checkpoint(model, args.iteration)
    ply15 = model / f"point_cloud/iteration_{args.iteration}/point_cloud.ply"
    ply20 = model / "point_cloud/iteration_20000/point_cloud.ply"
    ply30 = model / "point_cloud/iteration_30000/point_cloud.ply"
    valid = cfg.get("iterations") == 30000 and bool(ckpt) and ply15.exists() and ply20.exists() and ply30.exists()
    lines = [
        "# Scene 01 Mid-run Checkpoint Verification",
        "",
        f"- model_root: `{model}`",
        f"- cfg_iterations: `{cfg.get('iterations')}`",
        f"- checkpoint_iteration: `{args.iteration}`",
        f"- checkpoint_path: `{ckpt}`",
        f"- checkpoint_sha256: `{sha256_file(ckpt) if ckpt else None}`",
        f"- ply_15000_path: `{ply15}`",
        f"- ply_15000_sha256: `{sha256_file(ply15)}`",
        f"- ply_20000_path: `{ply20}` exists=`{ply20.exists()}`",
        f"- ply_30000_path: `{ply30}` exists=`{ply30.exists()}`",
        f"- is_midrun: `{valid}`",
        "",
        f"VALID_15K_MIDRUN_CHECKPOINT = `{str(valid).lower()}`",
    ]
    write_md(OUT_DEBUG / "scene01_midrun_checkpoint_report.md", lines)
    print(f"VALID_15K_MIDRUN_CHECKPOINT={valid}")
    if not valid:
        sys.exit(2)


if __name__ == "__main__":
    main()
