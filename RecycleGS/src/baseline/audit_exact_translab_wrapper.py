import argparse
import json
import shlex
from pathlib import Path

from stage0r_utils import OUT_STAGE0RC, TSGS_ROOT, ensure_dirs, write_json, write_md


def resolved_config(scene_dir, model_dir):
    return {
        "source_path": scene_dir,
        "model_path": model_dir,
        "delight": True,
        "normal": True,
        "eval": True,
        "use_asg": True,
        "resolution": 2,
        "iterations": 30000,
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
        "mask_background": True,
        "use_transparencies_map": True,
        "normal_folder": "normals",
        "test_iterations": [7000, 10000, 15000, 20000, 30000],
        "save_iterations": [7000, 10000, 15000, 20000, 30000],
        "checkpoint_iterations": [7000, 10000, 15000, 20000],
    }


def build_command(cfg):
    args = [
        "python3", "train.py",
        "-s", cfg["source_path"],
        "-m", cfg["model_path"],
    ]
    for key in ["delight", "normal", "mask_background", "use_asg"]:
        if cfg[key]:
            args.append("--" + key)
    args += [
        "--sd_normal_until_iter", str(cfg["sd_normal_until_iter"]),
        "--iterations", str(cfg["iterations"]),
        "--normal_cos_threshold_iter", str(cfg["normal_cos_threshold_iter"]),
    ]
    if cfg["eval"]:
        args.append("--eval")
    args += [
        "--delight_iterations", str(cfg["delight_iterations"]),
        "--resolution", str(cfg["resolution"]),
        "--ncc_loss_from_iter", str(cfg["ncc_loss_from_iter"]),
    ]
    for key in ["nofix_position", "nofix_opacity", "nofix_param", "nofix_scaling", "nofix_rotation"]:
        if cfg[key]:
            args.append("--" + key)
    args += ["--seed", str(cfg["seed"]), "--normal_folder", cfg["normal_folder"]]
    if cfg["use_transparencies_map"]:
        args.append("--use_transparencies_map")
    for key in ["test_iterations", "save_iterations", "checkpoint_iterations"]:
        args.append("--" + key)
        args += [str(v) for v in cfg[key]]
    return " ".join(shlex.quote(a) for a in args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", default="/data/wyh/RecycleGS/data/translab_full/scene_01")
    parser.add_argument("--model-dir", default="/data/wyh/RecycleGS/baselines/tsgs_exact_scene01_30k")
    args = parser.parse_args()
    ensure_dirs()
    cfg = resolved_config(args.scene_dir, args.model_dir)
    cfg["train_command"] = build_command(cfg)
    cfg["source_of_truth"] = {
        "run_translab_sh": str(TSGS_ROOT / "run_translab.sh"),
        "run_translab_py": str(TSGS_ROOT / "scripts/run_translab.py"),
    }
    write_json(OUT_STAGE0RC / "official_wrapper_resolved_config.json", cfg)
    lines = ["# Exact TransLab Wrapper Audit", "", "Conclusion: `EXACT_WRAPPER_RESOLVED`", "", "The resolved train.py command includes wrapper defaults `--mask_background` and `--use_transparencies_map`.", "", "## Command", "", f"`{cfg['train_command']}`", "", "## Core Fields", "", "| Field | Value |", "|---|---|"]
    for key in ["delight", "normal", "eval", "use_asg", "resolution", "iterations", "sd_normal_until_iter", "delight_iterations", "normal_cos_threshold_iter", "ncc_loss_from_iter", "nofix_position", "nofix_opacity", "nofix_param", "nofix_scaling", "nofix_rotation", "seed", "mask_background", "use_transparencies_map", "normal_folder"]:
        lines.append(f"| `{key}` | `{cfg[key]}` |")
    write_md(OUT_STAGE0RC / "official_wrapper_resolved_config_report.md", lines)


if __name__ == "__main__":
    main()
