import argparse
import json
import shlex
from pathlib import Path

from stage0r_utils import OUT_STAGE0RC, write_md


PRIMARY_TSGS_ROOT = Path("/data/wyh/repos/TSGS")
FALLBACK_TSGS_ROOT = Path("/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k_v4")


def execution_root():
    if (PRIMARY_TSGS_ROOT / "train.py").exists():
        return PRIMARY_TSGS_ROOT, "PRIMARY_TSGS_ROOT"
    if (FALLBACK_TSGS_ROOT / "train.py").exists():
        return FALLBACK_TSGS_ROOT, "SAVED_RUN_SOURCE_FALLBACK"
    raise SystemExit("No runnable TSGS train.py found in primary or fallback source roots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/data/wyh/RecycleGS/outputs/debug/stage0rc/official_wrapper_resolved_config.json")
    parser.add_argument("--out", default="/data/wyh/RecycleGS/scripts/stage0rc_train_exact_scene01.sh")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    command = cfg["train_command"]
    expected = {
        "mask_background": True,
        "use_transparencies_map": True,
        "iterations": 30000,
        "resolution": 2,
        "seed": 42,
    }
    mismatches = {k: {"actual": cfg.get(k), "expected": v} for k, v in expected.items() if cfg.get(k) != v}
    if mismatches:
        raise SystemExit(f"Resolved wrapper config mismatch: {mismatches}")
    root, root_kind = execution_root()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(root))}",
        "",
        "export CUDA_VISIBLE_DEVICES=2",
        f"export PYTHONPATH={shlex.quote(str(root / 'pytorch3d_stub'))}:{shlex.quote(str(root))}:/data/wyh/RecycleGS:${{PYTHONPATH:-}}",
        "",
        command,
        "",
    ]
    out = Path(args.out)
    out.write_text("\n".join(lines))
    out.chmod(0o755)
    (OUT_STAGE0RC / "materialized_train_command.txt").write_text(command + "\n")
    write_md(OUT_STAGE0RC / "stage0rc_execution_source_report.md", [
        "# Stage 0R-C Execution Source",
        "",
        f"- primary_tsgs_root: `{PRIMARY_TSGS_ROOT}`",
        f"- primary_train_py_exists: `{(PRIMARY_TSGS_ROOT / 'train.py').exists()}`",
        f"- fallback_saved_source_root: `{FALLBACK_TSGS_ROOT}`",
        f"- fallback_train_py_exists: `{(FALLBACK_TSGS_ROOT / 'train.py').exists()}`",
        f"- selected_execution_root: `{root}`",
        f"- selected_execution_root_kind: `{root_kind}`",
        "",
        "The exact wrapper command is unchanged. This selection only determines which available copy of the TSGS training code executes it.",
    ])
    print(command)


if __name__ == "__main__":
    main()
