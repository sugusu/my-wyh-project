import argparse
import json
import shlex
from pathlib import Path

from stage0r_utils import OUT_STAGE0RC


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
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "cd /data/wyh/repos/TSGS",
        "",
        "export CUDA_VISIBLE_DEVICES=2",
        "export PYTHONPATH=/data/wyh/repos/TSGS/pytorch3d_stub:/data/wyh/repos/TSGS:/data/wyh/RecycleGS:${PYTHONPATH:-}",
        "",
        command,
        "",
    ]
    out = Path(args.out)
    out.write_text("\n".join(lines))
    out.chmod(0o755)
    (OUT_STAGE0RC / "materialized_train_command.txt").write_text(command + "\n")
    print(command)


if __name__ == "__main__":
    main()
