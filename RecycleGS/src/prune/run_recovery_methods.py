#!/usr/bin/env python3
"""Run recovery training for all methods across both scenes."""
import argparse, json, os, subprocess, sys, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

SCENES_CONFIGS = {
    'scene_01': '/data/wyh/RecycleGS/configs/stage1/reliability_scene01.yaml',
    'scene_03': '/data/wyh/RecycleGS/configs/stage1/reliability_scene03.yaml',
}

RECOVERY_CONFIG = '/data/wyh/RecycleGS/configs/stage2/recovery_500_locked.yaml'
TRAIN_SCRIPT = '/data/wyh/RecycleGS/src/prune/train_pruned_recovery.py'

METHODS = ['schedule_control', 'random', 'low_opacity', 'low_contribution', 'mask_risk']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--override-steps', type=int, default=None,
                        help='Override recovery steps (for smoke test)')
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene to run (scene_01 or scene_03), default: both')
    args = parser.parse_args()

    with open(RECOVERY_CONFIG) as f:
        recovery_cfg = yaml.safe_load(f)

    scenes_to_run = [args.scene] if args.scene else list(SCENES_CONFIGS.keys())

    for scene_name in scenes_to_run:
        scene_config = SCENES_CONFIGS[scene_name]
        with open(scene_config) as f:
            scene_cfg = yaml.safe_load(f)

        for method in METHODS:
            output_dir = f'/data/wyh/RecycleGS/outputs/recovery/{scene_name}/{method}'

            cmd = [
                sys.executable, TRAIN_SCRIPT,
                '--scene-config', scene_config,
                '--recovery-config', RECOVERY_CONFIG,
                '--method', method,
                '--output-dir', output_dir,
                '--seed', str(recovery_cfg.get('seed', 0)),
            ]

            if method != 'schedule_control':
                removed_indices_pattern = recovery_cfg['methods'][method]['removed_indices']
                removed_indices_path = removed_indices_pattern.format(scene_name=scene_name)
                if os.path.exists(removed_indices_path):
                    cmd.extend(['--removed-indices', removed_indices_path])
                else:
                    print(f"  WARNING: Removed indices not found for {scene_name}/{method}: "
                          f"{removed_indices_path}")
                    print(f"  Skipping...")
                    continue

            if args.override_steps is not None:
                cmd.extend(['--override-steps', str(args.override_steps)])

            cmd_str = ' '.join(cmd)
            print(f"\n=== Running {scene_name}/{method} ===")
            print(f"  Output: {output_dir}")
            print(f"  Command: {cmd_str}")

            if not args.dry_run:
                os.makedirs(output_dir, exist_ok=True)
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
                if result.returncode != 0:
                    print(f"  ERROR (rc={result.returncode}): {result.stderr}")
                else:
                    print(f"  Completed successfully")

    print(f"\nAll recovery runs complete.")

if __name__ == '__main__':
    main()
