#!/usr/bin/env python3
"""Check if train_pruned_recovery.py already calls selective_learning_rate_control.
Print the status and whether the fix is needed."""
import os, sys

sys.path.insert(0, '/data/wyh/RecycleGS/src')

RECOVERY_SCRIPT = '/data/wyh/RecycleGS/src/prune/train_pruned_recovery.py'
TSGS_TRAIN = '/data/wyh/repos/TSGS/train.py'

def check_file_for_pattern(filepath, patterns):
    results = {}
    with open(filepath) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for pat in patterns:
            if pat in line:
                if pat not in results:
                    results[pat] = []
                results[pat].append((i + 1, line.strip()))
    return results

def main():
    print("=" * 60)
    print("VERIFY: selective_learning_rate_control in recovery training")
    print("=" * 60)

    rec_results = check_file_for_pattern(RECOVERY_SCRIPT, [
        'selective_learning_rate_control',
        'update_learning_rate',
    ])

    tsgs_results = check_file_for_pattern(TSGS_TRAIN, [
        'selective_learning_rate_control',
        'update_learning_rate',
    ])

    print(f"\nTSGS train.py ({TSGS_TRAIN}):")
    for pat, matches in tsgs_results.items():
        for line_no, line_text in matches:
            print(f"  Line {line_no}: {line_text}")

    print(f"\nRecovery train_pruned_recovery.py ({RECOVERY_SCRIPT}):")
    for pat, matches in rec_results.items():
        for line_no, line_text in matches:
            print(f"  Line {line_no}: {line_text}")
    if not rec_results:
        print("  NOT FOUND: selective_learning_rate_control or update_learning_rate")

    has_policy = 'selective_learning_rate_control' in rec_results
    has_lr_update = 'update_learning_rate' in rec_results

    print(f"\n{'='*60}")
    print("STATUS:")
    print(f"  selective_learning_rate_control in recovery: {has_policy}")
    print(f"  update_learning_rate in recovery: {has_lr_update}")
    print(f"  selective_learning_rate_control in TSGS: {'selective_learning_rate_control' in tsgs_results}")
    print(f"  update_learning_rate in TSGS: {'update_learning_rate' in tsgs_results}")
    print(f"\n  Fix needed: {not has_policy and not has_lr_update}")
    print(f"{'='*60}")

    if not has_policy and not has_lr_update:
        print("\nRECOMMENDED FIX:")
        print("  Add the following to train_pruned_recovery.py at the start of the training loop:")
        print("""  gaussians.selective_learning_rate_control(
        iteration, 15000,
        nofix_position=opt.nofix_position,
        nofix_opacity=opt.nofix_opacity,
        nofix_param=opt.nofix_param,
        nofix_scaling=opt.nofix_scaling,
        nofix_rotation=opt.nofix_rotation,
    )""")
    else:
        print("\nFix already applied or partially applied.")

if __name__ == '__main__':
    main()
