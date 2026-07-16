#!/usr/bin/env python3
"""Audit recovery fairness: verify all methods have same training config."""
import json, os, sys, numpy as np, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

SCENES = ['scene_01', 'scene_03']
METHODS = ['schedule_control', 'random', 'low_opacity', 'low_contribution', 'mask_risk']
RECOVERY_CONFIG = '/data/wyh/RecycleGS/configs/stage2/recovery_500_locked.yaml'

def main():
    out_dir = Path('/data/wyh/RecycleGS/outputs/debug/stage2b_fairness')
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(RECOVERY_CONFIG) as f:
        recovery_cfg = yaml.safe_load(f)

    expected_steps = recovery_cfg.get('recovery_steps', 500)
    expected_seed = recovery_cfg.get('seed', 0)

    audit = {}
    all_pass = True

    for scene_name in SCENES:
        scene_audit = {}
        for method in METHODS:
            log_path = f'/data/wyh/RecycleGS/outputs/recovery/{scene_name}/{method}/training_log.json'
            if not os.path.exists(log_path):
                scene_audit[method] = {'error': 'training_log.json not found'}
                all_pass = False
                continue

            with open(log_path) as f:
                log = json.load(f)

            if not log:
                scene_audit[method] = {'error': 'empty training log'}
                all_pass = False
                continue

            first_iter = log[0]['iteration']
            last_iter = log[-1]['iteration']
            total_steps = last_iter - first_iter + 1

            trace_path = f'/data/wyh/RecycleGS/outputs/recovery/{scene_name}/{method}/gaussian_count_trace.csv'
            count_stable = None
            if os.path.exists(trace_path):
                import csv
                traces = []
                with open(trace_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        traces.append(int(row['N']))
                if traces:
                    count_stable = len(set(traces)) == 1

            method_audit = {
                'training_steps': total_steps,
                'expected_steps': expected_steps,
                'steps_match': total_steps == expected_steps,
                'first_iter': first_iter,
                'last_iter': last_iter,
                'gaussian_count_stable': count_stable,
                'seed': expected_seed,
            }
            scene_audit[method] = method_audit

            if not method_audit['steps_match'] or not method_audit.get('gaussian_count_stable', False):
                all_pass = False

        audit[scene_name] = scene_audit

    # Cross-scene consistency
    cross_scene = {}
    for metric in ['training_steps', 'gaussian_count_stable']:
        cross_scene[metric] = {}
        for method in METHODS:
            s01 = audit.get('scene_01', {}).get(method, {}).get(metric)
            s03 = audit.get('scene_03', {}).get(method, {}).get(metric)
            cross_scene[metric][method] = {
                'scene_01': s01,
                'scene_03': s03,
                'consistent': s01 == s03,
            }

    overall_status = 'PASS' if all_pass else 'FAIL'

    json_path = out_dir / 'recovery_fairness_audit.json'
    with open(json_path, 'w') as f:
        json.dump({
            'audit': audit,
            'cross_scene': cross_scene,
            'overall_status': overall_status,
        }, f, indent=2, default=str)

    md_lines = [
        "# Recovery Fairness Audit",
        "",
        f"**Status: {overall_status}**",
        "",
        "## Per-Scene Per-Method Audit",
        "",
    ]
    for scene_name in SCENES:
        md_lines.append(f"### {scene_name}")
        md_lines.append("| Method | Steps | Expected | Match | Count Stable |")
        md_lines.append("|--------|-------|----------|-------|--------------|")
        for method in METHODS:
            ma = audit.get(scene_name, {}).get(method, {})
            steps = ma.get('training_steps', 'N/A')
            exp = ma.get('expected_steps', 'N/A')
            match = 'PASS' if ma.get('steps_match') else 'FAIL' if 'steps_match' in ma else 'N/A'
            stable = 'PASS' if ma.get('gaussian_count_stable') else 'FAIL' if 'gaussian_count_stable' in ma else 'N/A'
            md_lines.append(f"| {method} | {steps} | {exp} | {match} | {stable} |")
        md_lines.append("")

    md_lines.append("## Cross-Scene Consistency")
    md_lines.append("")
    for metric, methods in cross_scene.items():
        md_lines.append(f"### {metric}")
        for method, data in methods.items():
            md_lines.append(f"- {method}: s01={data['scene_01']}, s03={data['scene_03']}, "
                          f"consistent={'Yes' if data['consistent'] else 'No'}")
    md_lines.append("")
    md_lines.append(f"## Overall: {overall_status}")

    md_path = out_dir / 'recovery_fairness_audit_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f"Saved: {md_path}")
    print(f"Overall: {overall_status}")

if __name__ == '__main__':
    main()
