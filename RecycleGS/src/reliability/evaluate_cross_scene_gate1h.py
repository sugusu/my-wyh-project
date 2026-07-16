#!/usr/bin/env python3
"""Stage 1H final gate — determine PASS/CONDITIONAL/FAIL based on cross-scene feature validation."""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

METRICS_PATH = Path("/data/wyh/RecycleGS/outputs/reliability/stage1h_cross_scene_feature_metrics.json")
OUT_DIR = Path("/data/wyh/RecycleGS/outputs/reliability")

def apply_gate1h_rules(validation):
    """
    Section 12 rules for Stage 1H:
    - PASS: At least one feature passes ALL validity rules (rho>0.15, CI_lo>0, T10/B10>1.30, coverage>=30%, direction consistent)
    - CONDITIONAL: Direction consistent across all features, borderline magnitude
    - FAIL: No feature passes, or direction inconsistency across scenes
    """
    reasons = []
    passing_features = []
    failing_features = []
    direction_consistent_all = True

    for fname in sorted(validation.keys()):
        for ename in sorted(validation[fname].keys()):
            v = validation[fname][ename]
            if v.get('status') == 'SKIP':
                continue
            rules = v.get('rules', {})
            s01 = v.get('scene_01', {})
            s03 = v.get('scene_03', {})

            s01_rho = s01.get('spearman_rho', 0)
            s03_rho = s03.get('spearman_rho', 0)

            if rules.get('all_pass', False):
                passing_features.append((fname, ename, s01_rho, s03_rho))

            if not rules.get('direction_consistent', True):
                direction_consistent_all = False
                failing_features.append((fname, ename, 'direction_inconsistent'))

            if not rules.get('all_pass', False) and rules.get('direction_consistent', True):
                fail_reasons = []
                if not rules.get('rho_ok', False):
                    fail_reasons.append(f"rho(s01={s01_rho:.4f}, s03={s03_rho:.4f})")
                if not rules.get('ci_lower_ok', False):
                    fail_reasons.append(f"CI_lo(s01={s01.get('spearman_ci_95_lo', 0):.4f}, s03={s03.get('spearman_ci_95_lo', 0):.4f})")
                if not rules.get('t10b10_ok', False):
                    fail_reasons.append(f"T10/B10(s01={s01.get('top10_bottom10_ratio', 0):.4f}, s03={s03.get('top10_bottom10_ratio', 0):.4f})")
                if not rules.get('coverage_ok', False):
                    fail_reasons.append(f"coverage(s01={s01.get('valid_ratio', 0)*100:.1f}%, s03={s03.get('valid_ratio', 0)*100:.1f}%)")
                failing_features.append((fname, ename, '; '.join(fail_reasons)))

    for fname, ename, s01_rho, s03_rho in passing_features:
        reasons.append(f"PASS: {fname}/{ename} passes all criteria (rho s01={s01_rho:.4f}, s03={s03_rho:.4f})")

    for fname, ename, fail_reason in failing_features:
        reasons.append(f"FAIL: {fname}/{ename} — {fail_reason}")

    if len(passing_features) > 0:
        return "PASS", reasons

    if direction_consistent_all:
        return "CONDITIONAL", reasons + ["Direction consistent across all features, but no feature passes all magnitude criteria"]
    else:
        return "FAIL", reasons + ["Direction inconsistency detected across scenes"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default=str(METRICS_PATH))
    parser.add_argument('--out-dir', type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    validation = metrics.get('validation', {})

    print("=== Stage 1H Final Gate ===")
    verdict, reasons = apply_gate1h_rules(validation)

    print(f"\nVerdict: {verdict}")
    for r in reasons:
        print(f"  - {r}")

    # Determine overall status for each feature
    feature_status = {}
    for fname in sorted(validation.keys()):
        feature_status[fname] = {}
        for ename in sorted(validation[fname].keys()):
            v = validation[fname][ename]
            if v.get('status') == 'SKIP':
                feature_status[fname][ename] = 'SKIP'
            elif v.get('rules', {}).get('all_pass', False):
                feature_status[fname][ename] = 'PASS'
            else:
                feature_status[fname][ename] = 'FAIL'

    gate_result = {
        'gate': 'stage1h_final',
        'verdict': verdict,
        'reasons': reasons,
        'feature_status': feature_status,
        'passing_feature_count': sum(1 for fv in feature_status.values() for ev in fv.values() if ev == 'PASS'),
        'failing_feature_count': sum(1 for fv in feature_status.values() for ev in fv.values() if ev == 'FAIL'),
    }

    json_path = out_dir / 'stage1h_final_gate.json'
    with open(json_path, 'w') as f:
        json.dump(gate_result, f, indent=2)
    print(f"\nSaved: {json_path}")

    md_lines = [
        f"# Stage 1H Final Gate",
        f"",
        f"## Verdict: {verdict}",
        f"",
        f"### Evaluation Summary",
        f"",
        f"| Feature | Error Metric | Status |",
        f"|---------|-------------|--------|",
    ]
    for fname in sorted(feature_status.keys()):
        for ename in sorted(feature_status[fname].keys()):
            status = feature_status[fname][ename]
            md_lines.append(f"| {fname} | {ename} | {status} |")

    md_lines.extend([
        f"",
        f"### Reasons",
    ])
    for r in reasons:
        md_lines.append(f"- {r}")

    md_lines.extend([
        f"",
        f"### Validity Criteria",
        f"1. Spearman rho > 0.15 in both scenes",
        f"2. Bootstrap CI lower bound > 0 in both scenes",
        f"3. Top10/Bottom10 ratio > 1.30 in both scenes",
        f"4. Coverage >= 30% in both scenes",
        f"5. Direction consistency across scenes",
        f"",
        f"**Final Verdict: {verdict}**",
        f"",
        f"---",
        f"Generated by evaluate_cross_scene_gate1h.py",
    ])

    md_path = out_dir / 'stage1h_final_gate.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")

    print(f"\n{'='*60}")
    print(f"Stage 1H Final Verdict: {verdict}")
    print(f"{'='*60}")
    for r in reasons:
        print(f"  - {r}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
