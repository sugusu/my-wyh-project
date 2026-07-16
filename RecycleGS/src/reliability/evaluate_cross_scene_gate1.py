#!/usr/bin/env python3
"""Cross-scene Gate 1 validation: compare E_support_v2 across scene_01 and scene_03."""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

SCENE_01_METRICS = Path("/data/wyh/RecycleGS/outputs/reliability/scene_01/iter_15000/signed_feature_metrics.json")
SCENE_03_METRICS = Path("/data/wyh/RecycleGS/outputs/reliability/scene_03/iter_15000/signed_feature_metrics.json")
OUT_DIR = Path("/data/wyh/RecycleGS/outputs/reliability")

def load_metrics(path):
    if not path.exists():
        print(f"ERROR: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)

def extract_feature_metrics(metrics, feature_name, error_key='d_center_norm'):
    if not metrics or feature_name not in metrics:
        return None
    if error_key not in metrics[feature_name]:
        return None
    d = metrics[feature_name][error_key]
    if 'error' in d:
        return None
    return d

def apply_gate1_rules(s01_rho, s01_ci_lo, s01_t10b10, s01_cov,
                       s03_rho, s03_ci_lo, s03_t10b10, s03_cov):
    """
    Gate 1 rules:
    - PASS: both scenes have E_support_v2 rho>0.15, CI lower>0, T10/B10>1.30, coverage>=30%
    - CONDITIONAL: direction consistent but slightly weaker
    - FAIL: scene_03 direction opposite or near zero
    """
    reasons = []

    s01_ok = (s01_rho is not None and s01_rho > 0.15 and
              s01_ci_lo is not None and s01_ci_lo > 0 and
              s01_t10b10 is not None and s01_t10b10 > 1.30 and
              s01_cov is not None and s01_cov >= 30)

    s03_ok = (s03_rho is not None and s03_rho > 0.15 and
              s03_ci_lo is not None and s03_ci_lo > 0 and
              s03_t10b10 is not None and s03_t10b10 > 1.30 and
              s03_cov is not None and s03_cov >= 30)

    if s01_rho is None:
        reasons.append("scene_01: no valid E_support_v2 data")
    else:
        reasons.append(f"scene_01: rho={s01_rho:.4f}, CI_lo={s01_ci_lo:.4f}, "
                       f"T10/B10={s01_t10b10:.4f}, coverage={s01_cov:.1f}%")

    if s03_rho is None:
        reasons.append("scene_03: no valid E_support_v2 data")
    else:
        reasons.append(f"scene_03: rho={s03_rho:.4f}, CI_lo={s03_ci_lo:.4f}, "
                       f"T10/B10={s03_t10b10:.4f}, coverage={s03_cov:.1f}%")

    # Check direction consistency
    if s01_rho is not None and s03_rho is not None:
        same_direction = (s01_rho > 0) == (s03_rho > 0)
        if not same_direction:
            return "FAIL", reasons + ["scene_03 direction OPPOSITE to scene_01"]
        if s03_rho <= 0:
            return "FAIL", reasons + ["scene_03 rho near zero or negative"]

    if s01_ok and s03_ok:
        return "PASS", reasons + ["Both scenes pass all Gate 1 criteria"]
    elif s01_ok and s03_rho is not None and s03_rho > 0.10:
        return "CONDITIONAL", reasons + [
            f"scene_01 passes ({s01_rho:.4f}), scene_03 direction consistent "
            f"({s03_rho:.4f}) but slightly weaker. CONDITIONAL pass."
        ]
    elif s01_rho is not None and s03_rho is not None and abs(s03_rho) < 0.10:
        return "FAIL", reasons + [f"scene_03 rho={s03_rho:.4f} near zero"]
    elif s03_rho is None:
        return "FAIL", reasons + ["scene_03 evaluation incomplete"]
    else:
        return "FAIL", reasons + [f"Gate 1 criteria not met"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene01-metrics', type=str, default=str(SCENE_01_METRICS))
    parser.add_argument('--scene03-metrics', type=str, default=str(SCENE_03_METRICS))
    parser.add_argument('--out-dir', type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir / "scene_03", exist_ok=True)

    print("Loading scene_01 metrics...")
    s01 = load_metrics(Path(args.scene01_metrics))
    print("Loading scene_03 metrics...")
    s03 = load_metrics(Path(args.scene03_metrics))

    if s01 is None or s03 is None:
        print("ERROR: Could not load metrics for one or both scenes")
        sys.exit(1)

    features_to_eval = ['E_support_v2', 'E_scale_v2']
    error_key = 'd_center_norm'

    report = {
        "gate": "cross_scene_gate1",
        "scene_01": {"metrics_path": str(args.scene01_metrics)},
        "scene_03": {"metrics_path": str(args.scene03_metrics)},
        "feature_evaluation": {},
        "verdict": None,
        "reasons": [],
    }

    scene03_report_lines = [
        f"# Cross-Scene Gate 1 Validation - scene_03",
        f"",
        f"## Scene 01 Reference",
    ]

    for scene_name, metrics, label in [("scene_01", s01, "scene_01"), ("scene_03", s03, "scene_03")]:
        scene03_report_lines.append(f"### {label}")
        scene03_report_lines.append(f"| Feature | Spearman rho | 95% CI | T10/B10 Ratio | Valid Count | Coverage |")
        scene03_report_lines.append(f"|---------|-------------|--------|---------------|-------------|----------|")
        for fname in features_to_eval:
            d = extract_feature_metrics(metrics, fname, error_key)
            if d:
                cov = d.get('valid_ratio', 0) * 100
                scene03_report_lines.append(
                    f"| {fname} | {d['spearman_rho']:.4f} | "
                    f"[{d['spearman_ci_95_lo']:.4f}, {d['spearman_ci_95_hi']:.4f}] | "
                    f"{d['top10_bottom10_ratio']:.4f} | "
                    f"{d['valid_count']} | {cov:.1f}% |"
                )
            else:
                scene03_report_lines.append(f"| {fname} | N/A | N/A | N/A | N/A | N/A |")
        scene03_report_lines.append("")

    # Evaluate E_support_v2 (primary) and E_scale_v2 (auxiliary)
    print("\nEvaluating E_support_v2 (PRIMARY)...")
    s01_es = extract_feature_metrics(s01, 'E_support_v2', error_key)
    s03_es = extract_feature_metrics(s03, 'E_support_v2', error_key)

    s01_rho = s01_es['spearman_rho'] if s01_es else None
    s01_ci_lo = s01_es['spearman_ci_95_lo'] if s01_es else None
    s01_t10b10 = s01_es['top10_bottom10_ratio'] if s01_es else None
    s01_cov = s01_es['valid_ratio'] * 100 if s01_es and 'valid_ratio' in s01_es else None

    s03_rho = s03_es['spearman_rho'] if s03_es else None
    s03_ci_lo = s03_es['spearman_ci_95_lo'] if s03_es else None
    s03_t10b10 = s03_es['top10_bottom10_ratio'] if s03_es else None
    s03_cov = s03_es['valid_ratio'] * 100 if s03_es and 'valid_ratio' in s03_es else None

    verdict_primary, reasons_primary = apply_gate1_rules(
        s01_rho, s01_ci_lo, s01_t10b10, s01_cov,
        s03_rho, s03_ci_lo, s03_t10b10, s03_cov
    )

    report['feature_evaluation']['E_support_v2'] = {
        "primary": True,
        "scene_01": s01_es,
        "scene_03": s03_es,
        "verdict": verdict_primary,
        "reasons": reasons_primary,
    }

    print(f"  Verdict: {verdict_primary}")

    # Evaluate E_scale_v2 (auxiliary)
    print("\nEvaluating E_scale_v2 (AUXILIARY)...")
    s01_esc = extract_feature_metrics(s01, 'E_scale_v2', error_key)
    s03_esc = extract_feature_metrics(s03, 'E_scale_v2', error_key)

    s01_rho2 = s01_esc['spearman_rho'] if s01_esc else None
    s03_rho2 = s03_esc['spearman_rho'] if s03_esc else None

    report['feature_evaluation']['E_scale_v2'] = {
        "primary": False,
        "scene_01": s01_esc,
        "scene_03": s03_esc,
        "verdict": "INFO_ONLY",
        "reasons": ["E_scale_v2 is auxiliary only. Not used for Gate 1 decision."],
    }

    # Gate 1 final verdict based on E_support_v2
    report['verdict'] = verdict_primary
    report['reasons'] = reasons_primary

    # Save cross_scene_gate1_metrics.json
    metrics_out = out_dir / "cross_scene_gate1_metrics.json"
    with open(metrics_out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved metrics: {metrics_out}")

    # Generate cross_scene_gate1_report.md
    md_lines = [
        f"# Cross-Scene Gate 1 Validation Report",
        f"",
        f"## Verdict: {verdict_primary}",
        f"",
        f"### Reasons",
    ]
    for r in reasons_primary:
        md_lines.append(f"- {r}")
    md_lines.extend([
        f"",
        f"## Feature: E_support_v2 (PRIMARY)",
        f"",
        f"| Metric | scene_01 | scene_03 |",
        f"|--------|----------|----------|",
    ])
    for label, ds in [("Spearman rho", s01_es), ("Spearman rho", s03_es)]:
        pass
    md_lines.append(f"| Spearman rho | {s01_rho:.4f} | {s03_rho:.4f} |")
    md_lines.append(f"| 95% CI | [{s01_ci_lo:.4f}, {s01_es['spearman_ci_95_hi']:.4f}] | [{s03_ci_lo:.4f}, {s03_es['spearman_ci_95_hi']:.4f}] |")
    md_lines.append(f"| T10/B10 Ratio | {s01_t10b10:.4f} | {s03_t10b10:.4f} |")
    md_lines.append(f"| Coverage | {s01_cov:.1f}% | {s03_cov:.1f}% |")
    md_lines.append(f"| Valid Count | {s01_es['valid_count']} | {s03_es['valid_count']} |")
    md_lines.extend([
        f"",
        f"## Feature: E_scale_v2 (AUXILIARY)",
        f"",
        f"| Metric | scene_01 | scene_03 |",
        f"|--------|----------|----------|",
        f"| Spearman rho | {s01_rho2:.4f} | {s03_rho2:.4f} |" if s01_rho2 is not None and s03_rho2 is not None else f"| Spearman rho | N/A | N/A |",
        f"",
        f"## Gate 1 Rules Applied",
        f"1. E_support_v2 rho > 0.15 in both scenes",
        f"2. Bootstrap CI lower bound > 0 in both scenes",
        f"3. T10/B10 ratio > 1.30 in both scenes",
        f"4. Coverage >= 30% in both scenes",
        f"5. Direction consistency across scenes",
        f"",
        f"**Final Verdict: {verdict_primary}**",
        f"",
        f"---",
        f"Generated by evaluate_cross_scene_gate1.py",
    ])

    report_md = out_dir / "cross_scene_gate1_report.md"
    with open(report_md, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved report: {report_md}")

    # Generate scene-specific report
    scene03_md_lines = scene03_report_lines + [
        f"## Gate 1 Verdict for scene_03",
        f"",
        f"{verdict_primary}",
        f"",
    ]
    for r in reasons_primary:
        scene03_md_lines.append(f"- {r}")

    scene03_md = out_dir / "scene_03" / "gate1_scene03_report.md"
    with open(scene03_md, 'w') as f:
        f.write('\n'.join(scene03_md_lines))
    print(f"Saved scene report: {scene03_md}")

    print(f"\n{'='*60}")
    print(f"Cross-Scene Gate 1 Verdict: {verdict_primary}")
    print(f"{'='*60}")
    for r in reasons_primary:
        print(f"  - {r}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
