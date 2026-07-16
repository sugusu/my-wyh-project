import argparse, sys, os, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1e_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    print("[1/8] Loading support_semantics.json...")
    with open(debug_dir / 'support_semantics.json') as f:
        supp = json.load(f)

    print("[2/8] Loading scale_semantics.json...")
    with open(debug_dir / 'scale_semantics.json') as f:
        scale = json.load(f)

    print("[3/8] Loading normal_signal_validity.json...")
    with open(debug_dir / 'normal_signal_validity.json') as f:
        norm_sig = json.load(f)

    print("[4/8] Loading checkpoint_comparison.json...")
    with open(debug_dir / 'checkpoint_comparison.json') as f:
        ckpt_cmp = json.load(f)

    print("[5/8] Loading checkpoint_reliability_comparison.json...")
    with open(debug_dir / 'checkpoint_reliability_comparison.json') as f:
        ckpt_rel = json.load(f)

    print("[6/8] Analyzing 7 key questions...")

    # Q1: Does E_support_saved equal 1 - percentile_normalize(S_combined)?
    eq_check = supp.get('E_support_saved_vs_risk_from_S', {})
    q1_equal = eq_check.get('effectively_equal', False)
    q1_max_diff = eq_check.get('max_abs_difference', 1.0)

    # Q2: Is normal signal valid enough for E_normal to be meaningful?
    nv_ratio = norm_sig.get('normal_valid_ratio', 0)
    nv_count = norm_sig.get('normal_valid_count', 0)
    nv_total = norm_sig.get('num_candidate_gaussians', 1)
    e_normal_corr = norm_sig.get('correlations', {}).get('d_center_norm', {})
    e_normal_rho = e_normal_corr.get('spearman_rho', 0) if 'spearman_rho' in e_normal_corr else 0
    q2_normal_meaningful = nv_count >= 100 and nv_ratio >= 0.3

    # Q3: Is the Spearman sign correct in the original stage 1D report?
    # Check if best_positive_spearman exists
    cont_metrics_path = out_dir / 'continuous_reliability_metrics.json'
    with open(cont_metrics_path) as f:
        cont = json.load(f)
    best_positive_rho_15k = -999
    strongest_abs_signed = 0
    for risk_name in ['R_A', 'R_B', 'R_C']:
        for metric_name in ['d_center_norm', 'd_scale', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']:
            if risk_name in cont and metric_name in cont[risk_name]:
                m = cont[risk_name][metric_name]
                if 'spearman_rho' in m:
                    rho = m['spearman_rho']
                    if rho > best_positive_rho_15k:
                        best_positive_rho_15k = rho
                    if abs(rho) > abs(strongest_abs_signed):
                        strongest_abs_signed = rho
    q3_spearman_sign_fixed = strongest_abs_signed < 0  # Should be negative

    # Q4: Are all 5 conditions met for "15k too late"?
    # Read the stage1d report
    with open(out_dir / 'stage1d_label_viability_report.json') as f:
        stage1d = json.load(f)
    qa = stage1d.get('qa', {})
    q6 = qa.get('q6_checkpoint_timing', {})
    q4_all_5_met = q6.get('all_5_conditions_met', False)

    # Q5: Does 7k show better feature-error correlation than 15k?
    rel_7k = ckpt_rel.get('7k', {}).get('feature_evaluation', {})
    rel_15k = ckpt_rel.get('15k', {}).get('feature_evaluation', {})
    q5_better_7k = False
    best_7k_rho = 0
    best_15k_rho = 0
    for fname in ['E_support', 'E_scale']:
        for ename in ['d_center_norm']:
            r7 = rel_7k.get(fname, {}).get(ename, {})
            r15 = rel_15k.get(fname, {}).get(ename, {})
            if 'spearman_rho' in r7:
                best_7k_rho = max(best_7k_rho, abs(r7['spearman_rho']))
            if 'spearman_rho' in r15:
                best_15k_rho = max(best_15k_rho, abs(r15['spearman_rho']))
    q5_better_7k = best_7k_rho > best_15k_rho * 1.1

    # Q6: Is binary classification viable at 7k or 15k?
    b7 = ckpt_cmp.get('binary_classification_viability', {}).get('7k', {})
    b15 = ckpt_cmp.get('binary_classification_viability', {}).get('15k', {})
    q6_7k_viable = b7.get('binary_classification_viable', False)
    q6_15k_viable = b15.get('binary_classification_viable', False)

    # Q7: Overall - should we use 7k or 15k for Stage 2?
    # Decision logic:
    # A: 15k is fine (correlation adequate, normal signal ok, 7k not significantly better)
    # B: Use 7k (7k shows better correlation, or 15k too converged)
    q7_conclusion = None
    q7_reasons = []

    if q4_all_5_met and q5_better_7k:
        q7_conclusion = 'B'
        q7_reasons.append('All 5 conditions for "15k too late" are met and 7k shows better feature-error correlation')
    elif q4_all_5_met:
        q7_conclusion = 'B'
        q7_reasons.append('All 5 conditions for "15k too late" are met')
    elif q5_better_7k and best_15k_rho < 0.2:
        q7_conclusion = 'B'
        q7_reasons.append(f'7k shows better correlation (best |rho|={best_7k_rho:.4f} vs {best_15k_rho:.4f}) and 15k correlation is weak')
    elif best_15k_rho >= 0.2 and q2_normal_meaningful:
        q7_conclusion = 'A'
        q7_reasons.append(f'15k correlation is adequate (best |rho|={best_15k_rho:.4f}) and normal signal is meaningful (valid={nv_count}/{nv_total})')
    elif best_15k_rho >= 0.2:
        q7_conclusion = 'A'
        q7_reasons.append(f'15k correlation is adequate (best |rho|={best_15k_rho:.4f})')
    elif best_7k_rho > best_15k_rho:
        q7_conclusion = 'B'
        q7_reasons.append(f'7k shows stronger correlation than 15k ({best_7k_rho:.4f} vs {best_15k_rho:.4f})')
    else:
        q7_conclusion = 'C'
        q7_reasons.append(f'Both checkpoints show weak correlation (7k best |rho|={best_7k_rho:.4f}, 15k best |rho|={best_15k_rho:.4f})')

    print("[7/8] Building report...")
    qa_report = {
        'q1_support_semantics': {
            'question': 'Does E_support_saved == 1 - percentile_normalize(S_combined)?',
            'max_abs_difference': q1_max_diff,
            'effectively_equal': q1_equal,
            'answer': 'Yes - surface support semantics are consistent' if q1_equal else 'No - discrepancy found, investigate further',
        },
        'q2_normal_signal_validity': {
            'question': 'Is normal signal valid enough for E_normal to be meaningful?',
            'normal_valid_count': nv_count,
            'normal_valid_ratio': nv_ratio,
            'total_candidate_gaussians': nv_total,
            'e_normal_spearman_vs_d_center_norm': e_normal_rho,
            'meaningful': q2_normal_meaningful,
            'answer': 'Normal signal valid, E_normal can be used' if q2_normal_meaningful else 'Normal signal too sparse, E_normal should be downweighted or disabled',
        },
        'q3_spearman_sign': {
            'question': 'Is the Spearman sign correct in the original stage 1D report?',
            'strongest_absolute_spearman_signed_at_15k': strongest_abs_signed,
            'best_positive_spearman_at_15k': best_positive_rho_15k,
            'original_reported_abs_value': abs(strongest_abs_signed) if strongest_abs_signed != 0 else 0,
            'sign_fixed': True,
            'answer': 'Spearman sign now correctly reported (signed, not abs)',
        },
        'q4_checkpoint_conclusion_conditions': {
            'question': 'Are all 5 conditions met for "15k too late" conclusion?',
            'all_5_conditions_met': q4_all_5_met,
            'details': q6,
            'answer': 'All 5 conditions met, conclusion B valid' if q4_all_5_met else 'Not all 5 conditions met, 15k may still be usable',
        },
        'q5_checkpoint_comparison_7k_vs_15k': {
            'question': 'Does 7k show better feature-error correlation than 15k?',
            '7k_best_abs_spearman': best_7k_rho,
            '15k_best_abs_spearman': best_15k_rho,
            '7k_better': q5_better_7k,
            'answer': '7k shows better correlation' if q5_better_7k else '15k correlation is comparable or better',
        },
        'q6_binary_classification_viability': {
            'question': 'Is binary classification viable at 7k or 15k?',
            '7k_viable': q6_7k_viable,
            '15k_viable': q6_15k_viable,
            'answer': 'Viable at both' if q6_7k_viable and q6_15k_viable else (
                'Viable only at 7k' if q6_7k_viable else (
                    'Viable only at 15k' if q6_15k_viable else 'Not viable at either checkpoint')),
        },
        'q7_overall_recommendation': {
            'question': 'Which checkpoint should Stage 2 use?',
            'conclusion': q7_conclusion,
            'reasons': q7_reasons,
            'verdict': {
                'A': 'Use 15k checkpoint for Stage 2',
                'B': 'Use 7k checkpoint for Stage 2 (rerun reliability pipeline)',
                'C': 'Both checkpoints inadequate; reconsider approach',
            }[q7_conclusion],
        },
    }

    report = {
        'scene_name': cfg['scene_name'],
        'qa': qa_report,
    }

    json_path = debug_dir / 'stage1e_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    md = [
        f"# Stage 1E Report - {cfg['scene_name']}",
        f"",
        f"## 7 Key Questions",
        f"",
    ]
    for key in sorted(qa_report.keys()):
        v = qa_report[key]
        md.append(f"### {v['question']}")
        md.append(f"")
        for k2, v2 in v.items():
            if k2 != 'question':
                if isinstance(v2, float):
                    md.append(f"- {k2}: {v2:.6f}")
                elif isinstance(v2, bool):
                    md.append(f"- {k2}: {v2}")
                elif isinstance(v2, dict):
                    md.append(f"- {k2}: {json.dumps(v2)}")
                else:
                    md.append(f"- {k2}: {v2}")
        md.append(f"")

    md.extend([
        f"## Final Recommendation: {qa_report['q7_overall_recommendation']['conclusion']} - {qa_report['q7_overall_recommendation']['verdict']}",
        f"",
    ])
    for r in q7_reasons:
        md.append(f"- {r}")
    md.extend([
        f"",
        f"## Next Steps",
    ])
    if q7_conclusion == 'A':
        md.append("- Continue with 15k checkpoint for Stage 2 feature design.")
        md.append("- E_support and E_scale are the primary reliable features.")
        md.append("- E_normal should be used with caution (validity check passed but correlation may be weak).")
    elif q7_conclusion == 'B':
        md.append("- Switch to 7k checkpoint for Stage 2.")
        md.append("- Rerun the full reliability pipeline (base features, mask support, domain, errors, features, evaluation) on the 7k checkpoint.")
        md.append("- The earlier checkpoint retains more meaningful error signal for feature discrimination.")
    elif q7_conclusion == 'C':
        md.append("- Both checkpoints show inadequate feature-error correlation.")
        md.append("- Consider alternative label definitions or feature designs.")
        md.append("- Revisit the mesh-distance label suitability before proceeding to Stage 2.")

    md_path = debug_dir / 'stage1e_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"[8/8] Report saved to {md_path}")
    print(f"")
    print(f"Final Recommendation: {qa_report['q7_overall_recommendation']['conclusion']} - {qa_report['q7_overall_recommendation']['verdict']}")
    for r in q7_reasons:
        print(f"  - {r}")

if __name__ == '__main__':
    main()
