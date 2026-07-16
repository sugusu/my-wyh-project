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
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1d_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    print("[1/7] Reading domain_gt_distribution.json...")
    with open(debug_dir / 'domain_gt_distribution.json') as f:
        dist_report = json.load(f)

    print("[2/7] Reading candidate_domain_stats.json...")
    with open(out_dir / 'candidate_domain_stats.json') as f:
        cand_stats = json.load(f)

    print("[3/7] Reading candidate_geometry_errors_stats.json...")
    with open(out_dir / 'candidate_geometry_errors_stats.json') as f:
        geo_stats = json.load(f)

    print("[4/7] Reading continuous_reliability_metrics.json...")
    with open(out_dir / 'continuous_reliability_metrics.json') as f:
        cont_metrics = json.load(f)

    print("[5/7] Reading checkpoint comparison (if available)...")
    ckpt_data = None
    ckpt_path = debug_dir / 'checkpoint_7k_vs_15k.json'
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt_data = json.load(f)

    print("[6/7] Analyzing results for 7 key questions...")
    N = dist_report['all']['count']

    obj_ratio = dist_report['object']['ratio']
    bg_ratio = dist_report['background']['ratio']
    unc_ratio = dist_report['uncertain']['ratio']

    top5_obj = dist_report['top5_percent_highest_distance']['ratio_in_object']
    top5_bg = dist_report['top5_percent_highest_distance']['ratio_in_background']
    top5_unc = dist_report['top5_percent_highest_distance']['ratio_in_uncertain']
    top10_obj = dist_report['top10_percent_highest_distance']['ratio_in_object']
    top10_bg = dist_report['top10_percent_highest_distance']['ratio_in_background']
    top10_unc = dist_report['top10_percent_highest_distance']['ratio_in_uncertain']

    core_count = cand_stats['core_object']['count']
    candidate_count = cand_stats['candidate_object']['count']
    strong_bg_count = cand_stats['strong_background']['count']

    # Determine best continuous error metric — output signed Spearman, never abs()
    best_signed_rho = -999.0
    best_positive_rho = -999.0
    strongest_absolute_rho = 0.0
    strongest_absolute_metric = None
    strongest_absolute_signed = 0.0
    best_positive_metric = None
    for risk_name in ['R_A', 'R_B', 'R_C']:
        for metric_name in ['d_center_norm', 'd_scale', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']:
            if risk_name in cont_metrics and metric_name in cont_metrics[risk_name]:
                m = cont_metrics[risk_name][metric_name]
                if 'spearman_rho' in m:
                    rho = m['spearman_rho']
                    if rho > best_positive_rho:
                        best_positive_rho = rho
                        best_positive_metric = f"{risk_name}/{metric_name} (rho={rho:.4f})"
                    if abs(rho) > strongest_absolute_rho:
                        strongest_absolute_rho = abs(rho)
                        strongest_absolute_metric = f"{risk_name}/{metric_name}"
                        strongest_absolute_signed = rho

    has_ckpt = ckpt_data is not None and 'note' not in ckpt_data
    ckpt_7k_less_converged = False
    if has_ckpt:
        err_7k = ckpt_data['error_ratios']['7k']['gt_0.010']
        err_15k = ckpt_data['error_ratios']['15k']['gt_0.010']
        ckpt_7k_less_converged = err_7k > err_15k * 1.2

    # Answer 7 questions
    qa = {}

    qa['q1_domain_imbalance'] = {
        'question': 'Is the object domain too small/strict?',
        'object_ratio': obj_ratio,
        'object_count': dist_report['object']['count'],
        'answer': 'A: top5/10 high-error in uncertain' if top10_unc > 0.5 else 'B: object domain reasonable size',
    }

    qa['q2_high_error_concentration'] = {
        'question': 'Where do high-error Gaussians concentrate?',
        'top5_object_pct': top5_obj,
        'top5_background_pct': top5_bg,
        'top5_uncertain_pct': top5_unc,
        'top10_object_pct': top10_obj,
        'top10_background_pct': top10_bg,
        'top10_uncertain_pct': top10_unc,
        'judgment': 'High-error in uncertain -> domain too strict' if top10_unc > 0.5 else (
            'High-error in background -> near-surface leakage' if top10_bg > 0.5 else
            'Few high-error in object+uncertain -> 15k too converged'),
    }

    qa['q3_candidate_domain_expansion'] = {
        'question': 'Does the candidate domain capture more meaningful Gaussians?',
        'old_object_count': dist_report['object']['count'],
        'core_object_count': core_count,
        'candidate_object_count': candidate_count,
        'strong_background_count': strong_bg_count,
        'expansion_factor': candidate_count / max(core_count, 1),
        'adequate': candidate_count >= 100,
    }

    qa['q4_continuous_errors_meaningful'] = {
        'question': 'Are the continuous geometry error metrics meaningful?',
        'best_positive_correlation': best_positive_metric if best_positive_rho > -999 else 'N/A',
        'strongest_absolute_correlation': f"{strongest_absolute_metric} (signed rho={strongest_absolute_signed:.4f})",
        'num_candidate_gaussians': geo_stats['num_candidate_gaussians'],
        'd_center_norm_mean': geo_stats['d_center_norm']['mean'],
        'd_scale_mean': geo_stats['d_scale']['mean'],
        'meaningful': candidate_count >= 50,
    }

    qa['q5_risk_error_correlation'] = {
        'question': 'Do reliability scores correlate with continuous errors?',
        'best_positive_spearman': best_positive_rho if best_positive_rho > -999 else 0.0,
        'strongest_absolute_spearman_signed': strongest_absolute_signed,
        'interpretation': 'Strong' if strongest_absolute_rho > 0.3 else ('Moderate' if strongest_absolute_rho > 0.15 else 'Weak'),
        'adequate': strongest_absolute_rho > 0.15,
    }

    # For checkpoint conclusion, require ALL 5 conditions for '15k too late'
    # Condition 1: has_ckpt
    # Condition 2: ckpt_7k_less_converged
    # Condition 3: candidate_count >= 50
    # Condition 4: strongest_absolute_rho < 0.15 (weak correlation at 15k)
    # Condition 5: top10_unc > 0.5 (high-error in uncertain)
    all_5_conditions = (
        has_ckpt
        and ckpt_7k_less_converged
        and candidate_count >= 50
        and strongest_absolute_rho < 0.15
        and top10_unc > 0.5
    )

    qa['q6_checkpoint_timing'] = {
        'question': 'Is 15k too converged? Would 7k be better?',
        'has_7k_checkpoint': has_ckpt,
        '7k_less_converged': ckpt_7k_less_converged,
        'all_5_conditions_met': all_5_conditions,
        'condition_has_ckpt': has_ckpt,
        'condition_7k_less_converged': ckpt_7k_less_converged,
        'condition_candidate_count_ge_50': candidate_count >= 50,
        'condition_weak_correlation': strongest_absolute_rho < 0.15,
        'condition_high_error_in_uncertain': top10_unc > 0.5,
        'recommendation': 'Try 7k' if all_5_conditions else '15k is fine' if has_ckpt else 'Check not possible - no 7k',
    }

    # Overall conclusion
    reasons = []
    conclusion = None
    if candidate_count < 50 or geo_stats['num_candidate_gaussians'] < 50:
        conclusion = 'C'
        reasons.append('Too few candidate Gaussians for meaningful evaluation')
    elif all_5_conditions:
        conclusion = 'B'
        reasons.append('All 5 conditions met: 7k less converged, weak correlation at 15k, high-error in uncertain, enough candidates, and checkpoint comparison available')
    elif top10_unc > 0.7 and obj_ratio < 0.01 and not ckpt_7k_less_converged:
        conclusion = 'B'
        reasons.append(f'High-error Gaussians overwhelmingly in uncertain ({top10_unc*100:.0f}%), domain too strict; but 7k not less converged so earlier checkpoint may not help')
    elif strongest_absolute_rho < 0.1 and candidate_count > 100 and not ckpt_7k_less_converged:
        conclusion = 'C'
        reasons.append(f'Negligible correlation between risk scores and continuous errors (best |rho|={strongest_absolute_rho:.4f}) and 7k not less converged')
    elif ckpt_7k_less_converged and strongest_absolute_rho < 0.15 and not all_5_conditions:
        conclusion = 'B'
        reasons.append('7k checkpoint shows less converged Gaussians; 15k may be too late for meaningful discrimination (some conditions not met)')
    elif top10_unc > 0.7 and obj_ratio < 0.01:
        conclusion = 'B'
        reasons.append(f'High-error Gaussians overwhelmingly in uncertain ({top10_unc*100:.0f}%), domain too strict')
    elif strongest_absolute_rho < 0.1 and candidate_count > 100:
        conclusion = 'C'
        reasons.append(f'Negligible correlation between risk scores and continuous errors (best |rho|={strongest_absolute_rho:.4f})')
    elif ckpt_7k_less_converged and strongest_absolute_rho < 0.15:
        conclusion = 'B'
        reasons.append('7k checkpoint shows less converged Gaussians; 15k may be too late for meaningful discrimination')
    else:
        conclusion = 'A'
        reasons.append('Labels show reasonable separation; candidate domain captures sufficient Gaussians; risk-error correlation is meaningful')

    qa['q7_overall_conclusion'] = {
        'question': 'What is the final stage 1D conclusion?',
        'conclusion': conclusion,
        'reasons': reasons,
        'verdict': {
            'A': 'Labels valid, continue feature redesign',
            'B': '15k too late, try earlier checkpoint',
            'C': 'Single-Gaussian mesh-distance label unsuitable, pause H1',
        }[conclusion],
    }

    report_json = {
        'scene_name': cfg['scene_name'],
        'total_gaussians': N,
        'domain_statistics': {
            'object': {'count': dist_report['object']['count'], 'ratio': obj_ratio},
            'background': {'count': dist_report['background']['count'], 'ratio': bg_ratio},
            'uncertain': {'count': dist_report['uncertain']['count'], 'ratio': unc_ratio},
        },
        'candidate_domain': cand_stats,
        'geometry_errors': geo_stats,
        'qa': qa,
    }

    json_path = out_dir / 'stage1d_label_viability_report.json'
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2)

    md = [
        f"# Stage 1D Label Viability Report - {cfg['scene_name']}",
        f"",
        f"## Summary Statistics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Gaussians | {N} |",
        f"| Object domain | {dist_report['object']['count']} ({obj_ratio*100:.1f}%) |",
        f"| Background domain | {dist_report['background']['count']} ({bg_ratio*100:.1f}%) |",
        f"| Uncertain domain | {dist_report['uncertain']['count']} ({unc_ratio*100:.1f}%) |",
        f"| Core object (new) | {core_count} |",
        f"| Candidate object (new) | {candidate_count} |",
        f"| Strong background (new) | {strong_bg_count} |",
        f"",
        f"## 7 Key Questions",
        f"",
    ]
    for key in sorted(qa.keys()):
        v = qa[key]
        md.append(f"### {v['question']}")
        md.append(f"")
        for k2, v2 in v.items():
            if k2 != 'question':
                md.append(f"- {k2}: {v2}")
        md.append(f"")

    md.extend([
        f"## Final Conclusion: {qa['q7_overall_conclusion']['conclusion']} - {qa['q7_overall_conclusion']['verdict']}",
        f"",
    ])
    for r in reasons:
        md.append(f"- {r}")
    md.extend([
        f"",
        f"## Next Steps",
    ])
    if conclusion == 'A':
        md.append("- Labels valid. Continue to feature redesign (Stage 2 preparation).")
        md.append("- Keep candidate object domain as the evaluation domain.")
    elif conclusion == 'B':
        md.append("- Re-run all reliability computations on iteration_7000 checkpoint.")
        md.append("- The 15k Gaussians are too converged. Earlier checkpoint may retain more meaningful error signal.")
    elif conclusion == 'C':
        md.append("- Pause H1 direction. Single-Gaussian mesh-distance labels are not suitable.")
        md.append("- Consider alternative label definitions (e.g., patch-level, ray-level, or learning-based).")

    md_path = out_dir / 'stage1d_label_viability_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"[7/7] Report saved to {md_path}")
    print(f"")
    print(f"Final Conclusion: {qa['q7_overall_conclusion']['conclusion']} - {qa['q7_overall_conclusion']['verdict']}")
    print(f"Reasons:")
    for r in reasons:
        print(f"  - {r}")

if __name__ == '__main__':
    main()
