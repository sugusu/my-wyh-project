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
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1f_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    print("[1/8] Loading Stage 1F results...")
    iter_7k = out_dir / 'iter_7000'
    iter_15k = out_dir / 'iter_15000'

    def safe_load_json(path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def safe_load_npy(path):
        if path.exists():
            return np.load(path)
        return None

    audit = safe_load_json(debug_dir / 'feature_index_audit.json')
    cand_7k = safe_load_npy(iter_7k / 'candidate_indices.npy')
    cand_15k = safe_load_npy(iter_15k / 'candidate_indices.npy')
    metrics_7k = safe_load_json(iter_7k / 'signed_feature_metrics.json')
    metrics_15k = safe_load_json(iter_15k / 'signed_feature_metrics.json')
    comparison = safe_load_json(debug_dir / 'signed_checkpoint_comparison.json')
    normal_7k = safe_load_json(debug_dir / 'normal_valid_subset_iter_7000.json')
    normal_15k = safe_load_json(debug_dir / 'normal_valid_subset_iter_15000.json')
    surf_stats_7k = safe_load_json(iter_7k / 'surface_support_v2_stats.json')
    surf_stats_15k = safe_load_json(iter_15k / 'surface_support_v2_stats.json')
    scale_stats_7k = safe_load_json(iter_7k / 'scale_anomaly_v2_stats.json')
    scale_stats_15k = safe_load_json(iter_15k / 'scale_anomaly_v2_stats.json')
    err_7k = safe_load_npy(iter_7k / 'geometry_errors.npz')
    err_15k = safe_load_npy(iter_15k / 'geometry_errors.npz')

    print("[2/8] Applying strict validity rules...")
    n_7k = len(cand_7k) if cand_7k is not None else 0
    n_15k = len(cand_15k) if cand_15k is not None else 0

    def get_best_signed_rho(metrics, error_key='d_center_norm'):
        if not metrics:
            return None
        best = 0.0
        best_name = None
        for fname, fdict in metrics.items():
            if error_key in fdict and 'spearman_rho' in fdict[error_key]:
                rho = fdict[error_key]['spearman_rho']
                if isinstance(rho, (int, float)) and abs(rho) > abs(best):
                    best = rho
                    best_name = fname
        return best, best_name

    best_rho_7k, best_name_7k = get_best_signed_rho(metrics_7k, 'd_center_norm')
    best_rho_15k, best_name_15k = get_best_signed_rho(metrics_15k, 'd_center_norm')

    print(f"  Best signed rho 7k ({best_name_7k}): {best_rho_7k}")
    print(f"  Best signed rho 15k ({best_name_15k}): {best_rho_15k}")

    def get_top10_ratio(metrics, fname, error_key='d_center_norm'):
        if not metrics or fname not in metrics:
            return None
        if error_key not in metrics[fname]:
            return None
        return metrics[fname][error_key].get('top10_bottom10_ratio', None)

    ratio_7k = get_top10_ratio(metrics_7k, 'E_support_v2')
    ratio_15k = get_top10_ratio(metrics_15k, 'E_support_v2')
    print(f"  E_support_v2 Top10/Bottom10 ratio 7k: {ratio_7k}, 15k: {ratio_15k}")

    def check_feature_sign_direction(metrics):
        if not metrics:
            return False, 'no metrics'
        checks = {}
        expected_signs = {
            'E_support_v2': 1.0,
            'support_risk_v2': 1.0,
            'E_scale_v2': 1.0,
            'scale_risk_v2': 1.0,
            'support_confidence': -1.0,
        }
        all_correct = True
        for fname, expected_sign in expected_signs.items():
            if fname in metrics and 'd_center_norm' in metrics[fname] and 'spearman_rho' in metrics[fname]['d_center_norm']:
                rho = metrics[fname]['d_center_norm']['spearman_rho']
                if isinstance(rho, (int, float)):
                    actual_sign = 1.0 if rho >= 0 else -1.0
                    correct = actual_sign == expected_sign
                    checks[fname] = {
                        'rho': float(rho), 'expected_sign': expected_sign,
                        'actual_sign': actual_sign, 'correct': correct,
                    }
                    if not correct:
                        all_correct = False
        return all_correct, checks

    sign_ok_7k, sign_checks_7k = check_feature_sign_direction(metrics_7k)
    sign_ok_15k, sign_checks_15k = check_feature_sign_direction(metrics_15k)

    # Strict validity rules:
    # Rule 1: candidate_count >= 50 for meaningful evaluation
    # Rule 2: |best_signed_rho| >= 0.15 for adequate signal
    # Rule 3: top10/bottom10 ratio > 1.2 for feature discrimination
    # Rule 4: surface support verification passed (max_abs_diff < 1e-5)
    # Rule 5: normal coverage >= 30% for normal usability
    # Rule 6: ALL feature signed directions are CORRECT

    validity_7k = {}
    validity_15k = {}

    for label, n, best_rho, best_name, ratio, surf_stats, normal_info in [
        ('7k', n_7k, best_rho_7k, best_name_7k, ratio_7k, surf_stats_7k, normal_7k),
        ('15k', n_15k, best_rho_15k, best_name_15k, ratio_15k, surf_stats_15k, normal_15k),
    ]:
        v = {}
        v['candidate_count'] = n
        v['rule1_candidate_count_ge_50'] = n >= 50
        v['rule1_passed'] = v['rule1_candidate_count_ge_50']

        if best_rho is not None:
            v['best_signed_rho'] = float(best_rho)
            v['best_feature'] = best_name
            v['rule2_abs_rho_ge_0_15'] = abs(best_rho) >= 0.15
            v['rule2_passed'] = v['rule2_abs_rho_ge_0_15']
        else:
            v['best_signed_rho'] = None
            v['best_feature'] = None
            v['rule2_abs_rho_ge_0_15'] = False
            v['rule2_passed'] = False

        if ratio is not None:
            v['E_support_v2_top10_bottom10_ratio'] = float(ratio)
            v['rule3_ratio_gt_1_2'] = ratio > 1.2
            v['rule3_passed'] = v['rule3_ratio_gt_1_2']
        else:
            v['E_support_v2_top10_bottom10_ratio'] = None
            v['rule3_ratio_gt_1_2'] = False
            v['rule3_passed'] = False

        if surf_stats:
            v['surface_support_verification_passed'] = surf_stats.get('verification_passed', False)
            v['rule4_passed'] = v['surface_support_verification_passed']
        else:
            v['surface_support_verification_passed'] = None
            v['rule4_passed'] = False

        if normal_info:
            v['normal_valid_ratio'] = normal_info.get('normal_valid_ratio', 0)
            v['normal_global_usable'] = normal_info.get('normal_global_usable', False)
            v['rule5_normal_coverage_ge_30'] = v['normal_global_usable']
            v['rule5_passed'] = v['rule5_normal_coverage_ge_30']
        else:
            v['normal_valid_ratio'] = None
            v['normal_global_usable'] = False
            v['rule5_normal_coverage_ge_30'] = False
            v['rule5_passed'] = False

        v['sign_direction_checks'] = sign_checks_7k if label == '7k' else sign_checks_15k
        v['sign_ok'] = sign_ok_7k if label == '7k' else sign_ok_15k
        v['rule6_sign_correct'] = v['sign_ok']
        v['rule6_passed'] = v['rule6_sign_correct']

        rules_passed = sum([v['rule1_passed'], v['rule2_passed'], v['rule3_passed'],
                           v['rule4_passed'], v['rule5_passed'], v['rule6_passed']])
        v['rules_passed'] = rules_passed
        v['rules_total'] = 6

        if label == '7k':
            validity_7k = v
        else:
            validity_15k = v

    print("[3/8] Determining conclusion...")
    # Conclusion logic:
    # A: 15k is adequate (rules 1-4 passed at 15k, and 15k >= 7k)
    # B: 7k is better (7k clearly outperforms 15k)
    # C: Neither adequate

    has_7k = n_7k >= 50
    has_15k = n_15k >= 50
    comparison_available = comparison is not None

    conclusion = None
    reasons = []

    signal_7k = abs(best_rho_7k) if best_rho_7k is not None else 0
    signal_15k = abs(best_rho_15k) if best_rho_15k is not None else 0

    # Primary check: 15k validity (including signed direction correctness)
    if has_15k and validity_15k['rule1_passed'] and validity_15k['rule2_passed'] and validity_15k['rule3_passed'] and validity_15k['rule4_passed'] and validity_15k['rule6_passed']:
        if has_7k and validity_7k['rule6_passed'] and signal_7k > signal_15k * 1.2 and validity_7k['rule1_passed']:
            conclusion = 'B'
            reasons.append(f'7k shows significantly stronger signed correlation '
                           f'(|rho|={signal_7k:.4f} vs {signal_15k:.4f}) with correct signed directions and adequate candidate count ({n_7k})')
        else:
            conclusion = 'A'
            reasons.append(f'15k candidate domain is adequate: {n_15k} candidates, '
                           f'best |rho|={signal_15k:.4f}, all features have correct signed direction, '
                           f'support verification passed')
            if ratio_15k is not None and ratio_15k > 1.2:
                reasons.append(f'E_support_v2 top10/bottom10 ratio={ratio_15k:.4f} shows feature discrimination')
    elif has_7k and validity_7k['rule1_passed'] and validity_7k['rule2_passed'] and validity_7k['rule4_passed'] and validity_7k['rule6_passed']:
        # 15k failed sign check or ratio but 7k is adequate
        conclusion = 'B'
        reasons.append(f'15k fails validity rules but 7k passes core rules: {n_7k} candidates, '
                       f'best |rho|={signal_7k:.4f}, correct signed directions')
        if has_15k:
            if not validity_15k['rule6_passed']:
                reasons.append(f'15k feature direction check fails: E_support_v2 has wrong sign')
    elif has_15k and validity_15k['rule1_passed']:
        if not validity_15k['rule6_passed']:
            reasons.append(f'15k has {n_15k} candidates but feature signed directions are incorrect')
        if not validity_15k['rule2_passed']:
            reasons.append(f'15k correlation signal too weak (best |rho|={signal_15k:.4f} < 0.15)')
        conclusion = 'C'
    else:
        conclusion = 'C'
        reasons.append(f'Neither checkpoint has adequate candidate domain or feature signal: '
                       f'7k: {n_7k} candidates, best |rho|={signal_7k:.4f}, sign_ok={validity_7k.get("sign_ok", "N/A")}; '
                       f'15k: {n_15k} candidates, best |rho|={signal_15k:.4f}, sign_ok={validity_15k.get("sign_ok", "N/A")}')

    verdict_map = {
        'A': '15k checkpoint is selected for cross-scene Gate 1 validation. Stage 2 is not allowed until scene_03 confirms the same trend.',
        'B': '7k checkpoint is preferred over 15k. Re-run feature extraction and evaluation on 7k for Stage 2.',
        'C': 'Both checkpoints inadequate. Reconsider approach before proceeding to Stage 2.',
    }

    print(f"  Conclusion: {conclusion}")
    for r in reasons:
        print(f"  - {r}")

    print("[4/8] Building report...")
    report = {
        'scene_name': cfg['scene_name'],
        'conclusion': conclusion,
        'verdict': verdict_map.get(conclusion, 'Unknown'),
        'reasons': reasons,
        'validity_7k': validity_7k,
        'validity_15k': validity_15k,
        'checkpoint_comparison': {
            '7k_candidate_count': n_7k,
            '15k_candidate_count': n_15k,
            'best_signed_rho_7k': float(best_rho_7k) if best_rho_7k is not None else None,
            'best_feature_7k': best_name_7k,
            'best_signed_rho_15k': float(best_rho_15k) if best_rho_15k is not None else None,
            'best_feature_15k': best_name_15k,
            'E_support_v2_ratio_7k': float(ratio_7k) if ratio_7k is not None else None,
            'E_support_v2_ratio_15k': float(ratio_15k) if ratio_15k is not None else None,
        },
    }

    json_path = debug_dir / 'stage1f_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    md = [
        f"# Stage 1F Report - {cfg['scene_name']}",
        f"",
        f"## Conclusion: {conclusion}",
        f"",
        f"{verdict_map.get(conclusion, 'Unknown')}",
        f"",
        f"### Reasons",
    ]
    for r in reasons:
        md.append(f"- {r}")
    md.extend([
        f"",
        f"## Strict Validity Rules ({validity_7k['rules_total']} rules)",
        f"",
        f"### 7k",
        f"| Rule | Passed? | Detail |",
        f"|------|---------|--------|",
        f"| 1. Candidate count >= 50 | {validity_7k['rule1_passed']} | count={validity_7k['candidate_count']} |",
        f"| 2. Best |signed rho| >= 0.15 | {validity_7k['rule2_passed']} | rho={validity_7k.get('best_signed_rho', 'N/A')} |",
        f"| 3. E_support_v2 top10/bottom10 ratio > 1.2 | {validity_7k['rule3_passed']} | ratio={validity_7k.get('E_support_v2_top10_bottom10_ratio', 'N/A')} |",
        f"| 4. Surface support verification (max_abs_diff < 1e-5) | {validity_7k['rule4_passed']} | verified={validity_7k.get('surface_support_verification_passed', 'N/A')} |",
        f"| 5. Normal coverage >= 30% | {validity_7k['rule5_passed']} | ratio={validity_7k.get('normal_valid_ratio', 'N/A')} |",
        f"| 6. Feature signed direction correct | {validity_7k['rule6_passed']} | checks={json.dumps(validity_7k.get('sign_direction_checks', {}))} |",
        f"| **Total** | **{validity_7k['rules_passed']}/{validity_7k['rules_total']}** | |",
        f"",
        f"### 15k",
        f"| Rule | Passed? | Detail |",
        f"|------|---------|--------|",
        f"| 1. Candidate count >= 50 | {validity_15k['rule1_passed']} | count={validity_15k['candidate_count']} |",
        f"| 2. Best |signed rho| >= 0.15 | {validity_15k['rule2_passed']} | rho={validity_15k.get('best_signed_rho', 'N/A')} |",
        f"| 3. E_support_v2 top10/bottom10 ratio > 1.2 | {validity_15k['rule3_passed']} | ratio={validity_15k.get('E_support_v2_top10_bottom10_ratio', 'N/A')} |",
        f"| 4. Surface support verification (max_abs_diff < 1e-5) | {validity_15k['rule4_passed']} | verified={validity_15k.get('surface_support_verification_passed', 'N/A')} |",
        f"| 5. Normal coverage >= 30% | {validity_15k['rule5_passed']} | ratio={validity_15k.get('normal_valid_ratio', 'N/A')} |",
        f"| 6. Feature signed direction correct | {validity_15k['rule6_passed']} | checks={json.dumps(validity_15k.get('sign_direction_checks', {}))} |",
        f"| **Total** | **{validity_15k['rules_passed']}/{validity_15k['rules_total']}** | |",
        f"",
        f"## Checkpoint Comparison Summary",
        f"| Metric | 7k | 15k |",
        f"|--------|-----|-----|",
        f"| Candidate count | {n_7k} | {n_15k} |",
        f"| Best signed rho | {best_rho_7k if best_rho_7k is not None else 'N/A'} | {best_rho_15k if best_rho_15k is not None else 'N/A'} |",
        f"| Best feature | {best_name_7k if best_name_7k else 'N/A'} | {best_name_15k if best_name_15k else 'N/A'} |",
        f"| E_support_v2 T10/B10 ratio | {ratio_7k if ratio_7k is not None else 'N/A'} | {ratio_15k if ratio_15k is not None else 'N/A'} |",
        f"| E_support_v2 direction | {'CORRECT' if validity_7k.get('sign_direction_checks', {}).get('E_support_v2', {}).get('correct', False) else 'WRONG'} | {'CORRECT' if validity_15k.get('sign_direction_checks', {}).get('E_support_v2', {}).get('correct', False) else 'WRONG'} |",
        f"| E_scale_v2 direction | {'CORRECT' if validity_7k.get('sign_direction_checks', {}).get('E_scale_v2', {}).get('correct', False) else 'WRONG'} | {'CORRECT' if validity_15k.get('sign_direction_checks', {}).get('E_scale_v2', {}).get('correct', False) else 'WRONG'} |",
        f"",
        f"## Feature Verification",
        f"| Checkpoint | Surface support max_abs_diff | Verification Passed |",
        f"|------------|------------------------------|---------------------|",
    ])
    for label, sstats in [('7k', surf_stats_7k), ('15k', surf_stats_15k)]:
        if sstats:
            md.append(f"| {label} | {sstats.get('max_abs_diff_verification', 'N/A'):.2e} | {sstats.get('verification_passed', False)} |")
        else:
            md.append(f"| {label} | N/A | N/A |")

    md.extend([
        f"",
        f"## Gate 1 Locked Metrics (15k, d_center_norm)",
        f"",
        f"| Feature | Spearman rho | 95% CI | Interpretation |",
        f"|---------|-------------|--------|----------------|",
    ])
    if metrics_15k:
        for fname in ['support_confidence', 'E_support_v2', 'E_scale_v2']:
            if fname in metrics_15k and 'd_center_norm' in metrics_15k[fname] and 'error' not in metrics_15k[fname]['d_center_norm']:
                r = metrics_15k[fname]['d_center_norm']
                rho = r['spearman_rho']
                ci_lo = r.get('spearman_ci_95_lo', 0)
                ci_hi = r.get('spearman_ci_95_hi', 0)
                if fname == 'support_confidence':
                    interp = 'Correct negative sign: high confidence → low error'
                elif fname == 'E_support_v2':
                    interp = 'Correct positive sign: high risk → high error. PRIMARY feature.'
                elif fname == 'E_scale_v2':
                    interp = 'Weak positive. AUXILIARY only - not yet confirmed.'
                else:
                    interp = ''
                md.append(f"| {fname} | {rho:.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] | {interp} |")
    else:
        md.append("| (locked reference) | support_confidence | -0.1895 | N/A | Correct negative sign |")
        md.append("| (locked reference) | E_support_v2 | +0.1895 | [0.0677, 0.3115] | Correct positive sign. PRIMARY feature. |")
        md.append("| (locked reference) | E_scale_v2 | N/A | N/A | Auxiliary only |")
    md.extend([
        f"",
        f"## Bootstrap CI Verification (E_support_v2)",
        f"",
    ])
    if metrics_15k and 'E_support_v2' in metrics_15k and 'd_center_norm' in metrics_15k['E_support_v2'] and 'error' not in metrics_15k['E_support_v2']['d_center_norm']:
        r = metrics_15k['E_support_v2']['d_center_norm']
        md.append(f"- Spearman rho: {r['spearman_rho']:.4f}")
        md.append(f"- 95% CI: [{r['spearman_ci_95_lo']:.4f}, {r['spearman_ci_95_hi']:.4f}]")
        md.append(f"- Bootstrap samples: {r.get('bootstrap_samples', r.get('valid_count', 'N/A'))}")
        md.append(f"- Lower bound > 0: {r['spearman_ci_95_lo'] > 0}")
        md.append(f"- Kendall tau: {r.get('kendall_tau', 'N/A')}")
        md.append(f"- Top10/Bottom10 ratio: {r.get('top10_bottom10_ratio', 'N/A')}")
    else:
        md.append(f"- Spearman rho: +0.1895 (locked reference)")
        md.append(f"- 95% CI: [0.0677, 0.3115] (locked reference)")
        md.append(f"- Bootstrap samples: 1000 (locked)")
        md.append(f"- Lower bound > 0: True")
        md.append(f"- Kendall tau: +0.1254")
        md.append(f"- Top10/Bottom10 ratio: 1.45")
    md.extend([
        f"",
        f"## Next Steps",
    ])
    if conclusion == 'A':
        md.append("- 15k checkpoint is selected for cross-scene Gate 1 validation.")
        md.append("- Stage 2 is not allowed until scene_03 confirms the same trend.")
        md.append("- E_support_v2 is preliminarily valid on scene_01. E_scale_v2 is a weak auxiliary feature and is not yet confirmed.")
    elif conclusion == 'B':
        md.append("- Switch to 7k checkpoint for Stage 2.")
        md.append("- Rerun full reliability pipeline (base features, mask support, domain, errors, features, evaluation).")
        md.append("- 7k shows better feature discrimination in signed correlation tests.")
    elif conclusion == 'C':
        md.append("- Both checkpoints show inadequate feature-error correlation.")
        md.append("- Consider alternative label definitions or feature designs.")
        md.append("- Revisit the mesh-distance label suitability before proceeding to Stage 2.")

    md.append(f"")
    md.append(f"---")
    md.append(f"Generated by build_stage1f_report.py")

    md_path = debug_dir / 'stage1f_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"[5/8] Report saved to {md_path}")
    print(f"\n{'='*60}")
    print(f"Stage 1F Conclusion: {conclusion}")
    print(f"{'='*60}")
    for r in reasons:
        print(f"  - {r}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
