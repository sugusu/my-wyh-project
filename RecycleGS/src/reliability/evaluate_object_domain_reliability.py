import argparse, sys, os, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
sys.path.insert(0, '/data/wyh/RecycleGS/src')
from recyclegs.config import load_config, save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1c_scene01'
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading object domain data...")
    object_indices = np.load(out_dir / 'object_indices.npy')
    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = base['xyz']
    N = len(xyz)

    gt = np.load(out_dir / 'object_gaussian_gt_errors.npz')
    dist = gt['mesh_distance']
    d_norm = gt['normalized_mesh_distance']

    correct_th = cfg['mesh_evaluation']['correct_threshold_norm']
    wrong_th = cfg['mesh_evaluation']['wrong_threshold_norm']
    labels = np.where(d_norm < correct_th, 0, np.where(d_norm < wrong_th, 1, 2))
    is_wrong = (labels == 2).astype(int)

    print("[2/6] Loading risk scores...")
    R_A = np.load(out_dir / 'object_risk_A.npy')[object_indices]
    R_B = np.load(out_dir / 'object_risk_B.npy')[object_indices]
    R_C = np.load(out_dir / 'object_risk_C.npy')[object_indices]

    feat = np.load(out_dir / 'object_risk_features.npz')
    feature_names_obj = ['E_support_norm', 'E_scale_norm', 'E_normal_norm']

    print("[3/6] Computing metrics...")
    top_ratios = cfg['evaluation']['top_ratios']
    risk_names = ['R_A', 'R_B', 'R_C']
    risk_scores = {'R_A': R_A, 'R_B': R_B, 'R_C': R_C}

    metrics = {}
    bg_ratio = is_wrong.mean()

    for name in feature_names_obj + risk_names:
        if name in risk_scores:
            score = risk_scores[name]
        else:
            score = feat[name]
        valid = ~(np.isnan(score) | np.isinf(score))
        if valid.sum() < 10:
            metrics[name] = {'error': 'insufficient valid scores'}
            continue
        s, d = score[valid], dist[valid]
        w = is_wrong[valid]

        auroc = roc_auc_score(w, s) if len(np.unique(w)) > 1 else 0.5
        auprc = average_precision_score(w, s)
        rho, _ = spearmanr(s, d)

        metrics[name] = {
            'auroc': float(auroc),
            'auprc': float(auprc),
            'auprc_over_random': float(auprc - bg_ratio),
            'spearman': float(rho),
            'valid_count': int(valid.sum()),
        }

        for ratio in top_ratios:
            k = max(1, int(len(s) * ratio))
            top_k = np.argpartition(s, -k)[-k:]
            top_mean_dist = float(d[top_k].mean())
            bottom_k = np.argpartition(s, k)[:k]
            bottom_mean_dist = float(d[bottom_k].mean())
            top_wrong_rate = float(w[top_k].mean())
            bottom_wrong_rate = float(w[bottom_k].mean())
            metrics[name][f'top{int(ratio*100)}_mean_dist'] = top_mean_dist
            metrics[name][f'bottom{int(ratio*100)}_mean_dist'] = bottom_mean_dist
            metrics[name][f'top{int(ratio*100)}_wrong_rate'] = top_wrong_rate
            metrics[name][f'bottom{int(ratio*100)}_wrong_rate'] = bottom_wrong_rate

    metrics['random_baseline'] = {
        'auprc_random': float(bg_ratio),
        'overall_wrong_ratio': float(bg_ratio),
    }

    print("[4/6] Loading old full-scene results for comparison...")
    try:
        old_metrics = np.load(out_dir / 'reliability_metrics.json', allow_pickle=True)
        if isinstance(old_metrics, dict):
            metrics['old_full_scene'] = {
                'risk_auroc': old_metrics.get('risk_scores', {}).get('auroc', None),
                'risk_auprc': old_metrics.get('risk_scores', {}).get('auprc', None),
                'risk_spearman': old_metrics.get('risk_scores', {}).get('spearman', None),
            }
    except Exception as e:
        print(f"  Could not load old metrics: {e}")
        metrics['old_full_scene'] = None

    save_json(metrics, out_dir / 'object_reliability_metrics.json')

    print("[5/6] Generating report...")

    domain_stats_str = ""
    try:
        part_stats = np.load(out_dir / 'domain_partition_stats.json', allow_pickle=True)
        if isinstance(part_stats, dict):
            total = part_stats.get('total_gaussians', N)
            domain_stats_str = f"""## Domain Partition Stats
- Total Gaussians: {total}
- Object: {part_stats.get('object_supported', {}).get('count', 'N/A')} ({part_stats.get('object_supported', {}).get('ratio', 0)*100:.1f}%)
- Background: {part_stats.get('background_supported', {}).get('count', 'N/A')} ({part_stats.get('background_supported', {}).get('ratio', 0)*100:.1f}%)
- Uncertain: {part_stats.get('uncertain', {}).get('count', 'N/A')} ({part_stats.get('uncertain', {}).get('ratio', 0)*100:.1f}%)
"""
    except Exception as e:
        domain_stats_str = f"## Domain Partition Stats\n- Could not load: {e}\n"

    lines = [
        f"# Gate 1C Report - Object Domain Reliability - {cfg['scene_name']}",
        f"",
        f"## Domain Partition",
        f"",
        domain_stats_str,
    ]

    lines.extend([
        f"",
        f"## Object Domain Reliability Metrics",
        f"",
        f"| Metric | E_support | E_scale | E_normal | R_A | R_B | R_C |",
        f"|--------|-----------|---------|----------|-----|-----|-----|",
    ])
    for m in ['auroc', 'auprc', 'spearman']:
        vals = []
        for name in feature_names_obj + risk_names:
            if name in metrics and m in metrics[name]:
                vals.append(f"{metrics[name][m]:.4f}")
            else:
                vals.append("N/A")
        lines.append(f"| {m} | {' | '.join(vals)} |")
    lines.append(f"| random_baseline_auprc | {bg_ratio:.4f} |" + " |"*5)
    lines.append("")

    lines.append("### Top-10% Wrong Rate")
    vals = []
    for name in feature_names_obj + risk_names:
        if name in metrics and 'top10_wrong_rate' in metrics[name]:
            vals.append(f"{metrics[name]['top10_wrong_rate']:.4f}")
        else:
            vals.append("N/A")
    lines.append(f"| top10_wrong_rate | {' | '.join(vals)} |")

    lines.append("")
    lines.append("### Top-10% Mean Distance vs Bottom-10% Mean Distance")
    for name in feature_names_obj + risk_names:
        if name in metrics and 'top10_mean_dist' in metrics[name]:
            t = metrics[name]['top10_mean_dist']
            b = metrics[name]['bottom10_mean_dist']
            ratio = t / max(b, 1e-8)
            lines.append(f"- **{name}**: top={t:.4f}, bottom={b:.4f}, ratio={ratio:.2f}")

    if metrics.get('old_full_scene'):
        old = metrics['old_full_scene']
        lines.extend([
            "",
            "## Comparison with Full-Scene (Stage 1A)",
            f"- Old full-scene Risk AUROC: {old.get('risk_auroc', 'N/A')}",
            f"- Old full-scene Risk AUPRC: {old.get('risk_auprc', 'N/A')}",
        ])

    lines.extend([
        "",
        "## Gate 1C Check",
        "",
    ])
    checks = [
        ("1. R_C AUPRC > random ratio",
         metrics.get('R_C', {}).get('auprc', 0) > bg_ratio * 1.2),
        ("2. Top-10% wrong rate > overall * 1.5 (R_C)",
         metrics.get('R_C', {}).get('top10_wrong_rate', 0) > bg_ratio * 1.5),
        ("3. Top-10% dist > Bottom-10% dist (R_C)",
         metrics.get('R_C', {}).get('top10_mean_dist', 0) > metrics.get('R_C', {}).get('bottom10_mean_dist', 0) * 1.5),
    ]
    passed = 0
    for desc, ok in checks:
        lines.append(f"- {'✅' if ok else '❌'} {desc}")
        if ok:
            passed += 1
    lines.append(f"\n**Result: {passed}/{len(checks)} checks passed**")
    gate_passed = passed >= 2
    lines.append(f"\n**Gate 1C: {'PRELIMINARY PASS' if gate_passed else 'FAIL'}**")
    lines.append("")
    lines.append("## Next Steps")
    if gate_passed:
        lines.append("- scene_01 object domain passed gate.")
        lines.append("- Apply same parameters to scene_03.")
    else:
        lines.append("- Pause prune, re-evaluate single-Gaussian reliability labels and feature definitions.")
    lines.append("- Still not entering Stage 2.")

    report_path = debug_dir / 'gate1c_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[6/6] Report saved to {report_path}")

    print(f"\n{'='*60}")
    print(f"Gate 1C Results:")
    print(f"  R_C AUROC: {metrics.get('R_C', {}).get('auroc', 'N/A')}")
    print(f"  R_C AUPRC: {metrics.get('R_C', {}).get('auprc', 'N/A')}")
    print(f"  R_C Spearman: {metrics.get('R_C', {}).get('spearman', 'N/A')}")
    print(f"  Checks passed: {passed}/{len(checks)}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
