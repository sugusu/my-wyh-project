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
    fig_dir = Path(cfg['figure_output_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading risk scores and GT errors...")
    R = np.load(out_dir / 'risk_scores.npy')
    gt = np.load(out_dir / 'gaussian_gt_errors.npz')
    dist = gt['mesh_distance']
    d_norm = gt['normalized_mesh_distance']

    correct_th = cfg['mesh_evaluation']['correct_threshold_norm']
    wrong_th = cfg['mesh_evaluation']['wrong_threshold_norm']
    labels = np.where(d_norm < correct_th, 0, np.where(d_norm < wrong_th, 1, 2))
    is_wrong = (labels == 2).astype(int)

    feat = np.load(out_dir / 'reliability_features.npz')
    feature_names = ['E_mask', 'E_normal', 'E_depth', 'E_support', 'E_scale']
    top_ratios = cfg['evaluation']['top_ratios']

    print("[2/5] Computing metrics per feature...")
    metrics = {}
    bg_ratio = is_wrong.mean()
    for name in feature_names + ['risk_scores']:
        if name == 'risk_scores':
            score = R
        else:
            score = feat[name]

        auroc = roc_auc_score(is_wrong, score) if len(np.unique(is_wrong)) > 1 else 0.5
        auprc = average_precision_score(is_wrong, score)
        rho, _ = spearmanr(score, dist)
        metrics[name] = {
            'auroc': float(auroc),
            'auprc': float(auprc),
            'auprc_over_random': float(auprc - bg_ratio),
            'spearman': float(rho),
        }

        for ratio in top_ratios:
            k = max(1, int(len(score) * ratio))
            top_k = np.argpartition(score, -k)[-k:]
            top_mean_dist = float(dist[top_k].mean())
            bottom_k = np.argpartition(score, k)[:k]
            bottom_mean_dist = float(dist[bottom_k].mean())
            top_wrong_rate = float(is_wrong[top_k].mean())
            metrics[name][f'top{int(ratio*100)}_mean_dist'] = top_mean_dist
            metrics[name][f'bottom{int(ratio*100)}_mean_dist'] = bottom_mean_dist
            metrics[name][f'top{int(ratio*100)}_wrong_rate'] = top_wrong_rate

    metrics['random_baseline'] = {
        'auprc_random': float(bg_ratio),
        'overall_wrong_ratio': float(bg_ratio),
    }

    print("[3/5] Generating report...")
    lines = [f"# Gate 1A Reliability Report - {cfg['scene_name']}", ""]
    lines.append(f"| Metric | {' | '.join(feature_names)} | Risk |")
    lines.append(f"|--------|{'-'*20}|{' '*10}|")
    for m in ['auroc', 'auprc', 'spearman']:
        vals = [f"{metrics[n][m]:.4f}" for n in feature_names + ['risk_scores']]
        lines.append(f"| {m} | {' | '.join(vals)} |")
    lines.append(f"| random_baseline_auprc | {bg_ratio:.4f} |" + " |"*5)
    lines.append("")
    lines.append("### Gate 1A Check")
    checks = [
        ("1. AUPRC > random ratio", metrics['risk_scores']['auprc'] > bg_ratio * 1.2),
        ("2. Top-10% wrong rate > overall * 1.5",
         metrics['risk_scores']['top10_wrong_rate'] > bg_ratio * 1.5),
        ("3. Top-10% dist > Bottom-10% dist",
         metrics['risk_scores']['top10_mean_dist'] > metrics['risk_scores']['bottom10_mean_dist'] * 1.5),
    ]
    passed = 0
    for desc, ok in checks:
        lines.append(f"- {'✅' if ok else '❌'} {desc}")
        if ok:
            passed += 1
    lines.append(f"\n**Result: {passed}/{len(checks)} checks passed**")
    gate_passed = passed >= 2
    lines.append(f"\n**Gate 1A: {'PRELIMINARY PASS' if gate_passed else 'FAIL'}**")
    lines.append(f"\n**Note:** Gate 1A is single-scene only. Final Gate 1 requires scene_01 + scene_03.")

    with open(out_dir / 'gate1a_report.md', 'w') as f:
        f.write('\n'.join(lines))

    print("[4/5] Generating figures...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        order = np.argsort(R)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(R[order], dist[order], s=1, alpha=0.3)
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Mesh Distance')
        ax.set_title('Risk vs Mesh Distance')
        fig.savefig(fig_dir / 'risk_vs_mesh_distance.png', dpi=150)
        plt.close(fig)

        ratios = np.linspace(0.01, 0.5, 50)
        mean_dists = [dist[np.argpartition(R, -int(len(R)*r))[-int(len(R)*r):]].mean() for r in ratios]
        fig, ax = plt.subplots()
        ax.plot(ratios*100, mean_dists)
        ax.set_xlabel('Top Risk Ratio (%)')
        ax.set_ylabel('Mean Mesh Distance')
        ax.set_title('Risk Quantile Error Curve')
        fig.savefig(fig_dir / 'risk_quantile_error_curve.png', dpi=150)
        plt.close(fig)
        print("  Figures saved.")
    except Exception as e:
        print(f"  Figure generation skipped: {e}")

    save_json(metrics, out_dir / 'reliability_metrics.json')
    import csv
    with open(out_dir / 'reliability_table.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['feature', 'auroc', 'auprc', 'spearman', 'top10_wrong_rate', 'top10_mean_dist'])
        for name in feature_names + ['risk_scores']:
            m = metrics[name]
            w.writerow([name, m['auroc'], m['auprc'], m['spearman'], m.get('top10_wrong_rate', 0), m.get('top10_mean_dist', 0)])
    print(f"[5/5] Evaluation complete. Gate 1A: {'PASS' if gate_passed else 'FAIL'}")

if __name__ == '__main__':
    main()
