import argparse, sys, os, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

FEATURE_NAMES = ['E_mask', 'E_normal', 'E_depth', 'E_support', 'E_scale']
RISK_NAMES = FEATURE_NAMES + ['risk_scores']

def compute_percentiles(arr):
    p = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {str(k): float(v) for k, v in zip(p, np.percentile(arr, p))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir'])
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading all features, risk scores, and GT errors...")
    feat = np.load(out_dir / 'reliability_features.npz')
    R = feat['risk_scores']
    gt = np.load(out_dir / 'gaussian_gt_errors.npz')
    dist = gt['mesh_distance']
    d_norm = gt['normalized_mesh_distance']
    correct_th = cfg['mesh_evaluation']['correct_threshold_norm']
    wrong_th = cfg['mesh_evaluation']['wrong_threshold_norm']
    labels = np.where(d_norm < correct_th, 0, np.where(d_norm < wrong_th, 1, 2))
    is_wrong = (labels == 2).astype(int)
    n = len(dist)
    print(f"  Num Gaussians: {n}, wrong ratio: {is_wrong.mean():.4f}")

    print("[2/5] Computing per-feature statistics...")
    all_metrics = {}
    anomalies = []
    for name in RISK_NAMES:
        if name == 'risk_scores':
            arr = R
        else:
            arr = feat[name]
        arr = np.asarray(arr).ravel()
        stats = {
            'min': float(arr.min()),
            'max': float(arr.max()),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'percentiles': compute_percentiles(arr),
            'nan_count': int(np.isnan(arr).sum()),
            'inf_count': int(np.isinf(arr).sum()),
            'unique_count': int(len(np.unique(arr))),
            'zero_ratio': float((arr == 0).sum() / n),
            'one_ratio': float((arr == 1).sum() / n),
        }
        valid = arr[np.isfinite(arr)]
        if len(np.unique(is_wrong)) > 1 and len(np.unique(valid)) > 1:
            auroc = float(roc_auc_score(is_wrong, valid))
        else:
            auroc = 0.5
        auprc = float(average_precision_score(is_wrong, valid))
        rho, pval = spearmanr(valid, dist[np.isfinite(arr)])
        rho = float(rho) if not np.isnan(rho) else 0.0

        sorted_idx = np.argsort(arr)
        k10 = max(1, n // 10)
        top10_err = float(is_wrong[sorted_idx[-k10:]].mean())
        bottom10_err = float(is_wrong[sorted_idx[:k10]].mean())

        all_metrics[name] = {
            'stats': stats,
            'spearman_vs_mesh_distance': rho,
            'auroc_vs_wrong': auroc,
            'auprc_vs_wrong': auprc,
            'top10_error_rate': top10_err,
            'bottom10_error_rate': bottom10_err,
            'auprc_over_random': auprc - float(is_wrong.mean()),
        }

        if stats['std'] < 1e-4:
            anomalies.append(f"{name}: std={stats['std']:.2e} < 1e-4")
        if stats['nan_count'] > 0 or stats['inf_count'] > 0:
            anomalies.append(f"{name}: {stats['nan_count']} NaN, {stats['inf_count']} Inf")
        if rho < 0:
            anomalies.append(f"{name}: negative Spearman rho={rho:.4f}")
        if stats['zero_ratio'] > 0.90:
            anomalies.append(f"{name}: {stats['zero_ratio']*100:.1f}% zeros > 90%")

        print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, auroc={auroc:.4f}, auprc={auprc:.4f}, spearman={rho:.4f}")

    print("[3/5] Computing correlation matrix...")
    feat_arrays = []
    for name in FEATURE_NAMES:
        feat_arrays.append(np.asarray(feat[name]).ravel())
    feat_stack = np.stack(feat_arrays, axis=1)
    corr = np.corrcoef(feat_stack.T)
    for i in range(len(FEATURE_NAMES)):
        for j in range(i+1, len(FEATURE_NAMES)):
            if abs(corr[i, j]) > 0.95:
                anomalies.append(f"High correlation {FEATURE_NAMES[i]}-{FEATURE_NAMES[j]}: {corr[i,j]:.4f} > 0.95")

    print("[4/5] Saving CSV report and anomaly list...")
    import csv
    with open(debug_dir / 'feature_correlation.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow([''] + FEATURE_NAMES)
        for i, name in enumerate(FEATURE_NAMES):
            w.writerow([name] + [f'{corr[i,j]:.6f}' for j in range(len(FEATURE_NAMES))])

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(FEATURE_NAMES)))
        ax.set_yticks(range(len(FEATURE_NAMES)))
        ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(FEATURE_NAMES)
        for i in range(len(FEATURE_NAMES)):
            for j in range(len(FEATURE_NAMES)):
                ax.text(j, i, f'{corr[i,j]:.3f}', ha='center', va='center', fontsize=8)
        fig.colorbar(im)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        fig.savefig(str(debug_dir / 'feature_correlation.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  Correlation plot skipped: {e}")

    diagnosis = {
        'scene_name': cfg['scene_name'],
        'num_gaussians': n,
        'features': all_metrics,
        'correlation_matrix': {FEATURE_NAMES[i]: {FEATURE_NAMES[j]: float(corr[i,j]) for j in range(len(FEATURE_NAMES))} for i in range(len(FEATURE_NAMES))},
        'anomalies': anomalies,
    }
    with open(debug_dir / 'feature_diagnosis.json', 'w') as f:
        json.dump(diagnosis, f, indent=2)

    print("[5/5] Generating report...")
    md = [f"# Feature Diagnosis - {cfg['scene_name']}", ""]
    md.append(f"Total Gaussians: {n}")
    md.append(f"Wrong ratio: {is_wrong.mean():.4f}")
    md.append("")
    for name in RISK_NAMES:
        m = all_metrics[name]
        md.append(f"## {name}")
        md.append(f"| Stat | Value |")
        md.append(f"|------|-------|")
        for k, v in m['stats'].items():
            if isinstance(v, dict):
                md.append(f"| {k} | {v} |")
            else:
                md.append(f"| {k} | {v} |")
        md.append(f"| spearman | {m['spearman_vs_mesh_distance']:.4f} |")
        md.append(f"| auroc | {m['auroc_vs_wrong']:.4f} |")
        md.append(f"| auprc | {m['auprc_vs_wrong']:.4f} |")
        md.append(f"| auprc_over_random | {m['auprc_over_random']:.4f} |")
        md.append(f"| top10_error_rate | {m['top10_error_rate']:.4f} |")
        md.append(f"| bottom10_error_rate | {m['bottom10_error_rate']:.4f} |")
        md.append("")

    md.append("## Anomalies")
    if anomalies:
        for a in anomalies:
            md.append(f"- {a}")
    else:
        md.append("None detected.")
    md.append("")
    md.append("## Files Saved")
    md.append("- feature_diagnosis.json, feature_correlation.csv, feature_correlation.png")

    with open(debug_dir / 'feature_diagnosis_report.md', 'w') as f:
        f.write('\n'.join(md))

    print("Done.")

if __name__ == '__main__':
    main()
