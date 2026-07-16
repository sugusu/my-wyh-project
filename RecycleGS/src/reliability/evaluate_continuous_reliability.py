import argparse, sys, os, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

def bootstrap_spearman_ci(x, y, n_bootstrap=1000, ci_level=0.95):
    n = len(x)
    rhos = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            r, _ = spearmanr(x[idx], y[idx])
            rhos.append(r if not np.isnan(r) else 0.0)
        else:
            rhos.append(0.0)
    rhos = np.array(rhos)
    lo = np.percentile(rhos, (1 - ci_level) / 2 * 100)
    hi = np.percentile(rhos, (1 + ci_level) / 2 * 100)
    return float(lo), float(hi)

def ndcg_score(y_true, y_score, k):
    n = len(y_true)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    idx = np.argsort(y_score)[::-1][:k]
    d = y_true[idx]
    ideal = np.sort(y_true)[::-1][:k]
    dcg = d[0] + np.sum(d[1:] / np.log2(np.arange(2, len(d) + 1)))
    idcg = ideal[0] + np.sum(ideal[1:] / np.log2(np.arange(2, len(ideal) + 1)))
    return float(dcg / idcg) if idcg > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1d_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    print("[1/6] Loading data...")
    candidate_indices = np.load(out_dir / 'candidate_object_indices.npy')

    E_normal = np.load(out_dir / 'object_normal_conflict.npy')[candidate_indices]
    E_normal_valid = np.load(out_dir / 'object_normal_valid.npy')[candidate_indices]
    E_support = np.load(out_dir / 'object_surface_support_risk.npy')[candidate_indices]
    E_scale = np.load(out_dir / 'object_scale_anomaly.npy')[candidate_indices]

    err = np.load(out_dir / 'candidate_geometry_errors_v2.npz')
    continuous_metrics = {
        'd_center_norm': err['d_center_norm'],
        'd_scale': err['d_scale'],
        'd_surface_proxy_alpha1': err['d_surface_proxy_alpha1'],
        'd_surface_proxy_alpha2': err['d_surface_proxy_alpha2'],
    }

    print(f"  Candidate Gaussians: {len(candidate_indices)}")
    print(f"  Normal valid: {E_normal_valid.sum()}/{len(E_normal_valid)}")

    print("[2/6] Computing risk scores...")
    from scipy.stats import percentileofscore
    def percentile_clip_and_scale(arr):
        lo, hi = np.percentile(arr, [5, 95])
        clipped = arr.clip(lo, hi)
        scaled = (clipped - lo) / (hi - lo + 1e-8)
        return scaled.clip(0, 1)

    E_support_norm = percentile_clip_and_scale(E_support)
    E_scale_norm = percentile_clip_and_scale(E_scale)

    E_normal_norm = np.full_like(E_normal, np.nan)
    valid_mask = E_normal_valid
    if valid_mask.sum() > 0:
        E_normal_norm[valid_mask] = percentile_clip_and_scale(E_normal[valid_mask])

    R_A = 0.70 * E_support_norm + 0.30 * E_scale_norm

    R_B = np.full(len(candidate_indices), np.nan)
    if valid_mask.sum() > 0:
        R_B[valid_mask] = 0.45 * E_normal_norm[valid_mask] + 0.55 * E_support_norm[valid_mask]

    R_C = np.zeros(len(candidate_indices))
    R_C[valid_mask] = 0.35 * E_normal_norm[valid_mask] + 0.45 * E_support_norm[valid_mask] + 0.20 * E_scale_norm[valid_mask]
    invalid_c = ~valid_mask
    if invalid_c.sum() > 0:
        total_w = 0.45 + 0.20
        R_C[invalid_c] = 0.45 / total_w * E_support_norm[invalid_c] + 0.20 / total_w * E_scale_norm[invalid_c]

    risk_scores_map = {
        'E_normal': np.where(valid_mask, E_normal_norm, np.nan),
        'E_support': E_support_norm,
        'E_scale': E_scale_norm,
        'R_A': R_A,
        'R_B': R_B,
        'R_C': R_C,
    }

    print("[3/6] Computing correlation metrics...")
    top_ratios = cfg['evaluation']['top_ratios']
    bootstrap_samples = cfg['evaluation']['bootstrap_samples']
    ci_level = cfg['evaluation']['confidence_level']

    results = {}
    for score_name, score in risk_scores_map.items():
        results[score_name] = {}
        for metric_name, metric in continuous_metrics.items():
            valid = ~(np.isnan(score) | np.isinf(score) | np.isnan(metric))
            if valid.sum() < 5:
                results[score_name][metric_name] = {'error': 'insufficient valid data', 'valid_count': int(valid.sum())}
                continue

            s, m = score[valid], metric[valid]
            n_valid = len(s)

            rho, _ = spearmanr(s, m)
            rho = float(rho) if not np.isnan(rho) else 0.0
            tau, _ = kendalltau(s, m)
            tau = float(tau) if not np.isnan(tau) else 0.0
            rho_lo, rho_hi = bootstrap_spearman_ci(s, m, bootstrap_samples, ci_level)

            metric_res = {
                'valid_count': int(n_valid),
                'spearman_rho': rho,
                'spearman_ci_lo': rho_lo,
                'spearman_ci_hi': rho_hi,
                'kendall_tau': tau,
            }

            for ratio in top_ratios:
                k = max(1, int(n_valid * ratio))
                top_k = np.argpartition(s, -k)[-k:]
                bottom_k = np.argpartition(s, k)[:k]
                metric_res[f'top{int(ratio*100)}_mean_error'] = float(m[top_k].mean())
                metric_res[f'bottom{int(ratio*100)}_mean_error'] = float(m[bottom_k].mean())

            k10 = max(1, int(n_valid * 0.10))
            top10 = np.argpartition(s, -k10)[-k10:]
            bottom10 = np.argpartition(s, k10)[:k10]
            metric_res['top10_bottom10_ratio'] = float(m[top10].mean() / max(m[bottom10].mean(), 1e-8))

            for ratio in top_ratios:
                k = max(1, int(n_valid * ratio))
                ndcg = ndcg_score(m, s, k)
                metric_res[f'ndcg_{int(ratio*100)}pct'] = ndcg

            results[score_name][metric_name] = metric_res

    print("[4/6] Saving metrics...")
    with open(out_dir / 'continuous_reliability_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("[5/6] Generating report...")
    lines = [
        f"# Continuous Reliability Evaluation - {cfg['scene_name']}",
        f"",
        f"## Domain: Candidate Object ({len(candidate_indices)} Gaussians)",
        f"",
        f"### Correlation with Continuous Errors",
        f"",
    ]
    for score_name in ['E_normal', 'E_support', 'E_scale', 'R_A', 'R_B', 'R_C']:
        lines.append(f"#### {score_name}")
        lines.append(f"| Metric | Spearman rho | CI (95%) | Kendall tau | NDCG@5% | NDCG@10% | NDCG@20% |")
        lines.append(f"|--------|-------------|----------|-------------|---------|----------|----------|")
        for metric_name in ['d_center_norm', 'd_scale', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']:
            if score_name not in results or metric_name not in results[score_name]:
                continue
            r = results[score_name][metric_name]
            if 'error' in r:
                lines.append(f"| {metric_name} | {r['error']} |")
                continue
            lines.append(f"| {metric_name} | {r['spearman_rho']:.4f} | [{r['spearman_ci_lo']:.4f}, {r['spearman_ci_hi']:.4f}] | {r['kendall_tau']:.4f} | {r.get('ndcg_5pct', 'N/A')} | {r.get('ndcg_10pct', 'N/A')} | {r.get('ndcg_20pct', 'N/A')} |")
        lines.append(f"")

    lines.extend([
        f"### Top/Bottom Mean Error Ratios",
        f"| Score | Metric | Top5% | Bottom5% | Top10% | Bottom10% | Top20% | Bottom20% | Top10/Bottom10 |",
        f"|-------|--------|-------|----------|--------|-----------|--------|-----------|----------------|",
    ])
    for score_name in ['E_normal', 'E_support', 'E_scale', 'R_A', 'R_B', 'R_C']:
        for metric_name in ['d_center_norm', 'd_scale', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']:
            if score_name not in results or metric_name not in results[score_name]:
                continue
            r = results[score_name][metric_name]
            if 'error' in r:
                continue
            lines.append(f"| {score_name} | {metric_name} | {r.get('top5_mean_error', 'N/A'):.4f} | {r.get('bottom5_mean_error', 'N/A'):.4f} | {r.get('top10_mean_error', 'N/A'):.4f} | {r.get('bottom10_mean_error', 'N/A'):.4f} | {r.get('top20_mean_error', 'N/A'):.4f} | {r.get('bottom20_mean_error', 'N/A'):.4f} | {r.get('top10_bottom10_ratio', 'N/A'):.2f} |")

    report_path = out_dir / 'continuous_reliability_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Report saved to {report_path}")

    print("[6/6] Generating figures...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        quantiles = np.linspace(0.001, 0.999, 100)
        for name in ['R_A', 'R_B', 'R_C']:
            if name not in risk_scores_map:
                continue
            s = risk_scores_map[name]
            valid = np.isfinite(s)
            if valid.sum() > 10:
                qvals = np.quantile(s[valid], quantiles)
                ax1.plot(quantiles, qvals, label=name)
        ax1.set_xlabel('Quantile')
        ax1.set_ylabel('Risk Score')
        ax1.set_title('Candidate Domain Risk Score Quantile Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.savefig(str(debug_dir / 'candidate_risk_quantile_curve.png'), dpi=150)
        plt.close(fig1)

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        for idx, (risk_name, metric_name) in enumerate([
            ('R_A', 'd_center_norm'), ('R_B', 'd_center_norm'), ('R_C', 'd_center_norm')]):
            if risk_name not in risk_scores_map:
                continue
            s = risk_scores_map[risk_name]
            m = continuous_metrics[metric_name]
            valid = np.isfinite(s) & np.isfinite(m)
            if valid.sum() > 10:
                axes2[idx].scatter(s[valid], m[valid], s=1, alpha=0.3)
                axes2[idx].set_xlabel(f'{risk_name} score')
                axes2[idx].set_ylabel(metric_name)
                axes2[idx].set_title(f'{risk_name} vs {metric_name}')
                axes2[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(str(debug_dir / 'candidate_risk_vs_center_distance.png'), dpi=150)
        plt.close(fig2)

        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
        for idx, (risk_name, metric_name) in enumerate([
            ('R_A', 'd_surface_proxy_alpha1'), ('R_B', 'd_surface_proxy_alpha1'), ('R_C', 'd_surface_proxy_alpha1')]):
            if risk_name not in risk_scores_map:
                continue
            s = risk_scores_map[risk_name]
            m = continuous_metrics[metric_name]
            valid = np.isfinite(s) & np.isfinite(m)
            if valid.sum() > 10:
                axes3[idx].scatter(s[valid], m[valid], s=1, alpha=0.3)
                axes3[idx].set_xlabel(f'{risk_name} score')
                axes3[idx].set_ylabel(metric_name)
                axes3[idx].set_title(f'{risk_name} vs {metric_name}')
                axes3[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        fig3.savefig(str(debug_dir / 'candidate_risk_vs_surface_proxy.png'), dpi=150)
        plt.close(fig3)
        print(f"  Figures saved to {debug_dir}")
    except Exception as e:
        print(f"  Figure generation skipped: {e}")

    print("Done.")

if __name__ == '__main__':
    main()
