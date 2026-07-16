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
    parser.add_argument('--iteration', type=int, required=True, choices=[7000, 15000])
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1f_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    iteration = args.iteration
    iter_str = f"iter_{iteration}"
    iter_dir = out_dir / iter_str

    print(f"[{iter_str}] Loading data...")
    err = np.load(iter_dir / 'geometry_errors.npz')
    support_confidence = np.load(iter_dir / 'support_confidence_v2.npy')
    E_support_v2 = np.load(iter_dir / 'support_risk_v2.npy')

    scale_min = np.load(iter_dir / 'scale_min.npy') if (iter_dir / 'scale_min.npy').exists() else None
    scale_max = np.load(iter_dir / 'scale_max.npy') if (iter_dir / 'scale_max.npy').exists() else None
    scale_ratio = np.load(iter_dir / 'scale_ratio.npy') if (iter_dir / 'scale_ratio.npy').exists() else None
    scale_volume = np.load(iter_dir / 'scale_volume.npy') if (iter_dir / 'scale_volume.npy').exists() else None
    E_scale_v2 = np.load(iter_dir / 'scale_risk_v2.npy')

    normal_valid_path = debug_dir / f'normal_valid_subset_{iter_str}.json'
    normal_info = None
    if normal_valid_path.exists():
        with open(normal_valid_path) as f:
            normal_info = json.load(f)

    error_metrics = {
        'd_center_norm': err['d_center_norm'],
        'd_surface_proxy_alpha1': err['d_surface_proxy_alpha1'],
        'd_surface_proxy_alpha2': err['d_surface_proxy_alpha2'],
    }

    support_confidence_features = {
        'S_position': None,
        'S_normal': None,
        'S_combined': None,
    }

    features = {
        'support_confidence': support_confidence,
        'E_support_v2': E_support_v2,
        'E_scale_v2': E_scale_v2,
    }
    if scale_min is not None:
        features['scale_min'] = scale_min
    if scale_max is not None:
        features['scale_max'] = scale_max
    if scale_ratio is not None:
        features['scale_ratio'] = scale_ratio
    if scale_volume is not None:
        features['scale_volume'] = scale_volume

    if normal_info and normal_info.get('normal_global_usable', False):
        planarity_scores = normal_info.get('metrics', {})
        features['normal_planarity'] = None

    print(f"  Evaluating {len(features)} features...")

    results = {}
    for fname, farr in features.items():
        if farr is None:
            continue
        results[fname] = {}
        for ename, earr in error_metrics.items():
            valid = np.isfinite(farr) & np.isfinite(earr)
            if valid.sum() < 5:
                results[fname][ename] = {
                    'error': 'insufficient valid data',
                    'valid_count': int(valid.sum()),
                }
                continue

            s, m = farr[valid], earr[valid]
            nv = len(s)

            rho, _ = spearmanr(s, m)
            rho = float(rho) if not np.isnan(rho) else 0.0
            rho_lo, rho_hi = bootstrap_spearman_ci(s, m)

            tau, _ = kendalltau(s, m)
            tau = float(tau) if not np.isnan(tau) else 0.0

            metric_res = {
                'valid_count': int(nv),
                'valid_ratio': float(nv / max(len(s), 1)),
                'spearman_rho': rho,
                'spearman_ci_95_lo': rho_lo,
                'spearman_ci_95_hi': rho_hi,
                'kendall_tau': tau,
            }

            for k_pct in [5, 10, 20]:
                k = max(1, int(nv * k_pct / 100))
                top_k = np.argpartition(s, -k)[-k:]
                bottom_k = np.argpartition(s, k)[:k]
                metric_res[f'top{k_pct}_mean_error'] = float(m[top_k].mean())
                metric_res[f'bottom{k_pct}_mean_error'] = float(m[bottom_k].mean())

            k10 = max(1, int(nv * 0.10))
            top10 = np.argpartition(s, -k10)[-k10:]
            bottom10 = np.argpartition(s, k10)[:k10]
            metric_res['top10_bottom10_ratio'] = float(
                m[top10].mean() / max(m[bottom10].mean(), 1e-8)
            )

            for k_pct in [5, 10, 20]:
                k = max(1, int(nv * k_pct / 100))
                ndcg = ndcg_score(m, s, k)
                metric_res[f'ndcg_{k_pct}pct'] = ndcg

            results[fname][ename] = metric_res

    if normal_info and normal_info.get('normal_global_usable', False):
        print(f"  Normal is globally usable, including planarity-based evaluation...")

    json_path = iter_dir / 'signed_feature_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    md_lines = [
        f"# Signed Continuous Feature Metrics - {cfg['scene_name']} ({iter_str})",
        f"",
        f"## Features Evaluated",
    ]
    for fname in features:
        if features[fname] is not None:
            md_lines.append(f"- {fname}: {len(features[fname])} values")
    md_lines.extend([
        f"",
        f"## Signed Spearman Correlations",
        f"| Feature | Error Metric | Spearman rho | 95% CI | Kendall tau | NDCG@5 | NDCG@10 | NDCG@20 |",
        f"|--------|-------------|-------------|--------|-------------|--------|---------|---------|",
    ])
    for fname in results:
        for ename in error_metrics:
            if ename not in results[fname]:
                continue
            r = results[fname][ename]
            if 'error' in r:
                md_lines.append(f"| {fname} | {ename} | {r['error']} | | | | |")
            else:
                md_lines.append(
                    f"| {fname} | {ename} | {r['spearman_rho']:.4f} | "
                    f"[{r['spearman_ci_95_lo']:.4f}, {r['spearman_ci_95_hi']:.4f}] | "
                    f"{r['kendall_tau']:.4f} | "
                    f"{r.get('ndcg_5pct', 0):.4f} | {r.get('ndcg_10pct', 0):.4f} | "
                    f"{r.get('ndcg_20pct', 0):.4f} |"
                )

    md_lines.extend([
        f"",
        f"## Top/Bottom Mean Error",
        f"| Feature | Error | Top5 | Bottom5 | Top10 | Bottom10 | Top20 | Bottom20 | T10/B10 |",
        f"|--------|-------|------|---------|-------|----------|-------|----------|---------|",
    ])
    for fname in results:
        for ename in error_metrics:
            if ename not in results[fname]:
                continue
            r = results[fname][ename]
            if 'error' in r:
                continue
            md_lines.append(
                f"| {fname} | {ename} | "
                f"{r.get('top5_mean_error', 0):.6f} | {r.get('bottom5_mean_error', 0):.6f} | "
                f"{r.get('top10_mean_error', 0):.6f} | {r.get('bottom10_mean_error', 0):.6f} | "
                f"{r.get('top20_mean_error', 0):.6f} | {r.get('bottom20_mean_error', 0):.6f} | "
                f"{r.get('top10_bottom10_ratio', 0):.4f} |"
            )

    md_path = iter_dir / 'signed_feature_metrics.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"[{iter_str}] Signed feature metrics saved to {json_path}")
    print(f"[{iter_str}] Top signal summary:")
    for fname in results:
        if 'd_center_norm' in results[fname] and 'error' not in results[fname]['d_center_norm']:
            r = results[fname]['d_center_norm']
            print(f"  {fname}: rho={r['spearman_rho']:.4f} "
                  f"CI=[{r['spearman_ci_95_lo']:.4f}, {r['spearman_ci_95_hi']:.4f}] "
                  f"ratio={r['top10_bottom10_ratio']:.4f}")

if __name__ == '__main__':
    main()
