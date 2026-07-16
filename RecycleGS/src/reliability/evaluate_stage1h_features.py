#!/usr/bin/env python3
"""Stage 1H cross-scene feature evaluation."""
import argparse, json, os, sys, numpy as np, yaml
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, '/data/wyh/RecycleGS/src')
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

def evaluate_feature(feature, error, top_ratios=[5, 10, 20], bootstrap_samples=1000, ci_level=0.95):
    valid = np.isfinite(feature) & np.isfinite(error)
    if valid.sum() < 5:
        return {'error': 'insufficient valid data', 'valid_count': int(valid.sum())}

    s, m = feature[valid], error[valid]
    nv = len(s)

    rho, _ = spearmanr(s, m)
    rho = float(rho) if not np.isnan(rho) else 0.0
    rho_lo, rho_hi = bootstrap_spearman_ci(s, m, bootstrap_samples, ci_level)
    tau, _ = kendalltau(s, m)
    tau = float(tau) if not np.isnan(tau) else 0.0

    res = {
        'valid_count': int(nv),
        'valid_ratio': float(nv / max(len(feature), 1)),
        'spearman_rho': rho,
        'spearman_ci_95_lo': rho_lo,
        'spearman_ci_95_hi': rho_hi,
        'kendall_tau': tau,
    }

    for k_pct in top_ratios:
        k = max(1, int(nv * k_pct / 100))
        top_k = np.argpartition(s, -k)[-k:]
        bottom_k = np.argpartition(s, k)[:k]
        res[f'top{k_pct}_mean_error'] = float(m[top_k].mean())
        res[f'bottom{k_pct}_mean_error'] = float(m[bottom_k].mean())

    k10 = max(1, int(nv * 0.10))
    top10 = np.argpartition(s, -k10)[-k10:]
    bottom10 = np.argpartition(s, k10)[:k10]
    res['top10_bottom10_ratio'] = float(m[top10].mean() / max(m[bottom10].mean(), 1e-8))

    for k_pct in top_ratios:
        k = max(1, int(nv * k_pct / 100))
        res[f'ndcg_{k_pct}pct'] = ndcg_score(m, s, k)

    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene01-config', type=str, required=True)
    parser.add_argument('--scene03-config', type=str, required=True)
    parser.add_argument('--locked-config', type=str, required=True)
    args = parser.parse_args()

    with open(args.locked_config) as f:
        locked_cfg = yaml.safe_load(f)

    out_dir = Path(locked_cfg.get('project_root', '/data/wyh/RecycleGS')) / 'outputs' / 'reliability'
    os.makedirs(out_dir, exist_ok=True)

    scenes = {}
    for scene_key, cfg_path in [('scene_01', args.scene01_config), ('scene_03', args.scene03_config)]:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        scene_dir = Path(cfg['scene_dir'])
        rel_dir = Path(cfg['reliability_output_dir'])
        iter_dir = rel_dir / 'iter_15000'

        features = {}
        fnames = ['mask_risk_mean', 'mask_risk_variance', 'mask_risk_boundary', 'mask_risk_cv']
        for fn in fnames:
            fp = iter_dir / f'{fn}.npy'
            if fp.exists():
                features[fn] = np.load(fp)
            else:
                print(f"  WARNING: {fp} not found")

        pca_fp = iter_dir / 'pca_normal_conflict.npy'
        if pca_fp.exists():
            features['pca_normal_conflict'] = np.load(pca_fp)

        scale_fp = iter_dir / 'scale_risk_v2.npy'
        if scale_fp.exists():
            features['E_scale_v2'] = np.load(scale_fp)

        err_fp = iter_dir / 'geometry_errors.npz'
        if not err_fp.exists():
            print(f"  ERROR: {err_fp} not found")
            scenes[scene_key] = {'features': {}, 'errors': {}}
            continue

        err = np.load(err_fp)
        errors = {
            'd_center_norm': err['d_center_norm'],
            'd_surface_proxy_alpha1': err['d_surface_proxy_alpha1'],
            'd_surface_proxy_alpha2': err['d_surface_proxy_alpha2'],
        }

        # Mask features are only valid for candidate Gaussians, which match geometry errors dim
        # Ensure all features match the error array length
        for fn in list(features.keys()):
            if len(features[fn]) != len(errors['d_center_norm']):
                print(f"  WARNING: {fn} length {len(features[fn])} != error length {len(errors['d_center_norm'])}, dropping")
                del features[fn]

        scenes[scene_key] = {'features': features, 'errors': errors}
        print(f"  {scene_key}: {len(features)} features, {len(errors)} error metrics")

    bootstrap_samples = locked_cfg.get('evaluation', {}).get('bootstrap_samples', 1000)
    ci_level = locked_cfg.get('evaluation', {}).get('confidence_level', 0.95)
    validity = locked_cfg.get('stage1h_validity', {})
    min_rho = validity.get('min_rho', 0.15)
    min_ci_lower = validity.get('min_ci_lower', 0.0)
    min_t10b10 = validity.get('min_top10_bottom10', 1.30)
    min_cov = validity.get('min_coverage', 0.30)

    error_keys = ['d_center_norm', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']

    all_results = {}
    for scene_key in ['scene_01', 'scene_03']:
        scene_data = scenes[scene_key]
        results = {}
        for fname, farr in scene_data['features'].items():
            results[fname] = {}
            for ename in error_keys:
                if ename not in scene_data['errors']:
                    continue
                earr = scene_data['errors'][ename]
                res = evaluate_feature(farr, earr, bootstrap_samples=bootstrap_samples, ci_level=ci_level)
                results[fname][ename] = res
        all_results[scene_key] = results

    # Build cross-scene evaluation
    print("\nApplying cross-scene validity rules...")
    validation = {}
    all_feature_names = set()
    for scene_key in ['scene_01', 'scene_03']:
        all_feature_names.update(all_results[scene_key].keys())

    for fname in sorted(all_feature_names):
        validation[fname] = {}
        for ename in error_keys:
            s01 = all_results['scene_01'].get(fname, {}).get(ename, {})
            s03 = all_results['scene_03'].get(fname, {}).get(ename, {})

            if 'error' in s01 or 'error' in s03:
                validation[fname][ename] = {
                    'status': 'SKIP',
                    'reason': 'Insufficient data in one or both scenes',
                    'scene_01': s01.get('error', 'ok'),
                    'scene_03': s03.get('error', 'ok'),
                }
                continue

            cov_ok = (s01.get('valid_ratio', 0) >= min_cov) and (s03.get('valid_ratio', 0) >= min_cov)
            rho_ok = (s01.get('spearman_rho', 0) > min_rho) and (s03.get('spearman_rho', 0) > min_rho)
            ci_ok = (s01.get('spearman_ci_95_lo', -1) > min_ci_lower) and (s03.get('spearman_ci_95_lo', -1) > min_ci_lower)
            t10b10_ok = (s01.get('top10_bottom10_ratio', 1.0) > min_t10b10) and (s03.get('top10_bottom10_ratio', 1.0) > min_t10b10)
            direction_ok = (s01.get('spearman_rho', 0) > 0) == (s03.get('spearman_rho', 0) > 0)

            rules = {
                'coverage_ok': bool(cov_ok),
                'rho_ok': bool(rho_ok),
                'ci_lower_ok': bool(ci_ok),
                't10b10_ok': bool(t10b10_ok),
                'direction_consistent': bool(direction_ok),
                'all_pass': bool(cov_ok and rho_ok and ci_ok and t10b10_ok and direction_ok),
            }

            validation[fname][ename] = {
                'scene_01': {
                    'spearman_rho': s01.get('spearman_rho'),
                    'spearman_ci_95_lo': s01.get('spearman_ci_95_lo'),
                    'spearman_ci_95_hi': s01.get('spearman_ci_95_hi'),
                    'top10_bottom10_ratio': s01.get('top10_bottom10_ratio'),
                    'valid_ratio': s01.get('valid_ratio'),
                },
                'scene_03': {
                    'spearman_rho': s03.get('spearman_rho'),
                    'spearman_ci_95_lo': s03.get('spearman_ci_95_lo'),
                    'spearman_ci_95_hi': s03.get('spearman_ci_95_hi'),
                    'top10_bottom10_ratio': s03.get('top10_bottom10_ratio'),
                    'valid_ratio': s03.get('valid_ratio'),
                },
                'rules': rules,
            }

    # Save metrics JSON
    metrics_path = out_dir / 'stage1h_cross_scene_feature_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'all_results': {k: {fk: {ek: ev for ek, ev in fv.items() if 'error' not in ev} for fk, fv in v.items()} for k, v in all_results.items()},
            'validation': validation,
        }, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Generate MD report
    md_lines = [
        f"# Stage 1H Cross-Scene Feature Evaluation",
        f"",
        f"## Validation Rules",
        f"- Spearman rho > {min_rho} (both scenes)",
        f"- Bootstrap CI lower > {min_ci_lower} (both scenes)",
        f"- Top10/Bottom10 ratio > {min_t10b10} (both scenes)",
        f"- Coverage >= {min_cov*100:.0f}% (both scenes)",
        f"- Direction consistent across scenes",
        f"",
        f"## Feature Validation Summary",
        f"| Feature | Error Metric | Status | S01 rho | S03 rho | S01 CI lo | S03 CI lo | S01 T10/B10 | S03 T10/B10 | S01 Cov | S03 Cov | Direction |",
        f"|---------|-------------|--------|---------|---------|-----------|-----------|-------------|-------------|---------|---------|-----------|",
    ]
    for fname in sorted(all_feature_names):
        for ename in error_keys:
            if ename not in validation.get(fname, {}):
                continue
            v = validation[fname][ename]
            if v.get('status') == 'SKIP':
                md_lines.append(f"| {fname} | {ename} | SKIP | {v.get('reason', 'N/A')} | | | | | | | |")
                continue
            s01 = v['scene_01']
            s03 = v['scene_03']
            rules = v['rules']
            status = 'PASS' if rules['all_pass'] else 'FAIL'
            direction_str = 'Consistent' if rules['direction_consistent'] else 'OPPOSITE'
            md_lines.append(
                f"| {fname} | {ename} | {status} | "
                f"{s01.get('spearman_rho', 0):.4f} | {s03.get('spearman_rho', 0):.4f} | "
                f"{s01.get('spearman_ci_95_lo', 0):.4f} | {s03.get('spearman_ci_95_lo', 0):.4f} | "
                f"{s01.get('top10_bottom10_ratio', 0):.4f} | {s03.get('top10_bottom10_ratio', 0):.4f} | "
                f"{s01.get('valid_ratio', 0)*100:.1f}% | {s03.get('valid_ratio', 0)*100:.1f}% | "
                f"{direction_str} |"
            )

    md_lines.extend([
        f"",
        f"## Detailed Scene Metrics",
    ])
    for scene_key in ['scene_01', 'scene_03']:
        md_lines.extend([f"### {scene_key}", f"| Feature | Error Metric | Spearman rho | 95% CI | Kendall tau | T10/B10 | NDCG@5 | NDCG@10 | NDCG@20 | Valid Count | Valid Ratio |", f"|---------|-------------|-------------|--------|-------------|---------|--------|---------|---------|-------------|-------------|"])
        for fname in sorted(all_feature_names):
            for ename in error_keys:
                if fname not in all_results.get(scene_key, {}):
                    continue
                if ename not in all_results[scene_key][fname]:
                    continue
                r = all_results[scene_key][fname][ename]
                if 'error' in r:
                    md_lines.append(f"| {fname} | {ename} | {r['error']} | | | | | | | {r.get('valid_count', 0)} | N/A |")
                else:
                    md_lines.append(
                        f"| {fname} | {ename} | {r['spearman_rho']:.4f} | "
                        f"[{r['spearman_ci_95_lo']:.4f}, {r['spearman_ci_95_hi']:.4f}] | "
                        f"{r['kendall_tau']:.4f} | {r['top10_bottom10_ratio']:.4f} | "
                        f"{r.get('ndcg_5pct', 0):.4f} | {r.get('ndcg_10pct', 0):.4f} | "
                        f"{r.get('ndcg_20pct', 0):.4f} | {r['valid_count']} | "
                        f"{r['valid_ratio']*100:.1f}% |"
                    )
        md_lines.append("")

    report_path = out_dir / 'stage1h_cross_scene_feature_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved report: {report_path}")

if __name__ == '__main__':
    main()
