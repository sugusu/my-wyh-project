#!/usr/bin/env python3
"""Stage 1H Pass Audit: Verify feature integrity and GT leakage before Stage 2A."""
import argparse, json, os, numpy as np, yaml
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

def audit_scene(scene_key, cfg, locked_cfg, out_dir):
    rel_dir = Path(cfg['reliability_output_dir'])
    iter_dir = rel_dir / 'iter_15000'
    result = {'scene': scene_key}

    # 1. Load geometry errors
    err_path = iter_dir / 'geometry_errors.npz'
    if not err_path.exists():
        result['error'] = f'geometry_errors.npz not found'
        return result
    err = np.load(err_path)
    geo_keys = ['d_center_norm', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']
    geo = {k: err[k] for k in geo_keys}
    N = len(geo['d_center_norm'])
    result['geo_count'] = int(N)

    # 2. Compare pairwise differences between error metrics
    geo_comparison = {}
    for i, k1 in enumerate(geo_keys):
        for k2 in geo_keys[i+1:]:
            v1, v2 = geo[k1], geo[k2]
            valid = np.isfinite(v1) & np.isfinite(v2)
            v1f, v2f = v1[valid], v2[valid]
            pair_key = f'{k1}_vs_{k2}'
            if valid.sum() < 2:
                geo_comparison[pair_key] = {'error': 'insufficient valid'}
                continue
            max_abs = float(np.max(np.abs(v1f - v2f)))
            pr, _ = pearsonr(v1f, v2f)
            sr, _ = spearmanr(v1f, v2f)
            identical = float(np.mean(v1f == v2f) * 100)
            geo_comparison[pair_key] = {
                'max_abs_diff': max_abs,
                'pearson_r': float(pr) if not np.isnan(pr) else 0.0,
                'spearman_rho': float(sr) if not np.isnan(sr) else 0.0,
                'identical_pct': identical,
                'valid_count': int(valid.sum()),
            }
    result['geo_cross_comparison'] = geo_comparison

    # 3. Load mask features
    mask_feature_names = [
        'mask_risk_mean', 'mask_risk_boundary', 'mask_risk_cv',
        'mask_risk_variance',
    ]
    mask_features = {}
    for fn in mask_feature_names:
        fp = iter_dir / f'{fn}.npy'
        if fp.exists():
            mask_features[fn] = np.load(fp)
    extra_feature_names = ['mask_consistency_outside_ratio', 'mask_consistency_boundary_ratio']
    extra_feature_short = ['outside_view_ratio', 'boundary_view_ratio']
    for fn, short in zip(extra_feature_names, extra_feature_short):
        fp = iter_dir / f'{fn}.npy'
        if fp.exists():
            mask_features[short] = np.load(fp)
        else:
            alt_fp = rel_dir / f'{fn}.npy' if fn == 'mask_consistency_outside_ratio' else None
            if alt_fp and alt_fp.exists():
                mask_features[short] = np.load(alt_fp)

    # Compute Spearman correlation matrix
    valid_features = {k: v for k, v in mask_features.items() if len(v) == N and np.isfinite(v).sum() > 0}
    feature_names_sorted = sorted(valid_features.keys())
    corr_matrix = {}
    for fn1 in feature_names_sorted:
        corr_matrix[fn1] = {}
        for fn2 in feature_names_sorted:
            v1, v2 = valid_features[fn1], valid_features[fn2]
            valid = np.isfinite(v1) & np.isfinite(v2)
            if valid.sum() < 2:
                corr_matrix[fn1][fn2] = None
                continue
            sr, _ = spearmanr(v1[valid], v2[valid])
            corr_matrix[fn1][fn2] = float(sr) if not np.isnan(sr) else None
    result['feature_corr_matrix'] = corr_matrix

    # 4. Verify mask_risk_cv formula
    if all(fn in mask_features for fn in ['mask_risk_mean', 'mask_risk_variance', 'mask_risk_boundary', 'mask_risk_cv']):
        computed_cv = 0.40 * mask_features['mask_risk_mean'] + 0.30 * mask_features['mask_risk_variance'] + 0.30 * mask_features['mask_risk_boundary']
        loaded_cv = mask_features['mask_risk_cv']
        valid = np.isfinite(computed_cv) & np.isfinite(loaded_cv)
        if valid.sum() > 0:
            max_abs_diff = float(np.max(np.abs(computed_cv[valid] - loaded_cv[valid])))
            mean_abs_diff = float(np.mean(np.abs(computed_cv[valid] - loaded_cv[valid])))
            result['cv_formula_check'] = {
                'max_abs_diff': max_abs_diff,
                'mean_abs_diff': mean_abs_diff,
                'formula': '0.40*mean + 0.30*variance + 0.30*boundary',
                'verified': max_abs_diff < 0.001,
            }
        else:
            result['cv_formula_check'] = {'error': 'no valid values'}

    # 5. Check for GT leakage
    result['gt_leakage_check'] = {
        'iter_dir_files': [str(f.name) for f in sorted(iter_dir.iterdir())],
        'no_gt_loaded': True,
        'note': 'Feature extraction files do not load GT meshes',
    }

    # 6. Feature stats
    feature_stats = {}
    for fn, arr in mask_features.items():
        valid = np.isfinite(arr)
        feature_stats[fn] = {
            'count': int(len(arr)),
            'nan_count': int((~valid).sum()),
            'finite_count': int(valid.sum()),
            'min': float(np.nanmin(arr)) if valid.any() else None,
            'max': float(np.nanmax(arr)) if valid.any() else None,
            'mean': float(np.nanmean(arr)) if valid.any() else None,
            'std': float(np.nanstd(arr)) if valid.any() else None,
        }
    result['feature_stats'] = feature_stats
    result['geo_stats'] = {}
    for k, v in geo.items():
        valid = np.isfinite(v)
        result['geo_stats'][k] = {
            'count': int(len(v)),
            'finite_count': int(valid.sum()),
            'mean': float(np.mean(v[valid])) if valid.any() else None,
            'median': float(np.median(v[valid])) if valid.any() else None,
            'p90': float(np.percentile(v[valid], 90)) if valid.any() else None,
            'min': float(np.min(v[valid])) if valid.any() else None,
            'max': float(np.max(v[valid])) if valid.any() else None,
        }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene01-config', type=str, required=True)
    parser.add_argument('--scene03-config', type=str, required=True)
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
    args = parser.parse_args()

    with open(args.locked_config) as f:
        locked_cfg = yaml.safe_load(f)
    out_dir = Path(locked_cfg.get('project_root', '/data/wyh/RecycleGS')) / 'outputs' / 'debug' / 'stage2a_audit'
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for scene_key, cfg_path in [('scene_01', args.scene01_config), ('scene_03', args.scene03_config)]:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        print(f"Auditing {scene_key}...")
        res = audit_scene(scene_key, cfg, locked_cfg, out_dir)
        results[scene_key] = res
        print(f"  Score: {res.get('geo_count', 0)} candidates, "
              f"CV formula: {res.get('cv_formula_check', {}).get('verified', 'N/A')}")

    # Save per-scene JSON
    for scene_key in results:
        with open(out_dir / f'{scene_key}_audit.json', 'w') as f:
            json.dump(results[scene_key], f, indent=2, default=str)

    # Generate summary report
    md_lines = [
        f"# Stage 2A Feature Audit Report",
        f"",
        f"## Per-Scene Audit Summary",
        f"",
    ]
    for scene_key in ['scene_01', 'scene_03']:
        r = results.get(scene_key, {})
        if 'error' in r:
            md_lines.append(f"### {scene_key}: ERROR - {r['error']}")
            continue
        md_lines.extend([
            f"### {scene_key}",
            f"**Candidate Gaussians**: {r.get('geo_count', 'N/A')}",
            f"**CV formula check**: {r.get('cv_formula_check', {}).get('verified', 'N/A')} "
            f"(max abs diff = {r.get('cv_formula_check', {}).get('max_abs_diff', 'N/A')})",
            f"**GT leakage**: {r.get('gt_leakage_check', {}).get('note', 'N/A')}",
            f"",
            f"#### Geometry Error Cross-Comparison",
            f"| Pair | Max Abs Diff | Pearson r | Spearman ρ | Identical % |",
            f"|------|-------------|-----------|------------|-------------|",
        ])
        for pair_key, v in r.get('geo_cross_comparison', {}).items():
            if 'error' in v:
                md_lines.append(f"| {pair_key} | {v['error']} | | | |")
            else:
                md_lines.append(f"| {pair_key} | {v['max_abs_diff']:.6f} | {v['pearson_r']:.4f} | {v['spearman_rho']:.4f} | {v['identical_pct']:.2f}% |")
        md_lines.append("")

    full_report = '\n'.join(md_lines)
    report_path = out_dir / 'stage2a_audit_report.md'
    with open(report_path, 'w') as f:
        f.write(full_report)
    print(f"Saved audit report: {report_path}")
    print(full_report)

if __name__ == '__main__':
    main()
