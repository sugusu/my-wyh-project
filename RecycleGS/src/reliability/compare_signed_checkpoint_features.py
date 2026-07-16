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

    print("[1/4] Loading 7k and 15k results...")
    iter_7k = out_dir / 'iter_7000'
    iter_15k = out_dir / 'iter_15000'

    def load_metrics(iter_dir):
        metrics_path = iter_dir / 'signed_feature_metrics.json'
        err_path = iter_dir / 'geometry_errors.npz'
        cand_path = iter_dir / 'candidate_indices.npy'
        if not metrics_path.exists() or not err_path.exists():
            return None, None, None
        with open(metrics_path) as f:
            metrics = json.load(f)
        err = np.load(err_path)
        candidates = np.load(cand_path)
        return metrics, err, candidates

    metrics_7k, err_7k, cand_7k = load_metrics(iter_7k)
    metrics_15k, err_15k, cand_15k = load_metrics(iter_15k)

    if metrics_7k is None:
        print("  WARNING: 7k metrics not available")
    else:
        print(f"  7k: {len(cand_7k)} candidates, "
              f"{sum(1 for v in metrics_7k.values() for e in v.values() if 'spearman_rho' in e)} metric entries")

    if metrics_15k is None:
        print("  WARNING: 15k metrics not available")
    else:
        print(f"  15k: {len(cand_15k)} candidates, "
              f"{sum(1 for v in metrics_15k.values() for e in v.values() if 'spearman_rho' in e)} metric entries")

    print("[2/4] Comparing signed correlations...")
    features_to_compare = ['support_confidence', 'E_support_v2', 'E_scale_v2',
                           'scale_min', 'scale_max', 'scale_ratio', 'scale_volume']
    error_metrics = ['d_center_norm', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']

    comparison = {
        'scene_name': cfg['scene_name'],
        '7k': {
            'candidate_count': int(len(cand_7k)) if cand_7k is not None else 0,
        },
        '15k': {
            'candidate_count': int(len(cand_15k)) if cand_15k is not None else 0,
        },
        'signed_spearman_comparison': {},
        'top10_bottom10_ratio_comparison': {},
        'error_distribution': {},
        'effective_coverage': {},
    }

    if cand_7k is not None:
        comparison['7k']['candidate_ratio'] = float(len(cand_7k) / max(1, len(cand_7k)))
    if cand_15k is not None:
        comparison['15k']['candidate_ratio'] = float(len(cand_15k) / max(1, len(cand_15k)))

    for fname in features_to_compare:
        comparison['signed_spearman_comparison'][fname] = {}
        comparison['top10_bottom10_ratio_comparison'][fname] = {}
        for ename in error_metrics:
            entry_7k = None
            entry_15k = None
            if metrics_7k and fname in metrics_7k and ename in metrics_7k[fname]:
                entry_7k = metrics_7k[fname][ename]
            if metrics_15k and fname in metrics_15k and ename in metrics_15k[fname]:
                entry_15k = metrics_15k[fname][ename]

            comp = {
                '7k': entry_7k if entry_7k else None,
                '15k': entry_15k if entry_15k else None,
            }

            if entry_7k and entry_15k and 'spearman_rho' in entry_7k and 'spearman_rho' in entry_15k:
                rho_7k = entry_7k['spearman_rho']
                rho_15k = entry_15k['spearman_rho']
                comp['rho_difference_7k_minus_15k'] = float(rho_7k - rho_15k)
                comp['abs_rho_difference'] = float(abs(rho_7k) - abs(rho_15k))
                comp['absolute_rho_7k'] = float(abs(rho_7k))
                comp['absolute_rho_15k'] = float(abs(rho_15k))
                comp['7k_stronger_signed'] = abs(rho_7k) > abs(rho_15k) * 1.1
                comp['7k_more_positive'] = rho_7k > rho_15k

                if 'top10_bottom10_ratio' in entry_7k and 'top10_bottom10_ratio' in entry_15k:
                    comp['top10_bottom10_ratio_7k'] = entry_7k['top10_bottom10_ratio']
                    comp['top10_bottom10_ratio_15k'] = entry_15k['top10_bottom10_ratio']
                    comparison['top10_bottom10_ratio_comparison'][fname][ename] = {
                        '7k': entry_7k['top10_bottom10_ratio'],
                        '15k': entry_15k['top10_bottom10_ratio'],
                    }

            comparison['signed_spearman_comparison'][fname][ename] = comp

    print("[3/4] Computing error distributions...")
    for name, err, cand_label in [('7k', err_7k, '7k'), ('15k', err_15k, '15k')]:
        if err is None:
            comparison['error_distribution'][cand_label] = None
            continue
        comparison['error_distribution'][cand_label] = {
            'd_center_norm_mean': float(err['d_center_norm'].mean()),
            'd_center_norm_median': float(np.median(err['d_center_norm'])),
            'd_center_norm_std': float(err['d_center_norm'].std()),
            'd_surface_proxy_alpha1_mean': float(err['d_surface_proxy_alpha1'].mean()),
            'd_surface_proxy_alpha2_mean': float(err['d_surface_proxy_alpha2'].mean()),
            'd_center_norm_q25': float(np.percentile(err['d_center_norm'], 25)),
            'd_center_norm_q75': float(np.percentile(err['d_center_norm'], 75)),
        }

    comparison['effective_coverage'] = {
        '7k': {
            'candidate_count': int(len(cand_7k)) if cand_7k is not None else 0,
        },
        '15k': {
            'candidate_count': int(len(cand_15k)) if cand_15k is not None else 0,
        },
    }

    json_path = debug_dir / 'signed_checkpoint_comparison.json'
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    lines = [
        f"# Signed Checkpoint Comparison - {cfg['scene_name']}",
        f"",
        f"## Candidate Domain Size",
        f"| Checkpoint | Count |",
        f"|------------|-------|",
        f"| 7k | {comparison['7k']['candidate_count']} |",
        f"| 15k | {comparison['15k']['candidate_count']} |",
        f"",
        f"## Signed Spearman Correlation Comparison",
        f"| Feature | Error | 7k rho | 7k CI | 15k rho | 15k CI | Δrho | 7k stronger? |",
        f"|--------|-------|--------|-------|---------|-------|------|-------------|",
    ]
    for fname in features_to_compare:
        for ename in error_metrics:
            if fname not in comparison['signed_spearman_comparison']:
                continue
            comp = comparison['signed_spearman_comparison'][fname].get(ename, {})
            if not comp:
                continue
            r7 = comp.get('7k')
            r15 = comp.get('15k')
            if r7 is None and r15 is None:
                continue
            r7 = r7 or {}
            r15 = r15 or {}
            rho7 = f"{r7.get('spearman_rho', 'N/A'):.4f}" if isinstance(r7.get('spearman_rho'), (int, float)) else 'N/A'
            ci7 = ''
            if isinstance(r7.get('spearman_ci_95_lo'), (int, float)) and isinstance(r7.get('spearman_ci_95_hi'), (int, float)):
                ci7 = f"[{r7['spearman_ci_95_lo']:.4f},{r7['spearman_ci_95_hi']:.4f}]"
            rho15 = f"{r15.get('spearman_rho', 'N/A'):.4f}" if isinstance(r15.get('spearman_rho'), (int, float)) else 'N/A'
            ci15 = ''
            if isinstance(r15.get('spearman_ci_95_lo'), (int, float)) and isinstance(r15.get('spearman_ci_95_hi'), (int, float)):
                ci15 = f"[{r15['spearman_ci_95_lo']:.4f},{r15['spearman_ci_95_hi']:.4f}]"
            drho = f"{comp.get('rho_difference_7k_minus_15k', 0):.4f}" if 'rho_difference_7k_minus_15k' in comp else ''
            stronger = str(comp.get('7k_stronger_signed', False))
            lines.append(f"| {fname} | {ename} | {rho7} | {ci7} | {rho15} | {ci15} | {drho} | {stronger} |")

    lines.extend([
        f"",
        f"## Top10/Bottom10 Ratio Comparison",
        f"| Feature | Error | 7k ratio | 15k ratio |",
        f"|--------|-------|----------|-----------|",
    ])
    for fname in features_to_compare:
        for ename in error_metrics:
            if fname not in comparison['top10_bottom10_ratio_comparison']:
                continue
            rcomp = comparison['top10_bottom10_ratio_comparison'][fname].get(ename, {})
            if not rcomp:
                continue
            r7 = rcomp.get('7k', 'N/A')
            r15 = rcomp.get('15k', 'N/A')
            r7_str = f"{r7:.4f}" if isinstance(r7, (int, float)) else str(r7)
            r15_str = f"{r15:.4f}" if isinstance(r15, (int, float)) else str(r15)
            lines.append(f"| {fname} | {ename} | {r7_str} | {r15_str} |")

    lines.extend([
        f"",
        f"## Error Distribution",
        f"| Metric | 7k | 15k |",
        f"|--------|-----|-----|",
    ])
    for label in ['7k', '15k']:
        ed = comparison['error_distribution'].get(label, {})
        if ed:
            for key in ['d_center_norm_mean', 'd_center_norm_median', 'd_center_norm_std',
                         'd_center_norm_q25', 'd_center_norm_q75']:
                other_label = '15k' if label == '7k' else '7k'
                other_ed = comparison['error_distribution'].get(other_label, {})
                val = ed.get(key, 'N/A')
                oval = other_ed.get(key, 'N/A')
                val_str = f"{val:.6f}" if isinstance(val, (int, float)) else str(val)
                oval_str = f"{oval:.6f}" if isinstance(oval, (int, float)) else str(oval)
                if label == '7k':
                    lines.append(f"| {key} | {val_str} | {oval_str} |")

    lines.extend([
        f"",
        f"## Effective Coverage",
        f"| Checkpoint | Candidate Count |",
        f"|------------|----------------|",
        f"| 7k | {comparison['effective_coverage']['7k']['candidate_count']} |",
        f"| 15k | {comparison['effective_coverage']['15k']['candidate_count']} |",
    ])

    md_path = debug_dir / 'signed_checkpoint_comparison.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[4/4] Comparison saved to {json_path}")
    print(f"  Key differences:")
    for fname in features_to_compare:
        for ename in error_metrics:
            comp = comparison['signed_spearman_comparison'].get(fname, {}).get(ename, {})
            if comp and 'rho_difference_7k_minus_15k' in comp:
                drho = comp['rho_difference_7k_minus_15k']
                stronger = comp.get('7k_stronger_signed', False)
                print(f"  {fname}/{ename}: Δrho={drho:.4f}, 7k_stronger={stronger}")

if __name__ == '__main__':
    main()
