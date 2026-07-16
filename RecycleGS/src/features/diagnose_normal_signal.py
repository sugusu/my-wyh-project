import argparse, sys, os, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
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
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1e_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    print("[1/4] Loading data...")
    candidate_indices = np.load(out_dir / 'candidate_object_indices.npy')
    E_normal = np.load(out_dir / 'object_normal_conflict.npy')[candidate_indices]
    normal_valid = np.load(out_dir / 'object_normal_valid.npy')[candidate_indices]
    planarity = np.load(out_dir / 'object_planarity_confidence.npy')[candidate_indices]
    err = np.load(out_dir / 'candidate_geometry_errors_v2.npz')
    d_center_norm = err['d_center_norm']
    d_surface_proxy = err['d_surface_proxy_alpha1']

    print("[2/4] Computing validity statistics...")
    normal_valid_in_candidate = normal_valid
    valid_count = int(normal_valid_in_candidate.sum())
    valid_ratio = float(normal_valid_in_candidate.mean())

    planarity_stats = {
        'mean': float(planarity.mean()),
        'std': float(planarity.std()),
        'min': float(planarity.min()),
        'max': float(planarity.max()),
        'p25': float(np.percentile(planarity, 25)),
        'p50': float(np.percentile(planarity, 50)),
        'p75': float(np.percentile(planarity, 75)),
    }

    print(f"  Normal valid in candidate: {valid_count}/{len(candidate_indices)} ({valid_ratio*100:.1f}%)")

    print("[3/4] Computing Spearman for E_normal where normal_valid...")
    valid_mask = normal_valid_in_candidate
    correlations = {}
    for ename, earr in [('d_center_norm', d_center_norm), ('d_surface_proxy', d_surface_proxy)]:
        if valid_mask.sum() < 5:
            correlations[ename] = {'error': 'insufficient valid normal data', 'valid_count': int(valid_mask.sum())}
            continue
        s = E_normal[valid_mask]
        m = earr[valid_mask]
        valid_both = np.isfinite(s) & np.isfinite(m)
        if valid_both.sum() < 5:
            correlations[ename] = {'error': 'insufficient finite data', 'valid_count': int(valid_both.sum())}
            continue
        rho, pval = spearmanr(s[valid_both], m[valid_both])
        rho = float(rho) if not np.isnan(rho) else 0.0
        nv = int(valid_both.sum())
        k10 = max(1, nv // 10)
        top10_idx = np.argpartition(s[valid_both], -k10)[-k10:]
        bottom10_idx = np.argpartition(s[valid_both], k10)[:k10]
        top10_mean = float(m[valid_both][top10_idx].mean())
        bottom10_mean = float(m[valid_both][bottom10_idx].mean())
        ratio = top10_mean / max(bottom10_mean, 1e-8)
        correlations[ename] = {
            'valid_count': nv,
            'spearman_rho': rho,
            'spearman_pvalue': float(pval),
            'top10_mean_error': top10_mean,
            'bottom10_mean_error': bottom10_mean,
            'top10_bottom10_ratio': ratio,
        }

    print("[4/4] Saving...")
    normal_signal = {
        'num_candidate_gaussians': len(candidate_indices),
        'normal_valid_count': valid_count,
        'normal_valid_ratio': valid_ratio,
        'planarity_confidence': planarity_stats,
        'correlations': correlations,
    }

    json_path = debug_dir / 'normal_signal_validity.json'
    with open(json_path, 'w') as f:
        json.dump(normal_signal, f, indent=2)

    md = [
        f"# Normal Signal Validity - {cfg['scene_name']}",
        f"",
        f"## Normal Validity",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Candidate Gaussians | {len(candidate_indices)} |",
        f"| Normal valid count | {valid_count} |",
        f"| Normal valid ratio | {valid_ratio*100:.2f}% |",
        f"",
        f"## Planarity Confidence Distribution",
        f"| Stat | Value |",
        f"|------|-------|",
        f"| Mean | {planarity_stats['mean']:.4f} |",
        f"| Std | {planarity_stats['std']:.4f} |",
        f"| Min | {planarity_stats['min']:.4f} |",
        f"| P25 | {planarity_stats['p25']:.4f} |",
        f"| Median | {planarity_stats['p50']:.4f} |",
        f"| P75 | {planarity_stats['p75']:.4f} |",
        f"| Max | {planarity_stats['max']:.4f} |",
        f"",
        f"## E_normal Spearman (where normal_valid)",
        f"| Error Metric | Spearman rho | p-value | Top10 Mean | Bottom10 Mean | Ratio |",
        f"|-------------|-------------|--------|-----------|--------------|-------|",
    ]
    for ename in ['d_center_norm', 'd_surface_proxy']:
        r = correlations[ename]
        if 'error' in r:
            md.append(f"| {ename} | {r['error']} | | | | |")
        else:
            md.append(f"| {ename} | {r['spearman_rho']:.4f} | {r['spearman_pvalue']:.4e} | {r['top10_mean_error']:.6f} | {r['bottom10_mean_error']:.6f} | {r['top10_bottom10_ratio']:.4f} |")
    md.append(f"")

    md_path = debug_dir / 'normal_signal_validity.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"Saved to {json_path} and {md_path}")

if __name__ == '__main__':
    main()
