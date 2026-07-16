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
    base = np.load(out_dir / 'gaussian_base_features.npz')
    E_scale = np.load(out_dir / 'object_scale_anomaly.npy')[candidate_indices]
    err = np.load(out_dir / 'candidate_geometry_errors_v2.npz')
    d_center_norm = err['d_center_norm']
    d_surface_proxy_alpha1 = err['d_surface_proxy_alpha1']
    d_surface_proxy_alpha2 = err['d_surface_proxy_alpha2']

    print("[2/4] Extracting scale features...")
    scale_min = base['scale_min'][candidate_indices]
    scale_linear = base['scale_linear'][candidate_indices]
    scale_mid = np.median(scale_linear, axis=1)
    scale_max = base['scale_max'][candidate_indices]
    scale_ratio = base['scale_ratio'][candidate_indices]
    scale_volume = base['scale_volume'][candidate_indices]

    print("[3/4] Computing signed Spearman correlations...")
    scale_features = {
        'scale_min': scale_min,
        'scale_mid': scale_mid,
        'scale_max': scale_max,
        'scale_ratio': scale_ratio,
        'scale_volume': scale_volume,
    }
    error_metrics = {
        'd_center_norm': d_center_norm,
        'd_surface_proxy_alpha1': d_surface_proxy_alpha1,
        'd_surface_proxy_alpha2': d_surface_proxy_alpha2,
    }

    correlations = {}
    for sname, sarr in scale_features.items():
        correlations[sname] = {}
        for ename, earr in error_metrics.items():
            valid = np.isfinite(sarr) & np.isfinite(earr)
            if valid.sum() < 5:
                correlations[sname][ename] = {'error': 'insufficient data', 'valid_count': int(valid.sum())}
                continue
            rho, pval = spearmanr(sarr[valid], earr[valid])
            correlations[sname][ename] = {
                'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
                'spearman_pvalue': float(pval),
                'valid_count': int(valid.sum()),
            }

    correlations['E_scale'] = {}
    for ename, earr in error_metrics.items():
        valid = np.isfinite(E_scale) & np.isfinite(earr)
        if valid.sum() < 5:
            correlations['E_scale'][ename] = {'error': 'insufficient data', 'valid_count': int(valid.sum())}
            continue
        rho, pval = spearmanr(E_scale[valid], earr[valid])
        correlations['E_scale'][ename] = {
            'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
            'spearman_pvalue': float(pval),
            'valid_count': int(valid.sum()),
        }

    print("[4/4] Saving...")
    scale_semantics = {
        'num_candidate_gaussians': len(candidate_indices),
        'correlations': correlations,
    }

    json_path = debug_dir / 'scale_semantics.json'
    with open(json_path, 'w') as f:
        json.dump(scale_semantics, f, indent=2)

    md = [
        f"# Scale Semantics - {cfg['scene_name']}",
        f"",
        f"## Settings",
        f"- Candidate Gaussians: {len(candidate_indices)}",
        f"",
        f"## Signed Spearman Correlations with Geometry Errors",
        f"| Feature | Error Metric | Spearman rho | p-value | Valid Count |",
        f"|--------|-------------|-------------|--------|------------|",
    ]
    for sname in ['scale_min', 'scale_mid', 'scale_max', 'scale_ratio', 'scale_volume', 'E_scale']:
        for ename in ['d_center_norm', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']:
            r = correlations[sname][ename]
            if 'error' in r:
                md.append(f"| {sname} | {ename} | {r['error']} | | {r.get('valid_count', '')} |")
            else:
                md.append(f"| {sname} | {ename} | {r['spearman_rho']:.4f} | {r['spearman_pvalue']:.4e} | {r['valid_count']} |")
    md.append(f"")

    md_path = debug_dir / 'scale_semantics.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"Saved to {json_path} and {md_path}")

if __name__ == '__main__':
    main()
