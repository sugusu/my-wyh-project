import argparse, sys, os, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

def percentile_normalize(arr):
    lo, hi = np.percentile(arr, [5, 95])
    clipped = arr.clip(lo, hi)
    return (clipped - lo) / (hi - lo + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1e_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    print("[1/5] Loading data...")
    candidate_indices = np.load(out_dir / 'candidate_object_indices.npy')
    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz_all = base['xyz']
    normals_all = base['normal_world']

    xyz = xyz_all[candidate_indices]
    normals = normals_all[candidate_indices]
    N = len(xyz)
    k = cfg['surface_support']['knn']
    print(f"  Candidate Gaussians: {N}, k={k}")

    print("[2/5] Computing S_position, S_normal, S_combined with kNN on candidate domain...")
    tree = cKDTree(xyz)
    chunk = min(5000, N)
    S_position = np.zeros(N)
    S_normal = np.zeros(N)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        batch_xyz = xyz[start:end]
        batch_normals = normals[start:end]
        dists, idxs = tree.query(batch_xyz, k=k+1)
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]
        local_scales = np.linalg.norm(batch_xyz[:, None] - xyz[idxs], axis=2)
        sigma_p = local_scales.mean(axis=1, keepdims=True) * 3.0
        w_dist = np.exp(-dists**2 / (2 * sigma_p**2 + 1e-8))
        n_dot = np.abs((batch_normals[:, None] * normals[idxs]).sum(axis=2))
        S_position[start:end] = (w_dist * n_dot).sum(axis=1) / (w_dist.sum(axis=1) + 1e-8)
        S_normal[start:end] = n_dot.sum(axis=1) / k
        if start % 20000 == 0:
            print(f"  [{start}/{N}]")

    S_combined = S_position * (0.5 + 0.5 * S_normal)
    S_combined_norm = percentile_normalize(S_combined)
    risk_from_S = 1.0 - S_combined_norm.clip(0, 1)

    print("[3/5] Loading existing E_support_saved...")
    E_support_saved = np.load(out_dir / 'object_surface_support_risk.npy')[candidate_indices]

    print("[4/5] Loading geometry errors...")
    err = np.load(out_dir / 'candidate_geometry_errors_v2.npz')
    d_center_norm = err['d_center_norm']
    d_surface_proxy_alpha1 = err['d_surface_proxy_alpha1']
    d_surface_proxy_alpha2 = err['d_surface_proxy_alpha2']

    print("[5/5] Computing correlations and diagnostics...")
    features = {
        'S_position': S_position,
        'S_normal': S_normal,
        'S_combined': S_combined,
        'E_support_saved': E_support_saved,
        'risk_from_S_1_minus_norm': risk_from_S,
    }
    error_metrics = {
        'd_center_norm': d_center_norm,
        'd_surface_proxy_alpha1': d_surface_proxy_alpha1,
        'd_surface_proxy_alpha2': d_surface_proxy_alpha2,
    }

    results = {}
    for fname, farr in features.items():
        results[fname] = {}
        for ename, earr in error_metrics.items():
            valid = np.isfinite(farr) & np.isfinite(earr)
            if valid.sum() < 5:
                results[fname][ename] = {'error': 'insufficient valid data', 'valid_count': int(valid.sum())}
                continue
            s, m = farr[valid], earr[valid]
            nv = len(s)
            rho, pval = spearmanr(s, m)
            rho = float(rho) if not np.isnan(rho) else 0.0
            k10 = max(1, nv // 10)
            top10_idx = np.argpartition(s, -k10)[-k10:]
            bottom10_idx = np.argpartition(s, k10)[:k10]
            top10_mean = float(m[top10_idx].mean())
            bottom10_mean = float(m[bottom10_idx].mean())
            ratio = top10_mean / max(bottom10_mean, 1e-8)
            results[fname][ename] = {
                'valid_count': nv,
                'spearman_rho': rho,
                'spearman_pvalue': float(pval),
                'top10_mean_error': top10_mean,
                'bottom10_mean_error': bottom10_mean,
                'top10_bottom10_ratio': ratio,
            }

    # Check if E_support_saved == risk_from_S
    max_abs_diff = float(np.max(np.abs(E_support_saved - risk_from_S)))
    is_equal = max_abs_diff < 1e-5

    support_semantics = {
        'num_candidate_gaussians': N,
        'k': k,
        'correlations': results,
        'E_support_saved_vs_risk_from_S': {
            'max_abs_difference': max_abs_diff,
            'effectively_equal': is_equal,
            'note': 'E_support_saved comes from object-domain kNN; risk_from_S from candidate-domain kNN' if not is_equal else 'E_support_saved == 1 - percentile_normalize(S_combined) verified',
        },
    }

    json_path = debug_dir / 'support_semantics.json'
    with open(json_path, 'w') as f:
        json.dump(support_semantics, f, indent=2)

    print(f"  E_support_saved vs risk_from_S: max_abs_diff={max_abs_diff:.8f}, equal={is_equal}")

    import csv
    csv_path = debug_dir / 'support_semantics.csv'
    with open(csv_path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['feature', 'error_metric', 'spearman_rho', 'pvalue', 'top10_mean', 'bottom10_mean', 'top10_bottom10_ratio'])
        for fname, emap in results.items():
            for ename, vals in emap.items():
                if 'error' in vals:
                    w.writerow([fname, ename, vals['error'], '', '', '', ''])
                else:
                    w.writerow([fname, ename, vals['spearman_rho'], vals['spearman_pvalue'],
                                vals['top10_mean_error'], vals['bottom10_mean_error'], vals['top10_bottom10_ratio']])

    md = [
        f"# Surface Support Semantics - {cfg['scene_name']}",
        f"",
        f"## Settings",
        f"- Candidate Gaussians: {N}",
        f"- kNN k: {k}",
        f"- kNN built on candidate-object domain",
        f"",
        f"## Equality Check",
        f"- max_abs_diff(E_support_saved, risk_from_S): {max_abs_diff:.8f}",
        f"- Effectively equal (<1e-5): {is_equal}",
        f"",
        f"## Spearman Correlations with Geometry Errors",
        f"| Feature | Error Metric | Spearman rho | p-value | Top10 Mean | Bottom10 Mean | Ratio |",
        f"|--------|-------------|-------------|--------|-----------|--------------|-------|",
    ]
    for fname in ['S_position', 'S_normal', 'S_combined', 'E_support_saved', 'risk_from_S_1_minus_norm']:
        for ename in ['d_center_norm', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']:
            r = results[fname][ename]
            if 'error' in r:
                md.append(f"| {fname} | {ename} | {r['error']} | | | | |")
            else:
                md.append(f"| {fname} | {ename} | {r['spearman_rho']:.4f} | {r['spearman_pvalue']:.4e} | {r['top10_mean_error']:.6f} | {r['bottom10_mean_error']:.6f} | {r['top10_bottom10_ratio']:.4f} |")
    md.append(f"")

    md_path = debug_dir / 'support_semantics.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"Saved to {json_path} and {md_path}")

if __name__ == '__main__':
    main()
