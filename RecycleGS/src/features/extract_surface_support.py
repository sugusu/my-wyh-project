import argparse, sys, os, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, '/data/wyh/RecycleGS/src')
from recyclegs.config import load_config, save_np, save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])

    print("[1/3] Loading base features...")
    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = base['xyz']
    normals = base['normal_world']
    N = len(xyz)

    k = cfg['surface_support']['knn']
    radius_mult = cfg['surface_support']['radius_multiplier']

    print(f"[2/3] Building kNN tree (k={k})...")
    tree = cKDTree(xyz)

    chunk = min(5000, N)
    support = np.zeros(N)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        batch_xyz = xyz[start:end]
        batch_normals = normals[start:end]
        dists, idxs = tree.query(batch_xyz, k=k+1)
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]
        local_scales = np.linalg.norm(batch_xyz[:, None] - xyz[idxs], axis=2)
        sigma_p = local_scales.mean(axis=1, keepdims=True) * radius_mult
        w_dist = np.exp(-dists**2 / (2 * sigma_p**2 + 1e-8))
        n_dot = np.abs((batch_normals[:, None] * normals[idxs]).sum(axis=2))
        S = (w_dist * n_dot).sum(axis=1) / (w_dist.sum(axis=1) + 1e-8)
        support[start:end] = S
        if start % 20000 == 0:
            print(f"  [{start}/{N}]")

    S_min, S_max = np.percentile(support, [5, 95])
    support_norm = (support - S_min) / (S_max - S_min + 1e-8)
    support_risk = 1.0 - support_norm.clip(0, 1)

    save_np(support, out_dir / 'surface_support.npy')
    save_np(support_risk, out_dir / 'surface_support_risk.npy')
    save_json({'k': k, 'radius_multiplier': radius_mult}, out_dir / 'surface_support_stats.json')
    print(f"[3/3] Saved to {out_dir}")

if __name__ == '__main__':
    main()
