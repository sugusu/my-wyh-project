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

    print("[1/4] Loading object domain...")
    object_indices = np.load(out_dir / 'object_indices.npy')
    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz_all = base['xyz']
    normals_all = base['normal_world']

    xyz = xyz_all[object_indices]
    normals = normals_all[object_indices]
    N = len(xyz)
    k = cfg['surface_support']['knn']
    print(f"  Object Gaussians: {N}, k={k}")

    print("[2/4] Building kNN tree on object Gaussians only...")
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

    def percentile_normalize(arr):
        lo, hi = np.percentile(arr, [5, 95])
        clipped = arr.clip(lo, hi)
        return (clipped - lo) / (hi - lo + 1e-8)

    S_combined_norm = percentile_normalize(S_combined)
    risk = 1.0 - S_combined_norm.clip(0, 1)

    full_risk = np.zeros(len(xyz_all), dtype=np.float32)
    full_risk[object_indices] = risk

    full_support = np.zeros(len(xyz_all), dtype=np.float32)
    full_support[object_indices] = S_combined

    save_np(full_risk, out_dir / 'object_surface_support_risk.npy')
    save_np(full_support, out_dir / 'object_surface_support.npy')

    stats = {
        'k': k,
        'num_object_gaussians': N,
        'surface_support_mean': float(S_combined.mean()),
        'surface_support_std': float(S_combined.std()),
        'risk_mean': float(risk.mean()),
        'note': 'kNN built on object-supported Gaussians only',
    }
    save_json(stats, out_dir / 'object_surface_support_stats.json')
    print(f"[3/4] Surface support: mean={S_combined.mean():.4f}")
    print(f"[4/4] Saved")

if __name__ == '__main__':
    main()
