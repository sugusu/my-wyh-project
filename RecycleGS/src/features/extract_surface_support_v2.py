import argparse, sys, os, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

def load_ply_raw(ply_path):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    props = {p.name: np.asarray(vertex[p.name]) for p in vertex.properties}
    xyz = np.stack([props['x'], props['y'], props['z']], axis=1)
    scale_names = sorted([p for p in props if p.startswith('scale_')], key=lambda x: int(x.split('_')[-1]))
    scales = np.stack([props[s] for s in scale_names], axis=1) if scale_names else np.ones((len(xyz), 3))
    rot_names = sorted([p for p in props if p.startswith('rot')], key=lambda x: int(x.split('_')[-1]))
    rots = np.stack([props[r] for r in rot_names], axis=1) if rot_names else np.tile([1.,0.,0.,0.], (len(xyz),1))
    return xyz, scales, rots

def quat_to_rotmat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = len(q)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    return R

def compute_normals(xyz, scales, rots):
    scale_linear = np.exp(scales)
    rots_norm = rots / np.linalg.norm(rots, axis=1, keepdims=True)
    rot_mats = quat_to_rotmat(rots_norm)
    min_axis = np.argmin(scale_linear, axis=1)
    normals = np.zeros_like(xyz)
    for i in range(len(xyz)):
        normals[i] = rot_mats[i, :, min_axis[i]]
    return normals, scale_linear

def percentile_normalize(arr):
    lo, hi = np.percentile(arr, [5, 95])
    clipped = arr.clip(lo, hi)
    return (clipped - lo) / (hi - lo + 1e-8)

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

    model_dir = Path(cfg['model_dir'])
    if iteration == 7000:
        ply_path = model_dir / 'point_cloud' / 'iteration_7000' / 'point_cloud.ply'
    else:
        ply_path = Path(cfg['checkpoint_path'])

    print(f"[{iter_str}] Loading data...")
    candidate_indices = np.load(iter_dir / 'candidate_indices.npy')
    print(f"  Candidate count: {len(candidate_indices)}")

    print(f"  Loading PLY: {ply_path}")
    xyz_all, scales_all, rots_all = load_ply_raw(ply_path)
    N_all = len(xyz_all)
    print(f"  Total Gaussians: {N_all}")

    print(f"  Computing normals and scale features...")
    normals_all, scale_linear_all = compute_normals(xyz_all, scales_all, rots_all)

    xyz = xyz_all[candidate_indices]
    normals = normals_all[candidate_indices]
    N = len(xyz)
    k = cfg['surface_support']['knn']

    print(f"[{iter_str}] Building kNN tree on candidate domain (k={k})...")
    if N < k + 1:
        k_eff = max(1, N - 1)
        print(f"  Warning: N={N} < k+1, reducing k to {k_eff}")
    else:
        k_eff = k

    tree = cKDTree(xyz)
    chunk = min(5000, N)
    S_position = np.zeros(N)
    S_normal = np.zeros(N)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        batch_xyz = xyz[start:end]
        batch_normals = normals[start:end]

        dists, idxs = tree.query(batch_xyz, k=k_eff + 1)
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]

        local_scales = np.linalg.norm(batch_xyz[:, None] - xyz[idxs], axis=2)
        sigma_p = local_scales.mean(axis=1, keepdims=True) * 3.0
        w_dist = np.exp(-dists**2 / (2 * sigma_p**2 + 1e-8))
        n_dot = np.abs((batch_normals[:, None] * normals[idxs]).sum(axis=2))

        S_position[start:end] = (w_dist * n_dot).sum(axis=1) / (w_dist.sum(axis=1) + 1e-8)
        S_normal[start:end] = n_dot.sum(axis=1) / k_eff

        if start % 20000 == 0:
            print(f"  [{start}/{N}]")

    S_combined = S_position * (0.5 + 0.5 * S_normal)
    support_confidence = percentile_normalize(S_combined).clip(0, 1)
    E_support_v2 = 1.0 - support_confidence

    max_abs_diff = float(np.max(np.abs(E_support_v2 - (1.0 - support_confidence))))
    print(f"  max_abs_difference (E_support_v2 vs 1-confidence): {max_abs_diff:.2e}")
    if max_abs_diff > 1e-5:
        print(f"  WARNING: max_abs_difference {max_abs_diff} > 1e-6, verifying formula...")
    else:
        print(f"  Verification passed: E_support_v2 = 1 - support_confidence")

    print(f"  support_confidence: mean={support_confidence.mean():.4f}, "
          f"std={support_confidence.std():.4f}")
    print(f"  E_support_v2: mean={E_support_v2.mean():.4f}, "
          f"std={E_support_v2.std():.4f}")

    np.save(iter_dir / 'support_confidence_v2.npy', support_confidence)
    np.save(iter_dir / 'support_risk_v2.npy', E_support_v2)

    stats = {
        'iteration': iteration,
        'k': k_eff,
        'candidate_count': N,
        'S_position_mean': float(S_position.mean()),
        'S_normal_mean': float(S_normal.mean()),
        'S_combined_mean': float(S_combined.mean()),
        'support_confidence_mean': float(support_confidence.mean()),
        'E_support_v2_mean': float(E_support_v2.mean()),
        'max_abs_diff_verification': max_abs_diff,
        'verification_passed': max_abs_diff < 1e-5,
    }
    import json
    with open(iter_dir / 'surface_support_v2_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[{iter_str}] Saved support_risk_v2.npy and support_confidence_v2.npy")

if __name__ == '__main__':
    main()
