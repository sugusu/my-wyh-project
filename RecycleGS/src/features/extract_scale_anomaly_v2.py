import argparse, sys, os, numpy as np
from pathlib import Path
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
    return xyz, scales

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--iteration', type=int, required=True, choices=[7000, 15000])
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1f_scene01'
    os.makedirs(debug_dir, exist_ok=True)
    p_low = cfg['scale']['percentile_low']
    p_high = cfg['scale']['percentile_high']

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

    xyz_all, scales_all = load_ply_raw(ply_path)
    scale_linear = np.exp(scales_all)
    scale_min = scale_linear.min(axis=1)
    scale_mid = np.median(scale_linear, axis=1)
    scale_max = scale_linear.max(axis=1)
    scale_ratio = scale_max / (scale_min + 1e-8)
    scale_volume = np.prod(scale_linear, axis=1)

    cand_scale_min = scale_min[candidate_indices]
    cand_scale_mid = scale_mid[candidate_indices]
    cand_scale_max = scale_max[candidate_indices]
    cand_scale_ratio = scale_ratio[candidate_indices]
    cand_scale_volume = scale_volume[candidate_indices]

    def percentile_score(arr):
        lo, hi = np.percentile(arr, [p_low * 100, p_high * 100])
        clipped = arr.clip(lo, hi)
        return (clipped - lo) / (hi - lo + 1e-8)

    anisotropy = percentile_score(cand_scale_ratio)
    volume_log = np.log(cand_scale_volume + 1e-8)
    volume_extreme = percentile_score(volume_log)

    anisotropy_percentile = anisotropy
    volume_extreme_percentile = volume_extreme
    E_scale_v2 = np.maximum(anisotropy_percentile, volume_extreme_percentile)

    print(f"  scale_ratio: mean={cand_scale_ratio.mean():.4f}, "
          f"median={np.median(cand_scale_ratio):.4f}")
    print(f"  scale_volume: mean={cand_scale_volume.mean():.4e}, "
          f"median={np.median(cand_scale_volume):.4e}")
    print(f"  anisotropy_percentile: mean={anisotropy_percentile.mean():.4f}")
    print(f"  volume_extreme_percentile: mean={volume_extreme_percentile.mean():.4f}")
    print(f"  E_scale_v2: mean={E_scale_v2.mean():.4f}")

    base_features = {
        'scale_min': cand_scale_min,
        'scale_mid': cand_scale_mid,
        'scale_max': cand_scale_max,
        'scale_ratio': cand_scale_ratio,
        'scale_volume': cand_scale_volume,
        'anisotropy_percentile': anisotropy_percentile,
        'volume_extreme_percentile': volume_extreme_percentile,
    }

    np.save(iter_dir / 'scale_risk_v2.npy', E_scale_v2)
    for name, arr in base_features.items():
        np.save(iter_dir / f'scale_{name}.npy' if not name.startswith('scale_') else f'{name}.npy', arr)

    import json
    stats = {
        'iteration': iteration,
        'candidate_count': len(candidate_indices),
        'E_scale_v2_mean': float(E_scale_v2.mean()),
        'E_scale_v2_std': float(E_scale_v2.std()),
        'scale_ratio_mean': float(cand_scale_ratio.mean()),
        'scale_volume_mean': float(cand_scale_volume.mean()),
        'percentile_low': p_low,
        'percentile_high': p_high,
    }
    with open(iter_dir / 'scale_anomaly_v2_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"[{iter_str}] Saved scale_risk_v2.npy")

if __name__ == '__main__':
    main()
