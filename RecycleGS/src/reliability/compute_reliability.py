import argparse, sys, os, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
from recyclegs.config import load_config, save_np, save_npz, save_json
from plyfile import PlyData, PlyElement

def percentile_clip_and_scale(arr):
    lo, hi = np.percentile(arr, [5, 95])
    clipped = arr.clip(lo, hi)
    scaled = (clipped - lo) / (hi - lo + 1e-8)
    return scaled.clip(0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])

    print("[1/3] Loading risk features...")
    mask_risk = np.load(out_dir / 'mask_risk.npy')
    normal_conflict = np.load(out_dir / 'normal_conflict.npy')
    depth_conflict = np.load(out_dir / 'depth_order_conflict.npy')
    surface_risk = np.load(out_dir / 'surface_support_risk.npy')
    scale_anomaly = np.load(out_dir / 'scale_anomaly.npy')

    w = cfg['risk_weights']
    features = {
        'E_mask': percentile_clip_and_scale(mask_risk),
        'E_normal': percentile_clip_and_scale(normal_conflict),
        'E_depth': percentile_clip_and_scale(depth_conflict),
        'E_support': percentile_clip_and_scale(surface_risk),
        'E_scale': percentile_clip_and_scale(scale_anomaly),
    }

    R = (w['mask'] * features['E_mask']
         + w['normal'] * features['E_normal']
         + w['depth_order'] * features['E_depth']
         + w['surface_support'] * features['E_support']
         + w['scale'] * features['E_scale'])

    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = base['xyz']

    save_np(R, out_dir / 'risk_scores.npy')
    save_npz(out_dir / 'reliability_features.npz', **features, risk_scores=R)

    risk_pct = np.percentile(R, [0, 25, 50, 75, 90, 95, 99, 100])
    stats = {'weights': w, 'risk_percentiles': {str(k): float(v) for k, v in zip(['0','25','50','75','90','95','99','100'], risk_pct)}}
    save_json(stats, out_dir / 'risk_stats.json')

    R_norm = ((R - R.min()) / (R.max() - R.min() + 1e-8) * 255).astype(np.uint8)
    colored = np.zeros((len(xyz), 6))
    colored[:, :3] = xyz
    colored[:, 3] = R_norm
    colored[:, 4] = 0
    colored[:, 5] = 255 - R_norm
    ply_arr = np.array([tuple(r) for r in colored], dtype=[('x', '<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    PlyData([PlyElement.describe(ply_arr, 'vertex')]).write(str(out_dir / 'risk_colored.ply'))
    print(f"[2/3] Risk scores computed")
    print(f"[3/3] Saved to {out_dir}")

if __name__ == '__main__':
    main()
