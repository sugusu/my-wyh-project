import argparse, sys, os, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
from recyclegs.config import load_config, save_np, save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])

    print("[1/3] Loading object domain...")
    object_indices = np.load(out_dir / 'object_indices.npy')
    base = np.load(out_dir / 'gaussian_base_features.npz')
    scale_ratio = base['scale_ratio']
    scale_volume = base['scale_volume']

    obj_scale_ratio = scale_ratio[object_indices]
    obj_scale_volume = scale_volume[object_indices]

    p_low = cfg['scale']['percentile_low']
    p_high = cfg['scale']['percentile_high']

    def quantile_score_on_object(arr):
        lo, hi = np.percentile(arr, [p_low*100, p_high*100])
        clipped = arr.clip(lo, hi)
        return (clipped - lo) / (hi - lo + 1e-8)

    print("[2/3] Computing percentile normalization on object domain...")
    obj_r_anom = quantile_score_on_object(obj_scale_ratio)
    obj_v_anom = quantile_score_on_object(np.log(obj_scale_volume + 1e-8))
    obj_scale_anomaly = (obj_r_anom + obj_v_anom) / 2.0

    full_anomaly = np.zeros(len(scale_ratio), dtype=np.float32)
    full_anomaly[object_indices] = obj_scale_anomaly

    save_np(full_anomaly, out_dir / 'object_scale_anomaly.npy')

    stats = {
        'percentile_low': p_low,
        'percentile_high': p_high,
        'num_object_gaussians': len(object_indices),
        'object_scale_anomaly_mean': float(obj_scale_anomaly.mean()),
        'note': 'Percentile normalization computed only on object-supported Gaussians',
    }
    save_json(stats, out_dir / 'object_scale_anomaly_stats.json')
    print(f"[3/3] Object scale anomaly: mean={obj_scale_anomaly.mean():.4f}")
    print("  Saved.")

if __name__ == '__main__':
    main()
