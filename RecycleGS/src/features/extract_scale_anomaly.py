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

    base = np.load(out_dir / 'gaussian_base_features.npz')
    scale_ratio = base['scale_ratio']
    scale_volume = base['scale_volume']

    p_low = cfg['scale']['percentile_low']
    p_high = cfg['scale']['percentile_high']

    def quantile_score(arr):
        lo, hi = np.percentile(arr, [p_low*100, p_high*100])
        clipped = arr.clip(lo, hi)
        return (clipped - lo) / (hi - lo + 1e-8)

    r_anom = quantile_score(scale_ratio)
    v_anom = quantile_score(np.log(scale_volume + 1e-8))
    scale_anomaly = (r_anom + v_anom) / 2.0

    save_np(scale_anomaly, out_dir / 'scale_anomaly.npy')
    save_json({'percentile_low': p_low, 'percentile_high': p_high}, out_dir / 'scale_anomaly_stats.json')
    print("Scale anomaly saved.")

if __name__ == '__main__':
    main()
