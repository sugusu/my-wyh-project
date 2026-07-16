import argparse, sys, os, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
from recyclegs.config import load_config, save_np, save_npz, save_json

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

    print("[1/5] Loading data...")
    object_indices = np.load(out_dir / 'object_indices.npy')
    base = np.load(out_dir / 'gaussian_base_features.npz')
    N = len(base['xyz'])

    E_support = np.load(out_dir / 'object_surface_support_risk.npy')
    E_scale = np.load(out_dir / 'object_scale_anomaly.npy')
    E_normal = np.load(out_dir / 'object_normal_conflict.npy')
    normal_valid = np.load(out_dir / 'object_normal_valid.npy')

    E_support_obj = E_support[object_indices]
    E_scale_obj = E_scale[object_indices]
    E_normal_obj = E_normal[object_indices]
    normal_valid_obj = normal_valid[object_indices]

    E_support_norm = percentile_clip_and_scale(E_support_obj)
    E_scale_norm = percentile_clip_and_scale(E_scale_obj)
    E_normal_norm = percentile_clip_and_scale(E_normal_obj)

    print("[2/5] Computing Risk A...")
    R_A = 0.70 * E_support_norm + 0.30 * E_scale_norm

    print("[3/5] Computing Risk B (only where normal_valid)...")
    R_B = np.full(len(object_indices), np.nan, dtype=np.float32)
    valid_mask = normal_valid_obj
    if valid_mask.sum() > 0:
        R_B[valid_mask] = (
            0.45 * E_normal_norm[valid_mask]
            + 0.55 * E_support_norm[valid_mask]
        )

    print("[4/5] Computing Risk C (renormalize when normal invalid)...")
    R_C = np.zeros(len(object_indices), dtype=np.float32)
    valid_c = valid_mask
    R_C[valid_c] = (
        0.35 * E_normal_norm[valid_c]
        + 0.45 * E_support_norm[valid_c]
        + 0.20 * E_scale_norm[valid_c]
    )
    invalid_c = ~valid_c
    if invalid_c.sum() > 0:
        total_w = 0.45 + 0.20
        R_C[invalid_c] = (
            0.45 / total_w * E_support_norm[invalid_c]
            + 0.20 / total_w * E_scale_norm[invalid_c]
        )

    print(f"[5/5] Computing stats and saving...")
    full_RA = np.zeros(N, dtype=np.float32)
    full_RB = np.zeros(N, dtype=np.float32)
    full_RC = np.zeros(N, dtype=np.float32)
    full_RA[object_indices] = R_A
    full_RB[object_indices] = R_B
    full_RC[object_indices] = R_C

    save_np(full_RA, out_dir / 'object_risk_A.npy')
    save_np(full_RB, out_dir / 'object_risk_B.npy')
    save_np(full_RC, out_dir / 'object_risk_C.npy')

    save_npz(out_dir / 'object_risk_features.npz',
             E_support_obj=E_support_obj, E_support_norm=E_support_norm,
             E_scale_obj=E_scale_obj, E_scale_norm=E_scale_norm,
             E_normal_obj=E_normal_obj, E_normal_norm=E_normal_norm,
             R_A=R_A, R_B=R_B, R_C=R_C)

    stats = {
        'num_object_gaussians': int(len(object_indices)),
        'risk_A': {'mean': float(np.nanmean(R_A)), 'median': float(np.nanmedian(R_A))},
        'risk_B': {'mean': float(np.nanmean(R_B)), 'median': float(np.nanmedian(R_B)),
                   'valid_count': int(valid_mask.sum())},
        'risk_C': {'mean': float(R_C.mean()), 'median': float(np.median(R_C))},
        'normal_valid_count': int(valid_mask.sum()),
        'depth_feature_disabled': True,
        'depth_feature_reason': 'E_depth is constant and non-informative (AUROC=0.5)',
    }
    save_json(stats, out_dir / 'object_risk_stats.json')
    print(f"  Risk A: mean={np.nanmean(R_A):.4f}")
    print(f"  Risk B: mean={np.nanmean(R_B):.4f} (valid: {valid_mask.sum()})")
    print(f"  Risk C: mean={R_C.mean():.4f}")
    print("  Done.")

if __name__ == '__main__':
    main()
