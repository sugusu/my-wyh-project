#!/usr/bin/env python3
"""Evaluate random seed stability: compute removed set metrics across 10 seeds."""
import argparse, json, os, sys, numpy as np, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()
    n_seeds = args.seeds

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.locked_config) as f:
        locked_cfg = yaml.safe_load(f)

    scene_name = cfg.get('scene_name', 'scene_01')
    rel_dir = Path(cfg['reliability_output_dir'])
    iter_dir = rel_dir / 'iter_15000'
    ckpt_path = cfg['checkpoint_path']
    ratio = locked_cfg.get('prune_ratio', 0.005)
    ratio_str = f"ratio_{int(ratio*1000):03d}"

    out_base = Path(locked_cfg['project_root']) / 'outputs' / 'prune_only' / scene_name / ratio_str

    # Load eligible pool
    eligible = np.load(out_base / 'eligible_indices.npy')
    N_eligible = len(eligible)
    print(f"  Eligible pool: {N_eligible}")

    # Load total Gaussian info
    from plyfile import PlyData
    ply = PlyData.read(ckpt_path)
    N_total = ply['vertex'].count

    # Compute K using same formula as select_prune_indices
    K_target = min(
        round(0.5 / 100 * N_total),
        round(0.30 * N_eligible)
    )
    K_target = max(K_target, 1)
    print(f"  K = {K_target}")

    # Load feature data
    base_path = rel_dir / 'gaussian_base_features.npz'
    base = np.load(base_path)
    opacity_sigmoid = base['opacity_sigmoid']

    err = np.load(iter_dir / 'geometry_errors.npz')
    d_center_norm_full = np.full(N_total, np.nan, dtype=np.float32)
    cgi = np.load(iter_dir / 'candidate_global_indices.npy')
    if len(cgi) == len(err['d_center_norm']):
        d_center_norm_full[cgi] = err['d_center_norm']

    mask_risk_mean_full = np.full(N_total, np.nan, dtype=np.float32)
    mr = np.load(iter_dir / 'mask_risk_mean.npy')
    if len(cgi) == len(mr):
        mask_risk_mean_full[cgi] = mr

    mask_risk_boundary_full = np.full(N_total, np.nan, dtype=np.float32)
    mrb = np.load(iter_dir / 'mask_risk_boundary.npy')
    if len(cgi) == len(mrb):
        mask_risk_boundary_full[cgi] = mrb

    results_by_seed = {}
    all_d_center_means = []
    all_mask_risk_means = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        pool = eligible.copy()
        rng.shuffle(pool)
        prune_idx = pool[:K_target].copy()
        prune_idx = np.sort(prune_idx)

        out_dir = out_base / f'random_seed_{seed}'
        os.makedirs(out_dir, exist_ok=True)
        np.save(out_dir / 'removed_indices.npy', prune_idx)

        rem_d_center = d_center_norm_full[prune_idx]
        rem_mask_risk = mask_risk_mean_full[prune_idx]
        rem_mask_boundary = mask_risk_boundary_full[prune_idx]
        rem_opacity = opacity_sigmoid[prune_idx]

        valid_dc = np.isfinite(rem_d_center)
        valid_mr = np.isfinite(rem_mask_risk)
        valid_mb = np.isfinite(rem_mask_boundary)

        seed_metrics = {
            'seed': int(seed),
            'K': int(len(prune_idx)),
            'removed_d_center_norm_mean': float(np.nanmean(rem_d_center)) if valid_dc.any() else None,
            'removed_d_center_norm_median': float(np.nanmedian(rem_d_center)) if valid_dc.any() else None,
            'removed_d_center_norm_p90': float(np.nanpercentile(rem_d_center, 90)) if valid_dc.any() else None,
            'removed_mask_risk_mean': float(np.nanmean(rem_mask_risk)) if valid_mr.any() else None,
            'removed_mask_risk_median': float(np.nanmedian(rem_mask_risk)) if valid_mr.any() else None,
            'removed_mask_risk_boundary_mean': float(np.nanmean(rem_mask_boundary)) if valid_mb.any() else None,
            'removed_opacity_mean': float(rem_opacity.mean()),
            'removed_opacity_median': float(np.median(rem_opacity)),
        }
        results_by_seed[str(seed)] = seed_metrics

        with open(out_dir / 'removed_set_metrics.json', 'w') as f:
            json.dump(seed_metrics, f, indent=2)

        if valid_dc.any():
            all_d_center_means.append(np.nanmean(rem_d_center))
        if valid_mr.any():
            all_mask_risk_means.append(np.nanmean(rem_mask_risk))

        print(f"  Seed {seed}: d_center_mean={seed_metrics['removed_d_center_norm_mean']:.6f}, "
              f"mask_risk_mean={seed_metrics['removed_mask_risk_mean']:.4f}")

    # Aggregate across seeds
    agg = {
        'n_seeds': int(n_seeds),
        'K': int(K_target),
        'd_center_norm_mean': {
            'mean': float(np.mean(all_d_center_means)) if all_d_center_means else None,
            'std': float(np.std(all_d_center_means)) if all_d_center_means else None,
            'min': float(np.min(all_d_center_means)) if all_d_center_means else None,
            'max': float(np.max(all_d_center_means)) if all_d_center_means else None,
            'values': [float(v) for v in all_d_center_means],
        },
        'mask_risk_mean': {
            'mean': float(np.mean(all_mask_risk_means)) if all_mask_risk_means else None,
            'std': float(np.std(all_mask_risk_means)) if all_mask_risk_means else None,
            'values': [float(v) for v in all_mask_risk_means],
        },
    }

    # Compute mask_risk percentile of actual mask_risk method in random distribution
    mr_actual_path = out_base / 'mask_risk' / 'removed_set_metrics.json'
    if mr_actual_path.exists() and all_d_center_means:
        with open(mr_actual_path) as f:
            actual = json.load(f)
        mr_actual_d_center = actual.get('d_center_norm_mean', None)
        if mr_actual_d_center is not None:
            pct = np.mean(np.array(all_d_center_means) < mr_actual_d_center) * 100
            agg['mask_risk_d_center_percentile_in_random'] = float(pct)
            agg['mask_risk_d_center_actual'] = float(mr_actual_d_center)
            print(f"\n  mask_risk d_center_norm: {mr_actual_d_center:.6f} at {pct:.1f}th percentile of random distribution")

        mr_actual_risk = actual.get('mask_risk_mean', None)
        if mr_actual_risk is not None and all_mask_risk_means:
            pct_risk = np.mean(np.array(all_mask_risk_means) < mr_actual_risk) * 100
            agg['mask_risk_mask_risk_percentile_in_random'] = float(pct_risk)
            agg['mask_risk_mask_risk_actual'] = float(mr_actual_risk)
            print(f"  mask_risk mask_risk_mean: {mr_actual_risk:.4f} at {pct_risk:.1f}th percentile of random distribution")

    stability_path = out_base / 'random_seed_stability.json'
    with open(stability_path, 'w') as f:
        json.dump({'per_seed': results_by_seed, 'aggregate': agg}, f, indent=2)
    print(f"\nSaved random seed stability: {stability_path}")

    # Also save to stage2a5 path
    stage2a5_dir = Path(locked_cfg['project_root']) / 'outputs' / 'prune_only' / 'stage2a5'
    os.makedirs(stage2a5_dir, exist_ok=True)
    with open(stage2a5_dir / f'{scene_name}_random_seed_stability.json', 'w') as f:
        json.dump({'per_seed': results_by_seed, 'aggregate': agg}, f, indent=2)
    print(f"  Also saved to stage2a5/")

if __name__ == '__main__':
    main()
