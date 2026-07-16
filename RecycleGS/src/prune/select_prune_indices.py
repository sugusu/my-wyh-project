#!/usr/bin/env python3
"""Select prune indices for each method from eligible pool."""
import argparse, json, os, sys, numpy as np, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
    parser.add_argument('--methods', type=str, default='random,low_opacity,low_contribution,mask_risk,oracle')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.locked_config) as f:
        locked_cfg = yaml.safe_load(f)

    scene_name = cfg.get('scene_name', 'scene_01')
    rel_dir = Path(cfg['reliability_output_dir'])
    iter_dir = rel_dir / 'iter_15000'
    ratio = locked_cfg.get('prune_ratio', 0.005)
    ratio_str = f"ratio_{int(ratio*1000):03d}"

    out_base = Path(locked_cfg['project_root']) / 'outputs' / 'prune_only' / scene_name / ratio_str
    methods = [m.strip() for m in args.methods.split(',')]

    # 1. Load eligible pool
    eligible = np.load(out_base / 'eligible_indices.npy')
    N_eligible = len(eligible)
    print(f"  Eligible pool: {N_eligible}")

    # 2. Determine N_total from gaussian_base_features
    base_path = rel_dir / 'gaussian_base_features.npz'
    base = np.load(base_path)
    N_total = len(base['opacity_sigmoid'])
    opacity_sigmoid = base['opacity_sigmoid']
    print(f"  Total Gaussians: {N_total}")

    # 3. Load checkpoint PLY to get XYZ for total count
    ckpt_path = cfg['checkpoint_path']
    from plyfile import PlyData
    ply = PlyData.read(ckpt_path)
    N_total_ply = ply['vertex'].count
    if N_total != N_total_ply:
        print(f"  WARNING: base features count {N_total} != PLY count {N_total_ply}, using {max(N_total, N_total_ply)}")
    N_total = max(N_total, N_total_ply)

    # 4. Compute K
    K = min(
        round(0.5 / 100 * N_total),  # 0.5% of total
        round(0.30 * N_eligible)      # max 30% of eligible
    )
    K = max(K, 1)
    print(f"  K = min(0.5% * {N_total}, 30% * {N_eligible}) = {K}")

    # 5. Load data for each method
    # Load mask_risk_mean for mask_risk method
    mask_risk_mean = None
    mr_path = iter_dir / 'mask_risk_mean.npy'
    if mr_path.exists():
        mr = np.load(mr_path)
        mask_risk_mean = np.full(N_total, np.nan, dtype=np.float32)
        # Map to full array via candidate_global_indices
        cgi_path = iter_dir / 'candidate_global_indices.npy'
        if cgi_path.exists():
            cgi = np.load(cgi_path)
            if len(cgi) == len(mr):
                mask_risk_mean[cgi] = mr
        else:
            cand_path = rel_dir / 'candidate_object_indices.npy'
            if cand_path.exists():
                cand = np.load(cand_path)
                if len(cand) == len(mr):
                    mask_risk_mean[cand] = mr

    # Load contribution
    contribution = None
    contrib_path = rel_dir / 'contribution.npy'
    if contrib_path.exists():
        c = np.load(contrib_path)
        if len(c) == N_total:
            contribution = c

    # Load GT errors for oracle
    d_center_norm = None
    err_path = iter_dir / 'geometry_errors.npz'
    if err_path.exists():
        err = np.load(err_path)
        de = err['d_center_norm']
        d_center_norm = np.full(N_total, np.nan, dtype=np.float32)
        cgi_path = iter_dir / 'candidate_global_indices.npy'
        if cgi_path.exists():
            cgi = np.load(cgi_path)
            if len(cgi) == len(de):
                d_center_norm[cgi] = de

    # 6. Selection for each method
    rng = np.random.RandomState(locked_cfg.get('seed', 42))
    method_results = {}
    for method in methods:
        out_dir = out_base / method
        os.makedirs(out_dir, exist_ok=True)

        if method == 'random':
            rng.shuffle(eligible)
            prune_idx = eligible[:K].copy()

        elif method == 'low_opacity':
            # Lowest opacity among eligible
            sub_opac = opacity_sigmoid[eligible]
            order = np.argsort(sub_opac)
            prune_idx = eligible[order[:K]]

        elif method == 'low_contribution':
            if contribution is not None:
                sub_contrib = contribution[eligible]
                order = np.argsort(sub_contrib)
                prune_idx = eligible[order[:K]]
            else:
                print(f"  WARNING: contribution not available for {method}, using random")
                rng.shuffle(eligible)
                prune_idx = eligible[:K].copy()

        elif method == 'mask_risk':
            if mask_risk_mean is not None:
                sub_risk = mask_risk_mean[eligible]
                # Highest risk = most likely to be pruned
                # Sort descending (highest risk first)
                valid_risk = np.isfinite(sub_risk)
                if valid_risk.sum() > 0:
                    sub_risk_valid = sub_risk.copy()
                    sub_risk_valid[~valid_risk] = -np.inf
                    order = np.argsort(sub_risk_valid)[::-1]
                    n_avail = min(K, int(valid_risk.sum()))
                    prune_idx = eligible[order[:n_avail]]
                    if n_avail < K:
                        remaining = np.setdiff1d(eligible, prune_idx, assume_unique=True)
                        rng.shuffle(remaining)
                        extra = remaining[:K - n_avail]
                        prune_idx = np.concatenate([prune_idx, extra])
                else:
                    print(f"  WARNING: no valid mask_risk values, using random")
                    rng.shuffle(eligible)
                    prune_idx = eligible[:K].copy()
            else:
                print(f"  WARNING: mask_risk_mean not available for {method}, using random")
                rng.shuffle(eligible)
                prune_idx = eligible[:K].copy()

        elif method == 'oracle':
            if d_center_norm is not None:
                sub_err = d_center_norm[eligible]
                valid_err = np.isfinite(sub_err)
                if valid_err.sum() > 0:
                    sub_err_valid = sub_err.copy()
                    sub_err_valid[~valid_err] = -np.inf
                    order = np.argsort(sub_err_valid)[::-1]  # highest error first
                    n_avail = min(K, int(valid_err.sum()))
                    prune_idx = eligible[order[:n_avail]]
                    if n_avail < K:
                        remaining = np.setdiff1d(eligible, prune_idx, assume_unique=True)
                        rng.shuffle(remaining)
                        extra = remaining[:K - n_avail]
                        prune_idx = np.concatenate([prune_idx, extra])
                else:
                    print(f"  WARNING: no valid GT errors, using random")
                    rng.shuffle(eligible)
                    prune_idx = eligible[:K].copy()
            else:
                print(f"  WARNING: GT errors not available for oracle, using random")
                rng.shuffle(eligible)
                prune_idx = eligible[:K].copy()

        else:
            raise ValueError(f"Unknown method: {method}")

        prune_idx = np.sort(np.unique(prune_idx))
        actual_K = len(prune_idx)
        np.save(out_dir / 'prune_indices.npy', prune_idx)
        print(f"  {method}: pruned {actual_K} Gaussians")
        method_results[method] = {
            'method': method,
            'pruned_count': int(actual_K),
            'K_target': int(K),
        }

    # 7. Save metadata
    metadata = {
        'scene': scene_name,
        'ratio_str': ratio_str,
        'prune_ratio': ratio,
        'N_total': int(N_total),
        'N_eligible': int(N_eligible),
        'K': int(K),
        'methods': method_results,
    }
    with open(out_base / 'prune_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved prune_metadata.json")

if __name__ == '__main__':
    main()
