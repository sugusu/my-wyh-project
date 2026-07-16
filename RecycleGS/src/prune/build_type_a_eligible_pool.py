#!/usr/bin/env python3
"""Build Type-A eligible pool for mask-risk pruning."""
import argparse, json, os, sys, numpy as np, yaml
from pathlib import Path
from scipy.stats import rankdata
from plyfile import PlyData, PlyElement

sys.path.insert(0, '/data/wyh/RecycleGS/src')

def save_colored_ply(xyz, colors, save_path):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.empty(len(xyz), dtype=dtype)
    arr['x'] = xyz[:, 0]
    arr['y'] = xyz[:, 1]
    arr['z'] = xyz[:, 2]
    arr['red'] = (colors[:, 0] * 255).clip(0, 255).astype(np.uint8)
    arr['green'] = (colors[:, 1] * 255).clip(0, 255).astype(np.uint8)
    arr['blue'] = (colors[:, 2] * 255).clip(0, 255).astype(np.uint8)
    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False).write(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
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

    out_root = Path(locked_cfg['project_root']) / 'outputs' / 'prune_only' / scene_name / ratio_str
    os.makedirs(out_root, exist_ok=True)

    N_total = None
    gaussian_base = None

    # 1. Load candidate_object_indices
    cand_obj_path = rel_dir / 'candidate_object_indices.npy'
    if cand_obj_path.exists():
        candidate_indices = np.load(cand_obj_path)
    else:
        candidate_indices = np.load(iter_dir / 'candidate_indices.npy')
    print(f"  candidate_object_indices: {len(candidate_indices)}")

    # 2. Load valid_view_count and mask_support (full array)
    root_files = {}
    for fname in ['valid_view_count.npy', 'mask_support_unweighted.npy']:
        fp = rel_dir / fname
        if fp.exists():
            root_files[fname] = np.load(fp)
        else:
            fp = iter_dir / fname
            if fp.exists():
                root_files[fname] = np.load(fp)
    valid_view_count = root_files.get('valid_view_count.npy')
    mask_support = root_files.get('mask_support_unweighted.npy')

    full_len = None
    for arr in root_files.values():
        if full_len is None:
            full_len = len(arr)
        break

    # 3. Load mask_risk_variance, boundary_ratio from iter_15000
    mask_risk_variance = np.load(iter_dir / 'mask_risk_variance.npy') if (iter_dir / 'mask_risk_variance.npy').exists() else None

    boundary_ratio_path = iter_dir / 'mask_consistency_boundary_ratio.npy'
    if boundary_ratio_path.exists():
        boundary_view_ratio = np.load(boundary_ratio_path)
    else:
        boundary_view_ratio = None

    # 4. Load contribution from root
    contribution_path = rel_dir / 'contribution.npy'
    if contribution_path.exists():
        contribution = np.load(contribution_path)
    else:
        contribution = None

    # 5. Load gaussian_base_features
    base_path = rel_dir / 'gaussian_base_features.npz'
    if base_path.exists():
        base = np.load(base_path)
        N_total = len(base['opacity_sigmoid'])
        opacity_sigmoid = base['opacity_sigmoid']
        xyz_all = base['xyz']
    else:
        base = None
        q = np.load(cfg['checkpoint_path'].replace('.ply', '.npy')) if cfg.get('checkpoint_path', '').endswith('.ply') else None
        raise RuntimeError(f"gaussian_base_features.npz not found at {base_path}")

    print(f"  Total Gaussians: {N_total}")

    # Build full-size arrays for candidate-level features
    candidate_mask = np.zeros(N_total, dtype=bool)
    candidate_mask[candidate_indices] = True
    N_candidate = candidate_indices.shape[0]

    # Build full arrays for candidate-localized features
    cand_risk_var_full = np.full(N_total, np.nan, dtype=np.float32)
    cand_boundary_full = np.full(N_total, np.nan, dtype=np.float32)

    if mask_risk_variance is not None and len(mask_risk_variance) == N_candidate:
        cand_risk_var_full[candidate_indices] = mask_risk_variance
    elif mask_risk_variance is not None and len(mask_risk_variance) != N_candidate:
        print(f"  WARNING: mask_risk_variance length {len(mask_risk_variance)} != candidate count {N_candidate}")
        # Try to align via candidate_global_indices
        cgi = np.load(iter_dir / 'candidate_global_indices.npy') if (iter_dir / 'candidate_global_indices.npy').exists() else None
        if cgi is not None and len(cgi) == len(mask_risk_variance):
            cand_risk_var_full[cgi] = mask_risk_variance

    if boundary_view_ratio is not None and len(boundary_view_ratio) == N_candidate:
        cand_boundary_full[candidate_indices] = boundary_view_ratio
    elif boundary_view_ratio is not None and len(boundary_view_ratio) != N_candidate:
        cgi = np.load(iter_dir / 'candidate_global_indices.npy') if (iter_dir / 'candidate_global_indices.npy').exists() else None
        if cgi is not None and len(cgi) == len(boundary_view_ratio):
            cand_boundary_full[cgi] = boundary_view_ratio

    # 6. Compute contribution_percentile
    if contribution is not None:
        ranks = rankdata(contribution, method='average')
        contribution_percentile = ranks / len(contribution)
    else:
        contribution_percentile = np.ones(N_total)

    # 7. Apply eligibility criteria
    eligibility_cfg = locked_cfg.get('eligibility', {})
    min_valid_views = eligibility_cfg.get('min_valid_views', 5)
    max_mask_variance = eligibility_cfg.get('max_mask_variance', 0.20)
    max_boundary_ratio = eligibility_cfg.get('max_boundary_view_ratio', 0.50)
    max_contrib_pct = eligibility_cfg.get('max_contribution_percentile', 0.80)

    eligible = np.ones(N_total, dtype=bool)
    if candidate_mask is not None:
        eligible = eligible & candidate_mask
    if valid_view_count is not None:
        vvc = valid_view_count if len(valid_view_count) == N_total else np.full(N_total, 0)
        eligible = eligible & (vvc >= min_valid_views)
    if cand_risk_var_full is not None:
        eligible = eligible & (cand_risk_var_full <= max_mask_variance)
    if cand_boundary_full is not None:
        eligible = eligible & (cand_boundary_full <= max_boundary_ratio)
    if contribution_percentile is not None:
        eligible = eligible & (contribution_percentile <= max_contrib_pct)

    eligible_indices = np.where(eligible)[0]
    protected_indices = np.where(candidate_mask & ~eligible)[0]
    N_eligible = len(eligible_indices)

    print(f"  Eligible: {N_eligible} / {N_total} total ({100*N_eligible/N_total:.2f}%)")
    print(f"  Protected (candidate but not eligible): {len(protected_indices)}")

    # 8. Save
    np.save(out_root / 'eligible_indices.npy', eligible_indices)
    np.save(out_root / 'protected_indices.npy', protected_indices)

    stats = {
        'scene_name': scene_name,
        'total_gaussians': int(N_total),
        'candidate_count': int(N_candidate),
        'eligible_count': int(N_eligible),
        'protected_count': int(len(protected_indices)),
        'eligible_pct': float(100 * N_eligible / N_total),
        'criteria': {
            'candidate_object': True,
            'min_valid_views': min_valid_views,
            'max_mask_variance': max_mask_variance,
            'max_boundary_view_ratio': max_boundary_ratio,
            'max_contribution_percentile': max_contrib_pct,
        },
    }
    with open(out_root / 'eligible_pool_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats: {out_root / 'eligible_pool_stats.json'}")

    # Save colored PLYs
    if xyz_all is not None:
        # Color eligible green, others gray
        colors = np.ones((N_total, 3)) * 0.5
        if N_eligible > 0:
            colors[eligible_indices] = [0.0, 0.8, 0.0]
        save_colored_ply(xyz_all, colors, out_root / 'eligible_gaussians.ply')
        print(f"  Saved eligible_gaussians.ply: {N_eligible} green")

        if len(protected_indices) > 0:
            prot_colors = np.ones((len(protected_indices), 3)) * 0.5
            prot_colors[:, 0] = 0.8
            prot_colors[:, 1] = 0.4
            prot_colors[:, 2] = 0.0
            prot_xyz = xyz_all[protected_indices]
            save_colored_ply(prot_xyz, prot_colors, out_root / 'protected_gaussians.ply')
            print(f"  Saved protected_gaussians.ply: {len(protected_indices)} orange")
    else:
        print(f"  WARNING: xyz_all not available, skipping PLY export")

    print(f"  Done. Output: {out_root}")

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/data/wyh/RecycleGS/src')
    main()
