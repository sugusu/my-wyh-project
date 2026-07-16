#!/usr/bin/env python3
"""Evaluate pruned checkpoint: proxy geometry metrics (no mesh extraction)."""
import argparse, json, os, sys, numpy as np, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
    parser.add_argument('--all-methods', action='store_true')
    args = parser.parse_args()

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

    with open(out_base / 'prune_metadata.json') as f:
        metadata = json.load(f)
    methods = list(metadata['methods'].keys()) if args.all_methods else ['baseline']

    from plyfile import PlyData
    ply_orig = PlyData.read(ckpt_path)
    N_before = ply_orig['vertex'].count

    # Load GT mesh info for distance computation
    gt_mesh_path = None
    for candidate in ['meshes/gt.ply', 'mesh.ply', 'gt_mesh.ply', 'gt.ply']:
        p = Path(cfg.get('scene_dir', '')) / candidate
        if p.exists():
            gt_mesh_path = p
            break
    if gt_mesh_path is None:
        scene_dir = Path(cfg.get('scene_dir', ''))
        for p in sorted(scene_dir.rglob('*.ply')):
            print(f"  Checking {p}...")
        print(f"  WARNING: no GT mesh found in {scene_dir}")

    # Load feature data for proxy metrics
    base_path = rel_dir / 'gaussian_base_features.npz'
    base = np.load(base_path)
    xyz_all = base['xyz']

    err = np.load(iter_dir / 'geometry_errors.npz')
    d_center_norm_full = np.full(N_before, np.nan, dtype=np.float32)
    cgi = np.load(iter_dir / 'candidate_global_indices.npy')
    if len(cgi) == len(err['d_center_norm']):
        d_center_norm_full[cgi] = err['d_center_norm']

    mask_risk_mean_full = np.full(N_before, np.nan, dtype=np.float32)
    mr = np.load(iter_dir / 'mask_risk_mean.npy')
    if len(cgi) == len(mr):
        mask_risk_mean_full[cgi] = mr

    opacity_sigmoid = base['opacity_sigmoid']

    # Load scene GT xyz from original checkpoint xyz (full gaussian positions)
    orig_xyz = np.stack([np.asarray(ply_orig['vertex'][p]) for p in ['x', 'y', 'z']], axis=1)

    results = {}
    for method in methods:
        if method == 'baseline':
            ply_path = ckpt_path
            out_dir = out_base / method
            prune_idx = np.array([], dtype=np.int64)
        else:
            out_dir = out_base / method
            ply_path = out_dir / 'retained.ply'
            prune_idx = np.load(out_dir / 'prune_indices.npy')

        ply = PlyData.read(ply_path)
        N_after = ply['vertex'].count
        K = len(prune_idx)

        remaining_indices = np.setdiff1d(np.arange(N_before), prune_idx, assume_unique=True)

        # Proxy geometry metrics on remaining Gaussians
        rem_d_center = d_center_norm_full[remaining_indices]
        rem_risk = mask_risk_mean_full[remaining_indices]
        rem_opacity = opacity_sigmoid[remaining_indices]

        valid_dc = np.isfinite(rem_d_center)
        valid_risk = np.isfinite(rem_risk)

        geo_metrics = {
            'N_before': int(N_before),
            'N_after': int(N_after),
            'K': int(K),
        }

        # Surface Proximity Ratio: fraction of remaining Gaussians with d_center_norm < 0.02 (within 2cm of surface)
        if valid_dc.any():
            geo_metrics['surface_proximity_ratio_threshold_0_02'] = float(np.mean(rem_d_center[valid_dc] < 0.02))
            geo_metrics['surface_proximity_ratio_threshold_0_01'] = float(np.mean(rem_d_center[valid_dc] < 0.01))
            geo_metrics['remaining_d_center_norm_mean'] = float(np.nanmean(rem_d_center))
            geo_metrics['remaining_d_center_norm_median'] = float(np.nanmedian(rem_d_center))
            geo_metrics['remaining_d_center_norm_p90'] = float(np.nanpercentile(rem_d_center, 90))
        else:
            geo_metrics['remaining_d_center_norm_mean'] = None

        # Distribution shift: compare remaining vs full distribution
        full_dc_valid = np.isfinite(d_center_norm_full)
        if full_dc_valid.any() and valid_dc.any():
            full_mean = np.nanmean(d_center_norm_full)
            rem_mean = np.nanmean(rem_d_center)
            geo_metrics['d_center_norm_shift'] = float(rem_mean - full_mean)
            geo_metrics['d_center_norm_shift_pct'] = float((rem_mean - full_mean) / full_mean * 100) if full_mean > 0 else 0.0

        # Mask risk distribution shift
        full_risk_valid = np.isfinite(mask_risk_mean_full)
        if full_risk_valid.any() and valid_risk.any():
            full_risk_mean = np.nanmean(mask_risk_mean_full)
            rem_risk_mean = np.nanmean(rem_risk)
            geo_metrics['mask_risk_shift'] = float(rem_risk_mean - full_risk_mean)
            geo_metrics['remaining_mask_risk_mean'] = float(rem_risk_mean)
            geo_metrics['full_mask_risk_mean'] = float(full_risk_mean)

        # Opacity distribution of remaining
        geo_metrics['remaining_opacity_mean'] = float(rem_opacity.mean())
        geo_metrics['remaining_opacity_median'] = float(np.median(rem_opacity))

        # If GT mesh available, compute SPR using actual mesh distance
        if gt_mesh_path:
            try:
                import trimesh
                gt_mesh = trimesh.load(gt_mesh_path)
                gt_pts = np.asarray(gt_mesh.vertices)
                from scipy.spatial import cKDTree
                tree = cKDTree(gt_pts)
                rem_xyz = orig_xyz[remaining_indices]
                dists, _ = tree.query(rem_xyz)
                sp100 = float(np.mean(dists < 0.01))  # 1cm
                sp200 = float(np.mean(dists < 0.02))  # 2cm
                sp500 = float(np.mean(dists < 0.05))  # 5cm
                geo_metrics['spr_1cm'] = sp100
                geo_metrics['spr_2cm'] = sp200
                geo_metrics['spr_5cm'] = sp500
                geo_metrics['mean_distance_to_mesh'] = float(np.mean(dists))
                geo_metrics['median_distance_to_mesh'] = float(np.median(dists))
            except Exception as e:
                print(f"  GT mesh proximity computation failed: {e}")
                geo_metrics['mesh_proximity_error'] = str(e)

        results[method] = geo_metrics

        with open(out_dir / 'geometry_metrics.json', 'w') as f:
            json.dump(geo_metrics, f, indent=2)
        print(f"  {method}: SPR(2cm)={geo_metrics.get('surface_proximity_ratio_threshold_0_02', 'N/A'):.4f}, "
              f"d_shift={geo_metrics.get('d_center_norm_shift', 'N/A')}")

    summary_path = out_base / 'cross_method_geometry.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved cross-method geometry: {summary_path}")

if __name__ == '__main__':
    main()
