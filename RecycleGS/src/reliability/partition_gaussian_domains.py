import argparse, sys, os, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
from recyclegs.config import load_config, save_json
from plyfile import PlyData, PlyElement

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])

    print("[1/6] Loading data...")
    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = base['xyz']
    opacity_sigmoid = base['opacity_sigmoid']

    valid_view = np.load(out_dir / 'valid_view_count.npy')
    mask_support = np.load(out_dir / 'mask_support_unweighted.npy')

    contribution = np.load(out_dir / 'contribution.npy')
    N = len(xyz)

    print("[2/6] Computing percentiles for contribution and opacity...")
    p5_contrib = np.percentile(contribution[contribution > 0], 5) if contribution.sum() > 0 else 0
    p5_opacity = np.percentile(opacity_sigmoid, 5)
    print(f"  p5 contribution: {p5_contrib:.6f}")
    print(f"  p5 opacity: {p5_opacity:.6f}")

    has_min_views = valid_view >= 3

    object_mask = (
        has_min_views
        & (mask_support >= 0.50)
        & ((contribution >= p5_contrib) | (opacity_sigmoid >= p5_opacity))
    )

    background_mask = has_min_views & (mask_support <= 0.10)

    uncertain_mask = ~(object_mask | background_mask)

    object_indices = np.where(object_mask)[0]
    background_indices = np.where(background_mask)[0]
    uncertain_indices = np.where(uncertain_mask)[0]

    print(f"[3/6] Partition results:")
    print(f"  Object-supported: {len(object_indices)} ({len(object_indices)/N*100:.1f}%)")
    print(f"  Background-supported: {len(background_indices)} ({len(background_indices)/N*100:.1f}%)")
    print(f"  Uncertain: {len(uncertain_indices)} ({len(uncertain_indices)/N*100:.1f}%)")

    np.save(out_dir / 'object_indices.npy', object_indices)
    np.save(out_dir / 'background_indices.npy', background_indices)
    np.save(out_dir / 'uncertain_indices.npy', uncertain_indices)

    stats = {
        'total_gaussians': int(N),
        'object_supported': {'count': int(len(object_indices)), 'ratio': float(len(object_indices)/N)},
        'background_supported': {'count': int(len(background_indices)), 'ratio': float(len(background_indices)/N)},
        'uncertain': {'count': int(len(uncertain_indices)), 'ratio': float(len(uncertain_indices)/N)},
        'criteria': {
            'valid_view_min': 3,
            'mask_support_object_threshold': 0.50,
            'mask_support_background_threshold': 0.10,
            'p5_contribution': float(p5_contrib),
            'p5_opacity': float(p5_opacity),
        },
        'no_gt_mesh_used': True,
    }
    save_json(stats, out_dir / 'domain_partition_stats.json')

    print("[4/6] Creating colored PLY files...")
    def save_colored_ply(indices, color, name):
        pts = xyz[indices]
        colored = np.zeros((len(pts), 6))
        colored[:, :3] = pts
        colored[:, 3:] = color
        ply_arr = np.array([tuple(r) for r in colored],
                           dtype=[('x', '<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
        PlyData([PlyElement.describe(ply_arr, 'vertex')]).write(str(out_dir / name))

    save_colored_ply(object_indices, [0, 255, 0], 'object_domain.ply')
    save_colored_ply(background_indices, [255, 0, 0], 'background_domain.ply')
    save_colored_ply(uncertain_indices, [128, 128, 128], 'uncertain_domain.ply')

    full_colored = np.zeros((N, 6))
    full_colored[:, :3] = xyz
    full_colored[object_indices, 3:] = [0, 255, 0]
    full_colored[background_indices, 3:] = [255, 0, 0]
    full_colored[uncertain_indices, 3:] = [128, 128, 128]
    ply_arr = np.array([tuple(r) for r in full_colored],
                       dtype=[('x', '<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    PlyData([PlyElement.describe(ply_arr, 'vertex')]).write(str(out_dir / 'domain_colored.ply'))

    print(f"[5/6] Saved index files and PLY files")
    print(f"[6/6] Done")

if __name__ == '__main__':
    main()
