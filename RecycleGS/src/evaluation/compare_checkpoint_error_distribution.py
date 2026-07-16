import argparse, sys, os, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config
from plyfile import PlyData

def load_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    xyz = np.stack([x, y, z], axis=1)
    scale_names = [p.name for p in vertex.properties if p.name.startswith('scale_')]
    if scale_names:
        scales = np.stack([vertex[s] for s in sorted(scale_names)], axis=1)
    else:
        scales = None
    return xyz, scales

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1e_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    model_dir = Path(cfg['model_dir'])
    ckpt_7k_path = model_dir / 'point_cloud' / 'iteration_7000' / 'point_cloud.ply'
    ckpt_15k_path = Path(cfg['checkpoint_path'])

    if not ckpt_7k_path.exists():
        msg = "7k checkpoint not available at " + str(ckpt_7k_path)
        print(f"[SKIP] {msg}")
        result = {'note': msg, 'num_gaussians': {'7k': None, '15k': None}}
        with open(debug_dir / 'checkpoint_comparison.json', 'w') as f:
            json.dump(result, f, indent=2)
        return

    print("[1/6] Loading 7k and 15k point clouds...")
    xyz_7k, scales_7k = load_ply_points(ckpt_7k_path)
    xyz_15k, scales_15k = load_ply_points(ckpt_15k_path)
    print(f"  7k: {len(xyz_7k)} Gaussians")
    print(f"  15k: {len(xyz_15k)} Gaussians")

    print("[2/6] Loading GT mesh and computing distances...")
    import trimesh
    from scipy.spatial import cKDTree
    mesh_path = os.path.join(cfg['scene_dir'], 'meshes', 'scene_mesh.obj')
    mesh = trimesh.load(mesh_path, force='mesh')
    bbox = mesh.bounds
    obj_diameter = np.linalg.norm(bbox[1] - bbox[0])
    n_samples = cfg['mesh_evaluation']['sample_points']
    sampled, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    tree = cKDTree(sampled)

    dist_7k, _ = tree.query(xyz_7k)
    dist_15k, _ = tree.query(xyz_15k)
    d_norm_7k = dist_7k / obj_diameter
    d_norm_15k = dist_15k / obj_diameter

    print(f"  7k mean d_norm: {d_norm_7k.mean():.6f}")
    print(f"  15k mean d_norm: {d_norm_15k.mean():.6f}")

    print("[3/6] Loading candidate-object domain data (15k)...")
    candidate_indices = np.load(out_dir / 'candidate_object_indices.npy')
    mask_support = np.load(out_dir / 'mask_support_unweighted.npy')
    valid_view = np.load(out_dir / 'valid_view_count.npy')

    has_min_views = valid_view >= 3
    candidate_mask = has_min_views & (mask_support >= 0.20) & ~(has_min_views & (mask_support <= 0.05))
    print(f"  15k candidate-object: {candidate_mask.sum()}")

    print("[4/6] Computing per-checkpoint stats...")
    def array_stats(d):
        return {
            'mean': float(d.mean()), 'median': float(np.median(d)),
            'p90': float(np.percentile(d, 90)),
            'p95': float(np.percentile(d, 95)),
            'p99': float(np.percentile(d, 99)),
        }

    def positive_ratios(dn):
        return {
            'gt_0.005': float((dn > 0.005).mean()),
            'gt_0.010': float((dn > 0.010).mean()),
            'gt_0.020': float((dn > 0.020).mean()),
        }

    def binary_viability(dn, label=''):
        pos_ratio = float((dn > 0.010).mean())
        pos_count = int((dn > 0.010).sum())
        neg_count = int((dn <= 0.010).sum())
        viable = bool(pos_count >= 100 and neg_count >= 100 and 0.005 <= pos_ratio <= 0.50)
        return {
            'positive_threshold': 0.010,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'positive_ratio': pos_ratio,
            'binary_classification_viable': viable,
        }

    comparison = {
        'num_gaussians': {'7k': int(len(xyz_7k)), '15k': int(len(xyz_15k))},
        'candidate_object': {
            '15k': {
                'count': int(candidate_mask.sum()),
                'ratio': float(candidate_mask.mean()),
            }
        },
        'd_center_norm_stats': {
            '7k': array_stats(d_norm_7k),
            '15k': array_stats(d_norm_15k),
        },
        'd_surface_proxy_alpha1': {
            '7k': array_stats(np.maximum(0, d_norm_7k - 1.0 * (scales_7k.min(axis=1) / obj_diameter if scales_7k is not None else 0))),
            '15k': array_stats(np.maximum(0, d_norm_15k - 1.0 * (scales_15k.min(axis=1) / obj_diameter if scales_15k is not None else 0))),
        },
        'positive_ratios': {
            '7k': positive_ratios(d_norm_7k),
            '15k': positive_ratios(d_norm_15k),
        },
        'binary_classification_viability': {
            '7k': binary_viability(d_norm_7k, '7k'),
            '15k': binary_viability(d_norm_15k, '15k'),
        },
    }

    json_path = debug_dir / 'checkpoint_comparison.json'
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print("[5/6] Writing CSV...")
    import csv
    csv_path = debug_dir / 'checkpoint_comparison.csv'
    with open(csv_path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['metric', '7k', '15k'])
        w.writerow(['num_gaussians', comparison['num_gaussians']['7k'], comparison['num_gaussians']['15k']])
        for stat in ['mean', 'median', 'p90', 'p95', 'p99']:
            w.writerow([f'd_center_norm_{stat}', comparison['d_center_norm_stats']['7k'][stat], comparison['d_center_norm_stats']['15k'][stat]])
        for thresh in ['gt_0.005', 'gt_0.010', 'gt_0.020']:
            w.writerow([f'positive_ratio_{thresh}', comparison['positive_ratios']['7k'][thresh], comparison['positive_ratios']['15k'][thresh]])

    print("[6/6] Writing report...")
    ckpt = comparison
    d7 = ckpt['d_center_norm_stats']['7k']
    d15 = ckpt['d_center_norm_stats']['15k']
    p7 = ckpt['positive_ratios']['7k']
    p15 = ckpt['positive_ratios']['15k']
    b7 = ckpt['binary_classification_viability']['7k']
    b15 = ckpt['binary_classification_viability']['15k']

    md = [
        f"# Checkpoint Comparison (7k vs 15k) - {cfg['scene_name']}",
        f"",
        f"## Gaussian Count",
        f"| Checkpoint | Count |",
        f"|------------|-------|",
        f"| 7k | {ckpt['num_gaussians']['7k']} |",
        f"| 15k | {ckpt['num_gaussians']['15k']} |",
        f"",
        f"## d_center_norm Distribution",
        f"| Statistic | 7k | 15k |",
        f"|-----------|-----|-----|",
        f"| Mean | {d7['mean']:.6f} | {d15['mean']:.6f} |",
        f"| Median | {d7['median']:.6f} | {d15['median']:.6f} |",
        f"| P90 | {d7['p90']:.6f} | {d15['p90']:.6f} |",
        f"| P95 | {d7['p95']:.6f} | {d15['p95']:.6f} |",
        f"| P99 | {d7['p99']:.6f} | {d15['p99']:.6f} |",
        f"",
        f"## Positive Ratios",
        f"| Threshold | 7k | 15k |",
        f"|-----------|-----|-----|",
        f"| >0.005 | {p7['gt_0.005']*100:.2f}% | {p15['gt_0.005']*100:.2f}% |",
        f"| >0.010 | {p7['gt_0.010']*100:.2f}% | {p15['gt_0.010']*100:.2f}% |",
        f"| >0.020 | {p7['gt_0.020']*100:.2f}% | {p15['gt_0.020']*100:.2f}% |",
        f"",
        f"## Binary Classification Viability (threshold=0.010)",
        f"| Metric | 7k | 15k |",
        f"|--------|-----|-----|",
        f"| Positive count | {b7['positive_count']} | {b15['positive_count']} |",
        f"| Negative count | {b7['negative_count']} | {b15['negative_count']} |",
        f"| Positive ratio | {b7['positive_ratio']*100:.2f}% | {b15['positive_ratio']*100:.2f}% |",
        f"| Viable | {b7['binary_classification_viable']} | {b15['binary_classification_viable']} |",
    ]
    with open(debug_dir / 'checkpoint_comparison.md', 'w') as f:
        f.write('\n'.join(md))

    print(f"Comparison saved to {json_path}")

if __name__ == '__main__':
    main()
