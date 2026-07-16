import argparse, sys, os, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config, save_npz, save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1d_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    print("[1/5] Loading data...")
    candidate_indices = np.load(out_dir / 'candidate_object_indices.npy')
    base = np.load(out_dir / 'gaussian_base_features.npz')
    scale_linear = base['scale_linear']
    xyz = base['xyz']

    gt = np.load(out_dir / 'gaussian_gt_errors.npz')
    dist = gt['mesh_distance']
    obj_diameter = float(gt['obj_diameter'].item())

    print(f"  Candidate object indices: {len(candidate_indices)}")
    print(f"  Object diameter: {obj_diameter:.6f}")

    cand_dist = dist[candidate_indices]
    cand_scale = scale_linear[candidate_indices]
    cand_xyz = xyz[candidate_indices]

    eps = 1e-8

    print("[2/5] Computing continuous errors...")
    d_center_norm = cand_dist / obj_diameter

    normal_scale = np.min(cand_scale, axis=1)
    d_scale = cand_dist / (normal_scale + eps)

    d_surface_proxy_alpha1 = np.maximum(0, cand_dist - 1.0 * normal_scale)
    d_surface_proxy_alpha2 = np.maximum(0, cand_dist - 2.0 * normal_scale)

    print(f"  d_center_norm: mean={d_center_norm.mean():.6f}, std={d_center_norm.std():.6f}")
    print(f"  d_scale: mean={d_scale.mean():.6f}, std={d_scale.std():.6f}")
    print(f"  d_surface_proxy_alpha1: mean={d_surface_proxy_alpha1.mean():.6f}, nonzero={((d_surface_proxy_alpha1>0).mean()*100):.1f}%")
    print(f"  d_surface_proxy_alpha2: mean={d_surface_proxy_alpha2.mean():.6f}, nonzero={((d_surface_proxy_alpha2>0).mean()*100):.1f}%")

    save_npz(out_dir / 'candidate_geometry_errors_v2.npz',
             d_center_norm=d_center_norm,
             d_scale=d_scale,
             d_surface_proxy_alpha1=d_surface_proxy_alpha1,
             d_surface_proxy_alpha2=d_surface_proxy_alpha2,
             candidate_indices=candidate_indices,
             obj_diameter=np.array([obj_diameter]))

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    stats = {
        'num_candidate_gaussians': int(len(candidate_indices)),
        'obj_diameter': obj_diameter,
        'd_center_norm': {
            'mean': float(d_center_norm.mean()),
            'std': float(d_center_norm.std()),
            'percentiles': {str(p): float(np.percentile(d_center_norm, p)) for p in percentiles}
        },
        'd_scale': {
            'mean': float(d_scale.mean()),
            'std': float(d_scale.std()),
            'percentiles': {str(p): float(np.percentile(d_scale, p)) for p in percentiles}
        },
        'd_surface_proxy_alpha1': {
            'mean': float(d_surface_proxy_alpha1.mean()),
            'std': float(d_surface_proxy_alpha1.std()),
            'nonzero_ratio': float((d_surface_proxy_alpha1 > 0).mean()),
            'percentiles': {str(p): float(np.percentile(d_surface_proxy_alpha1, p)) for p in percentiles}
        },
        'd_surface_proxy_alpha2': {
            'mean': float(d_surface_proxy_alpha2.mean()),
            'std': float(d_surface_proxy_alpha2.std()),
            'nonzero_ratio': float((d_surface_proxy_alpha2 > 0).mean()),
            'percentiles': {str(p): float(np.percentile(d_surface_proxy_alpha2, p)) for p in percentiles}
        },
    }
    save_json(stats, out_dir / 'candidate_geometry_errors_stats.json')

    md = [
        f"# Candidate Geometry Error Labels V2 - {cfg['scene_name']}",
        f"",
        f"## Data",
        f"- Candidate object Gaussians: {len(candidate_indices)}",
        f"- Object diameter: {obj_diameter:.6f}",
        f"",
        f"## Error Definitions",
        f"1. d_center_norm = center_to_mesh_distance / object_diameter",
        f"2. d_scale = center_to_mesh_distance / min(scale_linear) + eps",
        f"3. d_surface_proxy_alpha1 = max(0, center_to_mesh_distance - 1.0 * min(scale_linear))",
        f"4. d_surface_proxy_alpha2 = max(0, center_to_mesh_distance - 2.0 * min(scale_linear))",
        f"",
        f"## Statistics",
        f"| Metric | Mean | Std | p1 | p5 | p25 | Median | p75 | p95 | p99 |",
        f"|--------|------|-----|----|----|-----|--------|-----|-----|-----|",
    ]
    for name, label in [('d_center_norm', 'd_center_norm'), ('d_scale', 'd_scale'),
                         ('d_surface_proxy_alpha1', 'd_surface_proxy_a1'), ('d_surface_proxy_alpha2', 'd_surface_proxy_a2')]:
        s = stats[name]
        p = s['percentiles']
        md.append(f"| {label} | {s['mean']:.6f} | {s['std']:.6f} | {p['1']:.6f} | {p['5']:.6f} | {p['25']:.6f} | {p['50']:.6f} | {p['75']:.6f} | {p['95']:.6f} | {p['99']:.6f} |")

    report_path = debug_dir / 'candidate_geometry_errors_v2_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"[5/5] Report saved to {report_path}")

if __name__ == '__main__':
    main()
