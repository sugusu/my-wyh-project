import argparse, sys, os, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

def percentile_report(arr, p_list):
    return {str(p): float(np.percentile(arr, p)) for p in p_list}

def error_ratio_at_thresholds(errors, thresholds):
    ratios = {}
    for th in thresholds:
        ratios[f'{th:.3f}'] = float((errors > th).mean())
    return ratios

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1d_scene01'
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading domain indices...")
    object_indices = np.load(out_dir / 'object_indices.npy')
    background_indices = np.load(out_dir / 'background_indices.npy')
    uncertain_indices = np.load(out_dir / 'uncertain_indices.npy')

    print("[2/6] Loading GT error data...")
    gt = np.load(out_dir / 'gaussian_gt_errors.npz')
    dist = gt['mesh_distance']
    d_norm = gt['normalized_mesh_distance']
    N = len(dist)
    obj_diameter = float(gt['obj_diameter'].item())

    print("[3/6] Loading mask support and valid view count...")
    mask_support = np.load(out_dir / 'mask_support_unweighted.npy')
    valid_view = np.load(out_dir / 'valid_view_count.npy')

    domains = {
        'all': np.arange(N),
        'object': object_indices,
        'background': background_indices,
        'uncertain': uncertain_indices,
    }
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    thresholds = [0.005, 0.010, 0.020, 0.030]

    results = {}
    print("[4/6] Computing per-domain statistics...")
    for name, idx in domains.items():
        d = dist[idx]
        dn = d_norm[idx]
        n_d = len(idx)
        results[name] = {
            'count': int(n_d),
            'ratio': float(n_d / N),
            'mesh_distance_percentiles': percentile_report(d, percentiles),
            'normalized_mesh_distance_percentiles': percentile_report(dn, percentiles),
            'error_ratios': error_ratio_at_thresholds(dn, thresholds),
            'mask_support_mean': float(mask_support[idx].mean()),
            'valid_view_mean': float(valid_view[idx].mean()),
        }

    print("[5/6] Top-5% and Top-10% high-error analysis...")
    sorted_by_dist = np.argsort(dist)
    top5_count = max(1, int(N * 0.05))
    top10_count = max(1, int(N * 0.10))
    top5_idx = sorted_by_dist[-top5_count:]
    top10_idx = sorted_by_dist[-top10_count:]

    def domain_overlap(indices, name):
        obj_overlap = len(np.intersect1d(indices, object_indices))
        bg_overlap = len(np.intersect1d(indices, background_indices))
        unc_overlap = len(np.intersect1d(indices, uncertain_indices))
        return {
            f'count_in_object': int(obj_overlap),
            f'ratio_in_object': float(obj_overlap / len(indices)),
            f'count_in_background': int(bg_overlap),
            f'ratio_in_background': float(bg_overlap / len(indices)),
            f'count_in_uncertain': int(unc_overlap),
            f'ratio_in_uncertain': float(unc_overlap / len(indices)),
        }

    results['top5_percent_highest_distance'] = domain_overlap(top5_idx, 'top5')
    results['top10_percent_highest_distance'] = domain_overlap(top10_idx, 'top10')
    results['top5_mean_distance'] = float(dist[top5_idx].mean())
    results['top10_mean_distance'] = float(dist[top10_idx].mean())

    save_path = debug_dir / 'domain_gt_distribution.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {save_path}")

    md = [
        f"# Domain GT Distribution Diagnosis - {cfg['scene_name']}",
        f"",
        f"Total Gaussians: {N}",
        f"Object Diameter: {obj_diameter:.6f}",
        f"",
        f"## Domain Sizes",
        f"| Domain | Count | Ratio | Mask Support Mean | Valid View Mean |",
        f"|--------|-------|-------|-------------------|-----------------|",
    ]
    for name in ['all', 'object', 'background', 'uncertain']:
        r = results[name]
        md.append(f"| {name} | {r['count']} | {r['ratio']*100:.2f}% | {r['mask_support_mean']:.4f} | {r['valid_view_mean']:.2f} |")

    md.extend([
        f"",
        f"## Mesh Distance Percentiles",
        f"| Domain | Min | p01 | p05 | p10 | p25 | Median | p75 | p90 | p95 | p99 | Max |",
        f"|--------|-----|-----|-----|-----|------|--------|-----|-----|-----|-----|",
    ])
    for name in ['all', 'object', 'background', 'uncertain']:
        r = results[name]
        p = r['mesh_distance_percentiles']
        md.append(f"| {name} | {p['1']:.6f} | {p['1']:.6f} | {p['5']:.6f} | {p['10']:.6f} | {p['25']:.6f} | {p['50']:.6f} | {p['75']:.6f} | {p['90']:.6f} | {p['95']:.6f} | {p['99']:.6f} | {r['mesh_distance_percentiles']['99']:.6f} |")

    md.extend([
        f"",
        f"## Normalized Mesh Distance Percentiles",
        f"| Domain | Min | p01 | p05 | p10 | p25 | Median | p75 | p90 | p95 | p99 | Max |",
        f"|--------|-----|-----|-----|-----|------|--------|-----|-----|-----|-----|",
    ])
    for name in ['all', 'object', 'background', 'uncertain']:
        p = results[name]['normalized_mesh_distance_percentiles']
        md.append(f"| {name} | {p['1']:.6f} | {p['1']:.6f} | {p['5']:.6f} | {p['10']:.6f} | {p['25']:.6f} | {p['50']:.6f} | {p['75']:.6f} | {p['90']:.6f} | {p['95']:.6f} | {p['99']:.6f} | {p['99']:.6f} |")

    md.extend([
        f"",
        f"## Error Ratios at Thresholds",
        f"| Domain | d_norm>0.005 | d_norm>0.010 | d_norm>0.020 | d_norm>0.030 |",
        f"|--------|-------------|-------------|-------------|-------------|",
    ])
    for name in ['all', 'object', 'background', 'uncertain']:
        er = results[name]['error_ratios']
        md.append(f"| {name} | {er['0.005']*100:.2f}% | {er['0.010']*100:.2f}% | {er['0.020']*100:.2f}% | {er['0.030']*100:.2f}% |")

    md.extend([
        f"",
        f"## High-Error Gaussian Domain Distribution",
        f"| Group | In Object | In Background | In Uncertain |",
        f"|-------|-----------|---------------|--------------|",
    ])
    t5 = results['top5_percent_highest_distance']
    t10 = results['top10_percent_highest_distance']
    md.append(f"| Top 5% ({top5_count}) | {t5['count_in_object']} ({t5['ratio_in_object']*100:.1f}%) | {t5['count_in_background']} ({t5['ratio_in_background']*100:.1f}%) | {t5['count_in_uncertain']} ({t5['ratio_in_uncertain']*100:.1f}%) |")
    md.append(f"| Top 10% ({top10_count}) | {t10['count_in_object']} ({t10['ratio_in_object']*100:.1f}%) | {t10['count_in_background']} ({t10['ratio_in_background']*100:.1f}%) | {t10['count_in_uncertain']} ({t10['ratio_in_uncertain']*100:.1f}%) |")
    md.append(f"| Top 5% mean dist | {results['top5_mean_distance']:.6f} |")
    md.append(f"| Top 10% mean dist | {results['top10_mean_distance']:.6f} |")

    high_bg = t10['ratio_in_background']
    high_obj = t10['ratio_in_object']
    high_unc = t10['ratio_in_uncertain']

    out_obj_ratio = results['object']['ratio']

    md.extend([
        f"",
        f"## Key Judgment",
        f"",
    ])
    if high_unc > 0.5:
        md.append(f"- High-error Gaussians are mostly in uncertain ({high_unc*100:.1f}%) -> Object domain too strict")
    if high_bg > 0.5:
        md.append(f"- High-error Gaussians are mostly in background ({high_bg*100:.1f}%) -> Need to distinguish background from near-surface leakage")
    if high_obj + high_unc < 0.1:
        md.append(f"- Object and uncertain both have few high-error Gaussians -> 15k may be too converged")

    report_path = debug_dir / 'domain_gt_distribution_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"[6/6] Report saved to {report_path}")

if __name__ == '__main__':
    main()
