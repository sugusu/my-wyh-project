import argparse, sys, os, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir'])
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading gaussian_gt_errors.npz...")
    data = np.load(out_dir / 'gaussian_gt_errors.npz')
    dist = data['mesh_distance']
    d_norm = data['normalized_mesh_distance']
    obj_diameter = float(data['obj_diameter'].item())
    n = len(dist)
    print(f"  Object diameter: {obj_diameter:.6f}")
    print(f"  Num Gaussians: {n}")

    print("[2/4] Reporting distance distributions...")
    d_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    dist_stats = {str(p): float(np.percentile(dist, p)) for p in d_percentiles}
    dist_stats['mean'] = float(dist.mean())
    dist_stats['std'] = float(dist.std())
    dist_stats['min'] = float(dist.min())
    dist_stats['max'] = float(dist.max())

    dn_stats = {str(p): float(np.percentile(d_norm, p)) for p in d_percentiles}
    dn_stats['mean'] = float(d_norm.mean())
    dn_stats['std'] = float(d_norm.std())
    dn_stats['min'] = float(d_norm.min())
    dn_stats['max'] = float(d_norm.max())

    print("[3/4] Computing label counts for 3 threshold schemes...")
    schemes = {
        'A': {'correct': 0.005, 'wrong': 0.020, 'name': 'A (0.005/0.020)'},
        'B': {'correct': 0.010, 'wrong': 0.030, 'name': 'B (0.010/0.030)'},
        'C': {'correct': 0.005, 'wrong': 0.010, 'name': 'C (0.005/0.010)'},
    }
    label_results = {}
    for key, sch in schemes.items():
        correct_th = sch['correct']
        wrong_th = sch['wrong']
        labels = np.where(d_norm < correct_th, 0, np.where(d_norm < wrong_th, 1, 2))
        n_correct = int((labels == 0).sum())
        n_ambiguous = int((labels == 1).sum())
        n_wrong = int((labels == 2).sum())
        p_correct = n_correct / n * 100
        p_ambiguous = n_ambiguous / n * 100
        p_wrong = n_wrong / n * 100
        label_results[key] = {
            'scheme': sch['name'],
            'correct_threshold': correct_th,
            'wrong_threshold': wrong_th,
            'correct_count': n_correct,
            'ambiguous_count': n_ambiguous,
            'wrong_count': n_wrong,
            'correct_pct': round(p_correct, 2),
            'ambiguous_pct': round(p_ambiguous, 2),
            'wrong_pct': round(p_wrong, 2),
        }
        print(f"  Scheme {sch['name']}: correct={n_correct}({p_correct:.1f}%) ambiguous={n_ambiguous}({p_ambiguous:.1f}%) wrong={n_wrong}({p_wrong:.1f}%)")

    print("[4/4] Saving results...")
    diagnosis = {
        'scene_name': cfg['scene_name'],
        'object_diameter': obj_diameter,
        'num_gaussians': n,
        'mesh_distance': dist_stats,
        'normalized_mesh_distance': dn_stats,
        'label_counts': label_results,
    }
    with open(debug_dir / 'gt_label_diagnosis.json', 'w') as f:
        json.dump(diagnosis, f, indent=2)

    md = [
        f"# GT Label Diagnosis - {cfg['scene_name']}",
        f"",
        f"## Object Diameter",
        f"- {obj_diameter:.6f}",
        f"",
        f"## Mesh Distance Distribution",
        f"| Percentile | Distance | Normalized |",
        f"|------------|----------|------------|",
    ]
    for p in d_percentiles:
        md.append(f"| p{p} | {dist_stats[str(p)]:.6f} | {dn_stats[str(p)]:.6f} |")
    md.append(f"| mean | {dist_stats['mean']:.6f} | {dn_stats['mean']:.6f} |")
    md.append(f"| std  | {dist_stats['std']:.6f} | {dn_stats['std']:.6f} |")
    md.append(f"")
    md.append(f"## Label Counts by Threshold Scheme")
    md.append(f"| Scheme | correct_th | wrong_th | Correct | Ambiguous | Wrong |")
    md.append(f"|--------|------------|----------|---------|-----------|-------|")
    for key in ['A', 'B', 'C']:
        r = label_results[key]
        md.append(f"| {r['scheme']} | {r['correct_threshold']} | {r['wrong_threshold']} | {r['correct_count']}({r['correct_pct']}%) | {r['ambiguous_count']}({r['ambiguous_pct']}%) | {r['wrong_count']}({r['wrong_pct']}%) |")
    md.append(f"")
    md.append(f"## Files Saved")
    md.append(f"- gt_label_diagnosis.json")
    md.append(f"- gt_label_report.md")

    with open(debug_dir / 'gt_label_report.md', 'w') as f:
        f.write('\n'.join(md))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, (key, sch) in enumerate([('A', 'A (0.005/0.020)'), ('B', 'B (0.010/0.030)'), ('C', 'C (0.005/0.010)')]):
            r = label_results[key]
            axes[idx].bar(['Correct', 'Ambiguous', 'Wrong'], [r['correct_count'], r['ambiguous_count'], r['wrong_count']],
                         color=['green', 'yellow', 'red'])
            axes[idx].set_title(f"Scheme {sch}")
            axes[idx].set_ylabel('Count')
        plt.tight_layout()
        fig.savefig(str(debug_dir / 'gt_label_histogram.png'), dpi=150)
        plt.close(fig)
        print(f"  Histogram saved.")
    except Exception as e:
        print(f"  Histogram skipped: {e}")

    print("Done.")

if __name__ == '__main__':
    main()
