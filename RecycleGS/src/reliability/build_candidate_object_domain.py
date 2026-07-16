import argparse, sys, os, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config, save_json
from plyfile import PlyData, PlyElement

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1d_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    print("[1/5] Loading data...")
    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = base['xyz']
    N = len(xyz)

    valid_view = np.load(out_dir / 'valid_view_count.npy')
    mask_support = np.load(out_dir / 'mask_support_unweighted.npy')
    object_indices_old = np.load(out_dir / 'object_indices.npy')
    uncertain_indices_old = np.load(out_dir / 'uncertain_indices.npy')
    background_indices_old = np.load(out_dir / 'background_indices.npy')

    has_min_views = valid_view >= 3

    print("[2/5] Building new domain definitions...")
    core_object = has_min_views & (mask_support >= 0.50)
    strong_background = has_min_views & (mask_support <= 0.05)
    candidate_object = has_min_views & (mask_support >= 0.20) & ~strong_background

    core_obj_idx = np.where(core_object)[0]
    candidate_obj_idx = np.where(candidate_object)[0]
    strong_bg_idx = np.where(strong_background)[0]

    print(f"  core_object: {len(core_obj_idx)} ({len(core_obj_idx)/N*100:.1f}%)")
    print(f"  candidate_object: {len(candidate_obj_idx)} ({len(candidate_obj_idx)/N*100:.1f}%)")
    print(f"  strong_background: {len(strong_bg_idx)} ({len(strong_bg_idx)/N*100:.1f}%)")
    print(f"  old object: {len(object_indices_old)}")
    print(f"  old uncertain: {len(uncertain_indices_old)}")

    uncertain_from_old = len(np.setdiff1d(uncertain_indices_old, np.concatenate([core_obj_idx, candidate_obj_idx, strong_bg_idx])))

    np.save(out_dir / 'core_object_indices.npy', core_obj_idx)
    np.save(out_dir / 'candidate_object_indices.npy', candidate_obj_idx)
    np.save(out_dir / 'strong_background_indices.npy', strong_bg_idx)

    print("[3/5] Saving colored PLY files...")
    def save_colored_ply(indices, color, name, path):
        pts = xyz[indices]
        colored = np.zeros((len(pts), 6))
        colored[:, :3] = pts
        colored[:, 3:] = color
        ply_arr = np.array([tuple(r) for r in colored],
                           dtype=[('x', '<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
        PlyData([PlyElement.describe(ply_arr, 'vertex')]).write(str(path / name))

    save_colored_ply(core_obj_idx, [0, 255, 0], 'core_object_domain.ply', out_dir)
    save_colored_ply(candidate_obj_idx, [0, 255, 255], 'candidate_object_domain.ply', out_dir)
    save_colored_ply(strong_bg_idx, [255, 0, 0], 'strong_background_domain.ply', out_dir)

    colored_all = np.zeros((N, 6))
    colored_all[:, :3] = xyz
    core_set = set(core_obj_idx.tolist())
    cand_set = set(candidate_obj_idx.tolist())
    bg_set = set(strong_bg_idx.tolist())
    for i in range(N):
        if i in core_set:
            colored_all[i, 3:] = [0, 255, 0]
        elif i in cand_set:
            colored_all[i, 3:] = [0, 255, 255]
        elif i in bg_set:
            colored_all[i, 3:] = [255, 0, 0]
        else:
            colored_all[i, 3:] = [128, 128, 128]
    ply_arr = np.array([tuple(r) for r in colored_all],
                       dtype=[('x', '<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    PlyData([PlyElement.describe(ply_arr, 'vertex')]).write(str(out_dir / 'candidate_domain_colored.ply'))

    stats = {
        'total_gaussians': int(N),
        'core_object': {'count': int(len(core_obj_idx)), 'ratio': float(len(core_obj_idx)/N)},
        'candidate_object': {'count': int(len(candidate_obj_idx)), 'ratio': float(len(candidate_obj_idx)/N)},
        'strong_background': {'count': int(len(strong_bg_idx)), 'ratio': float(len(strong_bg_idx)/N)},
        'uncertain_remaining': {'count': int(N - len(core_obj_idx) - len(candidate_obj_idx) - len(strong_bg_idx))},
        'old_partition': {
            'object_count': int(len(object_indices_old)),
            'uncertain_count': int(len(uncertain_indices_old)),
            'background_count': int(len(background_indices_old)),
            'uncertain_absorbed_into_candidate': int(len(np.intersect1d(uncertain_indices_old, candidate_obj_idx))),
            'background_absorbed_into_candidate': int(len(np.intersect1d(background_indices_old, candidate_obj_idx))),
        },
        'criteria': {
            'valid_view_min': 3,
            'core_mask_support_threshold': 0.50,
            'candidate_mask_support_threshold': 0.20,
            'strong_background_mask_support_threshold': 0.05,
        },
    }
    save_json(stats, out_dir / 'candidate_domain_stats.json')

    md = [
        f"# Candidate Object Domain Report - {cfg['scene_name']}",
        f"",
        f"## Domain Sizes",
        f"| Domain | Count | Ratio |",
        f"|--------|-------|-------|",
        f"| Core Object (mask>=0.50) | {stats['core_object']['count']} | {stats['core_object']['ratio']*100:.2f}% |",
        f"| Candidate Object (mask>=0.20) | {stats['candidate_object']['count']} | {stats['candidate_object']['ratio']*100:.2f}% |",
        f"| Strong Background (mask<=0.05) | {stats['strong_background']['count']} | {stats['strong_background']['ratio']*100:.2f}% |",
        f"| Uncertain Remaining | {stats['uncertain_remaining']} | {stats['uncertain_remaining']['count']/N*100:.2f}% |",
        f"",
        f"## Old Partition Intersection",
        f"| Old Domain | Count |",
        f"|------------|-------|",
        f"| Object | {stats['old_partition']['object_count']} |",
        f"| Background | {stats['old_partition']['background_count']} |",
        f"| Uncertain | {stats['old_partition']['uncertain_count']} |",
        f"| Old uncertain -> candidate | {stats['old_partition']['uncertain_absorbed_into_candidate']} |",
        f"| Old background -> candidate | {stats['old_partition']['background_absorbed_into_candidate']} |",
        f"",
        f"## Criteria",
        f"- valid_view >= 3",
        f"- core: mask_support >= 0.50",
        f"- candidate: mask_support >= 0.20 AND NOT strong_background",
        f"- strong_background: mask_support <= 0.05",
    ]
    report_path = out_dir / 'candidate_domain_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"[4/5] Report saved to {report_path}")
    print(f"[5/5] Done")

if __name__ == '__main__':
    main()
