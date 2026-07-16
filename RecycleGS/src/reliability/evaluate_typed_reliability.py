import argparse, sys, os, json, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config
from plyfile import PlyData, PlyElement

def compute_metrics(score, is_wrong, dist, name):
    if len(np.unique(is_wrong)) > 1 and len(np.unique(score[np.isfinite(score)])) > 1:
        auroc = float(roc_auc_score(is_wrong, score))
    else:
        auroc = 0.5
    auprc = float(average_precision_score(is_wrong, score))
    rho, _ = spearmanr(score, dist)
    rho = float(rho) if not np.isnan(rho) else 0.0
    bg_ratio = float(is_wrong.mean())

    n = len(score)
    k10 = max(1, n // 10)
    sorted_idx = np.argsort(score)
    top10_err = float(is_wrong[sorted_idx[-k10:]].mean())
    bottom10_err = float(is_wrong[sorted_idx[:k10]].mean())

    return {
        'auroc': auroc,
        'auprc': auprc,
        'auprc_over_random': auprc - bg_ratio,
        'spearman': rho,
        'top10_error_rate': top10_err,
        'bottom10_error_rate': bottom10_err,
        'top10_mean_dist': float(dist[sorted_idx[-k10:]].mean()),
        'bottom10_mean_dist': float(dist[sorted_idx[:k10]].mean()),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir'])
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading data...")
    feat = np.load(out_dir / 'reliability_features.npz')
    R = feat['risk_scores']
    gt = np.load(out_dir / 'gaussian_gt_errors.npz')
    dist = gt['mesh_distance']
    d_norm = gt['normalized_mesh_distance']
    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = base['xyz']
    correct_th = cfg['mesh_evaluation']['correct_threshold_norm']
    wrong_th = cfg['mesh_evaluation']['wrong_threshold_norm']
    labels = np.where(d_norm < correct_th, 0, np.where(d_norm < wrong_th, 1, 2))
    is_wrong = (labels == 2).astype(int)

    mask_score = 1.0 - feat['E_mask']
    normal_conflict = feat['E_normal']
    depth_conflict = feat['E_depth']
    support_risk = feat['E_support']
    scale_anomaly = feat['E_scale']

    print(f"  Gaussians: {len(xyz)}, wrong ratio: {is_wrong.mean():.4f}")

    print("[2/5] Branch A: background reliability (all Gaussians)...")
    R_background = 0.55 * feat['E_mask'] + 0.25 * feat['E_support'] + 0.20 * feat['E_scale']
    bg_metrics = compute_metrics(R_background, is_wrong, dist, 'background')
    print(f"  Background AUROC: {bg_metrics['auroc']:.4f}, AUPRC: {bg_metrics['auprc']:.4f}")

    print("[3/5] Branch B: internal reliability (mask >= 0.7 Gaussians)...")
    internal_mask = mask_score >= 0.7
    n_internal = internal_mask.sum()
    print(f"  Internal Gaussians (mask>=0.7): {n_internal}/{len(xyz)}")
    if n_internal > 100:
        R_internal = np.zeros(len(xyz))
        R_internal[internal_mask] = (
            0.40 * normal_conflict[internal_mask]
            + 0.35 * depth_conflict[internal_mask]
            + 0.25 * support_risk[internal_mask]
        )
        R_internal[~internal_mask] = 0.0
        int_metrics = compute_metrics(R_internal[internal_mask], is_wrong[internal_mask], dist[internal_mask], 'internal')
    else:
        int_metrics = {'note': 'too few internal Gaussians', 'n_internal': int(n_internal)}

    print("[4/5] Saving colored PLYs...")
    rng = np.random.RandomState(0)

    def save_colored_ply(scores, name, label):
        s_norm = scores - scores.min()
        s_range = s_norm.max() if s_norm.max() > 0 else 1.0
        s_norm = (s_norm / s_range * 255).astype(np.uint8)
        colored = np.zeros((len(xyz), 6))
        colored[:, :3] = xyz
        colored[:, 3] = 0
        colored[:, 4] = s_norm
        colored[:, 5] = 255 - s_norm
        ply_arr = np.array([tuple(r) for r in colored], dtype=[('x', '<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
        PlyData([PlyElement.describe(ply_arr, 'vertex')]).write(str(debug_dir / f'{name}_{label}.ply'))

    save_colored_ply(R_background, 'typed_reliability', 'background')
    if isinstance(int_metrics, dict) and 'note' not in int_metrics:
        save_colored_ply(R_internal, 'typed_reliability', 'internal')

    print("[5/5] Saving metrics and report...")
    all_metrics = {
        'scene_name': cfg['scene_name'],
        'branch_A_background': {
            'weights': {'E_mask': 0.55, 'E_support': 0.25, 'E_scale': 0.20},
            'num_gaussians': int(len(xyz)),
            'metrics': bg_metrics,
        },
        'branch_B_internal': int_metrics if isinstance(int_metrics, dict) and 'note' not in int_metrics else int_metrics,
        'random_baseline': {
            'wrong_ratio': float(is_wrong.mean()),
        },
    }
    with open(debug_dir / 'typed_reliability_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    lines = [
        f"# Typed Reliability Evaluation - {cfg['scene_name']}",
        f"",
        f"## Branch A: Background Reliability",
        f"R = 0.55*E_mask + 0.25*E_support + 0.20*E_scale (all {len(xyz)} Gaussians)",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in bg_metrics.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append(f"")
    lines.append(f"## Branch B: Internal Reliability")
    if isinstance(int_metrics, dict) and 'note' not in int_metrics:
        lines.append(f"R = 0.40*E_normal + 0.35*E_depth + 0.25*E_support ({n_internal} Gaussians, mask>=0.7)")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for k, v in int_metrics.items():
            lines.append(f"| {k} | {v:.4f} |")
    else:
        lines.append(f"- {int_metrics.get('note', 'N/A')} ({int_metrics.get('n_internal', 0)} internal)")
    lines.append(f"")
    lines.append(f"## Random Baseline")
    lines.append(f"- Wrong ratio: {is_wrong.mean():.4f}")
    lines.append(f"- Random AUPRC: {is_wrong.mean():.4f}")
    lines.append(f"")
    lines.append(f"## Files Saved")
    lines.append(f"- typed_reliability_metrics.json")
    lines.append(f"- typed_reliability_background.ply, typed_reliability_internal.ply")
    lines.append(f"- typed_reliability_report.md")

    with open(debug_dir / 'typed_reliability_report.md', 'w') as f:
        f.write('\n'.join(lines))

    print("Done.")

if __name__ == '__main__':
    main()
