import argparse, sys, os, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

def load_ply_raw(ply_path):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    props = {p.name: np.asarray(vertex[p.name]) for p in vertex.properties}
    xyz = np.stack([props['x'], props['y'], props['z']], axis=1)
    scale_names = sorted([p for p in props if p.startswith('scale_')], key=lambda x: int(x.split('_')[-1]))
    scales = np.stack([props[s] for s in scale_names], axis=1) if scale_names else np.ones((len(xyz), 3))
    rot_names = sorted([p for p in props if p.startswith('rot')], key=lambda x: int(x.split('_')[-1]))
    rots = np.stack([props[r] for r in rot_names], axis=1) if rot_names else np.tile([1.,0.,0.,0.], (len(xyz),1))
    return xyz, scales, rots

def quat_to_rotmat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = len(q)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    return R

def bootstrap_spearman_ci(x, y, n_bootstrap=1000, ci_level=0.95):
    n = len(x)
    rhos = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            r, _ = spearmanr(x[idx], y[idx])
            rhos.append(r if not np.isnan(r) else 0.0)
        else:
            rhos.append(0.0)
    rhos = np.array(rhos)
    lo = np.percentile(rhos, (1 - ci_level) / 2 * 100)
    hi = np.percentile(rhos, (1 + ci_level) / 2 * 100)
    return float(lo), float(hi)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--iteration', type=int, required=True, choices=[7000, 15000])
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1f_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    iteration = args.iteration
    iter_str = f"iter_{iteration}"
    iter_dir = out_dir / iter_str

    model_dir = Path(cfg['model_dir'])
    if iteration == 7000:
        ply_path = model_dir / 'point_cloud' / 'iteration_7000' / 'point_cloud.ply'
    else:
        ply_path = Path(cfg['checkpoint_path'])

    print(f"[{iter_str}] Loading candidate indices...")
    candidate_indices = np.load(iter_dir / 'candidate_indices.npy')
    N_cand = len(candidate_indices)
    print(f"  Candidate count: {N_cand}")

    xyz_all, scales_all, rots_all = load_ply_raw(ply_path)
    scale_linear = np.exp(scales_all)
    scales_cand = scale_linear[candidate_indices]

    rots_norm = rots_all / np.linalg.norm(rots_all, axis=1, keepdims=True)
    rot_mats = quat_to_rotmat(rots_norm)
    min_axis = np.argmin(scale_linear, axis=1)
    normals_all = np.zeros_like(xyz_all)
    for i in range(len(xyz_all)):
        normals_all[i] = rot_mats[i, :, min_axis[i]]
    normals_cand = normals_all[candidate_indices]

    planarity = (scales_cand.max(axis=1) - scales_cand.min(axis=1)) / (scales_cand.max(axis=1) + 1e-8)
    planarity_confidence = 1.0 - (scales_cand.min(axis=1) / (scales_cand.max(axis=1) + 1e-8))
    valid_normal_mask = planarity_confidence >= 0.20
    valid_count = int(valid_normal_mask.sum())
    coverage = valid_count / max(N_cand, 1)
    print(f"  valid_normal (planarity>=0.20): {valid_count}/{N_cand} ({coverage*100:.2f}%)")

    normal_global_usable = coverage >= 0.30
    print(f"  normal_global_usable: {normal_global_usable} (needs coverage>=30%)")

    err_path = iter_dir / 'geometry_errors.npz'
    if not err_path.exists():
        print(f"  SKIP: geometry errors not found at {err_path}")
        result = {
            'iteration': iteration,
            'normal_valid_count': valid_count,
            'normal_valid_ratio': float(coverage),
            'normal_global_usable': normal_global_usable,
            'error': 'geometry_errors.npz not found',
        }
        with open(debug_dir / f'normal_valid_subset_{iter_str}.json', 'w') as f:
            json.dump(result, f, indent=2)
        return

    err = np.load(err_path)
    d_center_norm = err['d_center_norm']
    d_surface_proxy_alpha1 = err['d_surface_proxy_alpha1']
    d_surface_proxy_alpha2 = err['d_surface_proxy_alpha2']

    print(f"  Computing normal conflict validity...")
    result = {
        'iteration': iteration,
        'candidate_count': N_cand,
        'normal_valid_count': valid_count,
        'normal_valid_ratio': float(coverage),
        'normal_global_usable': normal_global_usable,
        'metrics': {},
    }

    if valid_count < 5:
        result['metrics']['note'] = 'insufficient valid normals for correlation'
        print(f"  WARNING: insufficient valid normals ({valid_count})")
    else:
        error_metrics = {
            'd_center_norm': d_center_norm,
            'd_surface_proxy_alpha1': d_surface_proxy_alpha1,
            'd_surface_proxy_alpha2': d_surface_proxy_alpha2,
        }
        for ename, earr in error_metrics.items():
            s = planarity_confidence[valid_normal_mask]
            m = earr[valid_normal_mask]
            valid = np.isfinite(s) & np.isfinite(m)
            if valid.sum() < 5:
                result['metrics'][ename] = {'error': 'insufficient data', 'valid_count': int(valid.sum())}
                continue
            sv, mv = s[valid], m[valid]
            nv = len(sv)
            rho, _ = spearmanr(sv, mv)
            rho = float(rho) if not np.isnan(rho) else 0.0
            rho_lo, rho_hi = bootstrap_spearman_ci(sv, mv)
            tau, _ = kendalltau(sv, mv)
            tau = float(tau) if not np.isnan(tau) else 0.0

            k10 = max(1, nv // 10)
            top10_idx = np.argpartition(sv, -k10)[-k10:]
            bottom10_idx = np.argpartition(sv, k10)[:k10]
            top10_mean = float(mv[top10_idx].mean())
            bottom10_mean = float(mv[bottom10_idx].mean())
            ratio = top10_mean / max(bottom10_mean, 1e-8)

            result['metrics'][ename] = {
                'valid_count': nv,
                'spearman_rho': rho,
                'spearman_ci_lo': rho_lo,
                'spearman_ci_hi': rho_hi,
                'kendall_tau': tau,
                'top10_mean_error': top10_mean,
                'bottom10_mean_error': bottom10_mean,
                'top10_bottom10_ratio': ratio,
            }

    out_path = debug_dir / f'normal_valid_subset_{iter_str}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[{iter_str}] Normal valid subset evaluation saved to {out_path}")

if __name__ == '__main__':
    main()
