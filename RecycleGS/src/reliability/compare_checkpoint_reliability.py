import argparse, sys, os, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from plyfile import PlyData
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

def load_ply_full(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    props = {p.name: vertex[p.name] for p in vertex.properties}
    xyz = np.stack([props['x'], props['y'], props['z']], axis=1)
    scale_names = sorted([p for p in props if p.startswith('scale_')])
    scales = np.stack([props[s] for s in scale_names], axis=1) if scale_names else None
    opacity = np.asarray(props.get('opacity', np.ones(len(xyz)))).reshape(-1, 1)
    return xyz, scales, opacity

def gaussian_quat_to_rotmat(q):
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

def extract_base_features(xyz, scales, opacity, quats):
    N = len(xyz)
    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacity))
    scale_linear = np.exp(scales) if scales is not None else np.ones((N, 3))
    scale_min = scale_linear.min(axis=1)
    scale_max = scale_linear.max(axis=1)
    scale_ratio = scale_max / (scale_min + 1e-8)
    scale_volume = np.prod(scale_linear, axis=1)
    quats_norm = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    rots = gaussian_quat_to_rotmat(quats_norm)
    min_axis = np.argmin(scale_linear, axis=1)
    normal_world = np.zeros_like(xyz)
    for i in range(N):
        normal_world[i] = rots[i, :, min_axis[i]]
    return {
        'scale_linear': scale_linear, 'scale_min': scale_min, 'scale_max': scale_max,
        'scale_ratio': scale_ratio, 'scale_volume': scale_volume,
        'normal_world': normal_world, 'opacity_sigmoid': opacity_sigmoid,
    }

def compute_mask_support(xyz, cameras, scene_dir, cfg, device):
    import torch
    from PIL import Image
    N = len(xyz)
    total = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total)
    step = max(1, total // n_views)
    selected = list(range(0, total, step))[:n_views]
    mask_support = np.zeros(N)
    vis_count = np.zeros(N)
    xyz_t = torch.from_numpy(xyz).float().to(device)
    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        w2c = cam.world_view_transform[:3, :3].to(device)
        t = cam.world_view_transform[3, :3].to(device)
        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)
        pts_cam = xyz_t @ w2c.T + t.unsqueeze(0)
        depths = pts_cam[:, 2]
        valid = depths > cfg['projection']['min_positive_depth']
        pts_2d = pts_cam[:, :2] / depths.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid
        mask_path = os.path.join(scene_dir, 'transparent_masks', f'frame_{cam_idx+1:04d}.png')
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert('L')
            mask_arr = torch.from_numpy(np.array(mask_img)).float().to(device) / 255.0
            valid_idx = in_frame.nonzero(as_tuple=True)[0].cpu().numpy()
            sub_v = pts_px[valid_idx, 1].long().clamp(0, mask_arr.shape[0]-1).cpu().numpy()
            sub_u = pts_px[valid_idx, 0].long().clamp(0, mask_arr.shape[1]-1).cpu().numpy()
            mask_vals = mask_arr[sub_v, sub_u].cpu().numpy()
            mask_support[valid_idx] += mask_vals
            vis_count[valid_idx] += 1.0
        if vi % 16 == 0:
            print(f"    mask_support [{vi+1}/{len(selected)}]")
    vis_count = vis_count.clip(min=1)
    return mask_support / vis_count

def compute_gt_errors(xyz, mesh_path, obj_diameter=None):
    import trimesh
    from scipy.spatial import cKDTree
    mesh = trimesh.load(mesh_path, force='mesh')
    if obj_diameter is None:
        bbox = mesh.bounds
        obj_diameter = np.linalg.norm(bbox[1] - bbox[0])
    n_samples = 500000
    sampled, _ = trimesh.sample.sample_surface(mesh, n_samples)
    tree = cKDTree(sampled)
    dists, _ = tree.query(xyz)
    d_norm = dists / obj_diameter
    return dists, d_norm, obj_diameter

def compute_surface_support_risk(xyz, normals, k=24):
    N = len(xyz)
    if N < 3:
        return np.full(N, 0.5, dtype=np.float32)
    k_eff = min(k, N - 1)
    tree = cKDTree(xyz)
    chunk = min(5000, N)
    S_position = np.zeros(N)
    S_normal = np.zeros(N)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        batch_xyz = xyz[start:end]
        batch_normals = normals[start:end]
        dists, idxs = tree.query(batch_xyz, k=k_eff+1)
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]
        local_scales = np.linalg.norm(batch_xyz[:, None] - xyz[idxs], axis=2)
        sigma_p = local_scales.mean(axis=1, keepdims=True) * 3.0
        w_dist = np.exp(-dists**2 / (2 * sigma_p**2 + 1e-8))
        n_dot = np.abs((batch_normals[:, None] * normals[idxs]).sum(axis=2))
        S_position[start:end] = (w_dist * n_dot).sum(axis=1) / (w_dist.sum(axis=1) + 1e-8)
        S_normal[start:end] = n_dot.sum(axis=1) / k_eff
    S_combined = S_position * (0.5 + 0.5 * S_normal)
    lo, hi = np.percentile(S_combined, [5, 95])
    S_norm = ((S_combined - lo) / (hi - lo + 1e-8)).clip(0, 1)
    return 1.0 - S_norm

def compute_scale_anomaly(scale_ratio, scale_volume):
    lo_r, hi_r = np.percentile(scale_ratio, [1, 99])
    r_norm = ((scale_ratio - lo_r) / (hi_r - lo_r + 1e-8)).clip(0, 1)
    lo_v, hi_v = np.percentile(np.log(scale_volume + 1e-8), [1, 99])
    v_norm = ((np.log(scale_volume + 1e-8) - lo_v) / (hi_v - lo_v + 1e-8)).clip(0, 1)
    return (r_norm + v_norm) / 2.0

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

def evaluate_features_on_candidates(features, error_metrics):
    results = {}
    for fname, farr in features.items():
        results[fname] = {}
        for ename, earr in error_metrics.items():
            valid = np.isfinite(farr) & np.isfinite(earr)
            if valid.sum() < 5:
                results[fname][ename] = {'error': 'insufficient data', 'valid_count': int(valid.sum())}
                continue
            s, m = farr[valid], earr[valid]
            nv = len(s)
            rho, _ = spearmanr(s, m)
            rho = float(rho) if not np.isnan(rho) else 0.0
            rho_lo, rho_hi = bootstrap_spearman_ci(s, m)
            k10 = max(1, nv // 10)
            top10_idx = np.argpartition(s, -k10)[-k10:]
            bottom10_idx = np.argpartition(s, k10)[:k10]
            top10_mean = float(m[top10_idx].mean())
            bottom10_mean = float(m[bottom10_idx].mean())
            ratio = top10_mean / max(bottom10_mean, 1e-8)
            results[fname][ename] = {
                'valid_count': nv,
                'spearman_rho': rho,
                'spearman_ci_lo': rho_lo,
                'spearman_ci_hi': rho_hi,
                'top10_mean_error': top10_mean,
                'bottom10_mean_error': bottom10_mean,
                'top10_bottom10_ratio': ratio,
            }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1e_scene01'
    os.makedirs(debug_dir, exist_ok=True)
    device = cfg['device']
    scene_dir = cfg['scene_dir']
    mesh_path = os.path.join(scene_dir, 'meshes', 'scene_mesh.obj')

    model_dir = Path(cfg['model_dir'])
    ckpt_7k_path = model_dir / 'point_cloud' / 'iteration_7000' / 'point_cloud.ply'
    ckpt_15k_path = Path(cfg['checkpoint_path'])

    if not ckpt_7k_path.exists():
        msg = "7k checkpoint not available at " + str(ckpt_7k_path)
        print(f"[SKIP] {msg}")
        result = {'note': msg}
        with open(debug_dir / 'checkpoint_reliability_comparison.json', 'w') as f:
            json.dump(result, f, indent=2)
        return

    # Load 15k cameras (shared)
    print("[1/9] Loading scene for cameras...")
    from recyclegs.tsgs_loader import load_scene, get_train_cameras
    scene_15k, _, _ = load_scene(cfg, device)
    cameras = get_train_cameras(scene_15k)
    print(f"  {len(cameras)} cameras available")

    # ========== 15k (use precomputed) ==========
    print("[2/9] Loading 15k precomputed data...")
    candidate_indices_15k = np.load(out_dir / 'candidate_object_indices.npy')
    err_15k = np.load(out_dir / 'candidate_geometry_errors_v2.npz')
    d_center_15k = err_15k['d_center_norm']
    d_surf1_15k = err_15k['d_surface_proxy_alpha1']
    d_surf2_15k = err_15k['d_surface_proxy_alpha2']
    # Compute error metrics for ALL 15k Gaussians (for fair comparison)
    gt_15k = np.load(out_dir / 'gaussian_gt_errors.npz')
    d_norm_all_15k = gt_15k['normalized_mesh_distance']
    d_surf_all_15k = np.maximum(0, d_norm_all_15k - 1.0 * (np.load(out_dir / 'gaussian_base_features.npz')['scale_linear'].min(axis=1) / float(gt_15k['obj_diameter'].item())))

    E_normal_15k = np.load(out_dir / 'object_normal_conflict.npy')
    E_support_15k = np.load(out_dir / 'object_surface_support_risk.npy')
    E_scale_15k = np.load(out_dir / 'object_scale_anomaly.npy')
    normal_valid_15k = np.load(out_dir / 'object_normal_valid.npy')

    e_norm_15k = E_normal_15k[candidate_indices_15k]
    e_sup_15k = E_support_15k[candidate_indices_15k]
    e_scl_15k = E_scale_15k[candidate_indices_15k]
    nv_15k = normal_valid_15k[candidate_indices_15k]

    error_metrics_15k = {
        'd_center_norm': d_center_15k,
        'd_surface_proxy_alpha1': d_surf1_15k,
        'd_surface_proxy_alpha2': d_surf2_15k,
    }

    features_15k = {
        'E_normal': np.where(nv_15k, e_norm_15k, np.nan),
        'E_support': e_sup_15k,
        'E_scale': e_scl_15k,
    }
    results_15k = evaluate_features_on_candidates(features_15k, error_metrics_15k)

    # ========== 7k (compute from scratch) ==========
    print("[3/9] Loading 7k PLY...")
    xyz_7k, scales_7k, opacity_7k = load_ply_full(ckpt_7k_path)
    ply_7k = PlyData.read(ckpt_7k_path)
    vertex_7k = ply_7k['vertex']
    rot_names = [p.name for p in vertex_7k.properties if p.name.startswith('rot_')]
    if rot_names:
        quats_7k = np.stack([vertex_7k[r] for r in sorted(rot_names)], axis=1)
    else:
        quats_7k = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(xyz_7k), 1))
    print(f"  7k: {len(xyz_7k)} Gaussians")

    print("[4/9] Extracting base features for 7k...")
    base_7k = extract_base_features(xyz_7k, scales_7k, opacity_7k, quats_7k)

    print("[5/9] Computing mask_support for 7k...")
    mask_support_7k = compute_mask_support(xyz_7k, cameras, scene_dir, cfg, device)
    valid_view_7k = np.zeros(len(xyz_7k))
    mask_support_7k_copy = mask_support_7k.copy()
    has_min_views_7k = valid_view_7k >= 3
    # Since we didn't track visibility count separately, let valid_view_7k be a proxy
    # Recompute with visibility tracking
    import torch
    from PIL import Image
    N_7k = len(xyz_7k)
    total_cams = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total_cams)
    step = max(1, total_cams // n_views)
    selected = list(range(0, total_cams, step))[:n_views]
    vis_count_7k = np.zeros(N_7k)
    xyz_7k_t = torch.from_numpy(xyz_7k).float().to(device)
    mask_support_7k_t = np.zeros(N_7k)
    print(f"    Recomputing mask_support + visibility for 7k over {len(selected)} views...")
    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        w2c = cam.world_view_transform[:3, :3].to(device)
        t = cam.world_view_transform[3, :3].to(device)
        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)
        pts_cam = xyz_7k_t @ w2c.T + t.unsqueeze(0)
        depths = pts_cam[:, 2]
        valid = depths > cfg['projection']['min_positive_depth']
        pts_2d = pts_cam[:, :2] / depths.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid
        vis_count_7k += in_frame.cpu().numpy()
        mask_path = os.path.join(scene_dir, 'transparent_masks', f'frame_{cam_idx+1:04d}.png')
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert('L')
            mask_arr = torch.from_numpy(np.array(mask_img)).float().to(device) / 255.0
            valid_idx = in_frame.nonzero(as_tuple=True)[0].cpu().numpy()
            sub_v = pts_px[valid_idx, 1].long().clamp(0, mask_arr.shape[0]-1).cpu().numpy()
            sub_u = pts_px[valid_idx, 0].long().clamp(0, mask_arr.shape[1]-1).cpu().numpy()
            mask_vals = mask_arr[sub_v, sub_u].cpu().numpy()
            mask_support_7k_t[valid_idx] += mask_vals
        if vi % 16 == 0:
            print(f"    [{vi+1}/{len(selected)}]")
    vis_count_7k = vis_count_7k.clip(min=1)
    mask_support_mean_7k = mask_support_7k_t / vis_count_7k
    has_min_views_7k = vis_count_7k >= 3

    print("[6/9] Computing candidate-object domain for 7k...")
    strong_bg_7k = has_min_views_7k & (mask_support_mean_7k <= 0.05)
    candidate_7k = has_min_views_7k & (mask_support_mean_7k >= 0.20) & ~strong_bg_7k
    cand_idx_7k = np.where(candidate_7k)[0]
    print(f"  7k candidate-object: {len(cand_idx_7k)}")

    print("[7/9] Computing GT errors for 7k...")
    dist_7k, d_norm_7k, obj_diam = compute_gt_errors(xyz_7k, mesh_path)

    d_center_7k_cand = d_norm_7k[cand_idx_7k]
    min_scale_7k = base_7k['scale_min'][cand_idx_7k]
    d_surf1_7k_cand = np.maximum(0, d_center_7k_cand - 1.0 * min_scale_7k / obj_diam)
    d_surf2_7k_cand = np.maximum(0, d_center_7k_cand - 2.0 * min_scale_7k / obj_diam)

    print("[8/9] Computing features for 7k candidate domain...")
    normals_7k = base_7k['normal_world']
    e_support_7k_cand = compute_surface_support_risk(xyz_7k[cand_idx_7k], normals_7k[cand_idx_7k], k=min(cfg['surface_support']['knn'], max(1, len(cand_idx_7k)-1)))

    e_scale_7k_cand = compute_scale_anomaly(base_7k['scale_ratio'][cand_idx_7k], base_7k['scale_volume'][cand_idx_7k])

    e_normal_7k_cand = np.full(len(cand_idx_7k), np.nan)
    if len(cand_idx_7k) >= 3:
        print(f"    Computing normal conflict for 7k candidate Gaussians (N={len(cand_idx_7k)})...")
        try:
            normal_errors_list = []
            for vi, cam_idx in enumerate(selected):
                cam = cameras[cam_idx]
                w2c = cam.world_view_transform[:3, :3].to(device)
                t = cam.world_view_transform[3, :3].to(device)
                K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)
                pts_cam = xyz_7k_t @ w2c.T + t.unsqueeze(0)
                depths = pts_cam[:, 2]
                valid = depths > cfg['projection']['min_positive_depth']
                pts_2d = pts_cam[:, :2] / depths.clamp(min=1e-8).unsqueeze(1)
                pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
                u, v = pts_px[:, 0], pts_px[:, 1]
                in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid

                stem = cam.image_name.split('.')[0]
                normal_path = os.path.join(scene_dir, 'normals', f'{stem}_normal.png')
                if not os.path.exists(normal_path):
                    continue
                nimg = Image.open(normal_path).convert('RGB')
                nimg_resized = nimg.resize((cam.image_width, cam.image_height))
                n_arr = torch.from_numpy(np.array(nimg_resized)).float().to(device) / 255.0 * 2 - 1
                n_arr = n_arr / n_arr.norm(dim=2, keepdim=True).clamp(min=1e-8)

                normal_world_t = torch.from_numpy(normals_7k).float().to(device)
                normal_cam = normal_world_t @ w2c.T
                normal_cam = normal_cam / normal_cam.norm(dim=1, keepdim=True).clamp(min=1e-8)

                errs = np.full(N_7k, 0.5, dtype=np.float32)
                valid_idx = in_frame.nonzero(as_tuple=True)[0].cpu().numpy()
                sub_v = pts_px[valid_idx, 1].long().clamp(0, n_arr.shape[0]-1).cpu().numpy()
                sub_u = pts_px[valid_idx, 0].long().clamp(0, n_arr.shape[1]-1).cpu().numpy()
                prior = n_arr[sub_v, sub_u]
                dot = (normal_cam[valid_idx] * prior).sum(dim=1).abs()
                errs[valid_idx] = (1.0 - dot.clamp(max=1.0)).cpu().numpy()
                normal_errors_list.append(errs)
                if vi % 16 == 0:
                    print(f"      normal conflict [{vi+1}/{len(selected)}]")

            if normal_errors_list:
                err_stack = np.stack(normal_errors_list, axis=1)
                e_normal_all = np.median(err_stack, axis=1)
                e_normal_7k_cand = e_normal_all[cand_idx_7k]
        except Exception as exc:
            print(f"    Normal conflict computation failed: {exc}")
    else:
        print(f"    Skipping normal conflict: only {len(cand_idx_7k)} candidate Gaussians")

    error_metrics_7k = {
        'd_center_norm': d_center_7k_cand,
        'd_surface_proxy_alpha1': d_surf1_7k_cand,
        'd_surface_proxy_alpha2': d_surf2_7k_cand,
    }
    features_7k = {
        'E_normal': e_normal_7k_cand,
        'E_support': e_support_7k_cand,
        'E_scale': e_scale_7k_cand,
    }
    results_7k = evaluate_features_on_candidates(features_7k, error_metrics_7k)

    print("[9/9] Saving results...")
    comparison = {
        'scene_name': cfg['scene_name'],
        '7k': {
            'num_gaussians': int(len(xyz_7k)),
            'candidate_object_count': int(len(cand_idx_7k)),
            'candidate_object_ratio': float(len(cand_idx_7k) / len(xyz_7k)),
            'feature_evaluation': results_7k,
        },
        '15k': {
            'num_gaussians': int(len(gt_15k['mesh_distance'])),
            'candidate_object_count': int(len(candidate_indices_15k)),
            'candidate_object_ratio': float(len(candidate_indices_15k) / len(gt_15k['mesh_distance'])),
            'feature_evaluation': results_15k,
        },
    }

    json_path = debug_dir / 'checkpoint_reliability_comparison.json'
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    import csv
    csv_path = debug_dir / 'checkpoint_reliability_comparison.csv'
    with open(csv_path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['checkpoint', 'feature', 'error_metric', 'spearman_rho', 'ci_lo', 'ci_hi', 'top10_mean', 'bottom10_mean', 'ratio'])
        for ckpt_name, ckpt_data in [('7k', results_7k), ('15k', results_15k)]:
            for fname, emap in ckpt_data.items():
                for ename, vals in emap.items():
                    if 'error' in vals:
                        w.writerow([ckpt_name, fname, ename, vals['error'], '', '', '', '', ''])
                    else:
                        w.writerow([ckpt_name, fname, ename, vals['spearman_rho'],
                                    vals['spearman_ci_lo'], vals['spearman_ci_hi'],
                                    vals['top10_mean_error'], vals['bottom10_mean_error'],
                                    vals['top10_bottom10_ratio']])

    md = [
        f"# Checkpoint Reliability Comparison (7k vs 15k) - {cfg['scene_name']}",
        f"",
        f"## Overview",
        f"| Metric | 7k | 15k |",
        f"|--------|-----|-----|",
        f"| Total Gaussians | {comparison['7k']['num_gaussians']} | {comparison['15k']['num_gaussians']} |",
        f"| Candidate Object Count | {comparison['7k']['candidate_object_count']} | {comparison['15k']['candidate_object_count']} |",
        f"| Candidate Object Ratio | {comparison['7k']['candidate_object_ratio']*100:.2f}% | {comparison['15k']['candidate_object_ratio']*100:.2f}% |",
        f"",
        f"## Signed Spearman Correlations (candidate-object domain)",
        f"| Feature | Error Metric | 7k rho | 7k CI | 15k rho | 15k CI |",
        f"|--------|-------------|--------|-------|---------|-------|",
    ]
    for fname in ['E_normal', 'E_support', 'E_scale']:
        for ename in ['d_center_norm', 'd_surface_proxy_alpha1', 'd_surface_proxy_alpha2']:
            r7 = results_7k[fname][ename]
            r15 = results_15k[fname][ename]
            if 'error' in r7:
                row7 = r7['error']
            else:
                row7 = f"{r7['spearman_rho']:.4f} [{r7['spearman_ci_lo']:.4f}, {r7['spearman_ci_hi']:.4f}]"
            if 'error' in r15:
                row15 = r15['error']
            else:
                row15 = f"{r15['spearman_rho']:.4f} [{r15['spearman_ci_lo']:.4f}, {r15['spearman_ci_hi']:.4f}]"
            md.append(f"| {fname} | {ename} | {row7} | | {row15} | |")
    md.append(f"")

    md_path = debug_dir / 'checkpoint_reliability_comparison.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"Saved to {json_path} and {md_path}")

if __name__ == '__main__':
    main()
