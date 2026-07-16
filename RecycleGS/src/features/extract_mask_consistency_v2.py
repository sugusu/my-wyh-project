import argparse, sys, os, torch, numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config, save_np, save_json
from recyclegs.tsgs_loader import load_scene, get_train_cameras

def load_ply_xyz(ply_path):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    xyz = np.stack([np.asarray(vertex['x']), np.asarray(vertex['y']), np.asarray(vertex['z'])], axis=1)
    return xyz

def compute_signed_distance_to_edge(mask_binary_np):
    inside_dist = distance_transform_edt(mask_binary_np)
    outside_dist = distance_transform_edt(1 - mask_binary_np)
    signed = np.where(mask_binary_np > 0, inside_dist, -outside_dist)
    return signed

def percentile_normalize(arr, low=5, high=95):
    lo, hi = np.percentile(arr[~np.isnan(arr)], [low, high]) if np.isfinite(arr).sum() > 0 else (0, 1)
    clipped = arr.clip(lo, hi)
    return (clipped - lo) / (hi - lo + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--iteration', type=int, required=True, choices=[15000])
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    device = cfg['device']
    scene_dir = cfg['scene_dir']
    iter_str = f"iter_{args.iteration}"
    iter_dir = out_dir / iter_str
    os.makedirs(iter_dir, exist_ok=True)

    print("[1/6] Loading scene and cameras...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total)
    step = max(1, total // n_views)
    selected = list(range(0, total, step))[:n_views]
    print(f"  Using {len(selected)} cameras from {total} total")

    print("[2/6] Loading candidate Gaussians...")
    candidate_indices = np.load(iter_dir / 'candidate_indices.npy')
    ply_path = Path(cfg['checkpoint_path'])
    xyz_all = load_ply_xyz(str(ply_path))
    xyz = torch.from_numpy(xyz_all[candidate_indices]).float().to(device)
    N = len(xyz)
    print(f"  Candidate Gaussians: {N}")

    print(f"[3/6] Projecting onto {len(selected)} views...")
    mask_values = np.full((N, len(selected)), np.nan, dtype=np.float32)
    boundary_dist = np.full((N, len(selected)), np.nan, dtype=np.float32)

    mask_dir = os.path.join(scene_dir, 'transparent_masks')

    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        stem = cam.image_name.split('.')[0]

        w2c = cam.world_view_transform[:3, :3].to(device)
        tvec = cam.world_view_transform[3, :3].to(device)
        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]],
                         dtype=torch.float32, device=device)

        pts_cam = xyz @ w2c.T + tvec.unsqueeze(0)
        depths = pts_cam[:, 2]
        valid_depth = depths > cfg['projection']['min_positive_depth']
        pts_2d = pts_cam[:, :2] / depths.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid_depth

        mask_path = os.path.join(mask_dir, f'{stem}.png')
        if not os.path.exists(mask_path):
            continue

        mask_img = Image.open(mask_path).convert('L')
        mask_img_resized = mask_img.resize((cam.image_width, cam.image_height))
        mask_np = np.array(mask_img_resized, dtype=np.float32) / 255.0
        mask_t = torch.from_numpy(mask_np).float().to(device)

        mask_binary_np = (mask_np > 0.5).astype(np.uint8)
        signed_dist_np = compute_signed_distance_to_edge(mask_binary_np)

        valid_idx = in_frame.nonzero(as_tuple=True)[0].cpu().numpy()
        if len(valid_idx) == 0:
            continue

        sub_v = pts_px[valid_idx, 1].long().clamp(0, mask_t.shape[0]-1)
        sub_u = pts_px[valid_idx, 0].long().clamp(0, mask_t.shape[1]-1)

        v_cpu = sub_v.cpu().numpy()
        u_cpu = sub_u.cpu().numpy()

        mask_values[valid_idx, vi] = mask_np[v_cpu, u_cpu]
        boundary_dist[valid_idx, vi] = signed_dist_np[v_cpu, u_cpu]

        if vi % 16 == 0:
            print(f"  [{vi+1}/{len(selected)}]")

    print("[4/6] Computing per-Gaussian mask consistency statistics...")
    valid_view_count = np.sum(~np.isnan(mask_values), axis=1)
    min_valid = cfg.get('mask_consistency', {}).get('min_valid_views', 5)
    valid_mask = valid_view_count >= min_valid

    S_mask = np.full(N, np.nan, dtype=np.float32)
    V_mask = np.full(N, np.nan, dtype=np.float32)
    outside_ratio = np.full(N, np.nan, dtype=np.float32)
    boundary_ratio = np.full(N, np.nan, dtype=np.float32)
    inside_ratio = np.full(N, np.nan, dtype=np.float32)

    for i in range(N):
        if not valid_mask[i]:
            continue
        vals = mask_values[i, ~np.isnan(mask_values[i])]
        bd = boundary_dist[i, ~np.isnan(boundary_dist[i])]
        S_mask[i] = np.mean(vals)
        V_mask[i] = np.var(vals)

        inside = np.sum(bd > 3.0) / len(bd)
        boundary = np.sum(np.abs(bd) <= 3.0) / len(bd)
        outside = np.sum(bd < -3.0) / len(bd)
        inside_ratio[i] = inside
        boundary_ratio[i] = boundary
        outside_ratio[i] = outside

    print(f"  Valid Gaussians (>{min_valid} views): {valid_mask.sum()}/{N}")

    print("[5/6] Computing risk scores...")
    S_valid = S_mask[valid_mask]
    V_valid = V_mask[valid_mask]
    outside_valid = outside_ratio[valid_mask]
    boundary_valid = boundary_ratio[valid_mask]

    E_mask_mean = np.full(N, np.nan, dtype=np.float32)
    E_mask_variance = np.full(N, np.nan, dtype=np.float32)
    E_mask_boundary = np.full(N, np.nan, dtype=np.float32)
    E_mask_cv = np.full(N, np.nan, dtype=np.float32)

    if valid_mask.sum() > 0:
        E_mean_norm = 1.0 - percentile_normalize(S_valid)
        E_var_norm = percentile_normalize(V_valid)
        boundary_raw = 0.7 * outside_valid + 0.3 * boundary_valid
        E_boundary_norm = percentile_normalize(boundary_raw)

        E_mask_mean[valid_mask] = E_mean_norm
        E_mask_variance[valid_mask] = E_var_norm
        E_mask_boundary[valid_mask] = E_boundary_norm
        E_mask_cv[valid_mask] = 0.40 * E_mean_norm + 0.30 * E_var_norm + 0.30 * E_boundary_norm

    print(f"  E_mask_cv: mean={np.nanmean(E_mask_cv):.4f}, valid={np.isfinite(E_mask_cv).sum()}")

    print("[6/6] Saving outputs...")
    np.save(iter_dir / 'mask_consistency_S_mean.npy', S_mask.astype(np.float32))
    np.save(iter_dir / 'mask_consistency_V_var.npy', V_mask.astype(np.float32))
    np.save(iter_dir / 'mask_consistency_outside_ratio.npy', outside_ratio.astype(np.float32))
    np.save(iter_dir / 'mask_consistency_boundary_ratio.npy', boundary_ratio.astype(np.float32))
    np.save(iter_dir / 'mask_consistency_inside_ratio.npy', inside_ratio.astype(np.float32))
    np.save(iter_dir / 'mask_consistency_valid_view_count.npy', valid_view_count.astype(np.float32))
    np.save(iter_dir / 'mask_risk_mean.npy', E_mask_mean.astype(np.float32))
    np.save(iter_dir / 'mask_risk_variance.npy', E_mask_variance.astype(np.float32))
    np.save(iter_dir / 'mask_risk_boundary.npy', E_mask_boundary.astype(np.float32))
    np.save(iter_dir / 'mask_risk_cv.npy', E_mask_cv.astype(np.float32))

    stats = {
        'iteration': args.iteration,
        'num_views': len(selected),
        'num_candidate_gaussians': N,
        'num_valid_gaussians': int(valid_mask.sum()),
        'min_valid_views': min_valid,
        'S_mask_mean': float(np.nanmean(S_mask)),
        'V_mask_mean': float(np.nanmean(V_mask)),
        'E_mask_mean_mean': float(np.nanmean(E_mask_mean)),
        'E_mask_variance_mean': float(np.nanmean(E_mask_variance)),
        'E_mask_boundary_mean': float(np.nanmean(E_mask_boundary)),
        'E_mask_cv_mean': float(np.nanmean(E_mask_cv)),
    }
    save_json(stats, iter_dir / 'mask_consistency_stats.json')
    print("  Done.")

if __name__ == '__main__':
    main()
