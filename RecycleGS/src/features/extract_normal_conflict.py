import argparse, sys, os, torch, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config, save_np, save_json
from recyclegs.tsgs_loader import load_scene, get_train_cameras

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    device = cfg['device']

    print("[1/6] Loading model and object domain...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total)
    step = max(1, total // n_views)
    selected = list(range(0, total, step))[:n_views]

    base = np.load(out_dir / 'gaussian_base_features.npz')
    object_indices = np.load(out_dir / 'object_indices.npy')
    N_total = len(base['xyz'])

    normal_world = torch.from_numpy(base['normal_world']).float().to(device)
    normal_errors = []
    vis_counts = torch.zeros(N_total, device=device)

    scene_dir = cfg['scene_dir']

    print(f"[2/6] Computing normal conflict over {len(selected)} views (object-domain aware)...")
    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        w2c = cam.world_view_transform[:3, :3].to(device)
        t = cam.world_view_transform[3, :3].to(device)
        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)
        xyz = torch.from_numpy(base['xyz']).float().to(device)
        pts_cam = xyz @ w2c.T + t.unsqueeze(0)
        depths = pts_cam[:, 2]
        valid = depths > cfg['projection']['min_positive_depth']
        pts_2d = pts_cam[:, :2] / depths.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid

        stem = cam.image_name.split('.')[0]
        normal_path = os.path.join(scene_dir, 'normals', f'{stem}_normal.png')
        if os.path.exists(normal_path):
            nimg = Image.open(normal_path).convert('RGB')
            nimg_resized = nimg.resize((cam.image_width, cam.image_height))
            n_arr = torch.from_numpy(np.array(nimg_resized)).float().to(device) / 255.0 * 2 - 1
            n_arr = n_arr / n_arr.norm(dim=2, keepdim=True).clamp(min=1e-8)
            normal_cam = normal_world @ w2c.T
            normal_cam = normal_cam / normal_cam.norm(dim=1, keepdim=True).clamp(min=1e-8)

            errs = torch.full((N_total,), 0.5, device=device, dtype=torch.float32)
            valid_idx = in_frame.nonzero(as_tuple=True)[0]
            sub_v = pts_px[valid_idx, 1].long().clamp(0, n_arr.shape[0]-1)
            sub_u = pts_px[valid_idx, 0].long().clamp(0, n_arr.shape[1]-1)
            prior = n_arr[sub_v, sub_u]
            dot = (normal_cam[valid_idx] * prior).sum(dim=1).abs()
            errs[valid_idx] = 1.0 - dot.clamp(max=1.0)
            normal_errors.append(errs.cpu().numpy())
            vis_counts[valid_idx] += 1.0
        if vi % 16 == 0:
            print(f"  [{vi+1}/{len(selected)}]")

    err_stack = np.stack(normal_errors, axis=1)
    vis_mask_obj = np.zeros(N_total, dtype=bool)
    vis_mask_obj[object_indices] = True

    E_normal = np.median(err_stack, axis=1)
    V_normal = np.var(err_stack, axis=1)

    E_normal[~vis_mask_obj] = 0.0
    V_normal[~vis_mask_obj] = 0.0

    normal_valid = np.zeros(N_total, dtype=bool)
    for obj_idx in object_indices:
        if vis_counts[obj_idx].cpu().numpy() >= 3:
            normal_valid[obj_idx] = True

    save_np(E_normal, out_dir / 'object_normal_conflict.npy')
    save_np(normal_valid, out_dir / 'object_normal_valid.npy')

    scales_linear = base['scale_linear']
    scale_min = scales_linear.min(axis=1)
    scale_max = scales_linear.max(axis=1)
    planarity = 1.0 - scale_min / (scale_max + 1e-8)
    save_np(planarity, out_dir / 'object_planarity_confidence.npy')

    stats = {
        'num_views': len(selected),
        'num_object_gaussians': int(object_indices.sum()),
        'normal_valid_count': int(normal_valid.sum()),
        'planarity_confidence_mean': float(planarity.mean()),
        'planarity_confidence_min': float(planarity.min()),
        'planarity_confidence_max': float(planarity.max()),
        'note': 'Invalid projections NOT filled with 0. Only object-supported Gaussians evaluated.',
    }
    save_json(stats, out_dir / 'object_normal_conflict_stats.json')
    print(f"[3/6] Normal valid: {normal_valid.sum()}/{len(object_indices)} object Gaussians")
    print(f"[4/6] Planarity confidence: mean={planarity.mean():.4f}")
    print(f"[5/6] Object normal conflict computed")
    print(f"[6/6] Saved")

if __name__ == '__main__':
    main()
