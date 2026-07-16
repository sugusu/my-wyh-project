import argparse, sys, os, torch, numpy as np
from pathlib import Path
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

    print("[1/4] Loading model...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total)
    step = max(1, total // n_views)
    selected = list(range(0, total, step))[:n_views]

    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz_t = torch.from_numpy(base['xyz']).float().to(device)
    scales_t = torch.from_numpy(base['scale_linear']).float().to(device)
    N = xyz_t.shape[0]

    depth_errors = []
    vis_counts = torch.zeros(N, device=device)

    print(f"[2/4] Computing depth-order conflict...")
    ref_dir = Path(cfg['reference_output_dir'])
    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        w2c = cam.world_view_transform[:3, :3].to(device)
        t = cam.world_view_transform[3, :3].to(device)
        pts_cam = xyz_t @ w2c.T + t.unsqueeze(0)
        z_gauss = pts_cam[:, 2]
        valid = z_gauss > cfg['projection']['min_positive_depth']

        depth_path = ref_dir / 'depth_ref' / f'view_{cam_idx:04d}.npy'
        if not depth_path.exists():
            depth_errors.append(torch.zeros(N, device=device).cpu().numpy())
            continue
        depth_ref = torch.from_numpy(np.load(depth_path)).float().to(device)
        if depth_ref.dim() == 2:
            K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)
            pts_2d = pts_cam[:, :2] / z_gauss.clamp(min=1e-8).unsqueeze(1)
            pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
            u, v = pts_px[:, 0], pts_px[:, 1]
            in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid
            valid_idx = in_frame.nonzero(as_tuple=True)[0]
            sub_v = v[valid_idx].long().clamp(0, depth_ref.shape[0]-1)
            sub_u = u[valid_idx].long().clamp(0, depth_ref.shape[1]-1)
            depth_ref_vals = depth_ref[sub_v, sub_u]

            tau_abs = cfg['depth_order']['absolute_tolerance_scale'] * scales_t[valid_idx].mean(dim=1)
            tau_rel = cfg['depth_order']['relative_tolerance'] * depth_ref_vals
            tau = torch.max(tau_abs, tau_rel)
            behind = z_gauss[valid_idx] > depth_ref_vals + tau
            errs = torch.zeros(N, device=device)
            e_depth = (z_gauss[valid_idx] - depth_ref_vals) / (depth_ref_vals + 1e-8)
            errs[valid_idx] = torch.where(behind, e_depth, torch.zeros_like(e_depth))
            depth_errors.append(errs.cpu().numpy())
            vis_counts[valid_idx] += 1.0
        else:
            depth_errors.append(torch.zeros(N, device=device).cpu().numpy())
        if vi % 16 == 0:
            print(f"  [{vi+1}/{len(selected)}]")

    err_stack = np.stack(depth_errors, axis=1)
    E_depth = np.median(err_stack, axis=1)
    E_depth[vis_counts.cpu().numpy() == 0] = 0.0
    save_np(E_depth, out_dir / 'depth_order_conflict.npy')
    save_json({'note': 'depth_ref only for relative ordering, not GT geometry'}, out_dir / 'depth_order_stats.json')
    print(f"[3/4] Depth-order conflict computed")
    print(f"[4/4] Saved")

if __name__ == '__main__':
    main()
