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
    n_views = min(64, total)
    step = max(1, total // n_views)
    selected = list(range(0, total, step))[:n_views]

    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = torch.from_numpy(base['xyz']).float().to(device)
    opacity = torch.from_numpy(base['opacity_sigmoid']).float().to(device)
    scales = torch.from_numpy(base['scale_linear']).float().to(device)
    N = xyz.shape[0]

    contrib = torch.zeros(N, device=device)
    n_visible = torch.zeros(N, device=device)

    print(f"[2/4] Computing proxy contribution...")
    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        w2c = cam.world_view_transform[:3, :3].to(device)
        t = cam.world_view_transform[3, :3].to(device)
        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)
        pts_cam = xyz @ w2c.T + t.unsqueeze(0)
        z = pts_cam[:, 2]
        valid = z > cfg['projection']['min_positive_depth']
        pts_2d = pts_cam[:, :2] / z.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= -10) & (u < cam.image_width+10) & (v >= -10) & (v < cam.image_height+10) & valid
        proj_radius = torch.sqrt(scales[:, 0]**2 + scales[:, 1]**2) * 3.0
        contrib[in_frame] += opacity[in_frame] * proj_radius[in_frame].clamp(max=100.0)
        n_visible[in_frame] += 1.0
        if vi % 16 == 0:
            print(f"  [{vi+1}/{len(selected)}]")

    contrib_np = contrib.cpu().numpy()
    percentile = np.percentile(contrib_np[contrib_np > 0], np.arange(0, 101)) if contrib_np.sum() > 0 else np.zeros(101)
    save_np(contrib_np, out_dir / 'contribution.npy')
    save_np(percentile, out_dir / 'contribution_percentile.npy')
    save_json({'contribution_mode': 'proxy'}, out_dir / 'contribution_stats.json')
    print(f"[3/4] Contribution computed (proxy)")
    print(f"[4/4] Saved")

if __name__ == '__main__':
    main()
