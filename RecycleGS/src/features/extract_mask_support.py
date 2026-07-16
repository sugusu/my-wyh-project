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

    print("[1/4] Loading scene...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total)
    step = max(1, total // n_views)
    selected = list(range(0, total, step))[:n_views]

    xyz = gaussians.get_xyz.detach()
    N = xyz.shape[0]
    mask_support = torch.zeros(N, device=device)
    visibility_count = torch.zeros(N, device=device)

    scene_dir = cfg['scene_dir']
    print(f"[2/4] Projecting onto {len(selected)} views (chunked)...")

    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        w2c = cam.world_view_transform[:3, :3].to(device)
        t = cam.world_view_transform[3, :3].to(device)
        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)
        pts_cam = xyz @ w2c.T + t.unsqueeze(0)
        depths = pts_cam[:, 2]
        valid = depths > cfg['projection']['min_positive_depth']
        pts_2d = pts_cam[:, :2] / depths.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid
        vis_px = torch.stack([v, u], dim=1)

        mask_path = os.path.join(scene_dir, 'transparent_masks', f'frame_{cam_idx+1:04d}.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(scene_dir, 'transparent_masks', f'frame_{cam_idx+1:04d}.png')
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert('L')
            mask_arr = torch.from_numpy(np.array(mask_img)).float().to(device) / 255.0
            mask_vals = torch.zeros(N, device=device)
            valid_idx = in_frame.nonzero(as_tuple=True)[0]
            sub_v = vis_px[valid_idx, 0].long()
            sub_u = vis_px[valid_idx, 1].long()
            sub_v.clamp_(0, mask_arr.shape[0]-1)
            sub_u.clamp_(0, mask_arr.shape[1]-1)
            mask_vals[valid_idx] = mask_arr[sub_v, sub_u]
            mask_support += mask_vals.float()
            visibility_count += in_frame.float()
        if vi % 16 == 0:
            print(f"  [{vi+1}/{len(selected)}]")

    vis_count = visibility_count.clamp(min=1)
    mask_score = mask_support / vis_count
    mask_risk = 1.0 - mask_score

    out = {'contribution_mode': 'proxy'}
    save_np(mask_score.cpu().numpy(), out_dir / 'mask_support.npy')
    save_np(mask_risk.cpu().numpy(), out_dir / 'mask_risk.npy')
    save_json(out, out_dir / 'mask_support_stats.json')
    print(f"[3/4] Mask support computed")
    print(f"[4/4] Saved to {out_dir}")

if __name__ == '__main__':
    main()
