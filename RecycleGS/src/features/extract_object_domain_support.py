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

    print("[1/5] Loading scene...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total)
    step = max(1, total // n_views)
    selected = list(range(0, total, step))[:n_views]

    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = torch.from_numpy(base['xyz']).float().to(device)
    opacity_sigmoid = base['opacity_sigmoid']
    N = xyz.shape[0]

    valid_view_count = torch.zeros(N, device=device)
    mask_inside_count = torch.zeros(N, device=device)
    scene_dir = cfg['scene_dir']

    print(f"[2/5] Projecting onto {len(selected)} views...")
    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        stem = cam.image_name.split('.')[0]
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

        valid_view_count += in_frame.float()

        mask_path = os.path.join(scene_dir, 'transparent_masks', f'{stem}.png')
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert('L')
            mask_img_resized = mask_img.resize((cam.image_width, cam.image_height))
            mask_arr = torch.from_numpy(np.array(mask_img_resized)).float().to(device) / 255.0
            valid_idx = in_frame.nonzero(as_tuple=True)[0]
            sub_v = pts_px[valid_idx, 1].long().clamp(0, mask_arr.shape[0]-1)
            sub_u = pts_px[valid_idx, 0].long().clamp(0, mask_arr.shape[1]-1)
            mask_inside_count[valid_idx] += (mask_arr[sub_v, sub_u] > 0.5).float()
        if vi % 16 == 0:
            print(f"  [{vi+1}/{len(selected)}]")

    valid_view_np = valid_view_count.cpu().numpy()
    mask_inside_np = mask_inside_count.cpu().numpy()

    min_valid = 3
    mask_support = np.zeros(N, dtype=np.float32)
    valid_mask = valid_view_np >= min_valid
    mask_support[valid_mask] = mask_inside_np[valid_mask] / valid_view_np[valid_mask].clip(min=1)

    save_np(valid_view_np, out_dir / 'valid_view_count.npy')
    save_np(mask_inside_np, out_dir / 'mask_inside_count.npy')
    save_np(mask_support, out_dir / 'mask_support_unweighted.npy')

    stats = {
        'num_gaussians': N,
        'num_views': len(selected),
        'valid_view_stats': {
            'min': float(valid_view_np.min()),
            'max': float(valid_view_np.max()),
            'mean': float(valid_view_np.mean()),
            'median': float(np.median(valid_view_np)),
            'pct_ge_3': float((valid_view_np >= 3).mean()),
        },
        'mask_support_stats': {
            'min': float(mask_support.min()),
            'max': float(mask_support.max()),
            'mean': float(mask_support.mean()),
            'median': float(np.median(mask_support[mask_support > 0])),
        },
        'opacity_sigmoid_mean': float(opacity_sigmoid.mean()),
    }
    save_json(stats, out_dir / 'object_domain_support_stats.json')
    print(f"[3/5] valid_view: min={valid_view_np.min():.0f} max={valid_view_np.max():.0f}")
    print(f"[4/5] mask_support: mean={mask_support.mean():.4f}")
    print(f"[5/5] Saved to {out_dir}")

if __name__ == '__main__':
    main()
