import argparse, sys, os, json, torch, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config
from recyclegs.tsgs_loader import load_scene, get_train_cameras

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    debug_dir = Path(cfg['debug_output_dir'])
    out_dir = debug_dir / 'mask_alignment'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get('device', 'cuda:0')

    print("[1/4] Loading model and selecting views...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)
    n_views = 8
    rng = np.random.RandomState(42)
    selected_idx = rng.choice(total, min(n_views, total), replace=False)
    print(f"  Total cameras: {total}, selected {len(selected_idx)} views: {selected_idx.tolist()}")

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    print(f"  Gaussians: {len(xyz)}")

    print("[2/4] Loading images, masks, and projecting Gaussians...")
    scene_dir = cfg['scene_dir']
    image_dir = os.path.join(scene_dir, 'images')
    mask_dir = os.path.join(scene_dir, 'transparent_masks')

    total_cam = total
    total_mask_images = len(os.listdir(mask_dir)) if os.path.isdir(mask_dir) else 0
    stem_matched = 0
    missing_masks = []

    report_views = {}
    for vi, cam_idx in enumerate(selected_idx):
        cam = cameras[cam_idx]
        w2c = cam.world_view_transform[:3, :3].to(device)
        t = cam.world_view_transform[3, :3].to(device)
        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)

        xyz_t = torch.from_numpy(xyz).float().to(device)
        pts_cam = xyz_t @ w2c.T + t.unsqueeze(0)
        depths = pts_cam[:, 2]
        valid = depths > cfg['projection']['min_positive_depth']
        pts_2d = pts_cam[:, :2] / depths.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid
        proj_uv = torch.stack([u, v], dim=1)

        stem = f'frame_{cam_idx+1:04d}'
        img_path = os.path.join(image_dir, f'{stem}.png')
        mask_img_path = os.path.join(mask_dir, f'{stem}.png')

        if not os.path.exists(mask_img_path):
            missing_masks.append(stem)
            print(f"  View {cam_idx}: mask not found at {mask_img_path}")
            continue
        stem_matched += 1

        rgb = Image.open(img_path).convert('RGB')
        rgb_arr = np.array(rgb)
        mask_img = Image.open(mask_img_path).convert('L')
        mask_arr = np.array(mask_img)

        proj_u = proj_uv[in_frame, 0].cpu().numpy()
        proj_v = proj_uv[in_frame, 1].cpu().numpy()
        proj_u_i = np.clip(proj_u.astype(int), 0, rgb_arr.shape[1]-1)
        proj_v_i = np.clip(proj_v.astype(int), 0, rgb_arr.shape[0]-1)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(rgb_arr)
        axes[0].set_title(f'RGB View {cam_idx}')
        axes[0].scatter(proj_u_i, proj_v_i, s=1, c='red', alpha=0.3)

        axes[1].imshow(mask_arr, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f'Mask View {cam_idx}')
        axes[1].scatter(proj_u_i, proj_v_i, s=1, c='red', alpha=0.3)

        overlay = rgb_arr.copy().astype(float)
        overlay_r = overlay[:,:,0].copy()
        overlay_r[mask_arr > 128] = overlay_r[mask_arr > 128] * 0.5 + 255 * 0.5
        overlay[:,:,0] = overlay_r
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title(f'Overlay (mask region tinted red)')
        axes[2].scatter(proj_u_i, proj_v_i, s=1, c='cyan', alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(out_dir / f'view_{cam_idx:04d}_mask_alignment.png'), dpi=150)
        plt.close(fig)

        report_views[f'view_{cam_idx:04d}'] = {
            'image_path': img_path,
            'mask_path': mask_img_path,
            'num_gaussians_in_frame': int(in_frame.sum().item()),
            'image_shape': list(rgb_arr.shape),
            'mask_shape': list(mask_arr.shape),
        }
        print(f"  [{vi+1}/{len(selected_idx)}] View {cam_idx}: {in_frame.sum().item()} Gaussians projected")

    print("[3/4] Generating report...")
    report = {
        'scene_name': cfg['scene_name'],
        'total_cameras': total_cam,
        'total_mask_images': total_mask_images,
        'selected_views': int(len(selected_idx)),
        'stem_matched_count': stem_matched,
        'missing_masks': missing_masks,
        'missing_count': len(missing_masks),
        'views': report_views,
    }
    with open(debug_dir / 'mask_alignment_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    md = [
        f"# Mask Alignment Diagnosis - {cfg['scene_name']}",
        f"",
        f"## Summary",
        f"- Total cameras: {total_cam}",
        f"- Total mask images: {total_mask_images}",
        f"- Stem-matched count: {stem_matched}",
        f"- Missing masks: {missing_masks} ({len(missing_masks)} missing)",
        f"- Selected views for visualization: {len(selected_idx)}",
        f"",
        f"## Visualizations",
    ]
    for vi, cam_idx in enumerate(selected_idx):
        png_path = out_dir / f'view_{cam_idx:04d}_mask_alignment.png'
        md.append(f"- View {cam_idx}: {png_path}")

    with open(out_dir / 'mask_alignment_report.md', 'w') as f:
        f.write('\n'.join(md))

    print(f"[4/4] Done. Visualizations in {out_dir}")

if __name__ == '__main__':
    main()
