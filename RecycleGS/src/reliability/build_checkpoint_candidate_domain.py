import argparse, sys, os, torch, numpy as np
from pathlib import Path
from PIL import Image
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
    opacity = np.asarray(props.get('opacity', np.ones(len(xyz)))).reshape(-1, 1)
    return xyz, scales, rots, opacity

def compute_mask_support_checkpoint(xyz, cameras, scene_dir, cfg, device):
    N = len(xyz)
    total = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total)
    step = max(1, total // n_views)
    selected = list(range(0, total, step))[:n_views]

    valid_view_count = np.zeros(N, dtype=np.int32)
    mask_inside_count = np.zeros(N, dtype=np.int32)
    mask_support_sum = np.zeros(N, dtype=np.float32)
    xyz_t = torch.from_numpy(xyz).float().to(device)

    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        w2c = cam.world_view_transform[:3, :3].to(device)
        t = cam.world_view_transform[3, :3].to(device)
        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)

        pts_cam = xyz_t @ w2c.T + t.unsqueeze(0)
        depths = pts_cam[:, 2]
        valid_depth = depths > cfg['projection']['min_positive_depth']

        pts_2d = pts_cam[:, :2] / depths.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid_depth
        in_frame_np = in_frame.cpu().numpy()
        valid_view_count += in_frame_np.astype(np.int32)

        stem = cam.image_name.split('.')[0] if '.' in cam.image_name else cam.image_name
        mask_path = os.path.join(scene_dir, 'transparent_masks', f'{stem}.png')
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert('L')
            mask_resized = mask_img.resize((cam.image_width, cam.image_height), Image.NEAREST)
            mask_arr = torch.from_numpy(np.array(mask_resized)).float().to(device) / 255.0
            valid_idx = in_frame.nonzero(as_tuple=True)[0].cpu().numpy()
            if len(valid_idx) > 0:
                sub_v = pts_px[valid_idx, 1].long().clamp(0, mask_arr.shape[0]-1).cpu().numpy()
                sub_u = pts_px[valid_idx, 0].long().clamp(0, mask_arr.shape[1]-1).cpu().numpy()
                mask_vals = mask_arr[sub_v, sub_u].cpu().numpy()
                mask_inside_count[valid_idx] += 1
                mask_support_sum[valid_idx] += mask_vals

        if vi % 16 == 0:
            print(f"    view [{vi+1}/{len(selected)}]")

    vis_count = valid_view_count.clip(min=1).astype(np.float32)
    mask_support_unweighted = mask_support_sum / vis_count
    return valid_view_count, mask_inside_count, mask_support_unweighted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1f_scene01'
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(out_dir / 'iter_7000', exist_ok=True)
    os.makedirs(out_dir / 'iter_15000', exist_ok=True)
    device = cfg['device']
    scene_dir = cfg['scene_dir']

    model_dir = Path(cfg['model_dir'])
    ckpt_paths = {
        7000: model_dir / 'point_cloud' / 'iteration_7000' / 'point_cloud.ply',
        15000: Path(cfg['checkpoint_path']),
    }

    print("[1/4] Loading scene for cameras (shared)...")
    from recyclegs.tsgs_loader import load_scene, get_train_cameras
    scene_15k, _, _ = load_scene(cfg, device)
    cameras = get_train_cameras(scene_15k)
    print(f"  {len(cameras)} cameras available")

    results = {}
    for iteration in [7000, 15000]:
        ckpt_path = ckpt_paths[iteration]
        iter_str = f"iter_{iteration}"
        print(f"\n[2/4] Processing {iter_str}: {ckpt_path}")

        if not ckpt_path.exists():
            print(f"  SKIP: {ckpt_path} not found")
            results[iteration] = {'error': 'checkpoint not found'}
            continue

        xyz, scales, rots, opacity = load_ply_raw(ckpt_path)
        N = len(xyz)
        print(f"  {N} Gaussians loaded")

        print(f"  Computing mask support over {len(cameras)} cameras...")
        valid_view_count, mask_inside_count, mask_support_unweighted = compute_mask_support_checkpoint(
            xyz, cameras, scene_dir, cfg, device
        )

        has_min_views = valid_view_count >= 3
        strong_background = has_min_views & (mask_support_unweighted <= 0.05)
        candidate = has_min_views & (mask_support_unweighted >= 0.20) & ~strong_background
        candidate_indices = np.where(candidate)[0]
        print(f"  Candidate: {len(candidate_indices)}/{N} "
              f"({len(candidate_indices)/N*100:.2f}%)")
        print(f"  Strong BG: {strong_background.sum()}/{N} "
              f"({strong_background.sum()/N*100:.2f}%)")
        print(f"  No min views: {(~has_min_views).sum()}/{N} "
              f"({(~has_min_views).sum()/N*100:.2f}%)")

        np.save(out_dir / iter_str / 'candidate_indices.npy', candidate_indices)
        np.save(out_dir / iter_str / 'valid_view_count.npy', valid_view_count)
        np.save(out_dir / iter_str / 'mask_inside_count.npy', mask_inside_count)
        np.save(out_dir / iter_str / 'mask_support_unweighted.npy', mask_support_unweighted)

        results[iteration] = {
            'total_gaussians': int(N),
            'candidate_count': int(len(candidate_indices)),
            'candidate_ratio': float(len(candidate_indices) / N),
            'strong_background_count': int(strong_background.sum()),
            'no_min_views_count': int((~has_min_views).sum()),
            'mask_support_mean': float(mask_support_unweighted.mean()),
        }
        print(f"  Saved to {out_dir / iter_str / 'candidate_indices.npy'}")

    import json
    json_path = debug_dir / 'checkpoint_candidate_domain_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[3/4] Results saved to {json_path}")
    print(f"[4/4] Done")

if __name__ == '__main__':
    main()
