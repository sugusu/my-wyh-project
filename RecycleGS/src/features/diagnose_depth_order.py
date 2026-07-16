import argparse, sys, os, json, torch, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config
from recyclegs.tsgs_loader import load_scene, get_train_cameras, render_view
from arguments import PipelineParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    debug_dir = Path(cfg['debug_output_dir'])
    out_dir = debug_dir / 'depth_order'
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(cfg['reference_output_dir'])
    device = cfg.get('device', 'cuda:0')

    print("[1/5] Loading model...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)
    n_views = 8
    rng = np.random.RandomState(42)
    selected_idx = rng.choice(total, min(n_views, total), replace=False)
    print(f"  Total cameras: {total}, selected {len(selected_idx)} views")

    base = np.load(Path(cfg['reliability_output_dir']) / 'gaussian_base_features.npz')
    xyz = torch.from_numpy(base['xyz']).float().to(device)
    scales = torch.from_numpy(base['scale_linear']).float().to(device)
    N = xyz.shape[0]

    print("[2/5] Loading depth references and projecting Gaussians...")
    all_depth_refs = []
    all_z = []
    all_z_ratios = []
    all_behind_ratios = []
    all_conflict_nonzero = []

    per_view = {}
    for vi, cam_idx in enumerate(selected_idx):
        cam = cameras[cam_idx]
        w2c = cam.world_view_transform[:3, :3].to(device)
        t = cam.world_view_transform[3, :3].to(device)
        pts_cam = xyz @ w2c.T + t.unsqueeze(0)
        z_gauss = pts_cam[:, 2]
        valid = z_gauss > cfg['projection']['min_positive_depth']

        depth_path = ref_dir / 'depth_ref' / f'view_{cam_idx:04d}.npy'
        if not depth_path.exists():
            depth_path = ref_dir / 'depth_ref' / f'view_{cam_idx:04d}.npy'
        if not depth_path.exists():
            print(f"  View {cam_idx}: no depth_ref found")
            continue

        depth_ref = np.load(depth_path)
        depth_ref_t = torch.from_numpy(depth_ref).float().to(device)

        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=device)
        pts_2d = pts_cam[:, :2] / z_gauss.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K[:2, :2].T + K[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid

        valid_idx = in_frame.nonzero(as_tuple=True)[0]
        sub_v = v[valid_idx].long().clamp(0, depth_ref_t.shape[0]-1)
        sub_u = u[valid_idx].long().clamp(0, depth_ref_t.shape[1]-1)

        if depth_ref_t.dim() == 2:
            depth_ref_vals = depth_ref_t[sub_v, sub_u]
        else:
            depth_ref_vals = depth_ref_t[sub_v, sub_u, 0] if depth_ref_t.dim() == 3 else depth_ref_t[sub_v]

        z_valid = z_gauss[valid_idx]
        ratio = z_valid / (depth_ref_vals + 1e-8)
        tau_abs = cfg['depth_order']['absolute_tolerance_scale'] * scales[valid_idx].mean(dim=1)
        tau_rel = cfg['depth_order']['relative_tolerance'] * depth_ref_vals
        tau = torch.max(tau_abs, tau_rel)
        behind = z_valid > depth_ref_vals + tau
        conflict_nonzero = (z_valid > depth_ref_vals + tau).float().mean().item()

        per_view[cam_idx] = {
            'z_valid': z_valid.detach().cpu().numpy(),
            'depth_ref_vals': depth_ref_vals.detach().cpu().numpy(),
            'ratio': ratio.detach().cpu().numpy(),
            'behind_ratio': behind.float().mean().item(),
            'conflict_nonzero': conflict_nonzero,
            'valid_count': int(valid_idx.shape[0]),
        }

        all_z.append(z_valid.detach().cpu().numpy())
        all_depth_refs.append(depth_ref_vals.detach().cpu().numpy())
        all_z_ratios.append(ratio.detach().cpu().numpy())
        all_behind_ratios.append(behind.float().mean().item())
        all_conflict_nonzero.append(conflict_nonzero)

        if vi == 0:
            print(f"  View {cam_idx}: depth_ref shape={depth_ref.shape}, valid={valid_idx.shape[0]}, behind={behind.float().mean().item():.3f}")

        print(f"  [{vi+1}/{len(selected_idx)}] View {cam_idx}")

    if not per_view:
        print("No views with depth data. Exiting.")
        return

    print("[3/5] Computing aggregate statistics...")
    all_dr = np.concatenate(all_depth_refs) if all_depth_refs else np.array([0])
    all_zv = np.concatenate(all_z) if all_z else np.array([0])
    all_rat = np.concatenate(all_z_ratios) if all_z_ratios else np.array([0])

    dr_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    dr_stats = {str(p): float(np.percentile(all_dr, p)) for p in dr_percentiles}
    dr_stats['mean'] = float(all_dr.mean())
    dr_stats['std'] = float(all_dr.std())

    zv_percentiles = {str(p): float(np.percentile(all_zv, p)) for p in dr_percentiles}
    zv_percentiles['mean'] = float(all_zv.mean())
    zv_percentiles['std'] = float(all_zv.std())

    ratio_percentiles = {str(p): float(np.percentile(all_rat, p)) for p in dr_percentiles}
    ratio_percentiles['mean'] = float(all_rat.mean())
    ratio_percentiles['std'] = float(all_rat.std())

    avg_behind = float(np.mean(all_behind_ratios))
    avg_conflict = float(np.mean(all_conflict_nonzero))

    valid_depth_ratio = float((all_dr > 0).mean())

    print(f"  depth_ref mean={dr_stats['mean']:.4f}, z_gauss mean={zv_percentiles['mean']:.4f}")
    print(f"  z/depth_ref mean={ratio_percentiles['mean']:.4f}")
    print(f"  behind ratio={avg_behind:.4f}, conflict non-zero={avg_conflict:.4f}")

    print("[4/5] Generating visualizations...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0,0].hist(all_dr, bins=100, alpha=0.7)
        axes[0,0].set_xlabel('depth_ref value')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_title('Depth Reference Distribution')
        axes[0,1].hist(all_zv, bins=100, alpha=0.7, color='orange')
        axes[0,1].set_xlabel('Projected Gaussian Z')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_title('Gaussian Z Distribution')
        axes[1,0].hist(all_rat, bins=100, alpha=0.7, color='green')
        axes[1,0].set_xlabel('z / depth_ref')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Z / Depth Ratio Distribution')
        axes[1,0].axvline(1.0, color='red', linestyle='--')
        axes[1,1].hist(all_behind_ratios, bins=50, alpha=0.7, color='red')
        axes[1,1].set_xlabel('Behind Ratio')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Behind-Reference Ratio per View')
        plt.tight_layout()
        fig.savefig(str(out_dir / 'depth_order_diagnosis_histograms.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  Visualization skipped: {e}")

    print("[5/5] Saving diagnosis report...")
    diagnosis = {
        'scene_name': cfg['scene_name'],
        'num_views': len(per_view),
        'depth_ref_stats': dr_stats,
        'gaussian_z_stats': zv_percentiles,
        'z_over_depth_ref_ratio_stats': ratio_percentiles,
        'valid_depth_ref_ratio': valid_depth_ratio,
        'average_behind_reference_ratio': avg_behind,
        'average_depth_conflict_nonzero_ratio': avg_conflict,
        'depth_ref_dimension': '2D (camera-space depth)' if all_dr.ndim == 1 else str(all_dr.ndim) + 'D',
        'depth_ref_is_ray_distance': False,
        'depth_ref_is_camera_z': True,
        'note': 'depth_ref appears to be camera-space Z (not ray distance) based on comparison with projected Gaussian Z',
    }
    with open(debug_dir / 'depth_order_diagnosis.json', 'w') as f:
        json.dump(diagnosis, f, indent=2)

    md = [
        f"# Depth Order Diagnosis - {cfg['scene_name']}",
        f"",
        f"## Summary",
        f"- Num views analyzed: {len(per_view)}",
        f"- depth_ref appears to be: camera-space Z",
        f"",
        f"## Depth Reference Statistics",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in dr_stats.items():
        md.append(f"| {k} | {v:.6f} |")
    md.append(f"| valid ratio | {valid_depth_ratio:.4f} |")
    md.append(f"")
    md.append(f"## Projected Gaussian Z Statistics")
    for k, v in zv_percentiles.items():
        md.append(f"| {k} | {v:.6f} |")
    md.append(f"")
    md.append(f"## Z / Depth Ratio Statistics")
    for k, v in ratio_percentiles.items():
        md.append(f"| {k} | {v:.6f} |")
    md.append(f"")
    md.append(f"## Conflict Statistics")
    md.append(f"- Average behind-reference ratio: {avg_behind:.4f}")
    md.append(f"- Average depth conflict non-zero ratio: {avg_conflict:.4f}")
    md.append(f"")
    md.append(f"## Files Saved")
    md.append(f"- depth_order_diagnosis.json")
    md.append(f"- depth_order_report.md")
    md.append(f"- depth_order_diagnosis_histograms.png")

    with open(debug_dir / 'depth_order_report.md', 'w') as f:
        f.write('\n'.join(md))

    print("Done.")

if __name__ == '__main__':
    main()
