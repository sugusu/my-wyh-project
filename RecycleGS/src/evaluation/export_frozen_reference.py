import argparse, sys, os, torch, json, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config, save_json
from recyclegs.tsgs_loader import load_scene, get_train_cameras, render_view
from arguments import PipelineParams, ModelParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    ref_dir = Path(cfg['reference_output_dir'])
    (ref_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (ref_dir / 'depth_ref').mkdir(parents=True, exist_ok=True)
    (ref_dir / 'normal_render').mkdir(parents=True, exist_ok=True)
    (ref_dir / 'alpha').mkdir(parents=True, exist_ok=True)
    (ref_dir / 'mask').mkdir(parents=True, exist_ok=True)
    (ref_dir / 'normal_prior').mkdir(parents=True, exist_ok=True)
    (ref_dir / 'metadata').mkdir(parents=True, exist_ok=True)

    device = cfg['device']
    print("[1/3] Loading scene...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total)
    step = max(1, total // n_views)
    selected_indices = list(range(0, total, step))[:n_views]
    bg = torch.tensor([1,1,1], dtype=torch.float32, device=device)

    print(f"[2/3] Exporting {len(selected_indices)} reference views...")
    for idx, cam_idx in enumerate(selected_indices):
        cam = cameras[cam_idx]
        rendered = render_view(gaussians, cam, pipe, [1,1,1], device)
        rgb = rendered['render']
        rgb_np = (rgb.detach().cpu().numpy().transpose(1,2,0).clip(0,1)*255).astype(np.uint8)
        Image.fromarray(rgb_np).save(ref_dir / 'rgb' / f'view_{cam_idx:04d}.png')

        depth = rendered.get('depth', rendered.get('plane_depth', None))
        if depth is not None:
            depth_np = depth.detach().cpu().numpy().squeeze()
            np.save(ref_dir / 'depth_ref' / f'view_{cam_idx:04d}.npy', depth_np)
            d_norm = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(d_norm).save(ref_dir / 'depth_ref' / f'view_{cam_idx:04d}.png')

        alpha = rendered.get('rendered_alpha', rendered.get('alpha', None))
        if alpha is not None:
            a_np = alpha.detach().cpu().numpy().squeeze()
            np.save(ref_dir / 'alpha' / f'view_{cam_idx:04d}.npy', a_np)
            Image.fromarray((a_np.clip(0,1)*255).astype(np.uint8)).save(ref_dir / 'alpha' / f'view_{cam_idx:04d}.png')

        w2c = cam.world_view_transform.detach().cpu().numpy()
        proj = cam.full_proj_transform.detach().cpu().numpy() if hasattr(cam, 'full_proj_transform') else np.eye(4)
        meta = {
            'view_index': idx, 'camera_index': cam_idx,
            'image_name': getattr(cam, 'image_name', f'frame_{cam_idx+1:04d}.png'),
            'width': cam.image_width, 'height': cam.image_height,
            'fx': float(cam.Fx) if hasattr(cam, 'fx') else 0.0,
            'fy': float(cam.Fy) if hasattr(cam, 'fy') else 0.0,
            'cx': float(cam.Cx) if hasattr(cam, 'cx') else 0.0,
            'cy': float(cam.Cy) if hasattr(cam, 'cy') else 0.0,
            'world_view_transform': w2c.tolist(),
            'full_proj_transform': proj.tolist(),
            'camera_center': cam.camera_center.detach().cpu().tolist() if hasattr(cam, 'camera_center') else [],
        }
        save_json(meta, ref_dir / 'metadata' / f'view_{cam_idx:04d}.json')

        if idx % 16 == 0:
            print(f"  [{idx+1}/{len(selected_indices)}]")

    save_json({'n_views': len(selected_indices), 'selected_indices': selected_indices},
              ref_dir / 'reference_export_report.json')
    save_json({'selected_indices': selected_indices}, ref_dir / 'selected_views.json')
    print(f"[3/3] Done. Exported to {ref_dir}")

if __name__ == '__main__':
    main()
