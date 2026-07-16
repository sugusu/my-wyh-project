#!/usr/bin/env python3
"""Unified recovery evaluator: single evaluator for all recovery experiments.
Must produce exactly 22.39 for scene_01 baseline PLY.
Supports optional AppModel for full-bundle evaluation."""
import argparse, json, os, sys, numpy as np, yaml, torch
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.tsgs_loader import load_scene

def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

def load_app_model(model_dir, iteration=15000, device='cuda:0'):
    from scene.app_model import AppModel
    weights_path = os.path.join(model_dir, "app_model", f"iteration_{iteration}", "app.pth")
    if not os.path.exists(weights_path):
        return None
    app_model = AppModel()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    app_model.load_state_dict(state_dict)
    app_model.cuda()
    app_model.eval()
    return app_model

def tsgs_render(gaussians, cam, pipe, bg_color, device='cuda:0', app_model=None):
    from gaussian_renderer import render
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    return render(cam, gaussians, pipe, bg, app_model=app_model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-config', type=str, required=True)
    parser.add_argument('--ply-path', type=str, required=True, help='PLY file to evaluate')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--tag', type=str, default='eval', help='Tag for reporting')
    parser.add_argument('--use-test-cameras', action='store_true',
                        help='Use getTestCameras instead of default getTrainCameras')
    parser.add_argument('--white-bg', action='store_true', default=False,
                        help='Use white background [1,1,1] instead of black [0,0,0]')
    parser.add_argument('--num-views', type=int, default=64, help='Number of views to evaluate')
    parser.add_argument('--app-model-dir', type=str, default=None,
                        help='Directory containing app_model/iteration_N/app.pth')
    parser.add_argument('--app-model-iter', type=int, default=15000,
                        help='AppModel iteration to load')
    parser.add_argument('--enable-app-model', action='store_true', default=False,
                        help='Enable AppModel during rendering (requires gaussians.use_app=True)')
    args = parser.parse_args()

    with open(args.scene_config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda:0'
    bg_color = [1.0, 1.0, 1.0] if args.white_bg else [0.0, 0.0, 0.0]

    with torch.no_grad():
        scene, gaussians, pipe = load_scene(cfg, device)
        ply_path = args.ply_path
        gaussians.load_ply(str(ply_path))
        gaussians.active_sh_degree = 0

        app_model = None
        if args.app_model_dir and os.path.isdir(os.path.join(args.app_model_dir, "app_model")):
            app_model = load_app_model(args.app_model_dir, args.app_model_iter, device)
            if app_model is not None:
                print(f"  AppModel loaded (appear_ab non-zero: {(app_model.appear_ab != 0).sum().item()})")
        if args.enable_app_model and app_model is not None:
            gaussians.use_app = True
            print("  AppModel enabled for rendering")

        if args.use_test_cameras:
            cameras = scene.getTestCameras()
            if len(cameras) == 0:
                cameras = scene.getTrainCameras()
        else:
            cameras = scene.getTrainCameras()

        total_cams = len(cameras)
        n_eval = min(args.num_views, total_cams)

        psnrs = []
        ssims = []
        lpipss = []
        from skimage.metrics import structural_similarity
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)

        for ci in range(n_eval):
            cam = cameras[ci]
            rendered = tsgs_render(gaussians, cam, pipe, bg_color, device, app_model)
            if 'app_image' in rendered and args.enable_app_model:
                render_img = rendered['app_image']
            elif 'render' in rendered:
                render_img = rendered['render']
            elif 'rgb' in rendered:
                render_img = rendered['rgb']
            else:
                render_img = list(rendered.values())[0]

            gt_img = getattr(cam, 'original_image', None)
            if gt_img is None:
                gt_img = getattr(cam, 'image', None)
            if gt_img is not None and not torch.is_tensor(gt_img):
                gt_img = torch.tensor(gt_img, device=device)
            if gt_img is not None:
                gt_img = gt_img.to(device)

            if gt_img is not None:
                r = render_img[:3].detach().cpu().numpy().transpose(1, 2, 0)
                g = gt_img[:3].detach().cpu().numpy().transpose(1, 2, 0)
                r = r.clip(0, 1)
                g = g.clip(0, 1)
                psnr = compute_psnr(r, g)
                ssim = structural_similarity(r, g, channel_axis=2, data_range=1.0)
                psnrs.append(psnr)
                ssims.append(ssim)

                r_t = torch.tensor(r).permute(2, 0, 1).unsqueeze(0).to(device)
                g_t = torch.tensor(g).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    lpips_val = lpips_fn(r_t, g_t).item()
                lpipss.append(lpips_val)

        if psnrs:
            metrics = {
                'requested_ply': str(ply_path),
                'actual_loaded_count': gaussians.get_xyz.shape[0],
                'num_test_views': int(n_eval),
                'n_views': int(n_eval),
                'psnr_mean': round(float(np.mean(psnrs)), 8),
                'psnr_std': round(float(np.std(psnrs)), 8),
                'ssim_mean': round(float(np.mean(ssims)), 8),
                'ssim_std': round(float(np.std(ssims)), 8),
                'lpips_mean': round(float(np.mean(lpipss)), 8),
                'lpips_std': round(float(np.std(lpipss)), 8),
                'app_model_used': args.enable_app_model and app_model is not None,
                'app_model_path': os.path.join(args.app_model_dir, "app_model", f"iteration_{args.app_model_iter}", "app.pth") if args.app_model_dir else None,
            }
        else:
            metrics = {'error': 'no valid renderings', 'num_test_views': int(n_eval)}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Unified Recovery Evaluator [{args.tag}] ===")
    print(f"  PLY: {ply_path}")
    print(f"  Gaussians: {metrics.get('actual_loaded_count', 'N/A')}")
    print(f"  Views: {metrics.get('num_test_views', 'N/A')}")
    if 'psnr_mean' in metrics:
        print(f"  PSNR: {metrics['psnr_mean']:.8f} +- {metrics['psnr_std']:.8f}")
        print(f"  SSIM: {metrics['ssim_mean']:.8f} +- {metrics['ssim_std']:.8f}")
        print(f"  LPIPS: {metrics['lpips_mean']:.8f} +- {metrics['lpips_std']:.8f}")
        print(f"  AppModel: {'ENABLED' if metrics.get('app_model_used') else 'DISABLED'}")
    else:
        print(f"  ERROR: {metrics.get('error', 'unknown')}")
    print(f"  Output: {args.output}")

if __name__ == '__main__':
    main()
