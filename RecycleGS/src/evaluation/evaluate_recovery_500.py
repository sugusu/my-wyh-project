#!/usr/bin/env python3
"""Evaluate recovery 500: PSNR, SSIM, LPIPS by loading recovery PLY and rendering test views."""
import argparse, hashlib, json, os, sys, numpy as np, torch, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import PipelineParams, GroupParams
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

SCENE_CONFIGS = {
    'scene_01': '/data/wyh/RecycleGS/configs/stage1/reliability_scene01.yaml',
    'scene_03': '/data/wyh/RecycleGS/configs/stage1/reliability_scene03.yaml',
}
METHODS = ['schedule_control', 'random', 'low_opacity', 'low_contribution', 'mask_risk']
RATIO = 'ratio_005'
RECOVERY_ITER = 15500

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def make_pipe():
    pipe = GroupParams()
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    pipe.debug = False
    return pipe

def make_dataset(args, scene_dir, model_dir):
    d = GroupParams()
    d.source_path = scene_dir
    d.model_path = model_dir
    d.images = "images"
    d.resolution = 2
    d.sh_degree = 3
    d.asg_degree = 24
    d.eval = True
    d.preload_img = True
    d.white_background = False
    d.data_device = "cuda"
    d.delight = False
    d.normal = False
    d.normal_folder = "normals"
    d.mask_background = False
    d.use_delighted_normal = False
    d.use_transparencies_map = False
    d.not_delight_only_transparent = False
    d.load2gpu_on_the_fly = False
    d.is_real = False
    d.is_indoor = False
    d.add_val = False
    d.multi_view_num = 8
    d.multi_view_max_angle = 30
    d.multi_view_min_dis = 0.01
    d.multi_view_max_dis = 1.5
    d.ncc_scale = 1.0
    return d

@torch.no_grad()
def evaluate_ply(ply_path, scene_dir, model_dir, device='cuda:0'):
    dataset = make_dataset(None, scene_dir, model_dir)
    pipe = make_pipe()
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    gaussians.load_ply(str(ply_path))
    gaussians.active_sh_degree = 0
    print(f"  Loaded PLY: {ply_path}")
    print(f"  Gaussian count: {gaussians.get_xyz.shape[0]}")

    scene = Scene(dataset, gaussians, load_iteration=15000, shuffle=False)
    gaussians.load_ply(str(ply_path))
    gaussians.active_sh_degree = 0

    test_cams = scene.getTestCameras()
    if len(test_cams) == 0:
        test_cams = scene.getTrainCameras()
    print(f"  Test views: {len(test_cams)}")

    psnrs, ssims, lpipss = [], [], []
    for cam in test_cams:
        gt = cam.original_image
        if gt is None:
            gt, _, _, _, _ = cam.get_image()
        gt = gt.to(device)

        rendered = render(cam, gaussians, pipe, bg, app_model=None,
                          return_plane=False, return_depth_normal=False)
        image = rendered['render'].clamp(0.0, 1.0)

        psnrs.append(psnr(image, gt).mean().item())
        ssims.append(ssim(image, gt).mean().item())
        lpipss.append(lpips(image, gt, net_type='vgg').mean().item())

    n = len(psnrs)
    result = {
        'psnr': round(float(np.mean(psnrs)), 8),
        'psnr_std': round(float(np.std(psnrs)), 8),
        'ssim': round(float(np.mean(ssims)), 8),
        'ssim_std': round(float(np.std(ssims)), 8),
        'lpips': round(float(np.mean(lpipss)), 8),
        'lpips_std': round(float(np.std(lpipss)), 8),
        'num_views': n,
    }
    print(f"  PSNR: {result['psnr']:.8f} +- {result['psnr_std']:.8f}")
    print(f"  SSIM: {result['ssim']:.8f} +- {result['ssim_std']:.8f}")
    print(f"  LPIPS: {result['lpips']:.8f} +- {result['lpips_std']:.8f}")
    return result, gaussians.get_xyz.shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Scene config YAML (optional, auto-detects all)')
    parser.add_argument('--method', type=str, default=None, help='Method to evaluate (default: all)')
    parser.add_argument('--force-render', action='store_true', help='Re-render even if metrics exist')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    scenes_to_run = list(SCENE_CONFIGS.keys()) if args.config is None else ['scene_01' if 'scene_01' in args.config else 'scene_03']
    methods_to_run = METHODS if args.method is None else [args.method]

    for scene_name in scenes_to_run:
        scene_config_path = SCENE_CONFIGS[scene_name]
        with open(scene_config_path) as f:
            sc = yaml.safe_load(f)

        scene_dir = sc['scene_dir']
        model_dir = sc['model_dir']

        for method in methods_to_run:
            print(f"\n=== {scene_name}/{method} ===")

            # PLY is in outputs/recovery/ directory from training
            recovery_dir = Path(f'/data/wyh/RecycleGS/outputs/recovery/{scene_name}/{method}')
            ply_path = recovery_dir / 'point_cloud' / f'iteration_{RECOVERY_ITER}' / 'point_cloud.ply'

            if not ply_path.exists():
                print(f"  SKIP: PLY not found at {ply_path}")
                continue

            # Save metrics to outputs/prune_only/recovery_500/
            out_dir = Path(f'/data/wyh/RecycleGS/outputs/prune_only/{scene_name}/{RATIO}/{method}/recovery_500')
            out_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = out_dir / 'render_metrics.json'
            if not args.force_render and metrics_path.exists():
                print(f"  SKIP: render_metrics.json exists at {metrics_path} (use --force-render to re-run)")
                continue

            ply_sha256 = sha256_of(str(ply_path))
            metrics, gauss_count = evaluate_ply(str(ply_path), scene_dir, model_dir, args.device)

            metrics['scene'] = scene_name
            metrics['method'] = method
            metrics['iteration'] = RECOVERY_ITER
            metrics['model_ply_path'] = str(ply_path)
            metrics['model_ply_sha256'] = ply_sha256
            metrics['gaussian_count'] = gauss_count

            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"  Saved: {metrics_path}")

if __name__ == '__main__':
    main()
