#!/usr/bin/env python3
"""Train recovery: continue training from checkpoint after pruning for K steps."""
import argparse, json, os, sys, numpy as np, torch, yaml
from pathlib import Path
from random import randint
from tqdm import tqdm

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene import Scene, GaussianModel, SpecularModel
from scene.app_model import AppModel
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import OptimizationParams, ModelParams, PipelineParams
from argparse import ArgumentParser as ArgParser
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-config', type=str, required=True)
    parser.add_argument('--recovery-config', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--removed-indices', type=str, default=None)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--override-steps', type=int, default=None)
    parser.add_argument('--no-delight', action='store_true', default=False,
                        help='Disable delight (use original images, no AppModel)')
    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.scene_config) as f:
        scene_cfg = yaml.safe_load(f)
    with open(args.recovery_config) as f:
        recovery_cfg = yaml.safe_load(f)

    scene_name = scene_cfg.get('scene_name', 'scene_01')
    model_dir = scene_cfg['model_dir']
    scene_dir = scene_cfg['scene_dir']
    ckpt_path = os.path.join(model_dir, 'chkpnt15000.pth')
    app_ckpt_path = os.path.join(model_dir, 'app_model/iteration_15000/app.pth')

    device = torch.device('cuda:0')

    start_iter = recovery_cfg.get('start_iteration', 15001)
    end_iter = recovery_cfg.get('end_iteration', 15500)
    if args.override_steps is not None:
        end_iter = start_iter + args.override_steps - 1
    log_interval = recovery_cfg.get('log_interval', 25)

    # 1. Load training checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_params, iteration = ckpt
    print(f"  Loaded iteration {iteration}")

    # 2. Initialize models
    opt_parser = argparse.ArgumentParser()
    opt = OptimizationParams(opt_parser)
    opt_args = {
        'iterations': 30000,
        'position_lr_init': 0.00016,
        'position_lr_final': 0.0000016,
        'position_lr_delay_mult': 0.01,
        'position_lr_max_steps': 30000,
        'feature_lr': 0.0025,
        'opacity_lr': 0.05,
        'scaling_lr': 0.005,
        'rotation_lr': 0.001,
        'percent_dense': 0.001,
        'lambda_dssim': 0.2,
        'densification_interval': 100,
        'opacity_reset_interval': 3000,
        'densify_from_iter': 500,
        'densify_until_iter': 15000,
        'densify_grad_threshold': 0.0002,
        'scale_loss_weight': 100.0,
        'opacity_cull_threshold': 0.005,
        'densify_abs_grad_threshold': 0.0008,
        'abs_split_radii2D_threshold': 20,
        'max_abs_split_points': 50000,
        'max_all_points': 6000000,
        'random_background': False,
        'exposure_compensation': False,
        'wo_depth_normal_detach': False,
        'use_2dgsnormal_loss': False,
        'use_asg': False,
        'delight_iterations': 15000,
        'sd_normal_until_iter': 30000,
        'lambda_sd_normal': 0.05,
        'normal_cos_threshold_iter': 3000,
        'ncc_loss_from_iter': 7000,
        'single_view_weight': 0.015,
        'single_view_weight_from_iter': 7000,
        'multi_view_ncc_weight': 0.15,
        'multi_view_geo_weight': 0.03,
        'multi_view_weight_from_iter': 7000,
        'multi_view_patch_size': 3,
        'multi_view_sample_num': 102400,
        'multi_view_pixel_noise_th': 1.0,
        'wo_use_geo_occ_aware': False,
        'use_multi_view_trim': True,
        'T_threshold': 0.0001,
        'observe_T_threshold': 0.5,
        'bg_T_threshold': 0.98,
        'trans_binary_threshold': 0.5,
        'nofix_position': False,
        'nofix_opacity': False,
        'nofix_param': False,
        'nofix_scaling': False,
        'nofix_rotation': False,
    }
    for k, v in opt_args.items():
        setattr(opt, k, v)

    dataset_parser = argparse.ArgumentParser()
    dataset = ModelParams(dataset_parser)
    use_delight = not args.no_delight
    dataset_args = {
        'sh_degree': 3,
        'asg_degree': 24,
        'source_path': scene_dir,
        'model_path': model_dir,
        'images': 'images',
        'resolution': 2,
        'white_background': False,
        'data_device': 'cuda',
        'eval': False,
        'preload_img': True,
        'ncc_scale': 1.0,
        'multi_view_num': 8,
        'multi_view_max_angle': 30,
        'multi_view_min_dis': 0.01,
        'multi_view_max_dis': 1.5,
        'delight': use_delight,
        'normal': True,
        'normal_folder': 'normals',
        'mask_background': True,
        'use_delighted_normal': False,
        'use_transparencies_map': True,
        'not_delight_only_transparent': False,
        'load2gpu_on_the_fly': False,
        'is_real': False,
        'is_indoor': False,
        'add_val': False,
    }
    for k, v in dataset_args.items():
        setattr(dataset, k, v)

    gaussians = GaussianModel(dataset.sh_degree, None)  # original training used use_asg=False
    gaussians.restore(model_params, opt)

    # 3. Optionally prune
    if args.method != 'schedule_control' and args.removed_indices is not None:
        removed_idx = np.load(args.removed_indices)
        K = len(removed_idx)
        N_before = gaussians.get_xyz.shape[0]
        print(f"Pruning {K} Gaussians (method={args.method})")
        prune_mask = torch.zeros(N_before, dtype=bool, device=device)
        prune_mask[torch.tensor(removed_idx, device=device)] = True
        gaussians.prune_points(prune_mask)
        N_after = gaussians.get_xyz.shape[0]
        print(f"  After prune: {N_after} (expected {N_before - K})")
    else:
        print("Schedule control: no pruning")

    N_fixed = gaussians.get_xyz.shape[0]
    print(f"Fixed Gaussian count: {N_fixed}")

    # 4. Create scene (loads cameras, will overwrite gaussians with PLY)
    scene = Scene(dataset, gaussians, load_iteration=15000, shuffle=False)
    # Re-apply our restored+pruned state since Scene.load_ply overwrites gaussians
    gaussians.restore(model_params, opt)
    if args.method != 'schedule_control' and args.removed_indices is not None:
        removed_idx = np.load(args.removed_indices)
        prune_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=bool, device=device)
        prune_mask[torch.tensor(removed_idx, device=device)] = True
        gaussians.prune_points(prune_mask)

    app_model = None
    if use_delight:
        app_model = AppModel()
        if os.path.exists(app_ckpt_path):
            app_model.load_weights(model_dir)
        app_model.train()
        app_model.cuda()
        # Enable use_app so AppModel affects rendering (matches train.py behavior
        # when exposure_compensation is enabled at iteration > 1000)
        gaussians.use_app = True
        print("  AppModel loaded, gaussians.use_app set to True")

    # 5. Disable densification
    opt.densify_from_iter = 999999
    opt.densify_until_iter = 0

    # 6. Training loop
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    # Fixed camera order for reproducibility
    train_cams = scene.getTrainCameras()
    cam_indices = list(range(len(train_cams)))
    fixed_rng = np.random.RandomState(args.seed)
    fixed_rng.shuffle(cam_indices)
    cam_queue = [train_cams[i] for i in cam_indices]

    pipe_parser = argparse.ArgumentParser()
    pipe = PipelineParams(pipe_parser)
    pipe.debug = False
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False

    ema_loss = 0.0
    training_log = []
    gaussian_count_trace = []

    progress_bar = tqdm(range(start_iter, end_iter + 1), desc=f"Recovery ({args.method})")
    for iteration in progress_bar:
        # Stage 2B-Y fix: call selective_learning_rate_control like official TSGS train.py
        gaussians.selective_learning_rate_control(
            iteration, 15000,
            nofix_position=opt.nofix_position,
            nofix_opacity=opt.nofix_opacity,
            nofix_param=opt.nofix_param,
            nofix_scaling=opt.nofix_scaling,
            nofix_rotation=opt.nofix_rotation,
        )

        # Fixed camera order
        cam_idx = (iteration - start_iter) % len(cam_queue)
        viewpoint_cam = cam_queue[cam_idx]

        gt_image, gt_image_gray, gt_image_delight, gt_image_normal, transparencies_map = viewpoint_cam.get_image()
        if viewpoint_cam.mask is not None:
            gt_alpha = viewpoint_cam.mask.cuda()

        # Render
        if opt.use_asg:
            dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(
                gaussians.get_xyz.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            normal = gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
            mlp_color = 0  # No specular during simple recovery
        else:
            mlp_color = None

        bg = torch.rand((3), device='cuda') if opt.random_background else background
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
            return_plane=True, return_depth_normal=True,
            wo_depth_normal_detach=opt.wo_depth_normal_detach,
            mlp_color=mlp_color,
            T_threshold=opt.T_threshold,
            observe_T_threshold=opt.observe_T_threshold,
        )
        image = render_pkg['render']

        # Loss
        Ll1 = l1_loss(image, gt_image)
        ssim_loss_val = (1.0 - ssim(image, gt_image))
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss_val
        loss = image_loss.clone()

        # Simple normal loss (single-view)
        if iteration > opt.single_view_weight_from_iter:
            normal = render_pkg['rendered_normal']
            depth_normal = render_pkg['depth_normal']
            normal_loss = opt.single_view_weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += normal_loss

        # Alpha/mask loss
        if viewpoint_cam.mask is not None:
            alpha_loss = 0.1 * torch.nn.functional.binary_cross_entropy(
                render_pkg['rendered_alpha'], gt_alpha)
            loss += alpha_loss

        loss.backward()

        # Logging
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss if ema_loss > 0 else loss.item()
        N_curr = gaussians.get_xyz.shape[0]
        gaussian_count_trace.append({'iteration': iteration, 'N': int(N_curr)})

        if iteration % log_interval == 0 or iteration == start_iter:
            with torch.no_grad():
                log_entry = {
                    'iteration': iteration,
                    'loss': round(loss.item(), 6),
                    'ema_loss': round(ema_loss, 6),
                    'N_gaussians': int(N_curr),
                    'count_stable': N_curr == N_fixed,
                }
                training_log.append(log_entry)
                progress_bar.set_postfix({
                    'loss': f'{ema_loss:.4f}',
                    'N': str(N_curr),
                })

        # Optimizer step
        gaussians.optimizer.step()
        if app_model is not None:
            app_model.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        if app_model is not None:
            app_model.optimizer.zero_grad(set_to_none=True)

    progress_bar.close()

    # 7. Save outputs
    # Save checkpoint to output dir
    torch.save(
        (gaussians.capture(), end_iter),
        os.path.join(args.output_dir, 'chkpnt_recovery.pth')
    )
    # Save PLY to output dir
    out_ply = os.path.join(args.output_dir, 'point_cloud', f'iteration_{end_iter}', 'point_cloud.ply')
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    gaussians.save_ply(out_ply)
    # Save AppModel weights if app_model is active
    if app_model is not None:
        app_model.save_weights(args.output_dir, end_iter)
        print(f"  AppModel weights saved to {args.output_dir}/app_model/iteration_{end_iter}/")
    # Save training log
    with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)
    # Save gaussian count trace
    import csv
    trace_path = os.path.join(args.output_dir, 'gaussian_count_trace.csv')
    with open(trace_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['iteration', 'N'])
        w.writeheader()
        w.writerows(gaussian_count_trace)

    print(f"\nRecovery training complete for {args.method}")
    print(f"  Final loss: {ema_loss:.6f}")
    print(f"  Final Gaussian count: {N_curr} (stable: {N_curr == N_fixed})")
    print(f"  Output: {args.output_dir}")

if __name__ == '__main__':
    main()
