#!/usr/bin/env python3
"""Run 20 steps of recovery training for schedule_control (no pruning).
After each step: render a test view, compute PSNR.
Track: PSNR per step, loss per step, gradient norms, parameter changes."""
import argparse, csv, json, os, sys, torch, numpy as np, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene import Scene, GaussianModel
from scene.app_model import AppModel
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import OptimizationParams, ModelParams, PipelineParams
from argparse import ArgumentParser as ArgParser
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

SCENE_CONFIG = '/data/wyh/RecycleGS/configs/stage1/reliability_scene01.yaml'
CKPT_PATH = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full/chkpnt15000.pth'
PLY_PATH = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full/point_cloud/iteration_15000/point_cloud.ply'
MODEL_DIR = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full'
OUT_DIR = Path('/data/wyh/RecycleGS/outputs/debug/stage2b_recovery_collapse/scene_01')

parser = argparse.ArgumentParser()
parser.add_argument('--use-official-stage2-policy', action='store_true',
                    help='Call gaussians.selective_learning_rate_control at each step (like TSGS train.py)')
parser.add_argument('--unified-evaluator', action='store_true',
                    help='Use same render config for train loop and eval (both SH=0, eval=True, delight=False)')
cli_args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def make_opt():
    opt_parser = ArgParser()
    opt = OptimizationParams(opt_parser)
    for k, v in {
        'iterations': 30000, 'position_lr_init': 0.00016, 'position_lr_final': 0.0000016,
        'position_lr_delay_mult': 0.01, 'position_lr_max_steps': 30000, 'feature_lr': 0.0025,
        'opacity_lr': 0.05, 'scaling_lr': 0.005, 'rotation_lr': 0.001, 'percent_dense': 0.001,
        'lambda_dssim': 0.2, 'densification_interval': 100, 'opacity_reset_interval': 3000,
        'densify_from_iter': 999999, 'densify_until_iter': 0, 'densify_grad_threshold': 0.0002,
        'scale_loss_weight': 100.0, 'opacity_cull_threshold': 0.005, 'densify_abs_grad_threshold': 0.0008,
        'abs_split_radii2D_threshold': 20, 'max_abs_split_points': 50000, 'max_all_points': 6000000,
        'random_background': False, 'exposure_compensation': False, 'wo_depth_normal_detach': False,
        'use_2dgsnormal_loss': False, 'use_asg': False, 'delight_iterations': 15000,
        'sd_normal_until_iter': 30000, 'lambda_sd_normal': 0.05, 'normal_cos_threshold_iter': 3000,
        'ncc_loss_from_iter': 7000, 'single_view_weight': 0.015, 'single_view_weight_from_iter': 7000,
        'multi_view_ncc_weight': 0.15, 'multi_view_geo_weight': 0.03, 'multi_view_weight_from_iter': 7000,
        'multi_view_patch_size': 3, 'multi_view_sample_num': 102400, 'multi_view_pixel_noise_th': 1.0,
        'wo_use_geo_occ_aware': False, 'use_multi_view_trim': True, 'T_threshold': 0.0001,
        'observe_T_threshold': 0.5, 'bg_T_threshold': 0.98, 'trans_binary_threshold': 0.5,
        'nofix_position': False, 'nofix_opacity': False, 'nofix_param': False,
        'nofix_scaling': False, 'nofix_rotation': False,
    }.items():
        setattr(opt, k, v)
    return opt

def make_dataset(scene_dir):
    dataset_parser = ArgParser()
    dataset = ModelParams(dataset_parser)
    for k, v in {
        'sh_degree': 3, 'asg_degree': 24, 'source_path': scene_dir,
        'model_path': MODEL_DIR, 'images': 'images', 'resolution': 2,
        'white_background': False, 'data_device': 'cuda', 'eval': False,
        'preload_img': True, 'ncc_scale': 1.0, 'multi_view_num': 8,
        'multi_view_max_angle': 30, 'multi_view_min_dis': 0.01,
        'multi_view_max_dis': 1.5, 'delight': True, 'normal': True,
        'normal_folder': 'normals', 'mask_background': True,
        'use_delighted_normal': False, 'use_transparencies_map': True,
        'not_delight_only_transparent': False, 'load2gpu_on_the_fly': False,
        'is_real': False, 'is_indoor': False, 'add_val': False,
    }.items():
        setattr(dataset, k, v)
    return dataset

def get_param_snapshot(g):
    return {
        'xyz': g._xyz.detach().clone(),
        'features_dc': g._features_dc.detach().clone(),
        'features_rest': g._features_rest.detach().clone(),
        'opacity': g._opacity.detach().clone(),
        'scaling': g._scaling.detach().clone(),
        'rotation': g._rotation.detach().clone(),
    }

def compute_grad_norms(g):
    norms = {}
    for pg in g.optimizer.param_groups:
        p = pg['params'][0]
        if p.grad is not None:
            norms[pg['name']] = {
                'grad_norm': round(float(p.grad.norm().item()), 12),
                'grad_mean': round(float(p.grad.mean().item()), 12),
                'grad_max': round(float(p.grad.abs().max().item()), 12),
            }
        else:
            norms[pg['name']] = None
    return norms

def param_change(prev, curr):
    changes = {}
    for k in prev:
        diff = (curr[k] - prev[k]).abs()
        changes[k] = {
            'max_abs_change': round(float(diff.max().item()), 12),
            'mean_abs_change': round(float(diff.mean().item()), 12),
            'norm_change': round(float(diff.norm().item()), 12),
        }
    return changes

@torch.no_grad()
def eval_psnr(gaussians, cam, pipe, bg):
    gt = cam.original_image
    if gt is None:
        gt, _, _, _, _ = cam.get_image()
    gt = gt.cuda()
    rendered = render(cam, gaussians, pipe, bg, app_model=None,
                      return_plane=False, return_depth_normal=False)
    image = rendered['render'].clamp(0.0, 1.0)
    return float(psnr(image, gt).mean().item())

def main():
    setup_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0'

    with open(SCENE_CONFIG) as f:
        scene_cfg = yaml.safe_load(f)
    scene_dir = scene_cfg['scene_dir']

    # 1. Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model_params, iteration = ckpt
    print(f"  Iteration {iteration}")

    # 2. Initialize and restore
    opt = make_opt()
    dataset = make_dataset(scene_dir)
    gaussians = GaussianModel(dataset.sh_degree, None)
    gaussians.restore(model_params, opt)

    # 3. Create scene and re-restore (matching train_pruned_recovery.py flow)
    scene = Scene(dataset, gaussians, load_iteration=15000, shuffle=False)
    gaussians.restore(model_params, opt)

    # Unified evaluator: force SH=0 for both training and evaluation
    if cli_args.unified_evaluator:
        print("  Unified evaluator mode: setting active_sh_degree=0, disabling delight, using eval=True")
        gaussians.active_sh_degree = 0
        dataset.eval = True
        dataset.delight = False

    # 4. AppModel
    app_model = AppModel()
    app_ckpt_path = os.path.join(MODEL_DIR, 'app_model/iteration_15000/app.pth')
    if os.path.exists(app_ckpt_path):
        app_model.load_weights(MODEL_DIR)
    app_model.train()
    app_model.cuda()

    # 5. Setup camera queue
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    train_cams = scene.getTrainCameras()
    cam_indices = list(range(len(train_cams)))
    fixed_rng = np.random.RandomState(0)
    fixed_rng.shuffle(cam_indices)
    cam_queue = [train_cams[i] for i in cam_indices]

    pipe_parser = ArgParser()
    pipe = PipelineParams(pipe_parser)
    pipe.debug = False
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False

    # 6. Use a SEPARATE evaluation scene with eval=True, delight=False (matching evaluator)
    from arguments import GroupParams
    eval_ds = GroupParams()
    eval_ds.source_path = scene_dir; eval_ds.model_path = MODEL_DIR; eval_ds.images = "images"
    eval_ds.resolution = 2; eval_ds.sh_degree = 3; eval_ds.asg_degree = 24; eval_ds.eval = True
    eval_ds.preload_img = True; eval_ds.white_background = False; eval_ds.data_device = "cuda"
    eval_ds.delight = False; eval_ds.normal = False; eval_ds.mask_background = False
    eval_ds.use_delighted_normal = False; eval_ds.use_transparencies_map = False
    eval_ds.not_delight_only_transparent = False; eval_ds.load2gpu_on_the_fly = False
    eval_ds.is_real = False; eval_ds.is_indoor = False; eval_ds.add_val = False
    eval_ds.multi_view_num = 8; eval_ds.multi_view_max_angle = 30
    eval_ds.multi_view_min_dis = 0.01; eval_ds.multi_view_max_dis = 1.5
    eval_ds.ncc_scale = 1.0; eval_ds.normal_folder = "normals"

    eval_g = GaussianModel(eval_ds.sh_degree, eval_ds.asg_degree)
    eval_scene = Scene(eval_ds, eval_g, load_iteration=15000, shuffle=False)
    eval_g.load_ply(PLY_PATH)  # restore original PLY
    eval_g.active_sh_degree = 0
    eval_cams = eval_scene.getTestCameras()
    if len(eval_cams) == 0:
        eval_cams = eval_scene.getTrainCameras()
    eval_cam = eval_cams[0]
    print(f"  Using eval camera 0 (delight=False, eval=True, SH=0) for monitoring")

    # 7. Training loop (20 steps)
    start_iter = 15001
    end_iter = 15020

    trace = []
    prev_params = get_param_snapshot(gaussians)

    for iteration in range(start_iter, end_iter + 1):
        # Stage 2B-Y fix: call selective_learning_rate_control like official TSGS train.py
        if cli_args.use_official_stage2_policy:
            gaussians.selective_learning_rate_control(
                iteration, 15000,
                nofix_position=opt.nofix_position,
                nofix_opacity=opt.nofix_opacity,
                nofix_param=opt.nofix_param,
                nofix_scaling=opt.nofix_scaling,
                nofix_rotation=opt.nofix_rotation,
            )

        cam_idx = (iteration - start_iter) % len(cam_queue)
        viewpoint_cam = cam_queue[cam_idx]

        gt_image, gt_image_gray, gt_image_delight, gt_image_normal, transparencies_map = viewpoint_cam.get_image()
        if viewpoint_cam.mask is not None:
            gt_alpha = viewpoint_cam.mask.cuda()

        mlp_color = None  # use_asg=False

        bg = background
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
            return_plane=True, return_depth_normal=True,
            wo_depth_normal_detach=opt.wo_depth_normal_detach,
            mlp_color=mlp_color,
            T_threshold=opt.T_threshold,
            observe_T_threshold=opt.observe_T_threshold,
        )
        image = render_pkg['render']

        Ll1 = l1_loss(image, gt_image)
        ssim_loss_val = (1.0 - ssim(image, gt_image))
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss_val
        loss = image_loss.clone()

        # Alpha/mask loss
        if viewpoint_cam.mask is not None:
            alpha_loss = 0.1 * torch.nn.functional.binary_cross_entropy(
                render_pkg['rendered_alpha'], gt_alpha)
            loss += alpha_loss

        loss.backward()

        # Record gradient norms BEFORE optimizer step
        grad_norms = compute_grad_norms(gaussians)

        gaussians.optimizer.step()
        app_model.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        app_model.optimizer.zero_grad(set_to_none=True)

        # Evaluate PSNR on fixed test view
        current_params = get_param_snapshot(gaussians)
        p_changes = param_change(prev_params, current_params)
        test_psnr = eval_psnr(gaussians, eval_cam, pipe, background)

        entry = {
            'iteration': iteration,
            'loss': round(float(loss.item()), 8),
            'Ll1': round(float(Ll1.item()), 8),
            'test_psnr': round(test_psnr, 4),
            'grad_norms': grad_norms,
            'param_changes': p_changes,
        }
        trace.append(entry)
        print(f"  iter {iteration}: loss={entry['loss']:.6f}, test_psnr={entry['test_psnr']:.2f}")

        prev_params = current_params

    # Save trace
    json_path = OUT_DIR / 'divergence_trace.json'
    with open(json_path, 'w') as f:
        json.dump(trace, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    # CSV
    csv_path = OUT_DIR / 'divergence_trace.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'loss', 'Ll1', 'test_psnr', 'xyz_grad_norm', 'f_dc_grad_norm',
                     'f_rest_grad_norm', 'opacity_grad_norm', 'scaling_grad_norm', 'rotation_grad_norm',
                     'xyz_max_change', 'f_dc_max_change', 'opacity_max_change'])
        for e in trace:
            gn = e.get('grad_norms', {})
            pc = e.get('param_changes', {})
            w.writerow([
                e['iteration'], e['loss'], e['Ll1'], e['test_psnr'],
                gn.get('xyz', {}).get('grad_norm', ''),
                gn.get('f_dc', {}).get('grad_norm', ''),
                gn.get('f_rest', {}).get('grad_norm', ''),
                gn.get('opacity', {}).get('grad_norm', ''),
                gn.get('scaling', {}).get('grad_norm', ''),
                gn.get('rotation', {}).get('grad_norm', ''),
                pc.get('xyz', {}).get('max_abs_change', ''),
                pc.get('features_dc', {}).get('max_abs_change', ''),
                pc.get('opacity', {}).get('max_abs_change', ''),
            ])
    print(f"Saved: {csv_path}")

    # Analysis
    psnrs = [e['test_psnr'] for e in trace]
    drops = [(i, psnr_val) for i, psnr_val in enumerate(psnrs) if psnr_val < 20]
    print(f"\nPSNR trace: start={psnrs[0]:.2f}, end={psnrs[-1]:.2f}, min={min(psnrs):.2f}")
    if drops:
        first_below_20 = drops[0]
        print(f"  PSNR first dropped below 20 at step {first_below_20[0]} (PSNR={first_below_20[1]:.2f})")
    drops_15 = [(i, p) for i, p in enumerate(psnrs) if p < 15]
    if drops_15:
        first_below_15 = drops_15[0]
        print(f"  PSNR first dropped below 15 at step {first_below_15[0]} (PSNR={first_below_15[1]:.2f})")

    report = {
        'initial_psnr': psnrs[0],
        'final_psnr': psnrs[-1],
        'min_psnr': min(psnrs),
        'max_psnr': max(psnrs),
        'steps_to_below_20': drops[0][0] if drops else None,
        'steps_to_below_15': drops_15[0][0] if drops_15 else None,
        'psnr_drop': round(psnrs[0] - psnrs[-1], 4),
    }
    json_report = OUT_DIR / 'divergence_summary.json'
    with open(json_report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSummary: {json_report}")

if __name__ == '__main__':
    main()
