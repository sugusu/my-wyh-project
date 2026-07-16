#!/usr/bin/env python3
"""Load checkpoint, restore model + optimizer, record LR before and after selective_learning_rate_control at 15001."""
import json, os, sys, csv, torch

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene import GaussianModel
from arguments import OptimizationParams
from argparse import ArgumentParser as ArgParser

CKPT_PATH = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full/chkpnt15000.pth'
OUT_DIR = '/data/wyh/RecycleGS/outputs/debug/stage2b_recovery_collapse'

def make_opt():
    opt_parser = ArgParser()
    opt = OptimizationParams(opt_parser)
    for k, v in {
        'iterations': 30000, 'position_lr_init': 0.00016, 'position_lr_final': 0.0000016,
        'position_lr_delay_mult': 0.01, 'position_lr_max_steps': 30000, 'feature_lr': 0.0025,
        'opacity_lr': 0.05, 'scaling_lr': 0.005, 'rotation_lr': 0.001, 'percent_dense': 0.001,
        'lambda_dssim': 0.2, 'densification_interval': 100, 'opacity_reset_interval': 3000,
        'densify_from_iter': 500, 'densify_until_iter': 15000, 'densify_grad_threshold': 0.0002,
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

def get_lrs(gaussians):
    return {pg['name']: float(pg['lr']) for pg in gaussians.optimizer.param_groups}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model_params, iteration = ckpt
    print(f"Loaded checkpoint iteration {iteration}")

    opt = make_opt()
    gaussians = GaussianModel(3, None)
    gaussians.restore(model_params, opt)

    # LR before policy
    lr_before = get_lrs(gaussians)
    print("LR before selective_learning_rate_control:")
    for k, v in lr_before.items():
        print(f"  {k}: {v:.12e}")

    # Apply policy at iteration 15001 (just past freeze_iter=15000)
    gaussians.selective_learning_rate_control(
        15001, 15000,
        nofix_position=opt.nofix_position,
        nofix_opacity=opt.nofix_opacity,
        nofix_param=opt.nofix_param,
        nofix_scaling=opt.nofix_scaling,
        nofix_rotation=opt.nofix_rotation,
    )

    lr_after = get_lrs(gaussians)
    print("\nLR after selective_learning_rate_control at 15001:")
    for k, v in lr_after.items():
        print(f"  {k}: {v:.12e}")

    csv_path = os.path.join(OUT_DIR, 'lr_policy_before_after_15001.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['param_group', 'lr_before_policy', 'lr_after_policy', 'frozen', 'note'])
        for k in sorted(set(list(lr_before.keys()) + list(lr_after.keys()))):
            before = lr_before.get(k, 0)
            after = lr_after.get(k, 0)
            frozen = after == 0.0 and before > 0
            if k == 'xyz':
                note = "xyz updated by scheduler then frozen; LR already near final"
            elif k in ['f_dc', 'f_rest', 'transparency', 'f_asg']:
                note = "kept alive (in nofix_param_list)"
            elif frozen:
                note = "frozen to 0.0 (not in nofix_param_list)"
            else:
                note = ""
            w.writerow([k, f"{before:.12e}", f"{after:.12e}", frozen, note])
    print(f"Saved: {csv_path}")

    json_path = os.path.join(OUT_DIR, 'lr_policy_before_after_15001.json')
    with open(json_path, 'w') as f:
        json.dump({'lr_before': lr_before, 'lr_after': lr_after}, f, indent=2)
    print(f"Saved: {json_path}")

if __name__ == '__main__':
    main()
