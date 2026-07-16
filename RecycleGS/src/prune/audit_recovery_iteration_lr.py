#!/usr/bin/env python3
"""Load checkpoint, restore model, then record LR for step 0 and step 20.
Compare with expected TSGS LR schedule."""
import json, os, sys, torch, csv

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
    lrs = {}
    for pg in gaussians.optimizer.param_groups:
        lrs[pg['name']] = pg['lr']
    return lrs

def expected_xyz_lr(iteration, spatial_lr_scale=1.0):
    """Compute expected xyz LR from get_expon_lr_func"""
    from utils.general_utils import get_expon_lr_func
    func = get_expon_lr_func(
        lr_init=0.00016 * spatial_lr_scale,
        lr_final=0.0000016 * spatial_lr_scale,
        lr_delay_mult=0.01,
        max_steps=30000,
    )
    return func(iteration)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda:0'

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model_params, iteration = ckpt
    spatial_lr_scale = float(model_params[16])

    # Record LR from checkpoint state dict
    opt_dict = model_params[15]
    ckpt_lrs = {}
    for pg in opt_dict['param_groups']:
        ckpt_lrs[pg.get('name', '?')] = pg['lr']

    # Restore model and record LRs
    gaussians = GaussianModel(3, None)
    opt = make_opt()
    gaussians.restore(model_params, opt)

    # LRs immediately after restore (step 0)
    lrs_step0 = get_lrs(gaussians)

    # Simulate calling update_learning_rate like TSGS train.py does
    gaussians.update_learning_rate(15001)  # First recovery step
    lrs_stepped = get_lrs(gaussians)

    # Expected xyz LR
    exp_lr_15000 = expected_xyz_lr(15000, spatial_lr_scale)
    exp_lr_15001 = expected_xyz_lr(15001, spatial_lr_scale)
    exp_lr_15020 = expected_xyz_lr(15020, spatial_lr_scale)

    # Compute LR if we had called update_learning_rate at step 15001
    gaussians2 = GaussianModel(3, None)
    opt2 = make_opt()
    gaussians2.restore(model_params, opt2)
    gaussians2.update_learning_rate(15020)
    lrs_step20 = get_lrs(gaussians2)

    report = {
        'checkpoint_lrs': {k: round(float(v), 12) for k, v in ckpt_lrs.items()},
        'restored_step0_lrs': {k: round(float(v), 12) for k, v in lrs_step0.items()},
        'restored_with_lr_update_15001': {k: round(float(v), 12) for k, v in lrs_stepped.items()},
        'restored_with_lr_update_15020': {k: round(float(v), 12) for k, v in lrs_step20.items()},
        'expected_lrs': {
            'xyz_at_15000': round(float(exp_lr_15000), 12),
            'xyz_at_15001': round(float(exp_lr_15001), 12),
            'xyz_at_15020': round(float(exp_lr_15020), 12),
        },
        'spatial_lr_scale': spatial_lr_scale,
        'current_recovery_misses_lr_update': True,
        'note': 'train_pruned_recovery.py does NOT call update_learning_rate(). '
                'All LRs stay at their restored values throughout recovery.',
    }

    json_path = os.path.join(OUT_DIR, 'lr_audit_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    # CSV trace
    csv_path = os.path.join(OUT_DIR, 'lr_trace.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'param', 'lr_actual', 'lr_expected_xyz'])
        for step_name, lrs in [('step0(15000)', lrs_step0), ('step1(15001)', lrs_stepped), ('step20(15020)', lrs_step20)]:
            for pname, lr in lrs.items():
                w.writerow([step_name, pname, lr, exp_lr_15000 if step_name == 'step0(15000)' else (exp_lr_15001 if step_name == 'step1(15001)' else exp_lr_15020)])
        w.writerow(['expected_15000', 'xyz', '', exp_lr_15000])
        w.writerow(['expected_15001', 'xyz', '', exp_lr_15001])
        w.writerow(['expected_15020', 'xyz', '', exp_lr_15020])
    print(f"Saved: {csv_path}")

    # MD
    md = ["# Learning Rate Audit", "",
          f"Checkpoint: {CKPT_PATH}", f"Spatial LR scale: {spatial_lr_scale}", ""]
    md.append("## Checkpoint LRs (from state dict)")
    for k, v in report['checkpoint_lrs'].items():
        md.append(f"- {k}: {v:.12e}")
    md.append("")
    md.append("## Restored LRs (step 0, no update_learning_rate called)")
    for k, v in report['restored_step0_lrs'].items():
        md.append(f"- {k}: {v:.12e}")
    md.append("")
    md.append("## After calling update_learning_rate(15001)")
    for k, v in report['restored_with_lr_update_15001'].items():
        md.append(f"- {k}: {v:.12e}")
    md.append("")
    md.append("## After calling update_learning_rate(15020)")
    for k, v in report['restored_with_lr_update_15020'].items():
        md.append(f"- {k}: {v:.12e}")
    md.append("")
    md.append("## Expected XYZ LR (from get_expon_lr_func)")
    for k, v in report['expected_lrs'].items():
        md.append(f"- {k}: {v:.12e}")
    md.append("")
    md.append("## Diagnosis")
    md.append(f"- xyz LR at step 0 (restored): {lrs_step0.get('xyz', 'N/A')}")
    md.append(f"- xyz LR with update_learning_rate(15001): {lrs_stepped.get('xyz', 'N/A')}")
    md.append(f"- Expected xyz LR at 15001: {exp_lr_15001:.12e}")
    if abs(lrs_step0.get('xyz', 0) - exp_lr_15000) < 1e-10:
        md.append("- **OK**: Restored xyz LR matches expected LR at iteration 15000")
    else:
        md.append(f"- **ISSUE**: Restored xyz LR ({lrs_step0.get('xyz', 'N/A')}) differs from expected ({exp_lr_15000:.12e})")
    md.append(f"- **MISSING update_learning_rate**: Recovery training does not call update_learning_rate(). "
              f"xyz LR stays at {lrs_step0.get('xyz', 'N/A')} throughout instead of decaying to {exp_lr_15020:.12e} by step 20.")

    md_path = os.path.join(OUT_DIR, 'lr_audit_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
