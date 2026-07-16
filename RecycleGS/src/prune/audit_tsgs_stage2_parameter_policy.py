#!/usr/bin/env python3
"""Build table of all optimizer param groups comparing restored checkpoint vs official TSGS stage 2 policy."""
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

def get_official_lr_at_15001(gaussians, opt, spatial_lr_scale):
    from utils.general_utils import get_expon_lr_func
    xyz_scheduler = get_expon_lr_func(
        lr_init=0.00016 * spatial_lr_scale,
        lr_final=0.0000016 * spatial_lr_scale,
        lr_delay_mult=0.01,
        max_steps=30000,
    )
    official_lrs = {}
    for pg in gaussians.optimizer.param_groups:
        name = pg['name']
        if name == 'xyz':
            official_lrs[name] = float(xyz_scheduler(15001))
        else:
            official_lrs[name] = float(pg['lr'])
    return official_lrs

def apply_policy(gaussians, opt, spatial_lr_scale):
    gaussians.selective_learning_rate_control(
        15001, 15000,
        nofix_position=opt.nofix_position,
        nofix_opacity=opt.nofix_opacity,
        nofix_param=opt.nofix_param,
        nofix_scaling=opt.nofix_scaling,
        nofix_rotation=opt.nofix_rotation,
    )
    policy_lrs = {}
    policy_trainable = {}
    for pg in gaussians.optimizer.param_groups:
        name = pg['name']
        policy_lrs[name] = float(pg['lr'])
        policy_trainable[name] = pg['lr'] > 0
    return policy_lrs, policy_trainable

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model_params, iteration = ckpt
    spatial_lr_scale = float(model_params[16])

    opt = make_opt()
    gaussians = GaussianModel(3, None)
    gaussians.restore(model_params, opt)

    restored_lrs = {}
    restored_trainable = {}
    for pg in gaussians.optimizer.param_groups:
        name = pg['name']
        restored_lrs[name] = float(pg['lr'])
        restored_trainable[name] = True  # all trainable after restore

    official_lrs = get_official_lr_at_15001(gaussians, opt, spatial_lr_scale)

    policy_lrs, policy_trainable = apply_policy(gaussians, opt, spatial_lr_scale)

    NOFIX_PARAMS = ["transparency", "f_dc", "f_rest", "f_asg"]
    official_trainable = {}
    for name in sorted(policy_lrs.keys()):
        if name == 'xyz':
            official_trainable[name] = False  # xyz LR set to 0 by freezing logic
        elif name in NOFIX_PARAMS:
            official_trainable[name] = True
        else:
            official_trainable[name] = False

    rows = []
    for name in sorted(set(list(restored_lrs.keys()) + list(policy_lrs.keys()))):
        r_lr = restored_lrs.get(name, 'N/A')
        o_lr = official_lrs.get(name, 'N/A')
        p_lr = policy_lrs.get(name, 'N/A')
        o_tr = official_trainable.get(name, 'N/A')
        p_tr = policy_trainable.get(name, 'N/A')
        match = (abs(float(p_lr) - float(o_lr)) < 1e-12 if isinstance(p_lr, float) and isinstance(o_lr, float) else 'N/A')
        rows.append({
            'group_name': name,
            'lr_at_restored_checkpoint': r_lr,
            'official_lr_at_15001': o_lr,
            'official_trainable_at_15001': o_tr,
            'custom_recovery_lr_at_15001': p_lr,
            'custom_recovery_trainable_at_15001': p_tr,
            'match': match,
        })

    csv_path = os.path.join(OUT_DIR, 'stage2_parameter_policy_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['group_name', 'lr_at_restored_checkpoint', 'official_lr_at_15001',
                                           'official_trainable_at_15001', 'custom_recovery_lr_at_15001',
                                           'custom_recovery_trainable_at_15001', 'match'])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_path}")

    md = ["# Stage 2 Parameter Policy Comparison", "",
          f"Checkpoint: {CKPT_PATH}", f"Starting iteration: 15001", f"Freeze iteration: 15000", ""]

    md.append("| Param Group | LR Restored | Official LR @15001 | Official Trainable | Recovery LR @15001 | Recovery Trainable | Match |")
    md.append("|-------------|-------------|-------------------|-------------------|-------------------|-------------------|-------|")
    all_match = True
    for r in rows:
        m = r['match']
        if isinstance(m, bool) and not m:
            all_match = False
        md.append(f"| {r['group_name']} | {r['lr_at_restored_checkpoint']} | {r['official_lr_at_15001']} | {r['official_trainable_at_15001']} | {r['custom_recovery_lr_at_15001']} | {r['custom_recovery_trainable_at_15001']} | {m} |")

    md.append("")
    md.append("## Analysis")
    md.append(f"- `selective_learning_rate_control` at iteration 15001 (freeze_iter=15000):")
    md.append(f"  - xyz LR updated via scheduler, then frozen to 0.0 (not in nofix_param_list)")
    md.append(f"  - f_dc, f_rest, transparency, f_asg: keep their LR (in nofix_param_list)")
    md.append(f"  - opacity, scaling, rotation: frozen to 0.0 (not in nofix_param_list)")
    md.append(f"  - knn_f: frozen to 0.0 (not in nofix_param_list)")
    md.append(f"- **Without selective_learning_rate_control, all params keep peak training LRs**")

    if all_match:
        md.append(f"\n**VERDICT: Policy implementation matches TSGS official behavior**")
    else:
        mismatches = [r['group_name'] for r in rows if isinstance(r['match'], bool) and not r['match']]
        md.append(f"\n**ISSUE: Policy mismatch for groups: {mismatches}**")

    md_path = os.path.join(OUT_DIR, 'stage2_parameter_policy_comparison_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"Saved: {md_path}")

    json_path = os.path.join(OUT_DIR, 'stage2_parameter_policy_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Saved: {json_path}")

if __name__ == '__main__':
    main()
