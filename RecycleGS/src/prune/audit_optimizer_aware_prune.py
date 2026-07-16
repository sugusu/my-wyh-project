#!/usr/bin/env python3
"""Audit optimizer-aware pruning: verify prune_points preserves optimizer state correctly."""
import json, os, sys, torch, numpy as np
from pathlib import Path

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene.gaussian_model import GaussianModel
from arguments import OptimizationParams, ModelParams
from argparse import ArgumentParser

def main():
    out_dir = Path('/data/wyh/RecycleGS/outputs/debug/stage2b_preflight')
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_name = 'scene_01'
    ckpt_path = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full/chkpnt15000.pth'
    prune_indices_path = '/data/wyh/RecycleGS/outputs/prune_only/scene_01/ratio_005/mask_risk/prune_indices.npy'

    print(f"=== Optimizer-Aware Prune Audit ===")
    print(f"Scene: {scene_name}")
    print(f"Checkpoint: {ckpt_path}")

    # 1. Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    model_params, iteration = ckpt
    print(f"Iteration: {iteration}")

    # 2. Load prune indices
    prune_idx = np.load(prune_indices_path)
    K = len(prune_idx)
    print(f"Prune indices: {K} Gaussians to remove")

    # 3. Initialize GaussianModel
    gaussians = GaussianModel(sh_degree=3, asg_degree=None)
    parser = ArgumentParser()
    opt = OptimizationParams(parser)
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
        'use_asg': True,
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

    # 3. Restore from checkpoint
    gaussians.restore(model_params, opt)
    N_before = gaussians.get_xyz.shape[0]
    print(f"Gaussians before prune: {N_before}")

    # Record pre-prune optimizer state
    pre_opt_state = {
        group['name']: {
            'param_shape': list(group['params'][0].shape),
            'lr': group['lr'],
        }
        for group in gaussians.optimizer.param_groups
    }
    print("Pre-prune optimizer groups:")
    for name, info in pre_opt_state.items():
        print(f"  {name}: shape={info['param_shape']}, lr={info['lr']}")

    pre_state_dict = {}
    for group in gaussians.optimizer.param_groups:
        p = group['params'][0]
        name = group['name']
        state = gaussians.optimizer.state.get(p, None)
        if state and 'exp_avg' in state:
            pre_state_dict[name] = {
                'exp_avg_shape': list(state['exp_avg'].shape),
                'exp_avg_sq_shape': list(state['exp_avg_sq'].shape),
                'exp_avg_sample': state['exp_avg'][:5].detach().cpu().tolist(),
            }

    # 4. Create prune mask and prune
    prune_mask = torch.zeros(N_before, dtype=bool, device='cuda:0')
    prune_mask[torch.tensor(prune_idx, device='cuda:0')] = True
    gaussians.prune_points(prune_mask)

    N_after = gaussians.get_xyz.shape[0]
    print(f"Gaussians after prune: {N_after}")
    print(f"Expected: {N_before - K}, Got: {N_after}")
    count_check = (N_after == N_before - K)
    print(f"Count check: {'PASS' if count_check else 'FAIL'}")

    # 5. Check all tensors have consistent sizes
    param_names = ['_xyz', '_knn_f', '_features_dc', '_features_rest',
                   '_opacity', '_transparency', '_scaling', '_rotation']
    tensor_checks = {}
    for pn in param_names:
        t = getattr(gaussians, pn, None)
        if t is not None:
            tensor_checks[pn] = {
                'shape': list(t.shape),
                'N': t.shape[0],
                'consistent_with_count': t.shape[0] == N_after,
            }
        else:
            tensor_checks[pn] = {'shape': None, 'N': None}

    # Check extra buffers
    buffer_names = ['xyz_gradient_accum', 'xyz_gradient_accum_abs', 'denom', 'denom_abs', 'max_radii2D', 'max_weight']
    buffer_checks = {}
    for bn in buffer_names:
        t = getattr(gaussians, bn, None)
        if t is not None:
            buffer_checks[bn] = {
                'shape': list(t.shape),
                'N': t.shape[0],
                'consistent_with_count': t.shape[0] == N_after,
            }

    # Check optimizer state sizes
    post_opt_state = {}
    for group in gaussians.optimizer.param_groups:
        p = group['params'][0]
        name = group['name']
        state = gaussians.optimizer.state.get(p, None)
        if state and 'exp_avg' in state:
            e = state['exp_avg']
            eq = state['exp_avg_sq']
            post_opt_state[name] = {
                'param_shape': list(p.shape),
                'exp_avg_shape': list(e.shape),
                'exp_avg_sq_shape': list(eq.shape),
                'consistent': e.shape[0] == p.shape[0] == N_after,
            }

    all_consistent = all(
        v.get('consistent_with_count', True) for v in tensor_checks.values()
    ) and all(
        v.get('consistent_with_count', True) for v in buffer_checks.values()
    ) and all(
        v.get('consistent', True) for v in post_opt_state.values()
    )

    # 6. Build and save report
    audit = {
        'scene': scene_name,
        'checkpoint': ckpt_path,
        'iteration': iteration,
        'N_before': N_before,
        'K': K,
        'N_after': N_after,
        'N_expected': N_before - K,
        'count_check': count_check,
        'all_tensors_consistent': all_consistent,
        'pre_prune_optimizer_groups': pre_opt_state,
        'post_prune_tensor_shapes': tensor_checks,
        'post_prune_buffer_shapes': buffer_checks,
        'post_prune_optimizer_state': post_opt_state,
        'status': 'PASS' if (count_check and all_consistent) else 'FAIL',
    }

    json_path = out_dir / 'optimizer_aware_prune_audit.json'
    with open(json_path, 'w') as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nSaved audit: {json_path}")

    # Generate markdown report
    md_lines = [
        "# Optimizer-Aware Prune Audit",
        "",
        f"**Status**: {'PASS' if audit['status'] == 'PASS' else 'FAIL'}",
        "",
        f"- Scene: {scene_name}",
        f"- Checkpoint: {ckpt_path}",
        f"- Iteration: {iteration}",
        f"- N_before: {N_before}",
        f"- K (pruned): {K}",
        f"- N_after: {N_after}",
        f"- N_expected: {N_before - K}",
        f"- Count check: {'PASS' if count_check else 'FAIL'}"
           f" ({N_after} == {N_before - K})",
        f"- All tensors consistent: {'PASS' if all_consistent else 'FAIL'}",
        "",
        "### Post-Prune Tensor Shapes",
        "| Param | Shape | N | Consistent |",
        "|-------|-------|---|------------|",
    ]
    for pn, info in tensor_checks.items():
        shape_str = str(info.get('shape', 'N/A'))
        n = info.get('N', 'N/A')
        c = 'PASS' if info.get('consistent_with_count') else 'FAIL'
        md_lines.append(f"| {pn} | {shape_str} | {n} | {c} |")

    md_lines.extend([
        "",
        "### Post-Prune Buffer Shapes",
        "| Buffer | Shape | N | Consistent |",
        "|--------|-------|---|------------|",
    ])
    for bn, info in buffer_checks.items():
        shape_str = str(info.get('shape', 'N/A'))
        n = info.get('N', 'N/A')
        c = 'PASS' if info.get('consistent_with_count') else 'FAIL'
        md_lines.append(f"| {bn} | {shape_str} | {n} | {c} |")

    md_lines.extend([
        "",
        "### Post-Prune Optimizer State",
        "| Group | Param Shape | exp_avg Shape | exp_avg_sq Shape | Consistent |",
        "|-------|-------------|---------------|------------------|------------|",
    ])
    for name, info in post_opt_state.items():
        c = 'PASS' if info.get('consistent') else 'FAIL'
        md_lines.append(f"| {name} | {info['param_shape']} | {info['exp_avg_shape']} | {info['exp_avg_sq_shape']} | {c} |")

    md_path = out_dir / 'optimizer_aware_prune_audit_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f"Saved report: {md_path}")
    print(f"\nOverall: {audit['status']}")

if __name__ == '__main__':
    main()
