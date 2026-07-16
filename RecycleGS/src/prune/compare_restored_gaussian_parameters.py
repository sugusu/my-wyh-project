#!/usr/bin/env python3
"""Compare original 15k PLY vs checkpoint-restored model parameters."""
import json, os, sys, torch, numpy as np

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene import GaussianModel
from arguments import OptimizationParams
from argparse import ArgumentParser as ArgParser

PLY_PATH = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full/point_cloud/iteration_15000/point_cloud.ply'
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

def param_corr(a, b):
    a_f = a.flatten().float().cpu()
    b_f = b.flatten().float().cpu()
    if a_f.numel() == 0 or b_f.numel() == 0:
        return float('nan')
    a_centered = a_f - a_f.mean()
    b_centered = b_f - b_f.mean()
    denom = (a_centered.norm() * b_centered.norm())
    if denom < 1e-10:
        return 1.0
    return float((a_centered @ b_centered) / denom)

def analyze_param(name, ply_tensor, ckpt_tensor):
    diff = (ply_tensor - ckpt_tensor).abs()
    result = {
        'name': name,
        'shape': list(ply_tensor.shape),
        'dtype': str(ply_tensor.dtype),
        'ply': {
            'min': round(float(ply_tensor.min().cpu()), 8),
            'max': round(float(ply_tensor.max().cpu()), 8),
            'mean': round(float(ply_tensor.mean().cpu()), 8),
            'std': round(float(ply_tensor.std().cpu()), 8),
            'norm': round(float(ply_tensor.norm().cpu()), 8),
        },
        'checkpoint': {
            'min': round(float(ckpt_tensor.min().cpu()), 8),
            'max': round(float(ckpt_tensor.max().cpu()), 8),
            'mean': round(float(ckpt_tensor.mean().cpu()), 8),
            'std': round(float(ckpt_tensor.std().cpu()), 8),
            'norm': round(float(ckpt_tensor.norm().cpu()), 8),
        },
        'diff': {
            'max_abs': round(float(diff.max().cpu()), 8),
            'mean_abs': round(float(diff.mean().cpu()), 8),
            'max_rel_percent': round(float((diff.max() / (ply_tensor.abs().max() + 1e-10) * 100).cpu()), 4),
        },
        'correlation': round(param_corr(ply_tensor, ckpt_tensor), 8),
        'exact_match': bool((ply_tensor == ckpt_tensor).all()),
    }
    return result

def load_params_from_ply(ply_path):
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    xyz = np.stack([np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])], axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 1, 3))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 0, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 0, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
    extra_f_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")],
                           key=lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))
    scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
                         key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
                       key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    return {
        'xyz': torch.tensor(xyz, dtype=torch.float),
        'features_dc': torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous(),
        'features_rest': torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous(),
        'opacity': torch.tensor(opacities, dtype=torch.float),
        'scaling': torch.tensor(scales, dtype=torch.float),
        'rotation': torch.tensor(rots, dtype=torch.float),
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda:0'

    print("Loading PLY...")
    ply_params = load_params_from_ply(PLY_PATH)
    print(f"  Loaded {ply_params['xyz'].shape[0]} points from PLY")

    print("Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model_params, iteration = ckpt
    print(f"  Checkpoint iteration: {iteration}")

    ckpt_params = {
        'xyz': model_params[1].cpu(),
        'knn_f': model_params[2].cpu(),
        'features_dc': model_params[3].cpu(),
        'features_rest': model_params[4].cpu(),
        'scaling': model_params[5].cpu(),
        'rotation': model_params[6].cpu(),
        'opacity': model_params[7].cpu(),
        'transparency': model_params[8].cpu(),
    }

    # Compare PLY vs checkpoint for matching params
    report = {}
    for name in ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']:
        if name in ply_params and name in ckpt_params:
            report[name] = analyze_param(name, ply_params[name], ckpt_params[name])
            print(f"\n{name}:")
            r = report[name]
            print(f"  PLY: mean={r['ply']['mean']}, std={r['ply']['std']}")
            print(f"  CKPT: mean={r['checkpoint']['mean']}, std={r['checkpoint']['std']}")
            print(f"  Diff: max_abs={r['diff']['max_abs']}, mean_abs={r['diff']['mean_abs']}")
            print(f"  Correlation: {r['correlation']}")
            print(f"  Exact match: {r['exact_match']}")

    # Also restore model via restore() and compare
    print("\n\nRestoring model from checkpoint via restore()...")
    gaussians = GaussianModel(3, None)
    opt = make_opt()
    gaussians.restore(model_params, opt)
    restored_params = {
        'xyz': gaussians._xyz.detach().cpu(),
        'knn_f': gaussians._knn_f.detach().cpu(),
        'features_dc': gaussians._features_dc.detach().cpu(),
        'features_rest': gaussians._features_rest.detach().cpu(),
        'opacity': gaussians._opacity.detach().cpu(),
        'scaling': gaussians._scaling.detach().cpu(),
        'rotation': gaussians._rotation.detach().cpu(),
        'transparency': gaussians._transparency.detach().cpu(),
    }

    # Compare raw checkpoint params vs restored
    for name in ckpt_params:
        if name in restored_params:
            rdiff = (ckpt_params[name] - restored_params[name]).abs()
            report[f'raw_ckpt_vs_restored_{name}'] = {
                'name': name, 'max_abs_diff': round(float(rdiff.max().cpu()), 8),
                'mean_abs_diff': round(float(rdiff.mean().cpu()), 8),
                'exact_match': bool((ckpt_params[name] == restored_params[name]).all()),
            }

    # Compare optimizer state
    opt_dict = model_params[15]
    restored_opt = gaussians.optimizer
    opt_report = {}
    for pg in restored_opt.param_groups:
        name = pg['name']
        lr = pg['lr']
        state = restored_opt.state.get(pg['params'][0], None)
        opt_report[name] = {
            'lr': round(float(lr), 12),
            'has_state': state is not None,
            'exp_avg_shape': list(state['exp_avg'].shape) if state is not None else None,
            'exp_avg_sq_shape': list(state['exp_avg_sq'].shape) if state is not None else None,
        }
    report['optimizer_state'] = opt_report

    json_path = os.path.join(OUT_DIR, 'parameter_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Generate MD report
    md = ["# Parameter Comparison: PLY vs Checkpoint vs Restored", "",
          f"PLY: {PLY_PATH}", f"Checkpoint: {CKPT_PATH}", ""]
    for name in ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']:
        if name in report:
            r = report[name]
            md.append(f"## {name}")
            md.append(f"- Shape: {r['shape']}, dtype: {r['dtype']}")
            md.append(f"- PLY: [{r['ply']['min']:.6f}, {r['ply']['max']:.6f}] mean={r['ply']['mean']:.6f} std={r['ply']['std']:.6f}")
            md.append(f"- Checkpoint: [{r['checkpoint']['min']:.6f}, {r['checkpoint']['max']:.6f}] mean={r['checkpoint']['mean']:.6f} std={r['checkpoint']['std']:.6f}")
            md.append(f"- Max abs diff: {r['diff']['max_abs']:.8f}")
            md.append(f"- Correlation: {r['correlation']:.8f}")
            md.append(f"- Exact match: {r['exact_match']}")
            md.append("")

    md.append("## Raw Checkpoint vs Restored (via restore())")
    for name in ckpt_params:
        key = f'raw_ckpt_vs_restored_{name}'
        if key in report:
            r = report[key]
            md.append(f"- {name}: max_diff={r['max_abs_diff']:.8f}, mean_diff={r['mean_abs_diff']:.8f}, exact={r['exact_match']}")
    md.append("")

    md.append("## Optimizer State")
    for name, r in report.get('optimizer_state', {}).items():
        md.append(f"- {name}: lr={r['lr']:.12e}, has_state={r['has_state']}")
    md.append("")

    md_path = os.path.join(OUT_DIR, 'parameter_comparison_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
