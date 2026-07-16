#!/usr/bin/env python3
"""Check recovery checkpoint: load .pth, report keys, optimizer state."""
import json, os, sys, torch
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

SCENES = {
    'scene_01': '/data/wyh/RecycleGS/baselines/tsgs_scene01_full/chkpnt15000.pth',
    'scene_03': '/data/wyh/RecycleGS/baselines/tsgs_scene03_full/chkpnt15000.pth',
}

def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, tuple) and len(ckpt) == 2:
        model_params, iteration = ckpt
        info = {
            'format': 'tuple(model_params, iteration)',
            'iteration': iteration,
            'model_params_type': str(type(model_params)),
            'model_params_len': len(model_params) if isinstance(model_params, tuple) else None,
        }
        if isinstance(model_params, tuple):
            info['num_tensors'] = len(model_params)
            tensor_shapes = []
            for i, t in enumerate(model_params):
                if hasattr(t, 'shape'):
                    tensor_shapes.append(f"  [{i}] {type(t).__name__} shape={list(t.shape)}")
                else:
                    tensor_shapes.append(f"  [{i}] {type(t).__name__} value={t}")
            info['tensor_details'] = tensor_shapes
            # These are from capture(): (active_sh_degree, _xyz, _knn_f, _features_dc, _features_rest, _scaling, _rotation, _opacity, _transparency, max_radii2D, max_weight, xyz_gradient_accum, xyz_gradient_accum_abs, denom, denom_abs, opt_dict, spatial_lr_scale)
            gaussian_keys = ['active_sh_degree', '_xyz', '_knn_f', '_features_dc', '_features_rest',
                             '_scaling', '_rotation', '_opacity', '_transparency',
                             'max_radii2D', 'max_weight', 'xyz_gradient_accum', 'xyz_gradient_accum_abs',
                             'denom', 'denom_abs', 'optimizer_state_dict', 'spatial_lr_scale']
            for i, (k, t) in enumerate(zip(gaussian_keys, model_params)):
                if hasattr(t, 'shape'):
                    info[k] = str(list(t.shape))
                else:
                    info[k] = str(t)

            opt_dict = model_params[15] if len(model_params) > 15 else None
            if isinstance(opt_dict, dict) and 'state' in opt_dict:
                info['optimizer_state_present'] = True
                state = opt_dict['state']
                param_group_ids = list(state.keys())
                info['optimizer_param_group_ids'] = param_group_ids
                has_exp_avg = False
                has_exp_avg_sq = False
                for sid in param_group_ids:
                    s = state[sid]
                    if 'exp_avg' in s:
                        has_exp_avg = True
                        info[f'exp_avg_shape_{sid}'] = list(s['exp_avg'].shape)
                    if 'exp_avg_sq' in s:
                        has_exp_avg_sq = True
                        info[f'exp_avg_sq_shape_{sid}'] = list(s['exp_avg_sq'].shape)
                info['has_exp_avg'] = has_exp_avg
                info['has_exp_avg_sq'] = has_exp_avg_sq
                info['restore_possible'] = True
            else:
                info['optimizer_state_present'] = False
                info['restore_possible'] = False
        return info
    else:
        return {
            'format': 'unknown',
            'keys': list(ckpt.keys()) if isinstance(ckpt, dict) else str(type(ckpt)),
            'restore_possible': False,
        }

def main():
    out_dir = Path('/data/wyh/RecycleGS/outputs/debug/stage2b_preflight')
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = {}
    md_lines = ["# Checkpoint Inventory Report", "",
                "## Per-Scene Checkpoint Analysis", ""]

    for scene_name, ckpt_path in SCENES.items():
        print(f"\n=== {scene_name} ===")
        print(f"Checkpoint: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            print(f"  NOT FOUND")
            inventory[scene_name] = {'error': 'file not found'}
            md_lines.append(f"### {scene_name}")
            md_lines.append(f"**File NOT FOUND**: {ckpt_path}")
            continue

        info = inspect_checkpoint(ckpt_path)
        inventory[scene_name] = info

        md_lines.append(f"### {scene_name}")
        md_lines.append(f"- **File**: `{ckpt_path}`")
        md_lines.append(f"- **Format**: `{info.get('format', 'unknown')}`")
        md_lines.append(f"- **Iteration**: {info.get('iteration', 'N/A')}")
        md_lines.append(f"- **Optimizer state present**: {info.get('optimizer_state_present', False)}")
        md_lines.append(f"- **Has exp_avg**: {info.get('has_exp_avg', False)}")
        md_lines.append(f"- **Has exp_avg_sq**: {info.get('has_exp_avg_sq', False)}")
        md_lines.append(f"- **Restore possible**: {info.get('restore_possible', False)}")
        md_lines.append("")
        md_lines.append("**Tensor shapes:**")
        for detail in info.get('tensor_details', []):
            md_lines.append(f"  - `{detail}`")
        md_lines.append("")

    md_lines.append("## Restore Verdict")
    md_lines.append("")
    all_restorable = all(
        v.get('restore_possible', False)
        for v in inventory.values() if 'error' not in v
    )
    md_lines.append(f"**All scenes restorable: {'YES' if all_restorable else 'NO'}**")

    json_path = out_dir / 'checkpoint_inventory.json'
    with open(json_path, 'w') as f:
        json.dump(inventory, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    md_path = out_dir / 'checkpoint_restore_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
