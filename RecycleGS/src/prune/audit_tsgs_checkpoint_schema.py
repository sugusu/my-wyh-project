#!/usr/bin/env python3
"""Load chkpnt15000.pth, inspect all keys and their shapes.
Check capture/restore schema consistency."""
import json, os, sys, torch

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')

CKPT_PATH = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full/chkpnt15000.pth'
OUT_DIR = '/data/wyh/RecycleGS/outputs/debug/stage2b_recovery_collapse'

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    model_params, iteration = ckpt

    # Expected schema from capture():
    # (active_sh_degree, _xyz, _knn_f, _features_dc, _features_rest,
    #  _scaling, _rotation, _opacity, _transparency,
    #  max_radii2D, max_weight, xyz_gradient_accum, xyz_gradient_accum_abs,
    #  denom, denom_abs, optimizer.state_dict(), spatial_lr_scale)
    expected_names = [
        'active_sh_degree', '_xyz', '_knn_f', '_features_dc', '_features_rest',
        '_scaling', '_rotation', '_opacity', '_transparency',
        'max_radii2D', 'max_weight', 'xyz_gradient_accum', 'xyz_gradient_accum_abs',
        'denom', 'denom_abs', 'optimizer_state_dict', 'spatial_lr_scale',
    ]

    report = {
        'checkpoint_path': CKPT_PATH,
        'iteration': iteration,
        'num_elements': len(model_params),
        'expected_num': len(expected_names),
        'schema_match': len(model_params) == len(expected_names),
        'elements': [],
        'optimizer_analysis': {},
    }

    print(f"Checkpoint elements: {len(model_params)} (expected {len(expected_names)})")
    for i, (name, elem) in enumerate(zip(expected_names, model_params)):
        info = {'index': i, 'name': name}
        if isinstance(elem, torch.Tensor):
            info['type'] = 'Tensor'
            info['shape'] = list(elem.shape)
            info['dtype'] = str(elem.dtype)
            info['min'] = round(float(elem.min().item()), 8) if elem.numel() > 0 else None
            info['max'] = round(float(elem.max().item()), 8) if elem.numel() > 0 else None
            info['mean'] = round(float(elem.mean().item()), 8) if elem.numel() > 0 else None
            info['std'] = round(float(elem.std().item()), 8) if elem.numel() > 0 else None
            info['numel'] = int(elem.numel())
            info['has_nan'] = bool(torch.isnan(elem).any())
            info['has_inf'] = bool(torch.isinf(elem).any())
        elif isinstance(elem, dict):
            info['type'] = 'dict'
            info['keys'] = list(elem.keys())
            if 'param_groups' in elem:
                info['num_param_groups'] = len(elem['param_groups'])
                info['param_groups'] = []
                for j, pg in enumerate(elem['param_groups']):
                    pg_info = {
                        'group_index': j,
                        'lr': round(float(pg.get('lr', 0)), 12),
                        'name': pg.get('name', 'UNKNOWN'),
                        'num_params': len(pg.get('params', [])),
                        'betas': [round(float(b), 6) for b in pg.get('betas', [0,0])],
                        'eps': round(float(pg.get('eps', 0)), 12),
                        'weight_decay': round(float(pg.get('weight_decay', 0)), 12),
                    }
                    info['param_groups'].append(pg_info)
                info['state_entries'] = len(elem.get('state', {}))
                # Check state structure
                for pid, state in elem.get('state', {}).items():
                    if 'exp_avg' in state:
                        info.setdefault('state_shapes', {})[str(pid)] = {
                            'exp_avg_shape': list(state['exp_avg'].shape),
                            'exp_avg_sq_shape': list(state['exp_avg_sq'].shape),
                            'step': int(state.get('step', 0)) if 'step' in state else None,
                        }
            elif 'state' in elem:
                info['state_entry_count'] = len(elem['state'])
        elif isinstance(elem, (int, float)):
            info['type'] = type(elem).__name__
            info['value'] = elem
        else:
            info['type'] = type(elem).__name__
            info['value'] = str(elem)

        info['size_bytes'] = elem.element_size() * elem.numel() if isinstance(elem, torch.Tensor) else None
        report['elements'].append(info)
        print(f"  [{i:2d}] {name:25s}: {info['type']:8s} ", end="")
        if 'shape' in info:
            print(f"shape={info['shape']}, dtype={info['dtype']}, range=[{info.get('min','?')}, {info.get('max','?')}]")
        elif 'value' in info:
            print(f"value={info['value']}")
        elif 'num_param_groups' in info:
            print(f"param_groups={info['num_param_groups']}, state_entries={info['state_entries']}")
        else:
            print(f"keys={info.get('keys', '?')}")

    # Analyze optimizer structure
    opt_dict = model_params[15]
    state = opt_dict['state']
    param_groups = opt_dict['param_groups']
    report['optimizer_analysis'] = {
        'num_param_groups': len(param_groups),
        'num_state_entries': len(state),
        'all_params_have_state': len(state) == len(param_groups),
        'missing_state_for': [],
        'analysis': [],
    }
    for j, pg in enumerate(param_groups):
        pids = pg['params']
        for pid in pids:
            has_state = pid in state
            report['optimizer_analysis']['analysis'].append({
                'param_id': pid, 'group_name': pg.get('name', '?'),
                'group_index': j, 'has_state': has_state,
            })
            if not has_state:
                report['optimizer_analysis']['missing_state_for'].append(pg.get('name', '?'))

    # Check if schema matches capture()
    # Compare with scene/gaussian_model.py capture/restore
    capture_returns = 17  # From capture() code
    restore_accepts = 17   # From restore() code
    report['capture_restore_consistency'] = {
        'capture_returns': capture_returns,
        'restore_accepts': restore_accepts,
        'checkpoint_elements': len(model_params),
        'consistent': len(model_params) == capture_returns == restore_accepts,
        'note': 'capture() and restore() both expect 17 elements. Checkpoint has {} elements.'.format(len(model_params)),
    }

    json_path = os.path.join(OUT_DIR, 'checkpoint_schema_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Generate MD
    md = ["# Checkpoint Schema Audit", "", f"Checkpoint: {CKPT_PATH}", ""]
    for elem in report['elements']:
        md.append(f"## [{elem['index']}] {elem['name']}")
        md.append(f"- Type: {elem['type']}")
        if 'shape' in elem:
            md.append(f"- Shape: {elem['shape']}, dtype: {elem['dtype']}, numel: {elem.get('numel', '?')}")
            md.append(f"- Range: [{elem.get('min','?')}, {elem.get('max','?')}]")
            md.append(f"- Mean: {elem.get('mean','?')}, Std: {elem.get('std','?')}")
            md.append(f"- Has NaN: {elem.get('has_nan','?')}, Has Inf: {elem.get('has_inf','?')}")
        if 'param_groups' in elem:
            md.append(f"- Param groups: {elem['num_param_groups']}")
            for pg in elem['param_groups']:
                md.append(f"  - Group {pg['group_index']}: name={pg['name']}, lr={pg['lr']:.12e}, eps={pg['eps']}, betas={pg['betas']}")
            md.append(f"- State entries: {elem['state_entries']}")
        md.append("")

    md.append("## Optimizer Structure Analysis")
    oa = report['optimizer_analysis']
    md.append(f"- Total param groups: {oa['num_param_groups']}")
    md.append(f"- Total state entries: {oa['num_state_entries']}")
    md.append(f"- All params have state: {oa['all_params_have_state']}")
    if oa['missing_state_for']:
        md.append(f"- **MISSING state for**: {oa['missing_state_for']}")
    md.append("")

    md.append("## Capture/Restore Consistency")
    cr = report['capture_restore_consistency']
    md.append(f"- capture() returns {cr['capture_returns']} elements")
    md.append(f"- restore() accepts {cr['restore_accepts']} elements")
    md.append(f"- Checkpoint has {cr['checkpoint_elements']} elements")
    md.append(f"- Consistent: {cr['consistent']}")
    md.append(f"- Note: {cr['note']}")

    md_path = os.path.join(OUT_DIR, 'checkpoint_schema_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
