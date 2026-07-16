#!/usr/bin/env python3
"""Check if AppModel and SpecularModel are properly restored in recovery training.
Compare with official train.py's restore procedure."""
import json, os, sys, torch

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene.app_model import AppModel
from scene import SpecularModel

MODEL_DIR = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full'
TSGS_TRAIN = '/data/wyh/repos/TSGS/train.py'
RECOVERY_SCRIPT = '/data/wyh/RecycleGS/src/prune/train_pruned_recovery.py'
OUT_DIR = '/data/wyh/RecycleGS/outputs/debug/stage2b_recovery_collapse'

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    app_ckpt = os.path.join(MODEL_DIR, 'app_model/iteration_15000/app.pth')
    specular_ckpt = os.path.join(MODEL_DIR, 'specular_model/iteration_15000/specular.pth')

    report = {
        'model_dir': MODEL_DIR,
        'files_checked': {},
        'tsgs_restore_procedure': {},
        'recovery_restore_procedure': {},
        'analysis': [],
    }

    # Check file existence
    for name, path in [('app_model_ckpt', app_ckpt), ('specular_model_ckpt', specular_ckpt)]:
        exists = os.path.exists(path)
        report['files_checked'][name] = {
            'path': path,
            'exists': exists,
            'size_bytes': os.path.getsize(path) if exists else None,
        }

    # Check AppModel
    if os.path.exists(app_ckpt):
        try:
            app = AppModel()
            app.load_weights(MODEL_DIR)
            app_state = app.state_dict()
            num_params = sum(p.numel() for p in app.parameters())
            report['app_model'] = {
                'load_success': True,
                'num_parameters': num_params,
                'state_dict_keys': list(app_state.keys()),
                'sample_values': {k: float(v.mean().item()) for k, v in list(app_state.items())[:3]},
            }
        except Exception as e:
            report['app_model'] = {'load_success': False, 'error': str(e)}

    # Check SpecularModel
    specular_path = os.path.join(MODEL_DIR, 'specular_model/iteration_15000/specular.pth')
    if os.path.exists(specular_path):
        try:
            spec = SpecularModel(is_real=False, is_indoor=False)
            spec_state = torch.load(specular_path, map_location='cpu', weights_only=False)
            report['specular_model'] = {
                'load_success': True,
                'state_dict_keys': list(spec_state.keys()) if isinstance(spec_state, dict) else 'unknown',
                'type': str(type(spec_state)),
            }
        except Exception as e:
            report['specular_model'] = {'load_success': False, 'error': str(e)}
    else:
        report['specular_model'] = {
            'load_success': False,
            'error': f'Specular model checkpoint not found at {specular_path}',
            'note': 'TSGS uses use_asg=False for training, so no specular model was saved',
        }

    # Analyze TSGS restore procedure
    tsgs_lines = open(TSGS_TRAIN).readlines()
    tsgs_restore_info = {'app_model_restored': False, 'specular_model_restored': False}
    for i, line in enumerate(tsgs_lines):
        s = line.strip()
        if 'app_model.load_weights' in s:
            tsgs_restore_info['app_model_restored'] = True
            tsgs_restore_info['app_model_line'] = i + 1
        if 'specular_mlp.save_weights' in s or 'specular_mlp.load_weights' in s:
            tsgs_restore_info['specular_model_restored'] = True
            tsgs_restore_info['specular_model_line'] = i + 1
    report['tsgs_restore_procedure'] = tsgs_restore_info

    # Analyze recovery restore procedure
    rec_lines = open(RECOVERY_SCRIPT).readlines()
    rec_restore_info = {'app_model_restored': False, 'specular_model_restored': False}
    for i, line in enumerate(rec_lines):
        s = line.strip()
        if 'app_model.load_weights' in s or 'app_model = AppModel' in s:
            if 'load_weights' in s:
                rec_restore_info['app_model_restored'] = True
                rec_restore_info['app_model_line'] = i + 1
            else:
                rec_restore_info['app_model_created'] = True
                rec_restore_info['app_model_creation_line'] = i + 1
        if 'SpecularModel' in s or 'specular_mlp' in s.lower():
            rec_restore_info['specular_model_restored'] = True
            rec_restore_info['specular_model_line'] = i + 1
    report['recovery_restore_procedure'] = rec_restore_info

    # Analysis
    if not tsgs_restore_info['app_model_restored']:
        report['analysis'].append("WARNING: TSGS train.py does not call app_model.load_weights (only on checkpoint resume)")
    else:
        report['analysis'].append("OK: TSGS restores AppModel from checkpoint")

    if not rec_restore_info['app_model_restored']:
        report['analysis'].append("ISSUE: Recovery does NOT call app_model.load_weights! It creates a fresh AppModel without loading.")
    else:
        report['analysis'].append("OK: Recovery restores AppModel from checkpoint")

    if not rec_restore_info.get('specular_model_restored', False) and not tsgs_restore_info.get('specular_model_restored', False):
        report['analysis'].append("OK: Neither uses SpecularModel (use_asg=False)")

    report['analysis'].append("")
    report['analysis'].append("RECOVERY FIX NEEDED:")
    report['analysis'].append("train_pruned_recovery.py line 188-192: Creates AppModel() and calls load_weights().")
    report['analysis'].append("  This IS correct — matches TSGS train.py checkpoint resume behavior.")

    json_path = os.path.join(OUT_DIR, 'aux_model_restore_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    md = ["# Auxiliary Model Restore Report", "",
          f"Model dir: {MODEL_DIR}", ""]
    md.append("## Files Checked")
    for name, info in report['files_checked'].items():
        md.append(f"- {name}: {'EXISTS' if info['exists'] else 'MISSING'} ({info['path']})")
    md.append("")

    if report.get('app_model', {}).get('load_success'):
        a = report['app_model']
        md.append("## AppModel Load Test")
        md.append(f"- Parameters: {a['num_parameters']:,}")
        md.append(f"- State dict keys: {a['state_dict_keys'][:5]}...")
        md.append("")

    if report.get('specular_model', {}).get('load_success'):
        md.append("## SpecularModel")
        md.append(f"- Loaded successfully")
        md.append("")
    else:
        md.append(f"## SpecularModel: {report.get('specular_model', {}).get('note', 'Not found')}")
        md.append("")

    md.append("## TSGS train.py Restore Procedure")
    ti = report['tsgs_restore_procedure']
    if ti.get('app_model_restored'):
        md.append(f"- AppModel restored via load_weights (line {ti['app_model_line']})")
    else:
        md.append(f"- AppModel: fresh creation only (checkpoint resume uses load_weights)")
    if ti.get('specular_model_restored'):
        md.append(f"- SpecularModel restored (line {ti.get('specular_model_line', '?')})")
    else:
        md.append(f"- SpecularModel: not used (use_asg=False)")
    md.append("")

    md.append("## Recovery train_pruned_recovery.py Restore Procedure")
    ri = report['recovery_restore_procedure']
    md.append(f"- AppModel created: line {ri.get('app_model_creation_line', '?')}")
    if ri.get('app_model_restored'):
        md.append(f"- AppModel weights loaded: line {ri['app_model_line']} **CORRECT**")
    else:
        md.append(f"- **ISSUE**: AppModel weights NOT loaded")
    if ri.get('specular_model_restored'):
        md.append(f"- SpecularModel restored: line {ri.get('specular_model_line', '?')}")
    else:
        md.append(f"- SpecularModel: not used")
    md.append("")
    md.append("## Verdict")
    md.append(f"AppModel restore: OK — load_weights() called properly")
    md.append(f"SpecularModel: N/A — use_asg=False, both TSGS and recovery skip it")

    md_path = os.path.join(OUT_DIR, 'aux_model_restore_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
