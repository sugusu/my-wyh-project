#!/usr/bin/env python3
"""Analyze train_pruned_recovery.py restore call order vs TSGS official train.py."""
import json, os, sys

sys.path.insert(0, '/data/wyh/RecycleGS/src')

TRAIN_SCRIPT = '/data/wyh/RecycleGS/src/prune/train_pruned_recovery.py'
TSGS_TRAIN = '/data/wyh/repos/TSGS/train.py'
OUT_DIR = '/data/wyh/RecycleGS/outputs/debug/stage2b_recovery_collapse'

def extract_flow_tsgs():
    """Extract the initialization + training flow from TSGS train.py"""
    with open(TSGS_TRAIN) as f:
        lines = f.readlines()
    
    flow = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('gaussians = GaussianModel'):
            flow.append(('CREATE_GAUSSIANS', i+1, stripped))
        elif stripped.startswith('scene = Scene'):
            flow.append(('CREATE_SCENE', i+1, stripped[:100]))
        elif stripped.startswith('gaussians.training_setup'):
            flow.append(('TRAINING_SETUP', i+1, stripped))
        elif stripped.startswith('gaussians.restore'):
            flow.append(('RESTORE', i+1, stripped[:100]))
        elif stripped.startswith('app_model = AppModel'):
            flow.append(('CREATE_APP_MODEL', i+1, stripped))
        elif stripped.startswith('app_model.load_weights'):
            flow.append(('LOAD_APP_WEIGHTS', i+1, stripped))
        elif 'gaussians.optimizer' in stripped and 'step' in stripped:
            flow.append(('OPTIMIZER_STEP', i+1, stripped))
        elif 'update_learning_rate' in stripped:
            flow.append(('UPDATE_LR', i+1, stripped))
        elif stripped.startswith('render_pkg = render'):
            flow.append(('RENDER', i+1, stripped[:100]))
        elif 'loss.backward' in stripped:
            flow.append(('BACKWARD', i+1, stripped))
    return flow

def extract_flow_recovery():
    """Extract the initialization + training flow from train_pruned_recovery.py"""
    with open(TRAIN_SCRIPT) as f:
        lines = f.readlines()
    
    flow = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('gaussians = GaussianModel'):
            flow.append(('CREATE_GAUSSIANS', i+1, stripped))
        elif stripped.startswith('gaussians.restore'):
            flow.append(('RESTORE', i+1, stripped[:100]))
        elif stripped.startswith('scene = Scene'):
            flow.append(('CREATE_SCENE', i+1, stripped[:100]))
        elif stripped.startswith('app_model = AppModel'):
            flow.append(('CREATE_APP_MODEL', i+1, stripped))
        elif stripped.startswith('app_model.load_weights'):
            flow.append(('LOAD_APP_WEIGHTS', i+1, stripped))
        elif 'gaussians.optimizer' in stripped and 'step' in stripped:
            flow.append(('OPTIMIZER_STEP', i+1, stripped))
        elif 'update_learning_rate' in stripped:
            flow.append(('UPDATE_LR', i+1, stripped))
        elif stripped.startswith('render_pkg = render'):
            flow.append(('RENDER', i+1, stripped[:100]))
        elif 'loss.backward' in stripped:
            flow.append(('BACKWARD', i+1, stripped))
        elif 'prune' in stripped and 'mask' in stripped and 'gaussians' in stripped:
            flow.append(('PRUNE', i+1, stripped[:100]))
    return flow

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tsgs_flow = extract_flow_tsgs()
    rec_flow = extract_flow_recovery()

    report = {
        'tsgs_train_flow': [{'step': s[0], 'line': s[1], 'code': s[2]} for s in tsgs_flow],
        'recovery_flow': [{'step': s[0], 'line': s[1], 'code': s[2]} for s in rec_flow],
        'analysis': {},
    }

    # Key analysis: compare restore order
    tsgs_restore_idx = None
    tsgs_scene_idx = None
    tsgs_setup_idx = None
    for i, s in enumerate(tsgs_flow):
        if s[0] == 'RESTORE':
            tsgs_restore_idx = i
        if s[0] == 'CREATE_SCENE':
            tsgs_scene_idx = i
        if s[0] == 'TRAINING_SETUP':
            tsgs_setup_idx = i

    rec_restore_indices = [i for i, s in enumerate(rec_flow) if s[0] == 'RESTORE']
    rec_scene_indices = [i for i, s in enumerate(rec_flow) if s[0] == 'CREATE_SCENE']

    # Check if update_learning_rate is called in recovery
    has_lr_update = any(s[0] == 'UPDATE_LR' for s in rec_flow)
    tsgs_has_lr_update = any(s[0] == 'UPDATE_LR' for s in tsgs_flow)

    report['analysis'] = {
        'tsgs_restore_index_in_flow': tsgs_restore_idx,
        'tsgs_scene_created_before_restore': tsgs_scene_idx < tsgs_restore_idx if (tsgs_scene_idx is not None and tsgs_restore_idx is not None) else 'unknown',
        'tsgs_setup_before_restore': tsgs_setup_idx < tsgs_restore_idx if (tsgs_setup_idx is not None and tsgs_restore_idx is not None) else 'unknown',
        'recovery_num_restore_calls': len(rec_restore_indices),
        'recovery_restore_line_numbers': [rec_flow[i][1] for i in rec_restore_indices],
        'recovery_scene_lines': [rec_flow[i][1] for i in rec_scene_indices],
        'recovery_restore_after_scene': any(ri > si for ri in rec_restore_indices for si in rec_scene_indices),
        'recovery_has_update_learning_rate': has_lr_update,
        'tsgs_has_update_learning_rate': tsgs_has_lr_update,
        'recovery_missing_lr_update': not has_lr_update and tsgs_has_lr_update,
    }

    # Specific issues found
    issues = []
    if report['analysis']['recovery_missing_lr_update']:
        issues.append("CRITICAL: recovery training does NOT call update_learning_rate(), but TSGS train.py does. "
                      "This means position LR stays constant at the restored value instead of decaying.")
    if report['analysis']['recovery_num_restore_calls'] > 1:
        issues.append(f"WARNING: recovery calls restore() {report['analysis']['recovery_num_restore_calls']} times "
                      f"(lines {report['analysis']['recovery_restore_line_numbers']}). "
                      f"Each call re-creates the optimizer via training_setup() inside restore().")
    if report['analysis']['recovery_restore_after_scene']:
        issues.append("NOTE: recovery calls restore() AFTER Scene() constructor. "
                      "Scene.load_ply overwrites gaussian params, but the subsequent restore() should fix them.")

    report['analysis']['issues'] = issues

    json_path = os.path.join(OUT_DIR, 'restore_call_order_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    # MD
    md = ["# Restore Call Order Audit", "",
          f"Recovery script: {TRAIN_SCRIPT}", f"TSGS train.py: {TSGS_TRAIN}", ""]
    md.append("## TSGS train.py Flow")
    for s in tsgs_flow:
        md.append(f"  {s[0]:20s} line {s[1]:4d}: {s[2]}")
    md.append("")
    md.append("## Recovery train_pruned_recovery.py Flow")
    for s in rec_flow:
        md.append(f"  {s[0]:20s} line {s[1]:4d}: {s[2]}")
    md.append("")

    md.append("## Analysis")
    a = report['analysis']
    md.append(f"- TSGS restore at flow index: {a['tsgs_restore_index_in_flow']}")
    md.append(f"- TSGS Scene created before restore: {a['tsgs_scene_created_before_restore']}")
    md.append(f"- TSGS training_setup before restore: {a['tsgs_setup_before_restore']}")
    md.append(f"- Recovery restore calls: {a['recovery_num_restore_calls']} (lines {a['recovery_restore_line_numbers']})")
    md.append(f"- Recovery Scene created at lines: {a['recovery_scene_lines']}")
    md.append(f"- Recovery restore after Scene: {a['recovery_restore_after_scene']}")
    md.append(f"- Recovery has update_learning_rate: {a['recovery_has_update_learning_rate']}")
    md.append(f"- TSGS has update_learning_rate: {a['tsgs_has_update_learning_rate']}")
    md.append("")

    md.append("## Issues Found")
    for issue in issues:
        md.append(f"- {issue}")

    md_path = os.path.join(OUT_DIR, 'restore_call_order_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
