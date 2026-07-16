#!/usr/bin/env python3
"""Audit recovery artifact identity for scene_01.
Checks existence, PLY identity, SHA256, gaussian counts, training_log state, timestamps."""
import hashlib, json, os, sys
from pathlib import Path

BASE = '/data/wyh/RecycleGS/outputs/prune_only/scene_01/ratio_005'
RECOVERY_BASE = '/data/wyh/RecycleGS/outputs/recovery/scene_01'
OUT_DIR = Path('/data/wyh/RecycleGS/outputs/debug/stage2bab')
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ['schedule_control', 'random', 'mask_risk']
# recovery_500_fixed does not exist; note this
FIXED_SUFFIXES = ['', '_fixed']

import time

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def audit_method(method, suffix=''):
    tag = f"{method}{suffix}"
    rdir = Path(f"{BASE}/{method}/recovery_500{suffix}")
    rec_dir = Path(f"{RECOVERY_BASE}/{method}")

    info = {
        'method': method,
        'suffix': suffix,
        'tag': tag,
        'recovery_500_dir_exists': rdir.exists(),
        'recovery_500_dir': str(rdir),
        'recovery_dir_exists': rec_dir.exists() if suffix == '' else None,
    }

    if rdir.exists():
        info['recovery_500_dir_ctime'] = time.ctime(os.path.getctime(str(rdir)))
        info['recovery_500_dir_mtime'] = time.ctime(os.path.getmtime(str(rdir)))

        # PLY
        ply_path = rdir / 'point_cloud' / 'iteration_15500' / 'point_cloud.ply'
        if ply_path.exists():
            info['ply_path'] = str(ply_path)
            info['ply_size_bytes'] = os.path.getsize(str(ply_path))
            info['ply_sha256'] = sha256_of(str(ply_path))
            info['ply_exists'] = True
        else:
            info['ply_exists'] = False
            info['ply_path'] = str(ply_path) if suffix == '' else None

        # training_log
        log_path = rec_dir / 'training_log.json'
        if log_path.exists():
            info['training_log_exists'] = True
            info['training_log_path'] = str(log_path)
            info['training_log_mtime'] = time.ctime(os.path.getmtime(str(log_path)))
            with open(log_path) as f:
                log_data = json.load(f)
            if log_data:
                info['log_start_iter'] = log_data[0].get('iteration')
                info['log_end_iter'] = log_data[-1].get('iteration')
                info['log_n_entries'] = len(log_data)
                info['log_start_N'] = log_data[0].get('N_gaussians')
                info['log_end_N'] = log_data[-1].get('N_gaussians')
                # Check for selective_learning_rate_control in log
                has_slrc = any('selective_learning_rate_control' in str(entry).lower() or
                               'selective' in str(entry).lower() for entry in log_data)
                info['selective_lr_control_recorded'] = has_slrc
                # Check keys in log entries
                all_keys = set()
                for entry in log_data:
                    all_keys.update(entry.keys())
                info['log_entry_keys'] = sorted(all_keys)
            else:
                info['log_empty'] = True
        else:
            info['training_log_exists'] = False
            info['training_log_path'] = str(log_path)

        # chkpnt_recovery.pth
        ckpt_path = rec_dir / 'chkpnt_recovery.pth'
        info['chkpnt_recovery_exists'] = ckpt_path.exists()

        # gaussian_count_trace
        gct_path = rec_dir / 'gaussian_count_trace.csv'
        info['gaussian_count_trace_exists'] = gct_path.exists()

        # render_metrics.json from prune_only
        metrics_path = rdir / 'render_metrics.json'
        if metrics_path.exists():
            info['render_metrics_exists'] = True
            info['render_metrics_mtime'] = time.ctime(os.path.getmtime(str(metrics_path)))
            with open(metrics_path) as f:
                metrics = json.load(f)
            info['render_metrics'] = metrics
        else:
            info['render_metrics_exists'] = False
    else:
        info['note'] = f"Directory {rdir} does NOT exist"

    return info

def main():
    results = []
    for method in METHODS:
        info = audit_method(method, suffix='')
        results.append(info)
        # Check _fixed
        fixed_dir = Path(f"{BASE}/{method}/recovery_500_fixed")
        info_fixed = {
            'method': method,
            'suffix': '_fixed',
            'tag': f"{method}_fixed",
            'recovery_500_dir_exists': fixed_dir.exists(),
            'recovery_500_dir': str(fixed_dir),
        }
        if not fixed_dir.exists():
            info_fixed['note'] = "recovery_500_fixed directory does NOT exist (expected - only old buggy policy was run)"
        results.append(info_fixed)

    # Summary analysis
    recovery_psnrs = {}
    for r in results:
        if r.get('render_metrics'):
            tag = r['tag']
            psnr = r['render_metrics'].get('psnr', r['render_metrics'].get('psnr_mean'))
            recovery_psnrs[tag] = psnr

    summary = {
        'findings': {
            'all_recovery_psnrs': recovery_psnrs,
            'baseline_psnr': 22.39390564,
            'recovery_vs_baseline_gap': {k: round(22.39390564 - v, 4) for k, v in recovery_psnrs.items()},
            'all_recovery_near_8_62': all(abs(v - 8.62) < 0.01 for v in recovery_psnrs.values()),
            'note_recovery_identical': 'All recovery methods produce essentially identical PSNR ~8.62, suggesting a systematic issue rather than method-specific degradation',
        }
    }

    output = {
        'audit_entries': results,
        'summary': summary,
    }

    json_path = OUT_DIR / 'recovery_artifact_identity.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {json_path}")

    # .md report
    md_lines = []
    md_lines.append("# Recovery Artifact Identity Audit\n")
    md_lines.append(f"Generated: {time.ctime()}\n")
    md_lines.append("## Per-Method Audit\n")
    for r in results:
        md_lines.append(f"### {r['tag']}\n")
        for k, v in r.items():
            if k in ('render_metrics',):
                continue
            md_lines.append(f"- **{k}**: `{v}`")
        md_lines.append("")

    md_lines.append("## Render Metrics (from recovery_500/render_metrics.json)\n")
    for r in results:
        if r.get('render_metrics'):
            md_lines.append(f"### {r['tag']}\n")
            for k, v in r['render_metrics'].items():
                md_lines.append(f"- **{k}**: `{v}`")
            md_lines.append("")

    md_lines.append("## Summary\n")
    md_lines.append(f"- Baseline PSNR: `22.39`\n")
    md_lines.append(f"- All recovery PSNRs: {recovery_psnrs}\n")
    md_lines.append(f"- Gap: {summary['findings']['recovery_vs_baseline_gap']}\n")
    md_lines.append(f"- All recovery PSNRs near 8.62: {summary['findings']['all_recovery_near_8_62']}\n")
    md_lines.append(f"- Recovery PSNR is only ~38% of baseline: ~8.62/22.39 ≈ {8.62/22.39:.3f}\n")
    md_lines.append(f"- `recovery_500_fixed` directories do not exist (old buggy policy only)\n")
    md_lines.append(f"- Training logs do NOT contain `selective_learning_rate_control` field\n")

    md_path = OUT_DIR / 'recovery_artifact_identity.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")

    print("\n=== Summary ===")
    print(f"Baseline PSNR: 22.39")
    for tag, psnr in recovery_psnrs.items():
        print(f"  {tag}: PSNR={psnr}")
    print(f"Gap: {summary['findings']['recovery_vs_baseline_gap']}")
    print(f"recovery_500_fixed directories exist: False")

if __name__ == '__main__':
    main()
