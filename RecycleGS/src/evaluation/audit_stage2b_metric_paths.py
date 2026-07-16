#!/usr/bin/env python3
"""Audit stage 2B metric paths: check render_metrics.json files exist and have valid content."""
import json, os
from pathlib import Path

SCENES = ['scene_01', 'scene_03']
METHODS = ['schedule_control', 'random', 'low_opacity', 'low_contribution', 'mask_risk']
RATIO = 'ratio_005'

def check_json(path, label):
    info = {'path': str(path), 'label': label, 'file_exists': os.path.isfile(path)}
    if info['file_exists']:
        with open(path) as f:
            try:
                data = json.load(f)
            except Exception as e:
                info['parse_error'] = str(e)
                return info
        info['keys'] = list(data.keys())
        for k in ['psnr', 'psnr_mean', 'ssim', 'ssim_mean', 'lpips', 'lpips_mean']:
            if k in data:
                info[f'{k}_value'] = data[k]
    return info

def main():
    out_dir = Path('/data/wyh/RecycleGS/outputs/debug/stage2b_eval')
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for scene in SCENES:
        for method in METHODS:
            prune_base = f'/data/wyh/RecycleGS/outputs/prune_only/{scene}/{RATIO}'
            rec_base = f'/data/wyh/RecycleGS/outputs/recovery/{scene}/{method}'

            # baseline
            records.append(check_json(f'{prune_base}/baseline/render_metrics.json', f'{scene}/baseline'))
            # immediate (prune method)
            if method != 'schedule_control':
                records.append(check_json(f'{prune_base}/{method}/render_metrics.json', f'{scene}/{method}/immediate'))
            # recovery
            records.append(check_json(f'{prune_base}/{method}/recovery_500/render_metrics.json', f'{scene}/{method}/recovery'))

    json_path = out_dir / 'metric_path_audit.json'
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {json_path}")

    lines = ["# Metric Path Audit", "", "| Label | File Exists | Keys | PSNR | SSIM | LPIPS |", "|-------|-------------|------|------|------|-------|"]
    for r in records:
        psnr_val = r.get('psnr_mean', r.get('psnr', 'N/A'))
        ssim_val = r.get('ssim_mean', r.get('ssim', 'N/A'))
        lpips_val = r.get('lpips_mean', r.get('lpips', 'N/A'))
        if isinstance(psnr_val, float): psnr_val = f"{psnr_val:.8f}"
        if isinstance(ssim_val, float): ssim_val = f"{ssim_val:.8f}"
        if isinstance(lpips_val, float): lpips_val = f"{lpips_val:.8f}"
        exists = "YES" if r['file_exists'] else "NO"
        keys = ', '.join(r.get('keys', [])) if r.get('keys') else 'N/A'
        lines.append(f"| {r['label']} | {exists} | {keys} | {psnr_val} | {ssim_val} | {lpips_val} |")
    lines.append("")

    md_path = out_dir / 'metric_path_audit.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
