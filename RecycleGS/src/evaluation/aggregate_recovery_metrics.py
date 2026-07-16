#!/usr/bin/env python3
"""Aggregate recovery metrics across all scenes and methods."""
import csv, json, os
from pathlib import Path

SCENES = ['scene_01', 'scene_03']
METHODS = ['schedule_control', 'random', 'low_opacity', 'low_contribution', 'mask_risk']
RATIO = 'ratio_005'

def load_metrics(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None

def main():
    out_dir = Path('/data/wyh/RecycleGS/outputs/prune_only')
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in SCENES:
        for method in METHODS:
            base_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene}/{RATIO}/baseline/render_metrics.json'
            prune_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene}/{RATIO}/{method}/render_metrics.json'
            rec_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene}/{RATIO}/{method}/recovery_500/render_metrics.json'

            base = load_metrics(base_path)
            prune = load_metrics(prune_path) if method != 'schedule_control' else None
            rec = load_metrics(rec_path)

            def get_v(d, key):
                if d is None:
                    return None
                return d.get(key, d.get({
                    'psnr': 'psnr', 'psnr_mean': 'psnr_mean',
                    'ssim': 'ssim', 'ssim_mean': 'ssim_mean',
                    'lpips': 'lpips', 'lpips_mean': 'lpips_mean',
                }.get(key, key)))

            base_psnr = base.get('psnr_mean') if base else None
            base_ssim = base.get('ssim_mean') if base else None
            base_lpips = base.get('lpips_mean') if base else None

            if method == 'schedule_control':
                imm_psnr = base_psnr
                imm_ssim = base_ssim
                imm_lpips = base_lpips
            else:
                imm_psnr = prune.get('psnr_mean') if prune else None
                imm_ssim = prune.get('ssim_mean') if prune else None
                imm_lpips = prune.get('lpips_mean') if prune else None

            rec_psnr = rec.get('psnr', rec.get('psnr_mean')) if rec else None
            rec_ssim = rec.get('ssim', rec.get('ssim_mean')) if rec else None
            rec_lpips = rec.get('lpips', rec.get('lpips_mean')) if rec else None

            delta_prune = round(imm_psnr - base_psnr, 8) if (imm_psnr is not None and base_psnr is not None) else None
            delta_recovery = round(rec_psnr - imm_psnr, 8) if (rec_psnr is not None and imm_psnr is not None) else None
            total_delta = round(rec_psnr - base_psnr, 8) if (rec_psnr is not None and base_psnr is not None) else None

            row = {
                'scene': scene,
                'method': method,
                'base_psnr': round(base_psnr, 8) if base_psnr is not None else None,
                'base_ssim': round(base_ssim, 8) if base_ssim is not None else None,
                'base_lpips': round(base_lpips, 8) if base_lpips is not None else None,
                'immediate_psnr': round(imm_psnr, 8) if imm_psnr is not None else None,
                'immediate_ssim': round(imm_ssim, 8) if imm_ssim is not None else None,
                'immediate_lpips': round(imm_lpips, 8) if imm_lpips is not None else None,
                'recovery_psnr': round(rec_psnr, 8) if rec_psnr is not None else None,
                'recovery_ssim': round(rec_ssim, 8) if rec_ssim is not None else None,
                'recovery_lpips': round(rec_lpips, 8) if rec_lpips is not None else None,
                'delta_prune': delta_prune,
                'delta_recovery': delta_recovery,
                'total_delta': total_delta,
                'imm_is_baseline': method == 'schedule_control',
            }
            rows.append(row)

    json_path = out_dir / 'stage2b_cross_scene_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {json_path}")

    csv_path = out_dir / 'stage2b_cross_scene_metrics.csv'
    fieldnames = ['scene', 'method', 'base_psnr', 'base_ssim', 'base_lpips',
                  'immediate_psnr', 'immediate_ssim', 'immediate_lpips',
                  'recovery_psnr', 'recovery_ssim', 'recovery_lpips',
                  'delta_prune', 'delta_recovery', 'total_delta', 'imm_is_baseline']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_path}")

if __name__ == '__main__':
    main()
