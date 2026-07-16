#!/usr/bin/env python3
"""Audit recovery render load: verify all 10 recovery render results are valid."""
import hashlib, json, os
from pathlib import Path

SCENES = ['scene_01', 'scene_03']
METHODS = ['schedule_control', 'random', 'low_opacity', 'low_contribution', 'mask_risk']
RATIO = 'ratio_005'
ITERATION = 15500

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def gaussian_count_from_ply(path):
    from plyfile import PlyData
    return PlyData.read(path)['vertex'].count

def check_recovery(scene, method):
    entry = {'scene': scene, 'method': method}

    ply_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene}/{RATIO}/{method}/recovery_500/point_cloud/iteration_{ITERATION}/point_cloud.ply'
    entry['ply_path'] = ply_path
    entry['ply_exists'] = os.path.isfile(ply_path)
    if entry['ply_exists']:
        entry['ply_sha256'] = sha256_of(ply_path)
        entry['gaussian_count'] = int(gaussian_count_from_ply(ply_path))
    else:
        entry['ply_sha256'] = None
        entry['gaussian_count'] = None

    metrics_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene}/{RATIO}/{method}/recovery_500/render_metrics.json'
    entry['metrics_path'] = metrics_path
    entry['metrics_exists'] = os.path.isfile(metrics_path)
    if entry['metrics_exists']:
        with open(metrics_path) as f:
            data = json.load(f)
        entry['metrics_keys'] = list(data.keys())
        entry['psnr'] = data.get('psnr', data.get('psnr_mean'))
        entry['ssim'] = data.get('ssim', data.get('ssim_mean'))
        entry['lpips'] = data.get('lpips', data.get('lpips_mean'))
        entry['num_views'] = data.get('num_views', data.get('n_views'))
        entry['model_ply_path'] = data.get('model_ply_path')
        entry['model_ply_sha256'] = data.get('model_ply_sha256')
        entry['gaussian_count_metrics'] = data.get('gaussian_count')
        entry['all_values_finite'] = all(
            v is not None and isinstance(v, (int, float)) and v == v
            for v in [entry['psnr'], entry['ssim'], entry['lpips']]
        )
        entry['num_views_valid'] = isinstance(entry['num_views'], (int, float)) and entry['num_views'] > 0
    else:
        entry['metrics_keys'] = None
        entry['psnr'] = None
        entry['ssim'] = None
        entry['lpips'] = None
        entry['num_views'] = None
        entry['all_values_finite'] = False
        entry['num_views_valid'] = False

    sha_match = entry.get('model_ply_sha256') == entry.get('ply_sha256') if entry.get('model_ply_sha256') and entry.get('ply_sha256') else None
    entry['sha_matches'] = sha_match

    gc_match = entry.get('gaussian_count_metrics') == entry.get('gaussian_count') if entry.get('gaussian_count_metrics') is not None and entry.get('gaussian_count') is not None else None
    entry['gaussian_count_matches'] = gc_match

    return entry

def main():
    out_dir = Path('/data/wyh/RecycleGS/outputs/debug/stage2b_eval')
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for scene in SCENES:
        for method in METHODS:
            records.append(check_recovery(scene, method))

    json_path = out_dir / 'recovery_render_load_audit.json'
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {json_path}")

    lines = ["# Recovery Render Load Audit", "", "| Scene | Method | PLY Exists | Metrics Exists | PSNR | SSIM | LPIPS | Views>0 | Finite | SHA Matches | Gaussian Count Match |", "|-------|--------|------------|----------------|------|------|-------|---------|--------|-------------|---------------------|"]
    for r in records:
        pe = "YES" if r['ply_exists'] else "NO"
        me = "YES" if r['metrics_exists'] else "NO"
        psnr = f"{r['psnr']:.8f}" if r['psnr'] is not None else "N/A"
        ssim = f"{r['ssim']:.8f}" if r['ssim'] is not None else "N/A"
        lpips = f"{r['lpips']:.8f}" if r['lpips'] is not None else "N/A"
        nv = "YES" if r['num_views_valid'] else "NO"
        fin = "YES" if r['all_values_finite'] else "NO"
        sha = "YES" if r.get('sha_matches') else ("N/A" if r.get('sha_matches') is None else "NO")
        gc = "YES" if r.get('gaussian_count_matches') else ("N/A" if r.get('gaussian_count_matches') is None else "NO")
        lines.append(f"| {r['scene']} | {r['method']} | {pe} | {me} | {psnr} | {ssim} | {lpips} | {nv} | {fin} | {sha} | {gc} |")
    lines.append("")

    md_path = out_dir / 'recovery_render_load_audit.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
