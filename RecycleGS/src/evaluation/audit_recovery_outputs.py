#!/usr/bin/env python3
"""Audit recovery outputs: check all PLY files and training logs exist."""
import hashlib, json, os, sys
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')

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

def audit_recovery(scene, method):
    entry = {'scene': scene, 'method': method}
    recovery_dir = f'/data/wyh/RecycleGS/outputs/recovery/{scene}/{method}'
    entry['recovery_dir'] = recovery_dir

    ply_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene}/{RATIO}/{method}/recovery_500/point_cloud/iteration_{ITERATION}/point_cloud.ply'
    entry['ply_path'] = ply_path
    entry['ply_exists'] = os.path.isfile(ply_path)
    if entry['ply_exists']:
        entry['ply_sha256'] = sha256_of(ply_path)
        entry['gaussian_count'] = int(gaussian_count_from_ply(ply_path))
    else:
        entry['ply_sha256'] = None
        entry['gaussian_count'] = None

    log_path = os.path.join(recovery_dir, 'training_log.json')
    entry['training_log_exists'] = os.path.isfile(log_path)
    if entry['training_log_exists']:
        with open(log_path) as f:
            log = json.load(f)
        entry['training_start_iteration'] = log[0]['iteration'] if log else None
        entry['training_end_iteration'] = log[-1]['iteration'] if log else None
    else:
        entry['training_start_iteration'] = None
        entry['training_end_iteration'] = None

    return entry

def main():
    out_dir = Path('/data/wyh/RecycleGS/outputs/debug/stage2b_eval')
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for scene in SCENES:
        for method in METHODS:
            records.append(audit_recovery(scene, method))

    json_path = out_dir / 'recovery_output_inventory.json'
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {json_path}")

    lines = ["# Recovery Output Inventory", "", "| Scene | Method | PLY Exists | Gaussian Count | Training Log | Start Iter | End Iter |", "|-------|--------|------------|----------------|--------------|------------|----------|"]
    for r in records:
        pe = "YES" if r['ply_exists'] else "NO"
        te = "YES" if r['training_log_exists'] else "NO"
        gc = str(r['gaussian_count']) if r['gaussian_count'] is not None else "N/A"
        si = str(r['training_start_iteration']) if r['training_start_iteration'] is not None else "N/A"
        ei = str(r['training_end_iteration']) if r['training_end_iteration'] is not None else "N/A"
        lines.append(f"| {r['scene']} | {r['method']} | {pe} | {gc} | {te} | {si} | {ei} |")
    lines.append("")

    md_path = out_dir / 'recovery_output_inventory.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
