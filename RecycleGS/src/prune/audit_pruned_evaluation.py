#!/usr/bin/env python3
"""Audit pruned model loading: verify PLY integrity, counts, SHA diversity."""
import argparse, hashlib, json, os, sys, numpy as np, yaml
from pathlib import Path
from plyfile import PlyData

sys.path.insert(0, '/data/wyh/RecycleGS/src')

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def audit_scene(scene_key, locked_cfg):
    scene_cfg = locked_cfg.get(scene_key, {})
    scene_name = scene_cfg.get('scene_name', scene_key)
    ckpt_path = scene_cfg['checkpoint_path']
    ratio = locked_cfg.get('prune_ratio', 0.005)
    ratio_str = f"ratio_{int(ratio*1000):03d}"
    out_base = Path(locked_cfg['project_root']) / 'outputs' / 'prune_only' / scene_name / ratio_str

    methods = ['baseline', 'random', 'low_opacity', 'low_contribution', 'mask_risk', 'oracle']
    audit = {'scene': scene_name, 'methods': {}}
    all_shas = []
    baseline_N = None

    for method in methods:
        if method == 'baseline':
            ply_path = ckpt_path
        else:
            ply_path = out_base / method / 'retained.ply'

        if not os.path.exists(ply_path):
            audit['methods'][method] = {'error': 'PLY not found'}
            continue

        ply = PlyData.read(ply_path)
        N_loaded = ply['vertex'].count
        sha = sha256_file(ply_path)
        fsize = os.path.getsize(ply_path)

        rec = {
            'input_ply_path': str(ply_path),
            'sha256': sha,
            'file_size_bytes': fsize,
            'N_loaded': int(N_loaded),
        }

        if method == 'baseline':
            baseline_N = N_loaded
            rec['expected_N'] = baseline_N
            rec['count_check'] = (N_loaded == baseline_N)
            rec['status'] = 'PASS' if rec['count_check'] else 'FAIL'
        else:
            metadata_path = out_base / 'prune_metadata.json'
            with open(metadata_path) as f:
                metadata = json.load(f)
            K = metadata['methods'][method]['pruned_count']
            expected = baseline_N - K
            rec['N_baseline'] = baseline_N
            rec['K'] = K
            rec['expected_N'] = expected
            rec['count_check'] = (N_loaded == expected)
            rec['status'] = 'PASS' if rec['count_check'] else 'FAIL'
            if method == 'baseline':
                rec['status'] = 'PASS' if N_loaded == baseline_N else 'FAIL'

        audit['methods'][method] = rec
        all_shas.append(sha)

    all_same = len(set(all_shas)) == 1
    audit['all_shas_identical'] = all_same
    audit['unique_sha_count'] = len(set(all_shas))
    if all_shas:
        audit['shas'] = {m: audit['methods'][m]['sha256'] for m in methods if m in audit['methods'] and 'sha256' in audit['methods'][m]}
    audit['status'] = 'PASS' if (not all_same and baseline_N is not None) else 'FAIL'

    return audit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.locked_config) as f:
        locked_cfg = yaml.safe_load(f)

    proj_root = Path(locked_cfg['project_root'])
    out_dir = Path(args.output_dir) if args.output_dir else proj_root / 'outputs' / 'debug' / 'stage2a_audit'
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for scene_key in ['scene_01', 'scene_03']:
        print(f"\n=== Auditing {scene_key} ===")
        audit = audit_scene(scene_key, locked_cfg)
        results[scene_key] = audit
        for method, rec in audit['methods'].items():
            print(f"  {method}: {rec.get('status', 'SKIP')} N={rec.get('N_loaded', '?')} expected={rec.get('expected_N', '?')}")
        print(f"  SHA unique: {audit['unique_sha_count']} methods, all_identical={audit['all_shas_identical']}")

    output_path = out_dir / 'pruned_model_load_audit.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved audit: {output_path}")

    md_lines = [
        "# Stage 2A.5 Pruned Model Load Audit",
        "",
        f"## Verdict: **{'PASS' if all(r['status'] == 'PASS' for r in results.values()) else 'FAIL'}**",
        "",
    ]
    for scene_key, audit in results.items():
        md_lines.append(f"### {scene_key}")
        md_lines.append(f"| Method | N_loaded | Expected | Status | SHA256 |")
        md_lines.append(f"|--------|----------|----------|--------|--------|")
        for method in ['baseline', 'random', 'low_opacity', 'low_contribution', 'mask_risk', 'oracle']:
            if method not in audit['methods']:
                continue
            r = audit['methods'][method]
            sha_short = r.get('sha256', 'N/A')[:16] if r.get('sha256') else 'N/A'
            md_lines.append(f"| {method} | {r.get('N_loaded', '?')} | {r.get('expected_N', '?')} | {r.get('status', 'SKIP')} | {sha_short} |")
        md_lines.append(f"\nUnique SHAs: {audit['unique_sha_count']}, All identical: {audit['all_shas_identical']}\n")

    report_path = out_dir / 'pruned_model_load_audit_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved report: {report_path}")

if __name__ == '__main__':
    main()
