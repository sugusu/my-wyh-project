#!/usr/bin/env python3
"""Evaluate Stage 2A gate: compare mask_risk vs other methods across scenes."""
import argparse, json, os, sys, numpy as np, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

def load_method_results(scene_key, locked_cfg):
    scene_cfg = locked_cfg.get(scene_key, {})
    scene_name = scene_cfg.get('scene_name', scene_key)
    ratio = locked_cfg.get('prune_ratio', 0.005)
    ratio_str = f"ratio_{int(ratio*1000):03d}"
    base = Path(locked_cfg['project_root']) / 'outputs' / 'prune_only' / scene_name / ratio_str

    results = {}
    for method in ['random', 'low_opacity', 'low_contribution', 'mask_risk', 'oracle']:
        eval_path = base / method / 'evaluation_metrics.json'
        if eval_path.exists():
            with open(eval_path) as f:
                results[method] = json.load(f)
    return results, base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene01', type=str, default='configs/stage1/reliability_scene01.yaml')
    parser.add_argument('--scene03', type=str, default='configs/stage1/reliability_scene03.yaml')
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
    args = parser.parse_args()

    with open(args.locked_config) as f:
        locked_cfg = yaml.safe_load(f)

    out_dir = Path(locked_cfg['project_root']) / 'outputs' / 'prune_only'
    os.makedirs(out_dir, exist_ok=True)

    scene_results = {}
    for scene_key in ['scene_01', 'scene_03']:
        res, base = load_method_results(scene_key, locked_cfg)
        scene_results[scene_key] = res
        print(f"\n=== {scene_key} ===")
        for method, data in res.items():
            rm = data.get('removed_set_metrics', {})
            rv = data.get('render_metrics', {})
            print(f"  {method}: K={data.get('K', 0)}, "
                  f"removed_d_center_mean={rm.get('d_center_norm_mean', 'N/A')}, "
                  f"removed_mask_risk={rm.get('mask_risk_mean', 'N/A')}, "
                  f"PSNR={rv.get('psnr_mean', 'N/A')}")

    # Apply Gate 2A criteria
    criteria_results = {}
    for scene_key in ['scene_01', 'scene_03']:
        res = scene_results[scene_key]
        s_criteria = {}

        # 1. Removed mean GT error: mask_risk > random
        mr_rem = res.get('mask_risk', {}).get('removed_set_metrics', {}).get('d_center_norm_mean', None)
        rn_rem = res.get('random', {}).get('removed_set_metrics', {}).get('d_center_norm_mean', None)
        lo_rem = res.get('low_opacity', {}).get('removed_set_metrics', {}).get('d_center_norm_mean', None)
        lc_rem = res.get('low_contribution', {}).get('removed_set_metrics', {}).get('d_center_norm_mean', None)

        s_criteria['1_removed_gt_mask_risk_vs_random'] = {
            'mask_risk': mr_rem, 'random': rn_rem,
            'pass': mr_rem is not None and rn_rem is not None and mr_rem > rn_rem,
        }
        s_criteria['2_removed_gt_mask_risk_vs_low_opacity'] = {
            'mask_risk': mr_rem, 'low_opacity': lo_rem,
            'pass': mr_rem is not None and lo_rem is not None and mr_rem > lo_rem,
        }
        s_criteria['3_removed_gt_mask_risk_vs_low_contribution'] = {
            'mask_risk': mr_rem, 'low_contribution': lc_rem,
            'pass': mr_rem is not None and lc_rem is not None and mr_rem > lc_rem,
        }

        # 4. PSNR drop: mask_risk <= random
        mr_psnr = res.get('mask_risk', {}).get('render_metrics', {}).get('psnr_mean', None)
        rn_psnr = res.get('random', {}).get('render_metrics', {}).get('psnr_mean', None)

        s_criteria['4_psnr_drop_mask_risk_vs_random'] = {
            'mask_risk_psnr': mr_psnr, 'random_psnr': rn_psnr,
            'pass': mr_psnr is not None and rn_psnr is not None and mr_psnr >= rn_psnr,
        }

        criteria_results[scene_key] = s_criteria

    # Cross-scene consistency
    cross_scene = {}
    for criterion_key in ['1', '2', '3', '4']:
        s01_pass = criteria_results.get('scene_01', {}).get(f'{criterion_key}_removed_gt_mask_risk_vs_random' if criterion_key == '1' else
                                                            f'{criterion_key}_removed_gt_mask_risk_vs_low_opacity' if criterion_key == '2' else
                                                            f'{criterion_key}_removed_gt_mask_risk_vs_low_contribution' if criterion_key == '3' else
                                                            f'{criterion_key}_psnr_drop_mask_risk_vs_random', {}).get('pass', False)
        # Map correct key
        pass

    # Build proper cross-scene keys
    for ck in ['1_removed_gt_mask_risk_vs_random', '2_removed_gt_mask_risk_vs_low_opacity',
               '3_removed_gt_mask_risk_vs_low_contribution', '4_psnr_drop_mask_risk_vs_random']:
        s01_pass = criteria_results.get('scene_01', {}).get(ck, {}).get('pass', False)
        s03_pass = criteria_results.get('scene_03', {}).get(ck, {}).get('pass', False)
        cross_scene[ck] = {
            'scene_01_pass': s01_pass,
            'scene_03_pass': s03_pass,
            'consistent': s01_pass == s03_pass,
            'all_pass': s01_pass and s03_pass,
        }

    overall_pass = all(v['all_pass'] for v in cross_scene.values())

    # Save cross-scene metrics JSON
    metrics = {
        'per_scene_criteria': criteria_results,
        'cross_scene': cross_scene,
        'overall_pass': overall_pass,
    }
    metrics_path = out_dir / 'stage2a_cross_scene_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved cross-scene metrics: {metrics_path}")

    # Save CSV table
    csv_lines = [
        "criterion,scene_01_pass,scene_03_pass,consistent,all_pass"
    ]
    for ck, v in cross_scene.items():
        csv_lines.append(f"{ck},{v['scene_01_pass']},{v['scene_03_pass']},{v['consistent']},{v['all_pass']}")
    csv_path = out_dir / 'stage2a_cross_scene_table.csv'
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"Saved CSV: {csv_path}")

    # Generate MD report
    md_lines = [
        f"# Stage 2A Gate Report",
        f"",
        f"## Verdict: {'**PASS**' if overall_pass else '**FAIL**'}",
        f"",
        f"## Per-Scene Results",
        f"",
    ]
    for scene_key in ['scene_01', 'scene_03']:
        res = scene_results[scene_key]
        md_lines.extend([
            f"### {scene_key}",
            f"| Method | K | Removed d_center_norm (mean) | Removed mask_risk (mean) | PSNR | SSIM |",
            f"|--------|---|---------------------------|-------------------------|------|------|",
        ])
        baseline_psnr = res.get('baseline', {}).get('render_metrics', {}).get('psnr_mean', 'N/A')
        for method in ['mask_risk', 'random', 'low_opacity', 'low_contribution', 'oracle']:
            if method not in res:
                continue
            d = res[method]
            rm = d.get('removed_set_metrics', {})
            rv = d.get('render_metrics', {})
            k = d.get('K', 0)
            dc = rm.get('d_center_norm_mean', 'N/A')
            mr = rm.get('mask_risk_mean', 'N/A')
            psnr = rv.get('psnr_mean', 'N/A')
            ssim = rv.get('ssim_mean', 'N/A')
            if isinstance(dc, float):
                dc = f"{dc:.6f}"
            if isinstance(mr, float):
                mr = f"{mr:.4f}"
            if isinstance(psnr, float):
                psnr = f"{psnr:.2f}"
            if isinstance(ssim, float):
                ssim = f"{ssim:.4f}"
            md_lines.append(f"| {method} | {k} | {dc} | {mr} | {psnr} | {ssim} |")
        md_lines.append("")

    md_lines.extend([
        f"## Gate 2A Criteria",
        f"",
        f"| Criterion | Scene 01 | Scene 03 | Consistent | All Pass |",
        f"|-----------|----------|----------|------------|----------|",
    ])
    for ck, v in cross_scene.items():
        label = ck.replace('_', ' ').replace('1 ', '1. ').replace('2 ', '2. ').replace('3 ', '3. ').replace('4 ', '4. ').title()
        md_lines.append(f"| {label} | {'PASS' if v['scene_01_pass'] else 'FAIL'} | {'PASS' if v['scene_03_pass'] else 'FAIL'} | {'Yes' if v['consistent'] else 'No'} | {'PASS' if v['all_pass'] else 'FAIL'} |")

    md_lines.extend([
        f"",
        f"## Overall Verdict",
        f"",
        f"**{'PASS - Mask-risk pruning passes Gate 2A' if overall_pass else 'FAIL - Mask-risk pruning does not pass Gate 2A'}**",
        f"",
        f"### Criteria Details",
    ])
    for ck, v in cross_scene.items():
        if not v['all_pass']:
            md_lines.append(f"- {ck}: FAIL")
    if overall_pass:
        md_lines.append("- All criteria pass across both scenes")

    md_lines.append("")
    report_path = out_dir / 'stage2a_gate_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved gate report: {report_path}")
    print(f"\n{'='*60}")
    print(f"Stage 2A Gate Verdict: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
