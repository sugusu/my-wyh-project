#!/usr/bin/env python3
"""Gate 2B Final: unified metrics reader with proper PASS/CONDITIONAL/FAIL rules."""
import json, os, sys, numpy as np
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

SCENES = ['scene_01']
METHODS = ['schedule_control', 'random', 'mask_risk']
RATIO = 'ratio_005'

def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def main():
    out_dir = Path('/data/wyh/RecycleGS/outputs/prune_only')
    out_dir.mkdir(parents=True, exist_ok=True)

    unified_dir = Path('/data/wyh/RecycleGS/outputs/debug/stage2b_eval')

    criteria = {}
    for scene_name in SCENES:
        scene_criteria = {}

        # Baseline from unified evaluator
        base_path = unified_dir / f'unified_baseline_{scene_name.replace("_", "")}.json'
        base_rm = load_json(base_path)

        # Immediate prune metrics from existing cross_method_evaluation
        cross_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene_name}/{RATIO}/cross_method_evaluation.json'
        cross_data = load_json(cross_path)

        for method in METHODS:
            if method == 'schedule_control':
                imm_psnr = base_rm.get('psnr_mean') if base_rm else cross_data.get('baseline', {}).get('render_metrics', {}).get('psnr_mean')
            else:
                imm_rm = cross_data.get(method, {}).get('render_metrics', {}) if cross_data else {}
                imm_psnr = imm_rm.get('psnr_mean')

            short_scene = scene_name.replace('_', '')
            method_short = method.replace('_', '').replace('schedulecontrol', 'sc')
            rec_rm = load_json(unified_dir / f'{short_scene}_{method_short}500_final.json')
            rec_psnr = rec_rm.get('psnr_mean') if rec_rm else None
            rec_gaussians = rec_rm.get('actual_loaded_count') if rec_rm else None

            base_psnr = base_rm.get('psnr_mean') if base_rm else None

            # C1: recovery PSNR >= prune PSNR (for pruned methods)
            if method == 'schedule_control':
                c1_pass = 'N/A'
            elif rec_psnr is not None and imm_psnr is not None:
                c1_pass = bool(rec_psnr >= imm_psnr)
            else:
                c1_pass = 'PENDING_EVALUATION'

            # C2: schedule_control PSNR stable (delta within 0.5 dB)
            if method == 'schedule_control':
                if rec_psnr is not None and base_psnr is not None:
                    psnr_delta = abs(base_psnr - rec_psnr)
                    c2_pass = bool(psnr_delta <= 0.5)
                else:
                    c2_pass = 'PENDING_EVALUATION'
            else:
                c2_pass = 'N/A'

            # C3: mask_risk recovery PSNR >= random - 0.01
            c3_pass = None

            # C4: Gaussian count stable
            training_log_path = f'/data/wyh/RecycleGS/outputs/recovery/{scene_name}/{method}/training_log.json'
            log = load_json(training_log_path)
            if log:
                first_n = log[0]['N_gaussians']
                last_n = log[-1]['N_gaussians']
                c4_pass = bool(first_n == last_n)
            else:
                c4_pass = 'PENDING_EVALUATION'

            # C5: deletion quality (mask_risk removed d_center_norm > random)
            c5_pass = 'PENDING_EVALUATION'

            scene_criteria[method] = {
                'C1_recovery_improves_over_prune': {
                    'prune_psnr': imm_psnr,
                    'rec_psnr': rec_psnr,
                    'pass': c1_pass,
                },
                'C2_schedule_control_stable': {
                    'base_psnr': base_psnr,
                    'rec_psnr': rec_psnr,
                    'psnr_delta': round(abs(base_psnr - rec_psnr), 4) if (base_psnr is not None and rec_psnr is not None) else None,
                    'pass': c2_pass,
                },
                'C3_mask_risk_vs_random': {
                    'mask_risk_rec_psnr': None,
                    'random_rec_psnr': None,
                    'pass': None,
                },
                'C4_gaussian_count_stable': {
                    'first_n': first_n if log else None,
                    'last_n': last_n if log else None,
                    'pass': c4_pass,
                },
                'C5_deletion_quality': {
                    'pass': c5_pass,
                },
                'raw_values': {
                    'base_psnr': base_psnr,
                    'prune_psnr': imm_psnr if method != 'schedule_control' else base_psnr,
                    'rec_psnr': rec_psnr,
                    'rec_gaussians': rec_gaussians,
                },
            }

        criteria[scene_name] = scene_criteria

    # Cross-scene C3: mask_risk vs random
    for scene_name in SCENES:
        mr = criteria[scene_name].get('mask_risk', {})
        rn = criteria[scene_name].get('random', {})
        mr_rec = mr.get('raw_values', {}).get('rec_psnr')
        rn_rec = rn.get('raw_values', {}).get('rec_psnr')
        if mr_rec is not None and rn_rec is not None:
            passed = mr_rec >= rn_rec - 0.01
            mr['C3_mask_risk_vs_random'] = {
                'mask_risk_rec_psnr': mr_rec,
                'random_rec_psnr': rn_rec,
                'pass': passed,
            }
        else:
            mr['C3_mask_risk_vs_random'] = {
                'mask_risk_rec_psnr': mr_rec,
                'random_rec_psnr': rn_rec,
                'pass': 'PENDING_EVALUATION',
            }

    # Cross-scene C5: deletion quality
    for scene_name in SCENES:
        cross_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene_name}/{RATIO}/cross_method_evaluation.json'
        cross_data = load_json(cross_path)
        if cross_data:
            mr_removed = cross_data.get('mask_risk', {}).get('removed_set_metrics', {})
            rn_removed = cross_data.get('random', {}).get('removed_set_metrics', {})
            mr_dcenter = mr_removed.get('d_center_norm_mean')
            rn_dcenter = rn_removed.get('d_center_norm_mean')
            if mr_dcenter is not None and rn_dcenter is not None:
                c5_pass = mr_dcenter > rn_dcenter
            else:
                c5_pass = 'PENDING_EVALUATION'
        else:
            c5_pass = 'PENDING_EVALUATION'
        for method in ['mask_risk', 'random']:
            criteria[scene_name][method]['C5_deletion_quality']['pass'] = c5_pass

    # Compute overall verdict
    def is_pass(v):
        return v is True
    def is_pending(v):
        return v == 'PENDING_EVALUATION'

    all_c1 = []
    all_c2 = []
    all_c4 = []
    for scene_name in SCENES:
        for method in METHODS:
            c1 = criteria[scene_name][method]['C1_recovery_improves_over_prune']['pass']
            c2 = criteria[scene_name][method]['C2_schedule_control_stable']['pass']
            c4 = criteria[scene_name][method]['C4_gaussian_count_stable']['pass']
            if c1 != 'N/A':
                all_c1.append(c1)
            if c2 != 'N/A':
                all_c2.append(c2)
            all_c4.append(c4)

    has_metrics = any(isinstance(x, bool) for x in all_c1)
    all_c1_pass = all(is_pass(x) for x in all_c1 if isinstance(x, bool))
    all_c2_pass = all(is_pass(x) for x in all_c2 if isinstance(x, bool))
    all_c4_pass = all(is_pass(x) for x in all_c4 if isinstance(x, bool))

    pending_c1 = any(is_pending(x) for x in all_c1 if not isinstance(x, bool) and x != 'N/A')
    pending_c2 = any(is_pending(x) for x in all_c2 if not isinstance(x, bool) and x != 'N/A')
    pending_c4 = any(is_pending(x) for x in all_c4 if not isinstance(x, bool))

    if pending_c1 or pending_c2 or pending_c4:
        verdict = "PENDING_EVALUATION"
    elif all_c1_pass and all_c2_pass and all_c4_pass:
        verdict = "PASS"
    elif all_c1_pass and all_c2_pass:
        verdict = "CONDITIONAL_PASS"
    elif all_c1_pass:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "FAIL"

    json_path = out_dir / 'stage2b_final_gate_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(criteria, f, indent=2, default=str)

    md_lines = [
        "# Stage 2B Final Gate Report",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Gate 2B Criteria (Unified Evaluator)",
        "",
        "### C1: Recovery PSNR >= Immediate Prune PSNR (pruned methods)",
    ]
    for scene_name in SCENES:
        for method in METHODS:
            if method == 'schedule_control':
                md_lines.append(f"- {scene_name}/{method}: N/A (no pruning)")
                continue
            c = criteria[scene_name][method]['C1_recovery_improves_over_prune']
            ps = c.get('prune_psnr')
            rs = c.get('rec_psnr')
            result = c.get('pass')
            result_str = str(result)
            ps_str = f'{ps:.4f}' if ps is not None else 'N/A'
            rs_str = f'{rs:.4f}' if rs is not None else 'N/A'
            md_lines.append(f"- {scene_name}/{method}: prune_psnr={ps_str}, rec_psnr={rs_str}, **{result_str}**")
    md_lines.append("")

    md_lines.append("### C2: Schedule Control PSNR Stable (delta <= 0.5 dB)")
    for scene_name in SCENES:
        c = criteria[scene_name]['schedule_control']['C2_schedule_control_stable']
        bp = c.get('base_psnr')
        rp = c.get('rec_psnr')
        delta = c.get('psnr_delta')
        result = c.get('pass')
        result_str = str(result)
        bp_str = f'{bp:.4f}' if bp is not None else 'N/A'
        rp_str = f'{rp:.4f}' if rp is not None else 'N/A'
        delta_str = f'{delta:.4f}' if delta is not None else 'N/A'
        md_lines.append(f"- {scene_name}: base={bp_str}, rec={rp_str}, delta={delta_str}, **{result_str}**")
    md_lines.append("")

    md_lines.append("### C3: Mask Risk Recovery >= Random - 0.01")
    for scene_name in SCENES:
        c = criteria[scene_name].get('mask_risk', {}).get('C3_mask_risk_vs_random', {})
        mr = c.get('mask_risk_rec_psnr')
        rn = c.get('random_rec_psnr')
        result = c.get('pass')
        result_str = str(result)
        mr_str = f'{mr:.4f}' if mr is not None else 'N/A'
        rn_str = f'{rn:.4f}' if rn is not None else 'N/A'
        md_lines.append(f"- {scene_name}: mask_risk={mr_str}, random={rn_str}, **{result_str}**")
    md_lines.append("")

    md_lines.append("### C4: Gaussian Count Stable During Recovery")
    for scene_name in SCENES:
        for method in METHODS:
            c = criteria[scene_name][method]['C4_gaussian_count_stable']
            result = c.get('pass')
            result_str = str(result)
            fn = c.get('first_n')
            ln = c.get('last_n')
            md_lines.append(f"- {scene_name}/{method}: {fn} -> {ln}, **{result_str}**")
    md_lines.append("")

    md_lines.append("### C5: Deletion Quality (mask_risk d_center_norm > random)")
    for scene_name in SCENES:
        c = criteria[scene_name].get('mask_risk', {}).get('C5_deletion_quality', {})
        result = c.get('pass')
        result_str = str(result)
        md_lines.append(f"- {scene_name}: **{result_str}**")
    md_lines.append("")

    md_lines.extend([
        "## Summary Table",
        "",
        "| Scene | Method | Base PSNR | Prune PSNR | Rec PSNR | Rec N | C1 | C2 | C3 | C4 | C5 |",
        "|-------|--------|-----------|------------|----------|-------|-----|-----|-----|-----|-----|",
    ])
    for scene_name in SCENES:
        for method in METHODS:
            m = criteria[scene_name][method]
            raw = m.get('raw_values', {})
            c1 = str(m.get('C1_recovery_improves_over_prune', {}).get('pass', ''))
            c2 = str(m.get('C2_schedule_control_stable', {}).get('pass', ''))
            c3_val = m.get('C3_mask_risk_vs_random', {}).get('pass')
            c3 = str(c3_val) if c3_val is not None else 'N/A'
            if method not in ['mask_risk']:
                c3 = 'N/A'
            c4 = str(m.get('C4_gaussian_count_stable', {}).get('pass', ''))
            c5 = str(m.get('C5_deletion_quality', {}).get('pass', ''))
            def fmt(v, d=4):
                if v is None: return 'N/A'
                if isinstance(v, bool): return str(v)
                return f"{v:.{d}f}"
            bp = fmt(raw.get('base_psnr'))
            pp = fmt(raw.get('prune_psnr'))
            rp = fmt(raw.get('rec_psnr'))
            rn = str(raw.get('rec_gaussians', 'N/A'))
            md_lines.append(f"| {scene_name} | {method} | {bp} | {pp} | {rp} | {rn} | {c1} | {c2} | {c3} | {c4} | {c5} |")

    md_lines.extend([
        "",
        "## Overall Verdict",
        "",
        f"**{verdict}**",
        "",
        "### Criterion Details",
    ])
    if verdict == "PASS":
        md_lines.append("- Recovery PSNR >= Prune PSNR for all methods")
        md_lines.append("- Schedule Control PSNR delta <= 0.5 dB")
        md_lines.append("- Mask Risk >= Random - 0.01")
        md_lines.append("- Gaussian count stable")
        md_lines.append("- Deletion quality confirmed")
    elif verdict == "FAIL":
        if not all_c1_pass:
            md_lines.append("- C1 FAIL: Recovery PSNR does not exceed prune PSNR for some methods")
        if not all_c2_pass:
            md_lines.append("- C2 FAIL: Schedule Control PSNR delta exceeds 0.5 dB")
        if not all_c4_pass:
            md_lines.append("- C4 FAIL: Gaussian count not stable")
    elif "PENDING" in verdict:
        md_lines.append("- Some criteria could not be fully evaluated")
    md_lines.append("")

    md_path = out_dir / 'stage2b_final_gate_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f"Saved: {md_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Stage 2B Final Gate Verdict: {verdict}")
    print(f"{'='*60}")
    for scene_name in SCENES:
        for method in METHODS:
            raw = criteria[scene_name][method]['raw_values']
            bp = raw.get('base_psnr')
            pp = raw.get('prune_psnr')
            rp = raw.get('rec_psnr')
            bp_str = f'{bp:.2f}' if bp is not None else 'N/A'
            pp_str = f'{pp:.2f}' if pp is not None else 'N/A'
            rp_str = f'{rp:.2f}' if rp is not None else 'N/A'
            print(f"  {scene_name}/{method}: base={bp_str}, prune={pp_str}, rec={rp_str}")

if __name__ == '__main__':
    main()
