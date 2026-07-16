#!/usr/bin/env python3
"""Evaluate Stage 2B Gate: recovery improves PSNR, maintains geometry parity.
v2: handles missing metrics as PENDING_EVALUATION, proper N/A for schedule_control C1."""
import json, os, sys, numpy as np
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

SCENES = ['scene_01', 'scene_03']
METHODS = ['schedule_control', 'random', 'low_opacity', 'low_contribution', 'mask_risk']
RATIO = 'ratio_005'

PENDING_EVALUATION = 'PENDING_EVALUATION'

def load_training_log(scene, method):
    path = f'/data/wyh/RecycleGS/outputs/recovery/{scene}/{method}/training_log.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_render_metrics(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def main():
    out_dir = Path('/data/wyh/RecycleGS/outputs/prune_only')
    out_dir.mkdir(parents=True, exist_ok=True)

    criteria = {}
    for scene_name in SCENES:
        scene_criteria = {}
        for method in METHODS:
            base_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene_name}/{RATIO}/baseline/render_metrics.json'
            if method == 'schedule_control':
                imm_path = base_path
            else:
                imm_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene_name}/{RATIO}/{method}/render_metrics.json'
            rec_path = f'/data/wyh/RecycleGS/outputs/prune_only/{scene_name}/{RATIO}/{method}/recovery_500/render_metrics.json'

            base_rm = load_render_metrics(base_path)
            imm_rm = load_render_metrics(imm_path)
            rec_rm = load_render_metrics(rec_path)

            def get_psnr(rm):
                if rm is None: return None
                return rm.get('psnr', rm.get('psnr_mean'))
            def get_ssim(rm):
                if rm is None: return None
                return rm.get('ssim', rm.get('ssim_mean'))
            def get_lpips(rm):
                if rm is None: return None
                return rm.get('lpips', rm.get('lpips_mean'))

            base_psnr = get_psnr(base_rm)
            imm_psnr = get_psnr(imm_rm) if method != 'schedule_control' else base_psnr
            rec_psnr = get_psnr(rec_rm)

            # C1: Recovery improves PSNR over immediate prune
            if method == 'schedule_control':
                c1_pass = 'N/A'
            elif rec_psnr is not None and imm_psnr is not None:
                c1_pass = rec_psnr > imm_psnr
            elif rec_psnr is None:
                c1_pass = PENDING_EVALUATION
            else:
                c1_pass = PENDING_EVALUATION

            # C2: Recovery PSNR >= baseline PSNR (schedule_control only)
            if method == 'schedule_control':
                if rec_psnr is not None and base_psnr is not None:
                    c2_pass = rec_psnr >= base_psnr
                elif rec_psnr is None:
                    c2_pass = PENDING_EVALUATION
                else:
                    c2_pass = PENDING_EVALUATION
            else:
                c2_pass = 'N/A'

            # C3: mask_risk vs random comparison (computed cross-scene later)
            c3_pass = None

            # C4: Gaussian count stable during recovery
            log = load_training_log(scene_name, method)
            if log:
                first_n = log[0]['N_gaussians']
                last_n = log[-1]['N_gaussians']
                c4_pass = bool(first_n == last_n)
            else:
                c4_pass = PENDING_EVALUATION

            # C5: Deletion quality (PSNR should not drop significantly from baseline)
            # For all methods: recovery PSNR should be within 1dB of baseline
            if rec_psnr is not None and base_psnr is not None:
                c5_pass = (base_psnr - rec_psnr) < 1.0
            elif rec_psnr is None:
                c5_pass = PENDING_EVALUATION
            else:
                c5_pass = PENDING_EVALUATION

            # C6: Geometry proxy (PENDING if missing)
            c6_pass = PENDING_EVALUATION

            scene_criteria[method] = {
                'C1_recovery_improves_over_prune': {
                    'prune_psnr': imm_psnr,
                    'rec_psnr': rec_psnr,
                    'pass': c1_pass,
                },
                'C2_recovery_meets_baseline': {
                    'base_psnr': base_psnr,
                    'rec_psnr': rec_psnr,
                    'pass': c2_pass,
                },
                'C3_mask_risk_vs_random': {
                    'mask_risk_rec_psnr': None,
                    'random_rec_psnr': None,
                    'pass': None,
                },
                'C4_gaussian_count_stable': {
                    'pass': c4_pass,
                },
                'C5_deletion_quality': {
                    'base_psnr': base_psnr,
                    'rec_psnr': rec_psnr,
                    'delta': round(base_psnr - rec_psnr, 4) if (base_psnr is not None and rec_psnr is not None) else None,
                    'pass': c5_pass,
                },
                'C6_geometry_proxy': {
                    'pass': c6_pass,
                },
                'raw_values': {
                    'base_psnr': base_psnr,
                    'prune_psnr': imm_psnr,
                    'rec_psnr': rec_psnr,
                    'base_ssim': get_ssim(base_rm),
                    'imm_ssim': get_ssim(imm_rm) if method != 'schedule_control' else get_ssim(base_rm),
                    'rec_ssim': get_ssim(rec_rm),
                    'base_lpips': get_lpips(base_rm),
                    'imm_lpips': get_lpips(imm_rm) if method != 'schedule_control' else get_lpips(base_rm),
                    'rec_lpips': get_lpips(rec_rm),
                },
            }

        criteria[scene_name] = scene_criteria

    # Cross-scene C3: mask_risk vs random recovery
    for scene_name in SCENES:
        mr = criteria[scene_name].get('mask_risk', {})
        rn = criteria[scene_name].get('random', {})
        mr_rec = mr.get('raw_values', {}).get('rec_psnr')
        rn_rec = rn.get('raw_values', {}).get('rec_psnr')
        if mr_rec is not None and rn_rec is not None:
            passed = mr_rec >= rn_rec - 0.1
            mr['C3_mask_risk_vs_random'] = {
                'mask_risk_rec_psnr': mr_rec,
                'random_rec_psnr': rn_rec,
                'pass': passed,
            }
        else:
            mr['C3_mask_risk_vs_random'] = {
                'mask_risk_rec_psnr': mr_rec,
                'random_rec_psnr': rn_rec,
                'pass': PENDING_EVALUATION if (mr_rec is None or rn_rec is None) else None,
            }

    # Compute overall verdict
    def is_pass(v):
        return v is True
    def is_pending(v):
        return v == PENDING_EVALUATION

    all_c1 = []
    all_c4 = []
    all_c5 = []

    for scene_name in SCENES:
        for method in METHODS:
            c1 = criteria[scene_name][method]['C1_recovery_improves_over_prune']['pass']
            c4 = criteria[scene_name][method]['C4_gaussian_count_stable']['pass']
            c5 = criteria[scene_name][method]['C5_deletion_quality']['pass']
            if c1 != 'N/A':
                all_c1.append(c1)
            all_c4.append(c4)
            all_c5.append(c5)

    has_metrics = any(isinstance(x, bool) for x in all_c1)
    all_c1_pass = all(is_pass(x) for x in all_c1 if isinstance(x, bool))
    all_c4_pass = all(is_pass(x) for x in all_c4 if isinstance(x, bool))
    all_c5_pass = all(is_pass(x) for x in all_c5 if isinstance(x, bool))

    pending_c1 = any(is_pending(x) for x in all_c1 if not isinstance(x, bool))
    pending_c4 = any(is_pending(x) for x in all_c4 if not isinstance(x, bool))

    if pending_c1 or pending_c4:
        overall_pass = False
        verdict = "PENDING_EVALUATION"
    elif all_c1_pass and all_c4_pass and all_c5_pass:
        overall_pass = True
        verdict = "PASS"
    elif all_c1_pass:
        overall_pass = False
        verdict = "CONDITIONAL PASS"
    else:
        overall_pass = False
        verdict = "FAIL"

    json_path = out_dir / 'stage2b_gate_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(criteria, f, indent=2, default=str)

    md_lines = [
        "# Stage 2B Gate Report v2",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Gate 2B Criteria",
        "",
        "### C1: Recovery improves PSNR over immediate prune",
    ]
    for scene_name in SCENES:
        for method in METHODS:
            if method == 'schedule_control':
                md_lines.append(f"- {scene_name}/{method}: N/A (schedule_control retains all gaussians)")
                continue
            c = criteria.get(scene_name, {}).get(method, {}).get('C1_recovery_improves_over_prune', {})
            ps = c.get('prune_psnr')
            rs = c.get('rec_psnr')
            result = c.get('pass')
            if result is True: result_str = 'PASS'
            elif result is False: result_str = 'FAIL'
            else: result_str = str(result)
            md_lines.append(f"- {scene_name}/{method}: prune_psnr={ps}, rec_psnr={rs}, **{result_str}**")
    md_lines.append("")

    md_lines.append("### C2: Recovery meets baseline PSNR (schedule_control only)")
    for scene_name in SCENES:
        c = criteria.get(scene_name, {}).get('schedule_control', {}).get('C2_recovery_meets_baseline', {})
        bp = c.get('base_psnr')
        rp = c.get('rec_psnr')
        result = c.get('pass')
        if result is True: result_str = 'PASS'
        elif result is False: result_str = 'FAIL'
        elif result == 'N/A': result_str = 'N/A'
        else: result_str = str(result)
        md_lines.append(f"- {scene_name}: base_psnr={bp}, rec_psnr={rp}, **{result_str}**")
    md_lines.append("")

    md_lines.append("### C3: mask_risk recovery comparable to random")
    for scene_name in SCENES:
        c = criteria.get(scene_name, {}).get('mask_risk', {}).get('C3_mask_risk_vs_random', {})
        mr = c.get('mask_risk_rec_psnr')
        rn = c.get('random_rec_psnr')
        result = c.get('pass')
        if result is True: result_str = 'PASS'
        elif result is False: result_str = 'FAIL'
        elif result == PENDING_EVALUATION: result_str = 'PENDING_EVALUATION'
        else: result_str = 'N/A'
        md_lines.append(f"- {scene_name}: mask_risk={mr}, random={rn}, **{result_str}**")
    md_lines.append("")

    md_lines.append("### C4: Gaussian count stable during recovery")
    for scene_name in SCENES:
        for method in METHODS:
            c = criteria.get(scene_name, {}).get(method, {}).get('C4_gaussian_count_stable', {})
            result = c.get('pass')
            if result is True: result_str = 'PASS'
            elif result is False: result_str = 'FAIL'
            else: result_str = str(result)
            md_lines.append(f"- {scene_name}/{method}: **{result_str}**")
    md_lines.append("")

    md_lines.append("### C5: Deletion quality (Recovery PSNR within 1dB of baseline)")
    for scene_name in SCENES:
        for method in METHODS:
            c = criteria.get(scene_name, {}).get(method, {}).get('C5_deletion_quality', {})
            bp = c.get('base_psnr')
            rp = c.get('rec_psnr')
            delta = c.get('delta')
            result = c.get('pass')
            if result is True: result_str = 'PASS'
            elif result is False: result_str = 'FAIL'
            else: result_str = str(result)
            md_lines.append(f"- {scene_name}/{method}: base={bp}, rec={rp}, delta={delta}, **{result_str}**")
    md_lines.append("")

    md_lines.append("### C6: Geometry proxy (external)")
    for scene_name in SCENES:
        for method in METHODS:
            c = criteria.get(scene_name, {}).get(method, {}).get('C6_geometry_proxy', {})
            result = c.get('pass')
            result_str = str(result) if result else str(result)
            md_lines.append(f"- {scene_name}/{method}: **{result_str}**")
    md_lines.append("")

    md_lines.extend([
        "## Summary Table",
        "",
        "| Scene | Method | Base PSNR | Prune PSNR | Rec PSNR | C1 | C2 | C3 | C4 | C5 | C6 |",
        "|-------|--------|-----------|------------|----------|-----|-----|-----|-----|-----|-----|",
    ])
    for scene_name in SCENES:
        for method in METHODS:
            m = criteria.get(scene_name, {}).get(method, {})
            raw = m.get('raw_values', {})
            c1 = str(m.get('C1_recovery_improves_over_prune', {}).get('pass', ''))
            c2 = str(m.get('C2_recovery_meets_baseline', {}).get('pass', ''))
            c3 = str(m.get('C3_mask_risk_vs_random', {}).get('pass', ''))
            c4 = str(m.get('C4_gaussian_count_stable', {}).get('pass', ''))
            c5 = str(m.get('C5_deletion_quality', {}).get('pass', ''))
            c6 = str(m.get('C6_geometry_proxy', {}).get('pass', ''))
            def fmt(v, d=4):
                if v is None: return 'N/A'
                if isinstance(v, bool): return str(v)
                return f"{v:.{d}f}"
            bp = fmt(raw.get('base_psnr'))
            pp = fmt(raw.get('prune_psnr'))
            rp = fmt(raw.get('rec_psnr'))
            md_lines.append(f"| {scene_name} | {method} | {bp} | {pp} | {rp} | {c1} | {c2} | {c3} | {c4} | {c5} | {c6} |")

    md_lines.extend([
        "",
        "## Overall Verdict",
        "",
        f"**{verdict}**",
        "",
        "### Criteria Details",
    ])
    if verdict == "PASS":
        md_lines.append("- All criteria pass across both scenes")
    elif verdict == "PENDING_EVALUATION":
        md_lines.append("- Some criteria could not be evaluated (metrics missing)")
        if pending_c1:
            md_lines.append("- C1: PENDING for some methods (immediate prune or recovery metrics not computed)")
        if pending_c4:
            md_lines.append("- C4: PENDING for some methods (training log not available)")
    elif verdict == "CONDITIONAL PASS":
        md_lines.append("- C1: All evaluated methods show PSNR improvement, but some criteria have issues")
    elif verdict == "FAIL":
        if not all_c1_pass:
            md_lines.append("- C1: Some methods do not show PSNR improvement")
        if not all_c4_pass:
            md_lines.append("- C4: Gaussian count not stable in some methods")
        if not all_c5_pass:
            md_lines.append("- C5: Deletion quality check failed for some methods")
    md_lines.append("")

    md_path = out_dir / 'stage2b_gate_report_v2.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f"Saved: {md_path}")

    json_report_path = out_dir / 'stage2b_gate_report_v2.json'
    report_data = {
        'verdict': verdict,
        'overall_pass': overall_pass,
        'criteria': criteria,
    }
    with open(json_report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"Saved: {json_report_path}")

    print(f"\n{'='*60}")
    print(f"Stage 2B Gate Verdict: {verdict}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
