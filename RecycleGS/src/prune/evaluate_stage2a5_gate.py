#!/usr/bin/env python3
"""Evaluate Stage 2A.5 gate: read all audit results, check criteria."""
import argparse, json, os, sys, numpy as np, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')

def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
    args = parser.parse_args()

    with open(args.locked_config) as f:
        locked_cfg = yaml.safe_load(f)

    proj_root = Path(locked_cfg['project_root'])
    out_dir = proj_root / 'outputs' / 'prune_only'
    audit_dir = proj_root / 'outputs' / 'debug' / 'stage2a_audit'
    stage2a5_dir = out_dir / 'stage2a5'
    os.makedirs(stage2a5_dir, exist_ok=True)

    criteria = {}
    overall_pass = True
    details = []

    # ===== 1. All pruned methods have correct N_loaded =====
    for scene_key in ['scene_01', 'scene_03']:
        scene_cfg = locked_cfg.get(scene_key, {})
        scene_name = scene_cfg.get('scene_name', scene_key)
        ratio = locked_cfg.get('prune_ratio', 0.005)
        ratio_str = f"ratio_{int(ratio*1000):03d}"
        base = out_dir / scene_name / ratio_str

        with open(base / 'prune_metadata.json') as f:
            metadata = json.load(f)

        baseline_N = None
        ckpt_path = scene_cfg['checkpoint_path']
        from plyfile import PlyData
        ply = PlyData.read(ckpt_path)
        baseline_N = ply['vertex'].count

        for method in ['random', 'low_opacity', 'low_contribution', 'mask_risk', 'oracle']:
            out_m = base / method
            retained_path = out_m / 'retained.ply'
            if not retained_path.exists():
                details.append(f"FAIL [{scene_name}/{method}]: retained.ply not found")
                continue
            ply_m = PlyData.read(retained_path)
            N_loaded = ply_m['vertex'].count
            K = metadata['methods'][method]['pruned_count']
            expected = baseline_N - K
            if N_loaded != expected:
                details.append(f"FAIL [{scene_name}/{method}]: N_loaded={N_loaded} != expected={expected} (baseline_N={baseline_N}, K={K})")
                overall_pass = False
            else:
                details.append(f"PASS [{scene_name}/{method}]: N_loaded={N_loaded} == expected={expected}")

    c1 = all('N_loaded' in d and '!=' not in d.split(':')[-1] if 'N_loaded' in d else True for d in details[-12:])
    c1_pass = not any('FAIL' in d and 'N_loaded' in d for d in details)
    criteria['1_correct_N_loaded'] = {
        'pass': c1_pass,
        'detail': 'All methods have correct N_loaded = baseline_N - K',
    }

    # ===== 2. SHA differs between methods =====
    sha_sets = {}
    for scene_key in ['scene_01', 'scene_03']:
        scene_cfg = locked_cfg.get(scene_key, {})
        scene_name = scene_cfg.get('scene_name', scene_key)
        ratio = locked_cfg.get('prune_ratio', 0.005)
        ratio_str = f"ratio_{int(ratio*1000):03d}"
        base = out_dir / scene_name / ratio_str

        shas = {}
        for method in ['baseline', 'random', 'low_opacity', 'low_contribution', 'mask_risk', 'oracle']:
            if method == 'baseline':
                ply_path = scene_cfg['checkpoint_path']
            else:
                ply_path = base / method / 'retained.ply'
            if os.path.exists(ply_path):
                import hashlib
                h = hashlib.sha256()
                with open(ply_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(65536), b''):
                        h.update(chunk)
                shas[method] = h.hexdigest()
        sha_sets[scene_name] = shas

    sha_pass = True
    for scene_name, shas in sha_sets.items():
        vals = list(shas.values())
        if len(set(vals)) == 1 and len(vals) > 1:
            details.append(f"FAIL [{scene_name}]: All SHAs identical! {vals[0][:16]}...")
            sha_pass = False
        else:
            n_unique = len(set(vals))
            details.append(f"PASS [{scene_name}]: {n_unique} unique SHAs across {len(vals)} methods")
    criteria['2_sha_diverse'] = {
        'pass': sha_pass,
        'detail': 'SHA differs between methods in each scene',
    }

    # ===== 3. Render difference > 0 for pruned methods =====
    render_diff_pass = True
    for scene_key in ['scene_01', 'scene_03']:
        scene_cfg = locked_cfg.get(scene_key, {})
        scene_name = scene_cfg.get('scene_name', scene_key)
        ratio = locked_cfg.get('prune_ratio', 0.005)
        ratio_str = f"ratio_{int(ratio*1000):03d}"
        base = out_dir / scene_name / ratio_str

        baseline_metrics = load_json(base / 'baseline' / 'render_metrics.json')
        if baseline_metrics is None:
            baseline_metrics = load_json(base / 'baseline' / 'evaluation_metrics.json')
            if baseline_metrics:
                baseline_metrics = baseline_metrics.get('render_metrics', None)

        for method in ['random', 'low_opacity', 'low_contribution', 'mask_risk', 'oracle']:
            m_metrics = load_json(base / method / 'render_metrics.json')
            if m_metrics is None:
                m_metrics = load_json(base / method / 'evaluation_metrics.json')
                if m_metrics:
                    m_metrics = m_metrics.get('render_metrics', None)

            if m_metrics and baseline_metrics:
                b_psnr = baseline_metrics.get('psnr_mean', 0)
                m_psnr = m_metrics.get('psnr_mean', 0)
                diff = abs(m_psnr - b_psnr)
                if diff < 1e-10:
                    details.append(f"FAIL [{scene_name}/{method}]: render diff = {diff} (essentially zero)")
                    render_diff_pass = False
                else:
                    details.append(f"PASS [{scene_name}/{method}]: render diff = {diff:.6f}")
            else:
                details.append(f"SKIP [{scene_name}/{method}]: metrics not available")
    criteria['3_render_diff_nonzero'] = {
        'pass': render_diff_pass,
        'detail': 'Pruned methods have non-zero render difference from baseline',
    }

    # ===== 4. Baseline re-evaluated with same code =====
    # Check that baseline has render_metrics.json
    baseline_pass = True
    for scene_key in ['scene_01', 'scene_03']:
        scene_cfg = locked_cfg.get(scene_key, {})
        scene_name = scene_cfg.get('scene_name', scene_key)
        ratio = locked_cfg.get('prune_ratio', 0.005)
        ratio_str = f"ratio_{int(ratio*1000):03d}"
        base = out_dir / scene_name / ratio_str
        bm = load_json(base / 'baseline' / 'render_metrics.json')
        if bm is None:
            details.append(f"FAIL [{scene_name}]: baseline render_metrics.json not found")
            baseline_pass = False
        else:
            details.append(f"PASS [{scene_name}]: baseline evaluated, PSNR={bm.get('psnr_mean', 'N/A')}")
    criteria['4_baseline_evaluated'] = {
        'pass': baseline_pass,
        'detail': 'Baseline re-evaluated with same code',
    }

    # ===== 5. Random 10-seed stability computed =====
    seed_pass = True
    for scene_key in ['scene_01', 'scene_03']:
        scene_cfg = locked_cfg.get(scene_key, {})
        scene_name = scene_cfg.get('scene_name', scene_key)
        ratio = locked_cfg.get('prune_ratio', 0.005)
        ratio_str = f"ratio_{int(ratio*1000):03d}"
        base = out_dir / scene_name / ratio_str
        seed_path = base / 'random_seed_stability.json'
        if not seed_path.exists():
            details.append(f"FAIL [{scene_name}]: random_seed_stability.json not found")
            seed_pass = False
        else:
            with open(seed_path) as f:
                sd = json.load(f)
            n_seeds = sd.get('aggregate', {}).get('n_seeds', 0)
            if n_seeds < 10:
                details.append(f"FAIL [{scene_name}]: only {n_seeds} seeds, expected 10")
                seed_pass = False
            else:
                agg = sd.get('aggregate', {})
                dc_mean = agg.get('d_center_norm_mean', {})
                mr_mean = agg.get('mask_risk_mean', {})
                details.append(f"PASS [{scene_name}]: {n_seeds} seeds, "
                               f"d_center_mean={dc_mean.get('mean', 'N/A')} +- {dc_mean.get('std', 'N/A')}, "
                               f"mask_risk_mean={mr_mean.get('mean', 'N/A')} +- {mr_mean.get('std', 'N/A')}")
    criteria['5_random_10_seed_stability'] = {
        'pass': seed_pass,
        'detail': 'Random 10-seed stability computed for both scenes',
    }

    # ===== 6. Geometry metrics generated =====
    geo_pass = True
    for scene_key in ['scene_01', 'scene_03']:
        scene_cfg = locked_cfg.get(scene_key, {})
        scene_name = scene_cfg.get('scene_name', scene_key)
        ratio = locked_cfg.get('prune_ratio', 0.005)
        ratio_str = f"ratio_{int(ratio*1000):03d}"
        base = out_dir / scene_name / ratio_str
        for method in ['random', 'low_opacity', 'low_contribution', 'mask_risk', 'oracle']:
            gm = load_json(base / method / 'geometry_metrics.json')
            if gm is None:
                details.append(f"FAIL [{scene_name}/{method}]: geometry_metrics.json not found")
                geo_pass = False
            else:
                spr = gm.get('surface_proximity_ratio_threshold_0_02', 'N/A')
                dc_shift = gm.get('d_center_norm_shift', 'N/A')
                details.append(f"PASS [{scene_name}/{method}]: geo metrics, SPR(2cm)={spr}, d_shift={dc_shift}")
    criteria['6_geometry_metrics'] = {
        'pass': geo_pass,
        'detail': 'Proxy geometry metrics generated for all methods',
    }

    # ===== 7. mask_risk > random mean in both scenes =====
    mr_vs_random_pass = True
    for scene_key in ['scene_01', 'scene_03']:
        scene_cfg = locked_cfg.get(scene_key, {})
        scene_name = scene_cfg.get('scene_name', scene_key)
        ratio = locked_cfg.get('prune_ratio', 0.005)
        ratio_str = f"ratio_{int(ratio*1000):03d}"
        base = out_dir / scene_name / ratio_str

        # Load from removed_set_metrics in evaluation_metrics.json
        mr_eval = load_json(base / 'mask_risk' / 'evaluation_metrics.json')
        rn_eval = load_json(base / 'random' / 'evaluation_metrics.json')

        mr_dc = None
        rn_dc = None
        if mr_eval:
            mr_dc = mr_eval.get('removed_set_metrics', {}).get('d_center_norm_mean', None)
            mr_risk = mr_eval.get('removed_set_metrics', {}).get('mask_risk_mean', None)
        if rn_eval:
            rn_dc = rn_eval.get('removed_set_metrics', {}).get('d_center_norm_mean', None)

        if mr_dc is not None and rn_dc is not None:
            if mr_dc > rn_dc:
                details.append(f"PASS [{scene_name}]: mask_risk d_center={mr_dc:.6f} > random d_center={rn_dc:.6f}")
            else:
                details.append(f"FAIL [{scene_name}]: mask_risk d_center={mr_dc:.6f} <= random d_center={rn_dc:.6f}")
                mr_vs_random_pass = False
        else:
            details.append(f"SKIP [{scene_name}]: metrics not available")
    criteria['7_mask_risk_gt_random'] = {
        'pass': mr_vs_random_pass,
        'detail': 'mask_risk removed d_center_norm > random removed d_center_norm in both scenes',
    }

    # ===== 8. Cross-scene consistency =====
    cross_pass = True
    scene_results = {}
    for scene_key in ['scene_01', 'scene_03']:
        scene_cfg = locked_cfg.get(scene_key, {})
        scene_name = scene_cfg.get('scene_name', scene_key)
        ratio = locked_cfg.get('prune_ratio', 0.005)
        ratio_str = f"ratio_{int(ratio*1000):03d}"
        base = out_dir / scene_name / ratio_str

        mr_eval = load_json(base / 'mask_risk' / 'evaluation_metrics.json')
        rn_eval = load_json(base / 'random' / 'evaluation_metrics.json')
        lo_eval = load_json(base / 'low_opacity' / 'evaluation_metrics.json')
        lc_eval = load_json(base / 'low_contribution' / 'evaluation_metrics.json')

        def get_rm(d, key):
            if d:
                return d.get('removed_set_metrics', {}).get(key, None)
            return None

        scene_results[scene_name] = {
            'mask_risk_d_center': get_rm(mr_eval, 'd_center_norm_mean'),
            'random_d_center': get_rm(rn_eval, 'd_center_norm_mean'),
            'low_opacity_d_center': get_rm(lo_eval, 'd_center_norm_mean'),
            'low_contribution_d_center': get_rm(lc_eval, 'd_center_norm_mean'),
            'mask_risk_vs_random': get_rm(mr_eval, 'd_center_norm_mean',) is not None and get_rm(rn_eval, 'd_center_norm_mean') is not None and get_rm(mr_eval, 'd_center_norm_mean') > get_rm(rn_eval, 'd_center_norm_mean'),
            'mask_risk_vs_low_opacity': get_rm(mr_eval, 'd_center_norm_mean') is not None and get_rm(lo_eval, 'd_center_norm_mean') is not None and get_rm(mr_eval, 'd_center_norm_mean') > get_rm(lo_eval, 'd_center_norm_mean'),
            'mask_risk_vs_low_contribution': get_rm(mr_eval, 'd_center_norm_mean') is not None and get_rm(lc_eval, 'd_center_norm_mean') is not None and get_rm(mr_eval, 'd_center_norm_mean') > get_rm(lc_eval, 'd_center_norm_mean'),
        }

    # Check consistency
    consistency_checks = []
    for key in ['mask_risk_vs_random', 'mask_risk_vs_low_opacity', 'mask_risk_vs_low_contribution']:
        s01 = scene_results.get('scene_01', {}).get(key, False)
        s03 = scene_results.get('scene_03', {}).get(key, False)
        consistent = s01 == s03
        both_pass = s01 and s03
        consistency_checks.append({
            'criterion': key,
            'scene_01': s01,
            'scene_03': s03,
            'consistent': consistent,
            'both_pass': both_pass,
        })
        if not both_pass:
            cross_pass = False
            details.append(f"FAIL [cross-scene {key}]: scene_01={s01}, scene_03={s03}")
        else:
            details.append(f"PASS [cross-scene {key}]: both scenes pass")

    criteria['8_cross_scene_consistency'] = {
        'pass': cross_pass,
        'detail': 'Cross-scene consistency: mask_risk beats other methods in both scenes',
    }

    # Overall verdict
    overall_pass = all(c['pass'] for c in criteria.values())

    # Build cross-scene metrics JSON
    cross_scene_metrics = {
        'criteria': criteria,
        'details': details,
        'overall_pass': overall_pass,
        'scene_results': scene_results,
    }
    cross_metrics_path = stage2a5_dir / 'stage2a5_cross_scene_metrics.json'
    with open(cross_metrics_path, 'w') as f:
        json.dump(cross_scene_metrics, f, indent=2)
    print(f"Saved cross-scene metrics: {cross_metrics_path}")

    # Build MD report
    md_lines = [
        "# Stage 2A.5 Audit Gate Report",
        "",
        f"## Verdict: **{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "## Criteria Results",
        "",
        "| # | Criterion | Pass | Detail |",
        "|---|-----------|------|--------|",
    ]
    ckeys = [
        ('1', 'Correct N_loaded'),
        ('2', 'SHA diversity'),
        ('3', 'Render difference > 0'),
        ('4', 'Baseline evaluated'),
        ('5', 'Random 10-seed stability'),
        ('6', 'Geometry metrics'),
        ('7', 'mask_risk > random (both scenes)'),
        ('8', 'Cross-scene consistency'),
    ]
    for i, (num, label) in enumerate(ckeys):
        ck = list(criteria.keys())[i]
        c = criteria[ck]
        md_lines.append(f"| {num} | {label} | {'**PASS**' if c['pass'] else '**FAIL**'} | {c['detail']} |")

    md_lines.extend([
        "",
        "## Detail Log",
        "",
    ])
    for d in details:
        md_lines.append(f"- {d}")

    md_lines.extend([
        "",
        "## Per-Scene Summary",
        "",
        "| Scene | Metric | mask_risk | random | low_opacity | low_contribution | oracle |",
        "|-------|--------|-----------|--------|-------------|-------------------|--------|",
    ])
    for scene_name in ['scene_01', 'scene_03']:
        scene_cfg = locked_cfg.get(scene_name, {})
        sname = scene_cfg.get('scene_name', scene_name)
        ratio = locked_cfg.get('prune_ratio', 0.005)
        ratio_str = f"ratio_{int(ratio*1000):03d}"
        base = out_dir / sname / ratio_str

        mr = load_json(base / 'mask_risk' / 'evaluation_metrics.json')
        rn = load_json(base / 'random' / 'evaluation_metrics.json')
        lo = load_json(base / 'low_opacity' / 'evaluation_metrics.json')
        lc = load_json(base / 'low_contribution' / 'evaluation_metrics.json')
        orc = load_json(base / 'oracle' / 'evaluation_metrics.json')

        def get_dc(d):
            if d:
                return d.get('removed_set_metrics', {}).get('d_center_norm_mean', 'N/A')
            return 'N/A'

        md_lines.append(f"| {sname} | d_center_norm_mean | {get_dc(mr)} | {get_dc(rn)} | {get_dc(lo)} | {get_dc(lc)} | {get_dc(orc)} |")

        def get_psnr(d):
            if d:
                rm = d.get('render_metrics', {})
                if rm:
                    return rm.get('psnr_mean', 'N/A')
            return 'N/A'

        md_lines.append(f"| {sname} | PSNR | {get_psnr(mr)} | {get_psnr(rn)} | {get_psnr(lo)} | {get_psnr(lc)} | {get_psnr(orc)} |")

    md_lines.extend([
        "",
        "## Overall Verdict",
        "",
        f"**{'PASS - Stage 2A.5 audit confirms mask-risk pruning correctness' if overall_pass else 'FAIL - Stage 2A.5 audit reveals issues'}**",
        "",
    ])

    report_path = stage2a5_dir / 'stage2a5_gate_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved gate report: {report_path}")
    print(f"\n{'='*60}")
    print(f"Stage 2A.5 Gate Verdict: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*60}")

    return overall_pass

if __name__ == '__main__':
    main()
