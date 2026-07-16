import argparse, sys, os, json
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
from recyclegs.config import load_config

FAILURE_CATEGORIES = {
    'A': 'Coordinate misalignment: Gaussian vs mesh centers deviate >20% of mesh diameter',
    'B': 'Mesh quality: missing geometry, too sparse, or incorrect scale',
    'C': 'Gaussian quality: floaters, large offsets from mesh surface',
    'D': 'Normal convention: normal PNG decode differs from TSGS convention',
    'E': 'Mask alignment: masks do not align with projected Gaussians',
    'F': 'Depth order: reference depth is inconsistent with Gaussian z',
    'G': 'Feature collapse: one or more risk features have near-zero variance or NaN',
    'H': 'Gate 1A failure: risk score does not rank wrong Gaussians above random',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    debug_dir = Path(cfg['debug_output_dir'])
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Reading all diagnosis outputs...")
    findings = {}
    triggers = []

    # Coordinate alignment
    coord_path = debug_dir / 'coordinate_alignment_diagnosis.json'
    if coord_path.exists():
        with open(coord_path) as f:
            coord = json.load(f)
        findings['coordinate_alignment'] = coord
        cdist = coord.get('center_distance_gauss_mesh', 0)
        mdiam = coord.get('mesh_diameter', 1)
        ratio = cdist / mdiam if mdiam > 0 else 0
        if ratio > 0.20:
            triggers.append(f'A: center_distance/mesh_diameter={ratio*100:.1f}% > 20%')
        print(f"  Coordinate: center distance ratio = {ratio*100:.1f}%")
    else:
        triggers.append('A: coordinate_alignment_diagnosis.json not found')
        print(f"  Coordinate: NOT FOUND")

    # GT labels
    gt_path = debug_dir / 'gt_label_diagnosis.json'
    if gt_path.exists():
        with open(gt_path) as f:
            gt = json.load(f)
        findings['gt_labels'] = gt
        for sch_name, sch_data in gt.get('label_counts', {}).items():
            wrong_pct = sch_data.get('wrong_pct', 0)
            if wrong_pct > 80:
                triggers.append(f'C: Scheme {sch_name} wrong={wrong_pct}% > 80%')
        print(f"  GT labels: loaded")
    else:
        triggers.append('C: gt_label_diagnosis.json not found')
        print(f"  GT labels: NOT FOUND")

    # Features
    feat_path = debug_dir / 'feature_diagnosis.json'
    if feat_path.exists():
        with open(feat_path) as f:
            feat = json.load(f)
        findings['features'] = feat
        for anom in feat.get('anomalies', []):
            triggers.append(f'G: {anom}')
        print(f"  Features: {len(feat.get('anomalies', []))} anomalies")
    else:
        triggers.append('G: feature_diagnosis.json not found')
        print(f"  Features: NOT FOUND")

    # Normal convention
    norm_path = debug_dir / 'normal_convention_diagnosis.json'
    if norm_path.exists():
        with open(norm_path) as f:
            norm = json.load(f)
        findings['normal_convention'] = norm
        default_err = norm.get('default_mean_abs_cosine_error')
        best_err = norm.get('best_mean_abs_cosine_error', 0)
        if default_err is not None and best_err is not None and default_err > best_err * 1.5:
            triggers.append(f'D: default convention error={default_err:.4f} vs best={best_err:.4f}')
        print(f"  Normal convention: default err={default_err}, best={best_err}")
    else:
        triggers.append('D: normal_convention_diagnosis.json not found')
        print(f"  Normal convention: NOT FOUND")

    # Mask alignment
    mask_path = debug_dir / 'mask_alignment_report.json'
    if mask_path.exists():
        with open(mask_path) as f:
            mask = json.load(f)
        findings['mask_alignment'] = mask
        if mask.get('missing_count', 0) > 0:
            triggers.append(f'E: {mask["missing_count"]} masks missing')
        print(f"  Mask: loaded, {mask.get('missing_count', 0)} missing")
    else:
        triggers.append('E: mask_alignment_report.json not found')
        print(f"  Mask: NOT FOUND")

    # Depth order
    depth_path = debug_dir / 'depth_order_diagnosis.json'
    if depth_path.exists():
        with open(depth_path) as f:
            depth = json.load(f)
        findings['depth_order'] = depth
        behind = depth.get('average_behind_reference_ratio', 0)
        conflict = depth.get('average_depth_conflict_nonzero_ratio', 0)
        if behind > 0.5:
            triggers.append(f'F: behind_ref_ratio={behind:.3f} > 0.5')
        if conflict > 0.5:
            triggers.append(f'F: conflict_nonzero_ratio={conflict:.3f} > 0.5')
        print(f"  Depth: behind={behind:.3f}, conflict={conflict:.3f}")
    else:
        triggers.append('F: depth_order_diagnosis.json not found')
        print(f"  Depth: NOT FOUND")

    # Typed reliability
    typed_path = debug_dir / 'typed_reliability_metrics.json'
    if typed_path.exists():
        with open(typed_path) as f:
            typed = json.load(f)
        findings['typed_reliability'] = typed
        bg = typed.get('branch_A_background', {}).get('metrics', {})
        bg_auprc = bg.get('auprc', 0)
        rand_ratio = typed.get('random_baseline', {}).get('wrong_ratio', 1)
        if bg_auprc < rand_ratio * 1.2:
            triggers.append(f'H: background AUPRC={bg_auprc:.4f} < 1.2*random({rand_ratio:.4f})')
        print(f"  Typed reliability: background AUPRC={bg_auprc:.4f}, random={rand_ratio:.4f}")
    else:
        triggers.append('H: typed_reliability_metrics.json not found')
        print(f"  Typed reliability: NOT FOUND")

    # Gate 1A
    gate1a_path = out_dir / 'gate1a_report.md'
    if gate1a_path.exists():
        findings['gate1a_check'] = {'report_path': str(gate1a_path)}
        print(f"  Gate 1A: found")
    else:
        triggers.append('H: gate1a report not found')
        print(f"  Gate 1A: NOT FOUND")

    print(f"[2/5] Categorizing root causes...")
    triggered_categories = {}
    for t in triggers:
        cat_letter = t[0]
        if cat_letter not in triggered_categories:
            triggered_categories[cat_letter] = []
        triggered_categories[cat_letter].append(t)

    print(f"[3/5] Determining primary and secondary causes...")
    priority_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    primary = None
    secondary = []
    for letter in priority_order:
        if letter in triggered_categories:
            if primary is None:
                primary = letter
            else:
                secondary.append(letter)

    print(f"[4/5] Assigning severity...")
    n_triggers = len(triggers)
    if n_triggers >= 5:
        severity = 'CRITICAL'
    elif n_triggers >= 3:
        severity = 'HIGH'
    elif n_triggers >= 1:
        severity = 'MEDIUM'
    else:
        severity = 'LOW'

    print(f"[5/5] Writing final diagnosis...")
    diagnosis = {
        'scene_name': cfg['scene_name'],
        'pipeline_stage': 'Gate 1B',
        'overall_status': 'FAIL' if n_triggers > 0 else 'PASS',
        'severity': severity,
        'num_findings': len(findings),
        'num_triggers': n_triggers,
        'primary_root_cause': {
            'category': primary,
            'description': FAILURE_CATEGORIES.get(primary, 'None'),
        } if primary else None,
        'secondary_causes': [
            {'category': c, 'description': FAILURE_CATEGORIES.get(c, '')}
            for c in secondary
        ],
        'triggered_checks': triggers,
        'all_findings': {k: list(v.keys()) if isinstance(v, dict) else str(v) for k, v in findings.items()},
    }

    with open(debug_dir / 'gate1_failure_diagnosis.json', 'w') as f:
        json.dump(diagnosis, f, indent=2)

    md_lines = [
        f"# Gate 1 Failure Diagnosis - {cfg['scene_name']}",
        f"",
        f"## Overall Status: **{diagnosis['overall_status']}**",
        f"Severity: **{severity}**",
        f"",
        f"## Summary",
        f"- Pipeline stage: Gate 1B",
        f"- Total diagnostic findings loaded: {len(findings)}",
        f"- Triggered checks: {n_triggers}",
        f"",
    ]
    if primary:
        md_lines.append(f"## Primary Root Cause")
        md_lines.append(f"- **Category {primary}**: {FAILURE_CATEGORIES.get(primary, 'Unknown')}")
        md_lines.append(f"")
    if secondary:
        md_lines.append(f"## Secondary Causes")
        for c in secondary:
            md_lines.append(f"- **Category {c}**: {FAILURE_CATEGORIES.get(c, 'Unknown')}")
        md_lines.append(f"")

    md_lines.append(f"## Triggered Checks")
    md_lines.append(f"")
    for t in triggers:
        md_lines.append(f"- {t}")
    md_lines.append(f"")

    md_lines.append(f"## All Findings Summary")
    for k, v in findings.items():
        if isinstance(v, dict):
            md_lines.append(f"- **{k}**: {len(v)} fields")
        else:
            md_lines.append(f"- **{k}**: {v}")
    md_lines.append(f"")

    md_lines.append(f"## Failure Categories")
    for letter, desc in FAILURE_CATEGORIES.items():
        status = 'TRIGGERED' if letter in triggered_categories else 'OK'
        md_lines.append(f"- **{letter}**: {desc} — [{status}]")
    md_lines.append(f"")
    if diagnosis['overall_status'] == 'FAIL':
        md_lines.append(f"## Recommendation")
        if primary == 'A':
            md_lines.append(f"- Run ICP or adjust scene coordinate system to align Gaussian and mesh centers.")
        if primary == 'C' or 'C' in triggered_categories:
            md_lines.append(f"- Investigate Gaussian quality: check for floaters, large scales, or low opacity.")
        if 'D' in triggered_categories:
            md_lines.append(f"- Verify normal PNG decode convention matches TSGS code.")
        if 'E' in triggered_categories:
            md_lines.append(f"- Check mask availability and naming convention.")
        if 'F' in triggered_categories:
            md_lines.append(f"- Investigate depth reference generation.")
        if 'G' in triggered_categories:
            md_lines.append(f"- Check feature extraction pipeline for numerical issues.")
        if 'H' in triggered_categories:
            md_lines.append(f"- Risk weighting may need adjustment for this scene.")
        md_lines.append(f"")
    md_lines.append(f"## Files")
    md_lines.append(f"- gate1_failure_diagnosis.json")
    md_lines.append(f"- gate1_failure_diagnosis.md")

    with open(debug_dir / 'gate1_failure_diagnosis.md', 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"\nDiagnosis complete. Status: {diagnosis['overall_status']}, Severity: {severity}")
    print(f"Primary cause: {primary}, Secondary: {secondary}")

if __name__ == '__main__':
    main()
