import argparse, sys, os, json, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

FEATURE_FILES = [
    'gaussian_base_features.npz', 'valid_view_count.npy', 'mask_inside_count.npy',
    'mask_support_unweighted.npy', 'mask_support.npy',
    'surface_support.npy', 'surface_support_risk.npy',
    'scale_anomaly.npy',
    'normal_conflict.npy', 'normal_variance.npy',
    'object_indices.npy', 'uncertain_indices.npy', 'background_indices.npy',
    'core_object_indices.npy', 'candidate_object_indices.npy', 'strong_background_indices.npy',
    'object_surface_support_risk.npy', 'object_surface_support.npy',
    'object_scale_anomaly.npy', 'object_normal_conflict.npy', 'object_normal_valid.npy',
    'object_planarity_confidence.npy',
    'object_risk_A.npy', 'object_risk_B.npy', 'object_risk_C.npy',
    'candidate_geometry_errors_v2.npz',
    'gaussian_gt_errors.npz', 'object_gaussian_gt_errors.npz',
]

def sha256_of_npy(path):
    arr = np.load(path)
    data_bytes = arr.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1f_scene01'
    os.makedirs(debug_dir, exist_ok=True)

    candidate_indices_path = out_dir / 'candidate_object_indices.npy'
    if not candidate_indices_path.exists():
        msg = f"candidate_object_indices.npy not found at {candidate_indices_path}"
        print(f"[ERROR] {msg}")
        result = {'error': msg}
        with open(debug_dir / 'feature_index_audit.json', 'w') as f:
            json.dump(result, f, indent=2)
        with open(debug_dir / 'feature_index_audit.md', 'w') as f:
            f.write(f"# Feature Index Audit\n\n{msg}\n")
        return

    candidate_indices = np.load(candidate_indices_path)
    candidate_count = len(candidate_indices)
    candidate_hash = hashlib.sha256(candidate_indices.tobytes()).hexdigest()
    N_total = -1
    base_path = out_dir / 'gaussian_base_features.npz'
    if base_path.exists():
        base = np.load(base_path)
        N_total = len(base['xyz'])

    audit = {
        'candidate_indices_path': str(candidate_indices_path),
        'candidate_count': int(candidate_count),
        'candidate_sha256': candidate_hash,
        'total_gaussians': int(N_total) if N_total > 0 else 'unknown',
        'files': {},
        'mismatches': [],
    }

    for fname in FEATURE_FILES:
        fpath = out_dir / fname
        if not fpath.exists():
            audit['files'][fname] = {'status': 'missing'}
            audit['mismatches'].append({'file': fname, 'issue': 'file not found'})
            continue

        try:
            if fname.endswith('.npz'):
                data = np.load(fpath)
                names = list(data.keys())
                file_info = {'status': 'present', 'type': 'npz', 'arrays': {}}
                for arr_name in names:
                    arr = data[arr_name]
                    s = arr.shape
                    match = 'ok'
                    if s[0] == candidate_count:
                        match = 'matches_candidate_count'
                    elif s[0] == N_total:
                        match = 'matches_total_count'
                    else:
                        match = f'unexpected_first_dim_{s[0]}'
                        if fname == 'candidate_geometry_errors_v2.npz':
                            match = 'expected_candidate_count'
                        audit['mismatches'].append({
                            'file': fname, 'array': arr_name,
                            'shape': list(s), 'expected_first_dim': candidate_count,
                            'issue': match,
                        })
                    file_info['arrays'][arr_name] = {
                        'shape': list(s),
                        'match': match,
                    }
                audit['files'][fname] = file_info
            else:
                arr = np.load(fpath)
                s = arr.shape
                is_index_file = ('indices' in fname or 'index' in fname)
                file_info = {'status': 'present', 'type': 'npy', 'shape': list(s)}
                if len(s) == 0:
                    file_info['note'] = 'scalar'
                    match = 'ok'
                elif s[0] == candidate_count:
                    match = 'matches_candidate_count'
                    file_info['match'] = 'candidate_count'
                elif s[0] == N_total:
                    match = 'matches_total_count'
                    file_info['match'] = 'total_count'
                else:
                    match = f'unexpected_first_dim_{s[0]}'
                    file_info['match'] = match
                    audit['mismatches'].append({
                        'file': fname, 'shape': list(s),
                        'expected_first_dim': candidate_count,
                        'issue': match,
                    })

                if is_index_file and len(s) > 0 and arr.dtype in (np.int32, np.int64, np.uint32, np.uint64):
                    file_hash = hashlib.sha256(arr.tobytes()).hexdigest()
                    file_info['sha256'] = file_hash
                    match_hash = (file_hash == candidate_hash)
                    file_info['hash_matches_candidate'] = match_hash
                    if not match_hash:
                        overlap = len(np.intersect1d(arr, candidate_indices))
                        file_info['overlap_with_candidate'] = int(overlap)
                        if overlap < min(len(arr), candidate_count):
                            audit['mismatches'].append({
                                'file': fname, 'sha256_differs': True,
                                'overlap': int(overlap),
                                'issue': 'indices differ from candidate',
                            })

                audit['files'][fname] = file_info
        except Exception as e:
            audit['files'][fname] = {'status': 'error', 'error': str(e)}
            audit['mismatches'].append({'file': fname, 'issue': str(e)})

    report = {
        'summary': {
            'total_files_checked': len(FEATURE_FILES),
            'present': sum(1 for v in audit['files'].values() if v.get('status') == 'present'),
            'missing': sum(1 for v in audit['files'].values() if v.get('status') == 'missing'),
            'mismatches_found': len(audit['mismatches']),
        },
        'audit': audit,
    }

    json_path = debug_dir / 'feature_index_audit.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    md_lines = [
        f"# Feature Index Audit - {cfg['scene_name']}",
        f"",
        f"## Summary",
        f"- Candidate count: {candidate_count}",
        f"- Total Gaussians: {N_total}",
        f"- Files checked: {report['summary']['total_files_checked']}",
        f"- Present: {report['summary']['present']}",
        f"- Missing: {report['summary']['missing']}",
        f"- Mismatches: {report['summary']['mismatches_found']}",
        f"",
        f"## Mismatches",
    ]
    if audit['mismatches']:
        for m in audit['mismatches']:
            md_lines.append(f"- {m['file']}: {m.get('issue', 'unknown')}")
    else:
        md_lines.append("None found.")
    md_lines.append(f"")
    md_lines.append(f"## File Details")
    for fname, finfo in audit['files'].items():
        md_lines.append(f"- {fname}: {json.dumps(finfo)}")

    md_path = debug_dir / 'feature_index_audit.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"[AUDIT] {report['summary']['present']}/{report['summary']['total_files_checked']} files present, "
          f"{report['summary']['mismatches_found']} mismatches")
    print(f"[AUDIT] Saved to {json_path}")

if __name__ == '__main__':
    main()
