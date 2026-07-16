#!/usr/bin/env python3
"""Prune Gaussian PLY checkpoint: create removed and retained PLY files."""
import argparse, hashlib, json, os, sys, numpy as np, yaml
from pathlib import Path
from plyfile import PlyData, PlyElement

sys.path.insert(0, '/data/wyh/RecycleGS/src')

def ply_to_dict(ply):
    vertex = ply['vertex']
    props = {p.name: np.asarray(vertex[p.name]) for p in vertex.properties}
    return props, vertex.count

def dict_to_ply(props, save_path):
    dtype = [(k, props[k].dtype) for k in props]
    arr = np.empty(len(props[list(props.keys())[0]]), dtype=dtype)
    for k in props:
        arr[k] = props[k]
    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False).write(save_path)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
    parser.add_argument('--all-methods', action='store_true', help='Process all methods found in prune metadata')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.locked_config) as f:
        locked_cfg = yaml.safe_load(f)

    scene_name = cfg.get('scene_name', 'scene_01')
    ckpt_path = cfg['checkpoint_path']
    ratio = locked_cfg.get('prune_ratio', 0.005)
    ratio_str = f"ratio_{int(ratio*1000):03d}"

    out_base = Path(locked_cfg['project_root']) / 'outputs' / 'prune_only' / scene_name / ratio_str

    # 1. Load original checkpoint PLY
    print(f"Loading checkpoint: {ckpt_path}")
    ply = PlyData.read(ckpt_path)
    props, N_before = ply_to_dict(ply)
    print(f"  Total Gaussians: {N_before}")

    # Get original sha256
    orig_sha = sha256_file(ckpt_path)
    print(f"  Original SHA256: {orig_sha}")

    # Load metadata to get methods
    with open(out_base / 'prune_metadata.json') as f:
        metadata = json.load(f)
    methods = list(metadata['methods'].keys()) if args.all_methods else [args.methods]

    for method in methods:
        out_dir = out_base / method
        prune_path = out_dir / 'prune_indices.npy'

        if not prune_path.exists():
            print(f"  SKIP {method}: prune_indices.npy not found")
            continue

        prune_idx = np.load(prune_path)
        K = len(prune_idx)
        print(f"\n  [{method}] Pruning {K} Gaussians")

        # Create boolean mask
        keep_mask = np.ones(N_before, dtype=bool)
        keep_mask[prune_idx] = False
        N_after = keep_mask.sum()
        N_removed = N_before - N_after

        # Build retained props
        retained_props = {k: v[keep_mask].copy() for k, v in props.items()}
        removed_props = {k: v[prune_idx].copy() for k, v in props.items()}

        # Save retained PLY
        retained_path = out_dir / 'retained.ply'
        dict_to_ply(retained_props, retained_path)
        retained_sha = sha256_file(retained_path)

        # Save removed PLY
        removed_path = out_dir / 'removed.ply'
        dict_to_ply(removed_props, removed_path)
        removed_sha = sha256_file(removed_path)

        # Verification
        all_indices = np.concatenate([
            np.where(keep_mask)[0],
            prune_idx,
        ])
        all_indices.sort()
        is_identity = np.array_equal(all_indices, np.arange(N_before))
        no_intersection = len(np.intersect1d(np.where(keep_mask)[0], prune_idx)) == 0

        verification = {
            'N_before': int(N_before),
            'N_after': int(N_after),
            'N_removed': int(N_removed),
            'K_target': int(metadata['methods'].get(method, {}).get('K_target', K)),
            'K_actual': int(K),
            'union_is_all': bool(is_identity),
            'no_intersection': bool(no_intersection),
            'retained_sha256': retained_sha,
            'removed_sha256': removed_sha,
        }
        with open(out_dir / 'prune_verification.json', 'w') as f:
            json.dump(verification, f, indent=2)

        status = 'OK' if (is_identity and no_intersection and N_removed == K) else 'FAIL'
        print(f"    Retained: {N_after}, Removed: {N_removed}, Verification: {status}")

    print(f"\nDone.")

if __name__ == '__main__':
    main()
