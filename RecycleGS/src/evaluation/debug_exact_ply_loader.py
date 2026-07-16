#!/usr/bin/env python3
"""Load an exact PLY file using GaussianModel.load_ply() and print stats."""
import argparse, hashlib, json, os, sys, torch, numpy as np
from pathlib import Path

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene.gaussian_model import GaussianModel

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def load_and_diagnose(ply_path, sh_degree=3, asg_degree=24, device='cuda:0'):
    print(f"\n{'='*60}")
    print(f"Loading PLY: {ply_path}")
    print(f"{'='*60}")

    if not os.path.exists(ply_path):
        print(f"ERROR: PLY not found at {ply_path}")
        return None

    sha256 = sha256_of(ply_path)
    size_bytes = os.path.getsize(ply_path)
    print(f"  SHA256: {sha256}")
    print(f"  Size: {size_bytes} bytes ({size_bytes/1024/1024:.2f} MB)")

    gaussians = GaussianModel(sh_degree, asg_degree)
    gaussians.load_ply(str(ply_path))

    print(f"  active_sh_degree after load_ply: {gaussians.active_sh_degree}")
    print(f"  max_sh_degree: {gaussians.max_sh_degree}")

    N = gaussians.get_xyz.shape[0]
    print(f"  Gaussian count (N): {N}")

    # xyz stats
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    print(f"  xyz shape: {xyz.shape}")
    print(f"  xyz sum:   {xyz.sum(axis=0)}")
    print(f"  xyz mean:  {xyz.mean(axis=0)}")
    print(f"  xyz std:   {xyz.std(axis=0)}")
    print(f"  xyz min:   {xyz.min(axis=0)}")
    print(f"  xyz max:   {xyz.max(axis=0)}")

    # f_dc stats
    f_dc = gaussians._features_dc.detach().cpu().numpy()
    print(f"  f_dc shape: {f_dc.shape}")
    print(f"  f_dc sum:   {f_dc.sum():.6f}")
    print(f"  f_dc mean:  {f_dc.mean():.6f}")
    print(f"  f_dc std:   {f_dc.std():.6f}")

    # f_rest stats
    f_rest = gaussians._features_rest.detach().cpu().numpy()
    print(f"  f_rest shape: {f_rest.shape}")
    print(f"  f_rest sum:   {f_rest.sum():.6f}")
    print(f"  f_rest mean:  {f_rest.mean():.6f}")
    print(f"  f_rest std:   {f_rest.std():.6f}")

    # opacity stats
    opacity = gaussians._opacity.detach().cpu().numpy()
    print(f"  opacity shape: {opacity.shape}")
    print(f"  opacity sum:   {opacity.sum():.6f}")
    print(f"  opacity mean:  {opacity.mean():.6f}")
    print(f"  opacity std:   {opacity.std():.6f}")

    result = {
        'ply_path': str(ply_path),
        'ply_sha256': sha256,
        'ply_size_bytes': size_bytes,
        'N': int(N),
        'xyz_sum': xyz.sum(axis=0).tolist(),
        'xyz_mean': xyz.mean(axis=0).tolist(),
        'xyz_std': xyz.std(axis=0).tolist(),
        'xyz_min': xyz.min(axis=0).tolist(),
        'xyz_max': xyz.max(axis=0).tolist(),
        'f_dc_sum': float(f_dc.sum()),
        'f_dc_mean': float(f_dc.mean()),
        'f_dc_std': float(f_dc.std()),
        'f_rest_sum': float(f_rest.sum()),
        'f_rest_mean': float(f_rest.mean()),
        'f_rest_std': float(f_rest.std()),
        'opacity_sum': float(opacity.sum()),
        'opacity_mean': float(opacity.mean()),
        'opacity_std': float(opacity.std()),
        'active_sh_degree': int(gaussians.active_sh_degree),
        'max_sh_degree': int(gaussians.max_sh_degree),
    }

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply', type=str, required=True, help='PLY file to load')
    parser.add_argument('--tag', type=str, default='', help='Tag for output')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = 'cuda:0'

    result = load_and_diagnose(args.ply, device=device)

    if result and args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        result['tag'] = args.tag
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.output}")

if __name__ == '__main__':
    main()
