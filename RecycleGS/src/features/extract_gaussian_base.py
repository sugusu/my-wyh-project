import argparse, sys, os, torch, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config, save_npz, save_json
from recyclegs.tsgs_loader import load_scene

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])

    print("[1/3] Loading model...")
    scene, gaussians, pipe = load_scene(cfg, cfg['device'])
    device = cfg['device']

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach()
    opacity_sigmoid = opacity.cpu().numpy().squeeze()
    opacity_logit = np.log(opacity_sigmoid.clip(1e-8, 1-1e-8) / (1 - opacity_sigmoid.clip(1e-8, 1-1e-8)))

    scales = gaussians.get_scaling.detach()
    scale_log = scales.cpu().numpy()
    scale_linear = scales.exp().cpu().numpy()
    scale_min = scale_linear.min(axis=1)
    scale_max = scale_linear.max(axis=1)
    scale_ratio = scale_max / (scale_min + 1e-8)
    scale_volume = np.prod(scale_linear, axis=1)

    quats = gaussians.get_rotation.detach().cpu().numpy()
    quats_norm = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    rots = gaussian_quat_to_rotmat(quats_norm)
    min_axis = np.argmin(scale_linear, axis=1)
    normal_world = np.zeros_like(xyz)
    for i in range(len(xyz)):
        normal_world[i] = rots[i, :, min_axis[i]]

    save_npz(out_dir / 'gaussian_base_features.npz',
             xyz=xyz, opacity_logit=opacity_logit, opacity_sigmoid=opacity_sigmoid,
             scale_log=scale_log, scale_linear=scale_linear,
             scale_min=scale_min, scale_max=scale_max,
             scale_ratio=scale_ratio, scale_volume=scale_volume,
             rotation_quaternion=quats_norm, normal_world=normal_world)

    stats = {
        'num_gaussians': len(xyz),
        'xyz_mean': xyz.mean(axis=1).tolist() if xyz.ndim > 1 else xyz.mean().tolist(),
        'opacity_mean': float(opacity_sigmoid.mean()),
        'scale_ratio_mean': float(scale_ratio.mean()),
        'scale_volume_mean': float(scale_volume.mean()),
    }
    save_json(stats, out_dir / 'gaussian_base_stats.json')
    print(f"[2/3] Extracted {len(xyz)} Gaussians")
    print(f"[3/3] Saved to {out_dir / 'gaussian_base_features.npz'}")

def gaussian_quat_to_rotmat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = len(q)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    return R

if __name__ == '__main__':
    main()
