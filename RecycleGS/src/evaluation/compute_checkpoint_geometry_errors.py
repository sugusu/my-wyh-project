import argparse, sys, os, numpy as np
from pathlib import Path
import trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config

def load_ply_raw(ply_path):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    props = {p.name: np.asarray(vertex[p.name]) for p in vertex.properties}
    xyz = np.stack([props['x'], props['y'], props['z']], axis=1)
    scale_names = sorted([p for p in props if p.startswith('scale_')], key=lambda x: int(x.split('_')[-1]))
    scales = np.stack([props[s] for s in scale_names], axis=1) if scale_names else np.ones((len(xyz), 3))
    return xyz, scales

def compute_geometry_errors_for_indices(xyz_all, scales_all, candidate_indices, mesh_path, obj_diameter=None):
    xyz = xyz_all[candidate_indices]
    scales_linear = np.exp(scales_all[candidate_indices]) if scales_all is not None else np.ones((len(candidate_indices), 3))
    min_scale = scales_linear.min(axis=1)

    mesh = trimesh.load(mesh_path, force='mesh')
    if obj_diameter is None:
        bbox = mesh.bounds
        obj_diameter = np.linalg.norm(bbox[1] - bbox[0])
    n_samples = 500000
    sampled, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    sampled_normals = mesh.face_normals[face_idx]

    tree = cKDTree(sampled)
    dists, idxs = tree.query(xyz)
    nearest_normals = sampled_normals[idxs]
    d_norm = dists / obj_diameter

    d_center_norm = d_norm
    d_scale = dists / (np.linalg.norm(scales_linear, axis=1) + 1e-8)
    d_surface_proxy_alpha1 = np.maximum(0, d_center_norm - 1.0 * min_scale / obj_diameter)
    d_surface_proxy_alpha2 = np.maximum(0, d_center_norm - 2.0 * min_scale / obj_diameter)

    dot = np.abs((xyz * 0 + 1) * 0)  # placeholder for normal computation later
    try:
        from recyclegs.gaussian_io import gaussian_quat_to_rotmat
    except ImportError:
        pass

    return {
        'd_center_norm': d_center_norm,
        'd_scale': d_scale,
        'd_surface_proxy_alpha1': d_surface_proxy_alpha1,
        'd_surface_proxy_alpha2': d_surface_proxy_alpha2,
        'mesh_distance': dists,
        'nearest_gt_normal': nearest_normals,
        'obj_diameter': np.array([obj_diameter]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1f_scene01'
    os.makedirs(debug_dir, exist_ok=True)
    mesh_path = os.path.join(cfg['scene_dir'], 'meshes', 'scene_mesh.obj')
    model_dir = Path(cfg['model_dir'])

    mesh = trimesh.load(mesh_path, force='mesh')
    bbox = mesh.bounds
    obj_diameter = np.linalg.norm(bbox[1] - bbox[0])
    print(f"GT mesh: {mesh_path}")
    print(f"Object diameter: {obj_diameter:.4f}")

    ckpt_paths = {
        7000: model_dir / 'point_cloud' / 'iteration_7000' / 'point_cloud.ply',
        15000: Path(cfg['checkpoint_path']),
    }

    for iteration in [7000, 15000]:
        iter_str = f"iter_{iteration}"
        print(f"\n[{iter_str}] Processing...")
        ckpt_path = ckpt_paths[iteration]
        cand_path = out_dir / iter_str / 'candidate_indices.npy'

        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found at {ckpt_path}")
            continue
        if not cand_path.exists():
            print(f"  SKIP: candidate indices not found at {cand_path}")
            continue

        candidate_indices = np.load(cand_path)
        print(f"  Candidate count: {len(candidate_indices)}")

        xyz_all, scales_all = load_ply_raw(ckpt_path)
        print(f"  Total Gaussians: {len(xyz_all)}")

        print(f"  Computing geometry errors for candidate domain...")
        err = compute_geometry_errors_for_indices(
            xyz_all, scales_all, candidate_indices, mesh_path, obj_diameter
        )

        np.savez_compressed(out_dir / iter_str / 'geometry_errors.npz',
                            d_center_norm=err['d_center_norm'],
                            d_scale=err['d_scale'],
                            d_surface_proxy_alpha1=err['d_surface_proxy_alpha1'],
                            d_surface_proxy_alpha2=err['d_surface_proxy_alpha2'],
                            mesh_distance=err['mesh_distance'],
                            nearest_gt_normal=err['nearest_gt_normal'],
                            obj_diameter=err['obj_diameter'])

        np.save(out_dir / iter_str / 'candidate_global_indices.npy', candidate_indices)
        print(f"  Saved geometry_errors.npz and candidate_global_indices.npy")

        stats = {
            'iteration': iteration,
            'candidate_count': int(len(candidate_indices)),
            'd_center_norm_mean': float(err['d_center_norm'].mean()),
            'd_center_norm_median': float(np.median(err['d_center_norm'])),
            'd_scale_mean': float(err['d_scale'].mean()),
            'd_surface_proxy_alpha1_mean': float(err['d_surface_proxy_alpha1'].mean()),
            'd_surface_proxy_alpha2_mean': float(err['d_surface_proxy_alpha2'].mean()),
        }
        import json
        with open(out_dir / iter_str / 'geometry_errors_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats: {json.dumps(stats, indent=2)}")

    print(f"\nDone.")

if __name__ == '__main__':
    main()
