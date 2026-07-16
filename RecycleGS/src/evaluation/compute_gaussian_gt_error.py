import argparse, sys, os, numpy as np
from pathlib import Path
import trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, '/data/wyh/RecycleGS/src')
from recyclegs.config import load_config, save_npz, save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])

    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = base['xyz']
    normals = base['normal_world']

    print("[1/5] Loading GT mesh...")
    mesh_path = os.path.join(cfg['scene_dir'], 'meshes', 'scene_mesh.obj')
    mesh = trimesh.load(mesh_path, force='mesh')
    bbox = mesh.bounds
    obj_diameter = np.linalg.norm(bbox[1] - bbox[0])
    print(f"  Object diameter: {obj_diameter:.4f}")

    print("[2/5] Sampling mesh surface...")
    n_samples = cfg['mesh_evaluation']['sample_points']
    sampled, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    sampled_normals = mesh.face_normals[face_idx]
    print(f"  Sampled {len(sampled)} points")

    print("[3/5] Computing distances...")
    tree = cKDTree(sampled)
    dists, idxs = tree.query(xyz)
    nearest_normals = sampled_normals[idxs]
    dot = np.abs((normals * nearest_normals).sum(axis=1))
    normal_angular = np.arccos(dot.clip(0, 1))

    d_norm = dists / obj_diameter
    scale_norm = dists / (np.linalg.norm(base['scale_linear'], axis=1) + 1e-8)

    save_npz(out_dir / 'gaussian_gt_errors.npz',
             mesh_distance=dists, normalized_mesh_distance=d_norm,
             scale_normalized_distance=scale_norm,
             nearest_gt_normal=nearest_normals,
             normal_angular_error=normal_angular,
             obj_diameter=np.array([obj_diameter]))

    correct_th = cfg['mesh_evaluation']['correct_threshold_norm']
    wrong_th = cfg['mesh_evaluation']['wrong_threshold_norm']
    labels = np.where(d_norm < correct_th, 0, np.where(d_norm < wrong_th, 1, 2))
    stats = {
        'object_diameter': float(obj_diameter),
        'num_gaussians': len(xyz),
        'distance_stats': {
            'min': float(dists.min()), 'max': float(dists.max()),
            'mean': float(dists.mean()), 'median': float(np.median(dists)),
        },
        'labels': {
            'correct (<{:.3f})'.format(correct_th): int((labels == 0).sum()),
            'ambiguous ({:.3f}-{:.3f})'.format(correct_th, wrong_th): int((labels == 1).sum()),
            'wrong (>{:.3f})'.format(wrong_th): int((labels == 2).sum()),
        }
    }
    save_json(stats, out_dir / 'gt_error_stats.json')

    colored = np.zeros((len(xyz), 6))
    colored[:, :3] = xyz
    for i, label in enumerate(labels):
        if label == 0: colored[i, 3:] = [0, 1, 0]
        elif label == 1: colored[i, 3:] = [1, 1, 0]
        else: colored[i, 3:] = [1, 0, 0]
    from plyfile import PlyData, PlyElement
    ply_arr = np.array([tuple(r) for r in colored], dtype=[('x', '<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    PlyData([PlyElement.describe(ply_arr, 'vertex')]).write(str(out_dir / 'gt_distance_colored.ply'))
    print(f"[4/5] GT errors computed")
    print(f"[5/5] Saved to {out_dir}")

if __name__ == '__main__':
    main()
