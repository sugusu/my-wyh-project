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

    print("[1/6] Loading object domain indices...")
    object_indices = np.load(out_dir / 'object_indices.npy')
    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = base['xyz'][object_indices]
    normals = base['normal_world'][object_indices]
    scales_linear = base['scale_linear'][object_indices]
    print(f"  Object Gaussians: {len(xyz)}")

    print("[2/6] Loading GT mesh...")
    mesh_path = os.path.join(cfg['scene_dir'], 'meshes', 'scene_mesh.obj')
    mesh = trimesh.load(mesh_path, force='mesh')
    bbox = mesh.bounds
    obj_diameter = np.linalg.norm(bbox[1] - bbox[0])
    print(f"  Object diameter: {obj_diameter:.4f}")

    print("[3/6] Sampling mesh surface...")
    n_samples = cfg['mesh_evaluation']['sample_points']
    sampled, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    sampled_normals = mesh.face_normals[face_idx]

    print("[4/6] Computing distances (object domain only)...")
    tree = cKDTree(sampled)
    dists, idxs = tree.query(xyz)
    nearest_normals = sampled_normals[idxs]
    dot = np.abs((normals * nearest_normals).sum(axis=1))
    normal_angular = np.arccos(dot.clip(0, 1))

    d_norm = dists / obj_diameter
    scale_norm = dists / (np.linalg.norm(scales_linear, axis=1) + 1e-8)

    save_npz(out_dir / 'object_gaussian_gt_errors.npz',
             mesh_distance=dists, normalized_mesh_distance=d_norm,
             scale_normalized_distance=scale_norm,
             nearest_gt_normal=nearest_normals,
             normal_angular_error=normal_angular,
             obj_diameter=np.array([obj_diameter]))

    correct_th = cfg['mesh_evaluation']['correct_threshold_norm']
    wrong_th = cfg['mesh_evaluation']['wrong_threshold_norm']

    print("[5/6] Computing threshold schemes...")
    schemes = {
        'A_default': {'correct': correct_th, 'wrong': wrong_th},
        'B_strict': {'correct': correct_th * 0.5, 'wrong': wrong_th * 0.5},
        'C_lenient': {'correct': correct_th * 2.0, 'wrong': wrong_th * 2.0},
    }

    stats = {
        'object_diameter': float(obj_diameter),
        'num_object_gaussians': len(xyz),
        'distance_stats': {
            'min': float(dists.min()), 'max': float(dists.max()),
            'mean': float(dists.mean()), 'median': float(np.median(dists)),
        },
        'schemes': {},
    }

    for name, th in schemes.items():
        labels = np.where(d_norm < th['correct'], 0,
                         np.where(d_norm < th['wrong'], 1, 2))
        stats['schemes'][name] = {
            'thresholds': th,
            'correct': int((labels == 0).sum()),
            'ambiguous': int((labels == 1).sum()),
            'wrong': int((labels == 2).sum()),
            'wrong_ratio': float((labels == 2).mean()),
        }
        print(f"  {name}: correct={stats['schemes'][name]['correct']}, "
              f"ambiguous={stats['schemes'][name]['ambiguous']}, "
              f"wrong={stats['schemes'][name]['wrong']}")

    save_json(stats, out_dir / 'object_gt_error_stats.json')

    colored = np.zeros((len(xyz), 6))
    colored[:, :3] = xyz
    labels_default = np.where(d_norm < correct_th, 0, np.where(d_norm < wrong_th, 1, 2))
    for i, label in enumerate(labels_default):
        if label == 0: colored[i, 3:] = [0, 255, 0]
        elif label == 1: colored[i, 3:] = [255, 255, 0]
        else: colored[i, 3:] = [255, 0, 0]
    from plyfile import PlyData, PlyElement
    ply_arr = np.array([tuple(r) for r in colored], dtype=[('x', '<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    PlyData([PlyElement.describe(ply_arr, 'vertex')]).write(str(out_dir / 'object_gt_distance_colored.ply'))

    print(f"[6/6] Done. Saved to {out_dir}")

if __name__ == '__main__':
    main()
