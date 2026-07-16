import argparse, sys, os, numpy as np
from pathlib import Path
sys.path.insert(0, '/data/wyh/RecycleGS/src')
from recyclegs.config import load_config, save_json
from plyfile import PlyData, PlyElement
import trimesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    debug_dir = Path(cfg['debug_output_dir']).parent / 'stage1c_scene01'
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading object domain indices...")
    object_indices = np.load(out_dir / 'object_indices.npy')
    base = np.load(out_dir / 'gaussian_base_features.npz')
    xyz = base['xyz']
    obj_xyz = xyz[object_indices]

    print(f"  Object Gaussians: {len(obj_xyz)}")

    print("[2/5] Computing object Gaussian bbox...")
    obj_bbox_min = obj_xyz.min(axis=0)
    obj_bbox_max = obj_xyz.max(axis=0)
    obj_center = (obj_bbox_min + obj_bbox_max) / 2
    obj_diameter = np.linalg.norm(obj_bbox_max - obj_bbox_min)

    print(f"  Object bbox min: {obj_bbox_min}")
    print(f"  Object bbox max: {obj_bbox_max}")
    print(f"  Object center: {obj_center}")
    print(f"  Object diameter: {obj_diameter:.4f}")

    print("[3/5] Loading GT mesh...")
    mesh_path = os.path.join(cfg['scene_dir'], 'meshes', 'scene_mesh.obj')
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh_bbox = mesh.bounds
    mesh_center = mesh.bounds.mean(axis=0)
    mesh_diameter = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

    print(f"  Mesh bbox min: {mesh_bbox[0]}")
    print(f"  Mesh bbox max: {mesh_bbox[1]}")
    print(f"  Mesh center: {mesh_center}")
    print(f"  Mesh diameter: {mesh_diameter:.4f}")

    print("[4/5] Computing alignment metrics...")
    center_dist = np.linalg.norm(obj_center - mesh_center)
    center_dist_ratio = center_dist / max(obj_diameter, mesh_diameter, 1e-8)

    print(f"  Center distance: {center_dist:.4f}")
    print(f"  Center distance / max diameter: {center_dist_ratio:.4f}")

    alignment_info = {
        'object_gaussian_count': int(len(obj_xyz)),
        'object_gaussian_bbox': {
            'min': obj_bbox_min.tolist(),
            'max': obj_bbox_max.tolist(),
            'center': obj_center.tolist(),
            'diameter': float(obj_diameter),
        },
        'mesh_bbox': {
            'min': mesh_bbox[0].tolist(),
            'max': mesh_bbox[1].tolist(),
            'center': mesh_center.tolist(),
            'diameter': float(mesh_diameter),
        },
        'center_distance': float(center_dist),
        'center_distance_ratio': float(center_dist_ratio),
    }
    save_json(alignment_info, debug_dir / 'object_domain_alignment.json')

    print("[5/5] Creating overlay PLY...")
    n_obj = min(len(obj_xyz), 100000)
    n_mesh = min(50000, len(mesh.vertices))
    obj_pts = obj_xyz[:n_obj]
    mesh_pts = mesh.vertices[:n_mesh]

    combined = np.zeros((n_obj + n_mesh, 6))
    combined[:n_obj, :3] = obj_pts
    combined[:n_obj, 3:] = [0, 255, 0]
    combined[n_obj:, :3] = mesh_pts
    combined[n_obj:, 3:] = [0, 0, 255]
    ply_arr = np.array([tuple(r) for r in combined],
                       dtype=[('x', '<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    PlyData([PlyElement.describe(ply_arr, 'vertex')]).write(str(debug_dir / 'object_mesh_overlay.ply'))
    print(f"  Saved overlay to {debug_dir / 'object_mesh_overlay.ply'}")
    print(f"  Done")

if __name__ == '__main__':
    main()
