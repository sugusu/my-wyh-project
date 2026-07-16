import argparse, sys, os, json, torch
import numpy as np
from pathlib import Path
import trimesh
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config
from recyclegs.tsgs_loader import load_scene, get_train_cameras, render_view
from arguments import PipelineParams
from scipy.spatial import cKDTree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['debug_output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get('device', 'cuda:0')

    print("[1/6] Loading scene and Gaussian model...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    print(f"  Train cameras: {len(cameras)}")

    print("[2/6] Loading GT mesh...")
    mesh_path = os.path.join(cfg['scene_dir'], 'meshes', 'scene_mesh.obj')
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh_verts = mesh.vertices
    mesh_bbox_min = mesh_verts.min(axis=0)
    mesh_bbox_max = mesh_verts.max(axis=0)
    mesh_center = (mesh_bbox_min + mesh_bbox_max) / 2
    mesh_diameter = np.linalg.norm(mesh_bbox_max - mesh_bbox_min)
    print(f"  Mesh vertices: {len(mesh_verts)}, diameter: {mesh_diameter:.4f}")

    print("[3/6] Computing Gaussian bbox...")
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    gauss_bbox_min = xyz.min(axis=0)
    gauss_bbox_max = xyz.max(axis=0)
    gauss_center = (gauss_bbox_min + gauss_bbox_max) / 2
    gauss_diameter = np.linalg.norm(gauss_bbox_max - gauss_bbox_min)
    print(f"  Gaussians: {len(xyz)}, diameter: {gauss_diameter:.4f}")

    print("[4/6] Computing COLMAP camera centers bbox...")
    cam_centers = []
    for cam in cameras:
        cc = cam.camera_center.detach().cpu().numpy() if hasattr(cam, 'camera_center') else np.zeros(3)
        cam_centers.append(cc)
    cam_centers = np.array(cam_centers)
    cam_bbox_min = cam_centers.min(axis=0)
    cam_bbox_max = cam_centers.max(axis=0)
    cam_center_avg = cam_centers.mean(axis=0)
    print(f"  Cameras: {len(cam_centers)}")

    print("[5/6] Computing distance percentiles...")
    sampled = mesh.sample(500000)
    tree = cKDTree(sampled)
    dists, _ = tree.query(xyz)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    dist_percentiles = {str(p): float(np.percentile(dists, p)) for p in percentiles}

    print("[6/6] Saving results...")
    n_gauss_sample = min(5000, len(xyz))
    rng = np.random.RandomState(42)
    gauss_sample = xyz[rng.choice(len(xyz), n_gauss_sample, replace=False)]
    n_mesh_sample = min(5000, len(mesh_verts))
    mesh_sample = mesh_verts[rng.choice(len(mesh_verts), n_mesh_sample, replace=False)]
    n_cam_sample = min(500, len(cam_centers))
    cam_sample = cam_centers[rng.choice(len(cam_centers), n_cam_sample, replace=False)]

    def make_ply(points, r, g, b):
        arr = np.zeros(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        arr['x'] = points[:, 0]; arr['y'] = points[:, 1]; arr['z'] = points[:, 2]
        arr['r'] = r; arr['g'] = g; arr['b'] = b
        return arr
    all_ply = np.concatenate([
        make_ply(gauss_sample, 0, 0, 255),
        make_ply(mesh_sample, 0, 255, 0),
        make_ply(cam_sample, 255, 0, 0),
    ])
    from plyfile import PlyData, PlyElement
    PlyData([PlyElement.describe(all_ply, 'vertex')]).write(str(out_dir / 'cameras_gaussians_mesh_bbox.ply'))

    diagnosis = {
        'scene_name': cfg['scene_name'],
        'gaussian_bbox_min': gauss_bbox_min.tolist(),
        'gaussian_bbox_max': gauss_bbox_max.tolist(),
        'gaussian_center': gauss_center.tolist(),
        'gaussian_diameter': float(gauss_diameter),
        'mesh_bbox_min': mesh_bbox_min.tolist(),
        'mesh_bbox_max': mesh_bbox_max.tolist(),
        'mesh_center': mesh_center.tolist(),
        'mesh_diameter': float(mesh_diameter),
        'camera_bbox_min': cam_bbox_min.tolist(),
        'camera_bbox_max': cam_bbox_max.tolist(),
        'camera_center_avg': cam_center_avg.tolist(),
        'num_gaussians': int(len(xyz)),
        'num_mesh_vertices': int(len(mesh_verts)),
        'num_cameras': int(len(cam_centers)),
        'center_distance_gauss_mesh': float(np.linalg.norm(gauss_center - mesh_center)),
        'distance_percentiles': dist_percentiles,
    }
    with open(out_dir / 'coordinate_alignment_diagnosis.json', 'w') as f:
        json.dump(diagnosis, f, indent=2)

    md = [
        f"# Coordinate Alignment Diagnosis - {cfg['scene_name']}",
        f"",
        f"## Bounding Boxes",
        f"| Entity | Min | Max | Center |",
        f"|--------|-----|-----|--------|",
        f"| Gaussians | {gauss_bbox_min} | {gauss_bbox_max} | {gauss_center} |",
        f"| Mesh | {mesh_bbox_min} | {mesh_bbox_max} | {mesh_center} |",
        f"| Cameras | {cam_bbox_min} | {cam_bbox_max} | {cam_center_avg} |",
        f"",
        f"## Diameters",
        f"- Gaussian: {gauss_diameter:.4f}",
        f"- Mesh: {mesh_diameter:.4f}",
        f"- Ratio: {gauss_diameter / mesh_diameter:.4f}",
        f"",
        f"## Center Distance",
        f"- |gauss_center - mesh_center|: {diagnosis['center_distance_gauss_mesh']:.4f}",
        f"- Relative to mesh diameter: {diagnosis['center_distance_gauss_mesh'] / mesh_diameter * 100:.1f}%",
        f"",
        f"## Distance Percentiles (Gaussian to Mesh Surface)",
    ]
    for p, v in dist_percentiles.items():
        md.append(f"- p{p}: {v:.6f}")
    md.append(f"")
    md.append(f"## Files Saved")
    md.append(f"- cameras_gaussians_mesh_bbox.ply: blue=gaussian, green=mesh, red=camera")
    md.append(f"- coordinate_alignment_diagnosis.json")
    md.append(f"- coordinate_alignment_report.md")

    with open(out_dir / 'coordinate_alignment_report.md', 'w') as f:
        f.write('\n'.join(md))

    print(f"  Saved to {out_dir}")
    print("Done (no ICP applied).")

if __name__ == '__main__':
    main()
