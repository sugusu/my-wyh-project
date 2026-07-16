import argparse, yaml, os, json, sys, torch
import numpy as np
from pathlib import Path
import trimesh

sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.tsgs_loader import load_scene, get_train_cameras, render_view
from scene.gaussian_model import GaussianModel
from arguments import PipelineParams
from utils.graphics_utils import focal2fov
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    debug_dir = Path(cfg['debug_output_dir'])
    debug_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get('device', 'cuda:0')

    print("[1/9] Loading scene and Gaussian model...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    print(f"  Train cameras: {len(cameras)}")

    print("[2/9] Loading GT mesh...")
    mesh_path = os.path.join(cfg['scene_dir'], 'meshes', 'scene_mesh.obj')
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh_verts = mesh.vertices
    mesh_faces = mesh.faces
    mesh_bbox_min = mesh_verts.min(axis=0)
    mesh_bbox_max = mesh_verts.max(axis=0)
    mesh_center = (mesh_bbox_min + mesh_bbox_max) / 2
    mesh_diameter = np.linalg.norm(mesh_bbox_max - mesh_bbox_min)

    print(f"  Mesh vertices: {len(mesh_verts)}")
    print(f"  Mesh bbox center: {mesh_center}")
    print(f"  Mesh diameter: {mesh_diameter:.4f}")

    print("[3/9] Computing Gaussian bbox...")
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    gauss_bbox_min = xyz.min(axis=0)
    gauss_bbox_max = xyz.max(axis=0)
    gauss_center = (gauss_bbox_min + gauss_bbox_max) / 2
    gauss_diameter = np.linalg.norm(gauss_bbox_max - gauss_bbox_min)

    print(f"  Gaussians: {len(xyz)}")
    print(f"  Gaussian bbox center: {gauss_center}")
    print(f"  Gaussian diameter: {gauss_diameter:.4f}")

    print("[4/9] Computing camera centers bbox...")
    cam_centers = []
    for cam in cameras:
        cc = cam.camera_center.detach().cpu().numpy() if hasattr(cam, 'camera_center') else np.zeros(3)
        cam_centers.append(cc)
    cam_centers = np.array(cam_centers)
    cam_bbox_min = cam_centers.min(axis=0)
    cam_bbox_max = cam_centers.max(axis=0)
    cam_bbox_center = (cam_bbox_min + cam_bbox_max) / 2

    center_dist = np.linalg.norm(gauss_center - mesh_center)
    center_distance_over_mesh_diameter = center_dist / mesh_diameter
    print(f"  Center distance: {center_dist:.4f} ({center_distance_over_mesh_diameter*100:.1f}% of mesh diameter)")

    print("[5/9] Computing distance to mesh surface...")
    sampled = mesh.sample(500000)
    from scipy.spatial import cKDTree
    tree = cKDTree(sampled)
    dists, _ = tree.query(xyz)
    median_dist = np.median(dists)
    mean_dist = np.mean(dists)
    p95_dist = np.percentile(dists, 95)

    print(f"  Median distance: {median_dist:.4f}")
    print(f"  Mean distance: {mean_dist:.4f}")
    print(f"  95% distance: {p95_dist:.4f}")

    print("[6/9] Rendering 4 representative views...")
    n_views = min(4, len(cameras))
    step = max(1, len(cameras) // n_views)
    bg_color = [1, 1, 1]
    view_data = {}

    for i in range(n_views):
        idx = i * step
        cam = cameras[idx]
        rendered = render_view(gaussians, cam, pipe, bg_color, device)
        rgb = rendered['render'].detach().cpu().numpy().transpose(1, 2, 0)
        rgb_255 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        rgb_path = debug_dir / f'view_{idx:04d}_render.png'
        Image.fromarray(rgb_255).save(rgb_path)
        view_data[f'view_{idx:04d}'] = {
            'rgb_path': str(rgb_path),
            'camera_index': idx,
        }
        print(f"  Saved view {idx}")

    print("[7/9] Saving cameras-gaussians-mesh bbox PLY...")
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
    PlyData([PlyElement.describe(all_ply, 'vertex')]).write(str(debug_dir / 'cameras_gaussians_mesh_bbox.ply'))

    print("[8/9] Generating consistency report...")
    stop_flags = []
    if center_distance_over_mesh_diameter > 0.20:
        stop_flags.append(f"CENTER_DIST_FAIL: center_distance_over_mesh_diameter {center_distance_over_mesh_diameter*100:.1f}% > 20%")
    if median_dist > 0.20 * mesh_diameter:
        stop_flags.append(f"MEDIAN_DIST_FAIL: median dist {median_dist:.4f} > 20% mesh diameter {0.20*mesh_diameter:.4f}")
    if center_distance_over_mesh_diameter > 0.50:
        stop_flags.append(f"SEVERE_MISALIGNMENT: center distance ratio {center_distance_over_mesh_diameter*100:.1f}% > 50%")

    report = {
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
        'camera_bbox_center': cam_bbox_center.tolist(),
        'num_cameras': len(cameras),
        'center_distance': float(center_dist),
        'center_distance_ratio': float(center_distance_over_mesh_diameter),
        'distance_to_mesh': {
            'median': float(median_dist),
            'mean': float(mean_dist),
            'p95': float(p95_dist),
            'min': float(dists.min()),
            'max': float(dists.max()),
        },
        'num_gaussians': int(len(xyz)),
        'stop_conditions': stop_flags,
        'passed': len(stop_flags) == 0,
        'views_rendered': view_data,
    }

    with open(debug_dir / 'coordinate_consistency.json', 'w') as f:
        json.dump(report, f, indent=2)

    md_lines = [
        f"# Coordinate Check Report - {cfg['scene_name']}",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Gaussian bbox center | {gauss_center} |",
        f"| Mesh bbox center | {mesh_center} |",
        f"| Camera bbox center | {cam_bbox_center} |",
        f"| Center distance | {center_dist:.4f} |",
        f"| center_distance / mesh_diameter | {center_distance_over_mesh_diameter*100:.1f}% |",
        f"| Gaussian diameter | {gauss_diameter:.4f} |",
        f"| Mesh diameter | {mesh_diameter:.4f} |",
        f"| Num Gaussians | {len(xyz)} |",
        f"| Num Cameras | {len(cameras)} |",
        f"| Median mesh distance | {median_dist:.4f} |",
        f"| Mean mesh distance | {mean_dist:.4f} |",
        f"| 95% mesh distance | {p95_dist:.4f} |",
        f"",
        f"## Stop Conditions",
        f"",
    ]
    if stop_flags:
        md_lines.append("### FAILED")
        for s in stop_flags:
            md_lines.append(f"- {s}")
    else:
        md_lines.append("### PASSED - All checks passed.")

    md_lines.append(f"\n## Views Rendered\n")
    for k, v in view_data.items():
        md_lines.append(f"- {k}: [RGB]({v['rgb_path']})")

    with open(debug_dir / 'coordinate_check_report.md', 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"\nReport saved to {debug_dir / 'coordinate_consistency.json'}")
    print(f"MD report saved to {debug_dir / 'coordinate_check_report.md'}")
    print(f"BBox PLY saved to {debug_dir / 'cameras_gaussians_mesh_bbox.ply'}")

    print("[9/9] Final stop check...")
    if stop_flags:
        print("\nSTOP CONDITIONS TRIGGERED:")
        for s in stop_flags:
            print(f"  - {s}")
        print("Exiting with code 1.")
        sys.exit(1)
    else:
        print("\nAll coordinate checks passed.")

if __name__ == '__main__':
    main()
