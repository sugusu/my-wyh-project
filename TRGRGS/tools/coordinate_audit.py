#!/usr/bin/env python3
import argparse, hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trgr.camera_geometry import load_colmap_cameras, project, roundtrip, world_to_camera


def rasterize(vertices, faces, camera, scale=0.5):
    xyz = world_to_camera(vertices, camera.world_to_camera)
    uv, z = project(xyz, camera.K)
    uv *= scale
    good = np.all(z[faces] > 1e-8, axis=1)
    h, w = round(camera.height * scale), round(camera.width * scale)
    polys_float = uv[faces[good]]
    finite = np.all(np.isfinite(polys_float), axis=(1, 2))
    polys_float = np.clip(polys_float[finite], [-2*w, -2*h], [3*w, 3*h])
    polys = np.rint(polys_float).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    # Silhouette is the union of projected front-of-camera triangles; ICP/alignment is forbidden.
    if len(polys):
        cv2.fillPoly(mask, polys, 255)
    return mask


def load_transparent_mask(path, shape):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim == 3:
        # Transparent-mask PNGs may be RGB/RGBA; use alpha only when it is informative.
        alpha = raw[:, :, 3] if raw.shape[2] == 4 else None
        raw = alpha if alpha is not None and np.ptp(alpha) > 0 else np.max(raw[:, :, :3], axis=2)
    raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (raw > 127).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs/scene01_dev.yaml", type=Path)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    paths, acfg = cfg["paths"], cfg["coordinate_audit"]
    scene, tsgs = Path(paths["scene_path"]), Path(paths["tsgs_root"])
    out = Path(paths["output_root"]) / "coordinate_audit"; out.mkdir(parents=True, exist_ok=True)
    mesh_scene = trimesh.load(paths["gt_mesh"], force="scene", process=False)
    # The OBJ contains the complete room. The transparent mask corresponds only
    # to submeshes whose MTL material has d < 1 (scene_01: Material.011).
    mtl = Path(paths["gt_mesh"]).with_suffix(".mtl").read_text(errors="replace").splitlines()
    transparent_materials, current = set(), None
    for line in mtl:
        fields = line.split()
        if fields[:1] == ["newmtl"]: current = fields[1]
        elif fields[:1] == ["d"] and current and float(fields[1]) < 0.999:
            transparent_materials.add(current)
    parts = [g for g in mesh_scene.geometry.values()
             if getattr(getattr(g.visual, "material", None), "name", None) in transparent_materials]
    if not parts:
        raise RuntimeError(f"No transparent GT submesh found from {sorted(transparent_materials)}")
    mesh = trimesh.util.concatenate(parts)
    vertices_gt, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    diameter = float(np.linalg.norm(vertices_gt.max(0) - vertices_gt.min(0)))
    # TransLab's official evaluator converts predictions COLMAP -> Blender as
    # (x,y,z)->(x,z,-y). For projection into COLMAP cameras apply its exact inverse.
    # This fixed dataset convention is not ICP and never changes model coordinates.
    vertices = vertices_gt[:, [0, 2, 1]].copy()
    vertices[:, 1] *= -1
    cameras = load_colmap_cameras(scene, tsgs)
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    points = rng.uniform(vertices.min(0), vertices.max(0), (acfg["roundtrip_samples"], 3))
    rt = []
    for cam in cameras:
        error = np.linalg.norm(roundtrip(points, cam) - points, axis=1)
        rt.append(float(np.median(error)))
    ious, views = [], []
    for idx, cam in enumerate(cameras):
        pred = rasterize(vertices, faces, cam, args.scale)
        gt = load_transparent_mask(scene / acfg["mask_folder"] / Path(cam.name).with_suffix(".png"), pred.shape)
        inter = np.count_nonzero((pred > 0) & (gt > 0)); union = np.count_nonzero((pred > 0) | (gt > 0))
        iou = float(inter / union) if union else 1.0
        ious.append(iou); views.append({"name": cam.name, "iou": iou})
        image = cv2.imread(str(scene / "images" / cam.name), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (pred.shape[1], pred.shape[0]))
        overlay = image.copy(); overlay[pred > 0] = (0, 255, 0); overlay[gt > 0] = (0, 0, 255)
        overlay[(pred > 0) & (gt > 0)] = (0, 255, 255)
        cv2.addWeighted(overlay, acfg["overlay_alpha"], image, 1-acfg["overlay_alpha"], 0,
                        dst=overlay)
        cv2.imwrite(str(out / f"view_{idx:03d}_mesh_mask_overlay.png"), overlay)
    med_rt, med_iou = float(np.median(rt)), float(np.median(ious))
    rt_pass = med_rt < acfg["roundtrip_relative_tolerance"] * diameter
    iou_pass = med_iou >= acfg["median_iou_threshold"]
    report = {"stage": "0.5", "status": "PASS" if rt_pass and iou_pass else "FAIL",
              "prohibitions": {"icp_used": False, "gt_modified_model_coordinates": False},
              "gt_mesh_to_camera_convention": "official inverse Blender-to-COLMAP: (x,y,z)->(x,-z,y)",
              "transparent_gt_materials": sorted(transparent_materials),
              "camera_count": len(cameras), "mesh_vertices": len(vertices), "mesh_faces": len(faces),
              "scene_diameter": diameter, "median_roundtrip_error": med_rt,
              "roundtrip_relative_error": med_rt / diameter, "roundtrip_pass": rt_pass,
              "median_mesh_mask_iou": med_iou, "iou_threshold": acfg["median_iou_threshold"],
              "iou_pass": iou_pass, "views": views,
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "reports/stage05_coordinate_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "median_roundtrip_error", "roundtrip_relative_error", "median_mesh_mask_iou")}, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__": main()
