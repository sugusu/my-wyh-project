#!/usr/bin/env python3
"""Export TSPE-GS depth candidates to Parquet.

This wrapper keeps the official TSPE-GS training and renderer code unchanged.
It reuses `gaussian_renderer.render_threshold` and the peak-selection logic
from `mesh_extract_opa_hotfix.py::remove_adjacent_duplicates`, but writes
machine-readable candidate rows instead of fusing a mesh.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, peak_widths
from scipy.stats import gaussian_kde


TSPE_ROOT = Path("/data/wyh/ReliablePeakGS/external/TSPE-GS")
if str(TSPE_ROOT) not in sys.path:
    sys.path.insert(0, str(TSPE_ROOT))

from arguments import GroupParams, PipelineParams  # noqa: E402
from gaussian_renderer import render_threshold  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


def make_dataset_args(source_path: Path, model_path: Path, resolution: int, eval_split: bool) -> GroupParams:
    args = GroupParams()
    args.sh_degree = 3
    args.source_path = str(source_path.resolve())
    args.model_path = str(model_path.resolve())
    args.images = "images"
    args.dataset = ""
    args.resolution = resolution
    args.white_background = False
    args.data_device = "cuda"
    args.eval = eval_split
    args.use_decoupled_appearance = True
    args.use_coord_map = False
    args.disable_filter3D = False
    args.kernel_size = 0.0
    return args


def select_official_peak_indices(depth_stack: torch.Tensor, prominence: float) -> dict[str, np.ndarray]:
    """Mirror TSPE-GS mesh_extract_opa_hotfix.py peak selection.

    The official code averages each threshold depth map over image pixels,
    performs KDE over that threshold sequence, and selects threshold indices
    nearest to KDE local maxima above a prominence-scaled threshold.
    """
    threshold_mean_depth = depth_stack.mean(dim=(1, 2)).detach().cpu().numpy()
    if not np.isfinite(threshold_mean_depth).all() or np.ptp(threshold_mean_depth) <= 0:
        return {
            "indices": np.array([], dtype=np.int64),
            "peak_positions": np.array([], dtype=np.float64),
            "peak_height": np.array([], dtype=np.float64),
            "peak_prominence": np.array([], dtype=np.float64),
            "peak_width": np.array([], dtype=np.float64),
            "density_x": np.array([], dtype=np.float64),
            "density_y": np.array([], dtype=np.float64),
        }

    kde = gaussian_kde(threshold_mean_depth)
    value_range = threshold_mean_depth.max() - threshold_mean_depth.min()
    x_eval = np.linspace(threshold_mean_depth.min() - value_range * 0.02, threshold_mean_depth.max(), 1000)
    density = kde(x_eval)
    density = density / density.sum()

    peaks, props = find_peaks(density, prominence=prominence * (density.max() - density.min()))
    peak_positions = x_eval[peaks]
    indices = np.array([int(np.argmin(np.abs(threshold_mean_depth - pos))) for pos in peak_positions], dtype=np.int64)
    widths = peak_widths(density, peaks, rel_height=0.5)[0] if len(peaks) else np.array([], dtype=np.float64)
    return {
        "indices": indices,
        "peak_positions": peak_positions.astype(np.float64),
        "peak_height": density[peaks].astype(np.float64),
        "peak_prominence": props.get("prominences", np.zeros(len(peaks))).astype(np.float64),
        "peak_width": widths.astype(np.float64),
        "density_x": x_eval.astype(np.float64),
        "density_y": density.astype(np.float64),
    }


def depth_to_world(view, depth: torch.Tensor) -> torch.Tensor:
    height, width = depth.shape
    fx = width / (2.0 * math.tan(view.FoVx / 2.0))
    fy = height / (2.0 * math.tan(view.FoVy / 2.0))
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=depth.device) + 0.5,
        torch.arange(width, dtype=torch.float32, device=depth.device) + 0.5,
        indexing="ij",
    )
    x_cam = (x - width / 2.0) / fx * depth
    y_cam = (y - height / 2.0) / fy * depth
    ones = torch.ones_like(depth)
    pts_cam = torch.stack([x_cam, y_cam, depth, ones], dim=-1).reshape(-1, 4)
    c2w = torch.inverse(view.world_view_transform)
    pts_world = pts_cam @ c2w
    pts_world = pts_world[:, :3] / pts_world[:, 3:].clamp_min(1e-8)
    return pts_world.reshape(height, width, 3)


def export_view(scene_name: str, view, gaussians, pipe, out_path: Path, thresholds: np.ndarray, prominence: float) -> dict:
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        depth_stack = render_threshold(
            view,
            gaussians,
            pipe,
            background,
            kernel_size=0.0,
            thresholds=thresholds,
        )
        if view.gt_mask is not None:
            depth_stack[:, (view.gt_mask < 0.5).squeeze(0)] = 0

        peak_info = select_official_peak_indices(depth_stack, prominence)
        selected_indices = peak_info["indices"][:8]
        rows = []
        height, width = int(view.image_height), int(view.image_width)
        pixel_x = np.tile(np.arange(width, dtype=np.int32), height)
        pixel_y = np.repeat(np.arange(height, dtype=np.int32), width)
        ray_id = np.arange(height * width, dtype=np.int64)
        mask_np = (
            view.gt_mask.squeeze().detach().cpu().numpy().reshape(-1).astype(bool)
            if view.gt_mask is not None
            else np.ones(height * width, dtype=bool)
        )

        for local_id, threshold_index in enumerate(selected_indices):
            depth = depth_stack[int(threshold_index)].contiguous()
            world = depth_to_world(view, depth).detach().cpu().numpy().reshape(-1, 3)
            depth_np = depth.detach().cpu().numpy().reshape(-1).astype(np.float32)
            valid = np.isfinite(depth_np) & (depth_np > 0)
            rows.append(
                pd.DataFrame(
                    {
                        "scene_id": scene_name,
                        "view_id": str(view.image_name),
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                        "ray_id": ray_id,
                        "candidate_id": np.full(height * width, local_id, dtype=np.int16),
                        "threshold_index": np.full(height * width, int(threshold_index), dtype=np.int16),
                        "threshold_value": np.full(height * width, float(thresholds[int(threshold_index)]), dtype=np.float32),
                        "depth": depth_np,
                        "world_x": world[:, 0].astype(np.float32),
                        "world_y": world[:, 1].astype(np.float32),
                        "world_z": world[:, 2].astype(np.float32),
                        "peak_height": np.full(height * width, float(peak_info["peak_height"][local_id]), dtype=np.float32),
                        "peak_width": np.full(height * width, float(peak_info["peak_width"][local_id]), dtype=np.float32),
                        "peak_prominence": np.full(height * width, float(peak_info["peak_prominence"][local_id]), dtype=np.float32),
                        "accumulated_opacity": np.full(height * width, np.nan, dtype=np.float32),
                        "accumulated_transmittance": np.full(height * width, np.nan, dtype=np.float32),
                        "contribution_mass": np.full(height * width, np.nan, dtype=np.float32),
                        "contributing_gaussian_count": np.full(height * width, -1, dtype=np.int32),
                        "top_contributing_gaussian_ids": "",
                        "official_candidate_probability": np.full(height * width, np.nan, dtype=np.float32),
                        "candidate_order_by_depth": np.full(height * width, local_id, dtype=np.int16),
                        "candidate_order_by_score": np.full(height * width, local_id, dtype=np.int16),
                        "foreground_mask": mask_np,
                        "transparent_region_mask": mask_np,
                        "ray_candidate_count": np.full(height * width, len(selected_indices), dtype=np.int16),
                        "near_clip": np.full(height * width, float(view.znear), dtype=np.float32),
                        "far_clip": np.full(height * width, float(view.zfar), dtype=np.float32),
                        "validity_flag": valid,
                    }
                )
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        table.to_parquet(out_path, index=False)
        return {
            "view_id": str(view.image_name),
            "path": str(out_path),
            "ray_count": height * width,
            "candidate_count": int(len(table)),
            "selected_threshold_indices": selected_indices.tolist(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--sample-number", type=int, default=512)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--prominence", type=float, default=0.01)
    args = parser.parse_args()

    dataset = make_dataset_args(args.source_path, args.model_path, args.resolution, args.eval)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    pipe = GroupParams()
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    pipe.debug = False

    thresholds = np.arange(0.0, args.threshold_max, args.threshold_max / args.sample_number, dtype=np.float32)
    manifest = []
    for view in scene.getTrainCameras():
        out_path = args.output_dir / args.scene_name / f"{view.image_name}_candidates.parquet"
        manifest.append(export_view(args.scene_name, view, gaussians, pipe, out_path, thresholds, args.prominence))

    manifest_path = args.output_dir / args.scene_name / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
