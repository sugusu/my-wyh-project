"""Exact CUDA-matched projection for TSGS FirstSurface rasterizer.

Reproduces the CUDA transformPoint4x4 + perspective divide + ndc2Pix.
"""
from __future__ import annotations
import torch


def project_points_cuda_exact(
    xyz: torch.Tensor,
    camera,
) -> dict[str, torch.Tensor]:
    """
    Reproduce current TSGS CUDA rasterizer projection.

    Args:
        xyz: [N,3] world-space points (float).
        camera: TSGS Camera with .full_proj_transform (4x4),
                .image_width, .image_height (int).

    Returns:
        dict with keys:
          pixel_x, pixel_y  [N]  float pixel coordinates
          ndc_z             [N]  NDC depth
          hom_w             [N]  homogeneous w (positive → in front)
          in_frame          [N]  bool: pixel inside [0,W)x[0,H)
          finite            [N]  bool: all values finite
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be [N,3], got {tuple(xyz.shape)}")

    xyz = xyz.to(
        device=camera.full_proj_transform.device,
        dtype=camera.full_proj_transform.dtype,
    )

    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)  # [N,4]

    M = camera.full_proj_transform  # [4,4]
    if M.shape != (4, 4):
        raise ValueError(f"full_proj_transform must be [4,4], got {tuple(M.shape)}")

    # CUDA transformPoint4x4 reads column-major matrix.
    # In TSGS Camera, full_proj_transform is already transposed
    # so that xyz_h @ M equals transformPoint4x4(p_orig, projmatrix).
    p_hom = xyz_h @ M  # [N,4]

    inv_w = 1.0 / (p_hom[:, 3] + 1e-7)
    ndc_x = p_hom[:, 0] * inv_w
    ndc_y = p_hom[:, 1] * inv_w
    ndc_z = p_hom[:, 2] * inv_w

    W = int(camera.image_width)
    H = int(camera.image_height)

    # Exact CUDA ndc2Pix: ((v + 1.0) * S - 1.0) * 0.5
    pixel_x = ((ndc_x + 1.0) * W - 1.0) * 0.5
    pixel_y = ((ndc_y + 1.0) * H - 1.0) * 0.5

    in_frame = (
        (pixel_x >= 0.0) & (pixel_x < W) &
        (pixel_y >= 0.0) & (pixel_y < H)
    )

    finite = (
        torch.isfinite(pixel_x) & torch.isfinite(pixel_y) &
        torch.isfinite(ndc_z) & torch.isfinite(p_hom[:, 3])
    )

    return {
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "ndc_z": ndc_z,
        "hom_w": p_hom[:, 3],
        "in_frame": in_frame,
        "finite": finite,
    }


def assert_no_double_view_transform(module_paths: list[str]) -> list[str]:
    """Scan source files for double-view-transform patterns."""
    import re
    issues = []
    patterns = [
        (r"world_view_transform.*full_proj_transform", "double view + full_proj"),
        (r"full_proj_transform.*world_view_transform", "full_proj + double view"),
        (r"projection_matrix.*world_view_transform", "proj @ view (use full_proj)"),
        (r"world_view_transform.*projection_matrix", "view @ proj (use full_proj)"),
    ]
    for path in module_paths:
        try:
            with open(path) as f:
                content = f.read()
            for pat, desc in patterns:
                if re.search(pat, content):
                    issues.append(f"{path}: {desc}")
        except FileNotFoundError:
            issues.append(f"{path}: not found")
    return issues
