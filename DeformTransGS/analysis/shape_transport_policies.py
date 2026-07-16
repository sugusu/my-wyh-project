"""Shape transport policies: P0 fixed, P1 rigid, P2 full affine, P3 oracle."""
from __future__ import annotations
import torch

from analysis.validated_deformation_transport import (
    GaussianState, validate_state,
    covariance_from_scale_rotation, transport_covariance, covariance_to_scale_rotation,
    quaternion_wxyz_to_matrix, rotation_matrix_to_quaternion_wxyz,
)


def polar_rotation(F: torch.Tensor) -> torch.Tensor:
    """Polar decomposition F = R U → extract proper rotation R."""
    U, _, Vh = torch.linalg.svd(F)
    R = U @ Vh
    det = torch.linalg.det(R)
    neg = det < 0
    if torch.any(neg):
        U = U.clone()
        U[neg, :, -1] *= -1.0
        R = U @ Vh
    det_after = torch.linalg.det(R)
    if torch.max(torch.abs(det_after - 1.0)) > 1e-5:
        raise RuntimeError("polar rotation determinant invalid")
    ortho = R @ R.transpose(1, 2)
    eye3 = torch.eye(3, device=F.device, dtype=F.dtype).unsqueeze(0)
    if torch.max(torch.abs(ortho - eye3)) > 1e-5:
        raise RuntimeError("polar rotation not orthogonal")
    return R


def transport_p0_fixed(g: GaussianState, xyz_def: torch.Tensor) -> GaussianState:
    """P0: only xyz changes, scale/rotation fixed."""
    return GaussianState(xyz=xyz_def, scale=g.scale.clone(), rotation=g.rotation.clone(),
                         tau=g.tau.clone(), color=g.color.clone(), material_id=g.material_id.clone())


def transport_p1_rigid(g: GaussianState, xyz_def: torch.Tensor, F: torch.Tensor) -> GaussianState:
    """P1: polar rotation of covariance only."""
    Sigma_can = covariance_from_scale_rotation(g.scale, g.rotation)
    R_pol = polar_rotation(F)
    Sigma_def = R_pol @ Sigma_can @ R_pol.transpose(1, 2)
    Sigma_def = 0.5 * (Sigma_def + Sigma_def.transpose(1, 2))
    scale_def, rot_def = covariance_to_scale_rotation(Sigma_def)
    return GaussianState(xyz=xyz_def, scale=scale_def, rotation=rot_def,
                         tau=g.tau.clone(), color=g.color.clone(), material_id=g.material_id.clone())


def transport_p2_full(g: GaussianState, xyz_def: torch.Tensor, F: torch.Tensor) -> GaussianState:
    """P2: full F Sigma F^T transport."""
    Sigma_def = transport_covariance(g.scale, g.rotation, F)
    scale_def, rot_def = covariance_to_scale_rotation(Sigma_def)
    return GaussianState(xyz=xyz_def, scale=scale_def, rotation=rot_def,
                         tau=g.tau.clone(), color=g.color.clone(), material_id=g.material_id.clone())


def transport_p3_oracle(g: GaussianState, xyz_def: torch.Tensor, F: torch.Tensor, Js: torch.Tensor) -> GaussianState:
    """P3: P2 geometry + tau' = tau / Js (oracle diagnostic only)."""
    p2 = transport_p2_full(g, xyz_def, F)
    Js_clamped = Js.reshape(-1).clamp_min(1e-8)
    p2.tau = g.tau.reshape(-1).clone() / Js_clamped
    return p2
