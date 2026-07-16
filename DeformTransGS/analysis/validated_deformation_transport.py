"""Validated deformation transport: F Sigma F^T with covariance roundtrip."""
from __future__ import annotations
import torch
import numpy as np


class GaussianState:
    def __init__(self, xyz, scale, rotation, tau, color, material_id):
        self.xyz = xyz
        self.scale = scale
        self.rotation = rotation
        self.tau = tau
        self.color = color
        self.material_id = material_id

    @property
    def n(self): return self.xyz.shape[0]

    def clone(self):
        return GaussianState(self.xyz.clone(), self.scale.clone(), self.rotation.clone(),
                             self.tau.clone(), self.color.clone(), self.material_id.clone())


def validate_state(g: GaussianState):
    n = g.xyz.shape[0]
    for name, shape in [("xyz", (n,3)), ("scale", (n,3)), ("rotation", (n,4)), ("color", (n,3))]:
        t = getattr(g, name)
        if tuple(t.shape) != shape:
            raise ValueError(f"{name} expected {shape} got {tuple(t.shape)}")
        if not torch.isfinite(t).all():
            raise ValueError(f"{name} non-finite")
    if g.tau.reshape(-1).shape[0] != n:
        raise ValueError("tau length")
    if g.material_id.reshape(-1).shape[0] != n:
        raise ValueError("material_id length")
    if torch.any(g.scale <= 0):
        raise ValueError("scale must be positive")
    qn = torch.linalg.norm(g.rotation, dim=1)
    if torch.max(torch.abs(qn - 1.0)) > 1e-5:
        raise ValueError("quaternion not normalized")


def quaternion_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=1)
    R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:,0,0] = 1-2*(y*y+z*z); R[:,0,1] = 2*(x*y-w*z); R[:,0,2] = 2*(x*z+w*y)
    R[:,1,0] = 2*(x*y+w*z); R[:,1,1] = 1-2*(x*x+z*z); R[:,1,2] = 2*(y*z-w*x)
    R[:,2,0] = 2*(x*z-w*y); R[:,2,1] = 2*(y*z+w*x); R[:,2,2] = 1-2*(x*x+y*y)
    return R


def rotation_matrix_to_quaternion_wxyz(R: torch.Tensor) -> torch.Tensor:
    batch = R.shape[0]
    q = torch.zeros((batch, 4), device=R.device, dtype=R.dtype)
    tr = R[:,0,0] + R[:,1,1] + R[:,2,2]
    mask = tr > 0
    # Standard method for each case
    q_r = torch.zeros((batch, 4), device=R.device, dtype=R.dtype)
    # Case 1: trace > 0
    s = torch.sqrt(tr[mask] + 1.0) * 2
    q_r[mask, 0] = 0.25 * s
    q_r[mask, 1] = (R[mask,2,1] - R[mask,1,2]) / s
    q_r[mask, 2] = (R[mask,0,2] - R[mask,2,0]) / s
    q_r[mask, 3] = (R[mask,1,0] - R[mask,0,1]) / s

    # Cases for trace <= 0
    not_mask = ~mask
    if not_mask.any():
        R_n = R[not_mask]
        # Find largest diagonal
        diag = torch.diagonal(R_n, dim1=1, dim2=2)
        max_diag = torch.argmax(diag, dim=1)
        q_n = torch.zeros((not_mask.sum(), 4), device=R.device, dtype=R.dtype)
        for case in range(3):
            c_mask = max_diag == case
            if not c_mask.any(): continue
            R_c = R_n[c_mask]
            if case == 0:  # max at [0,0]
                s = torch.sqrt(1.0 + R_c[:,0,0] - R_c[:,1,1] - R_c[:,2,2]) * 2
                q_n[c_mask, 0] = (R_c[:,2,1] - R_c[:,1,2]) / s
                q_n[c_mask, 1] = 0.25 * s
                q_n[c_mask, 2] = (R_c[:,0,1] + R_c[:,1,0]) / s
                q_n[c_mask, 3] = (R_c[:,0,2] + R_c[:,2,0]) / s
            elif case == 1:  # max at [1,1]
                s = torch.sqrt(1.0 + R_c[:,1,1] - R_c[:,0,0] - R_c[:,2,2]) * 2
                q_n[c_mask, 0] = (R_c[:,0,2] - R_c[:,2,0]) / s
                q_n[c_mask, 1] = (R_c[:,0,1] + R_c[:,1,0]) / s
                q_n[c_mask, 2] = 0.25 * s
                q_n[c_mask, 3] = (R_c[:,1,2] + R_c[:,2,1]) / s
            else:  # case 2: max at [2,2]
                s = torch.sqrt(1.0 + R_c[:,2,2] - R_c[:,0,0] - R_c[:,1,1]) * 2
                q_n[c_mask, 0] = (R_c[:,1,0] - R_c[:,0,1]) / s
                q_n[c_mask, 1] = (R_c[:,0,2] + R_c[:,2,0]) / s
                q_n[c_mask, 2] = (R_c[:,1,2] + R_c[:,2,1]) / s
                q_n[c_mask, 3] = 0.25 * s
        q_r[not_mask] = q_n
    return q_r / torch.linalg.norm(q_r, dim=1, keepdim=True).clamp_min(1e-12)


def covariance_from_scale_rotation(scale: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    if torch.any(scale <= 0):
        raise ValueError("Scale must be positive")
    R = quaternion_wxyz_to_matrix(rotation)
    D = torch.diag_embed(scale * scale)
    Sigma = R @ D @ R.transpose(1, 2)
    return 0.5 * (Sigma + Sigma.transpose(1, 2))


def transport_covariance(scale: torch.Tensor, rotation: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    Sigma = covariance_from_scale_rotation(scale, rotation)
    if F.shape != Sigma.shape:
        raise ValueError(f"F shape {F.shape} != Sigma shape {Sigma.shape}")
    Sigma_def = F @ Sigma @ F.transpose(1, 2)
    Sigma_def = 0.5 * (Sigma_def + Sigma_def.transpose(1, 2))
    if not torch.isfinite(Sigma_def).all():
        raise ValueError("Sigma_def non-finite")
    return Sigma_def


def covariance_to_scale_rotation(Sigma: torch.Tensor):
    Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2))
    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    eigvals = torch.clamp(eigvals, min=1e-12)
    scale = torch.sqrt(eigvals)
    R = eigvecs
    det = torch.linalg.det(R)
    neg = det < 0
    if torch.any(neg):
        R = R.clone()
        R[neg, :, 0] *= -1.0
    rotation = rotation_matrix_to_quaternion_wxyz(R)
    rotation = rotation / torch.linalg.norm(rotation, dim=1, keepdim=True).clamp_min(1e-12)
    return scale, rotation


def transport_gaussians_validated(g: GaussianState, deformation, material_u: torch.Tensor, material_v: torch.Tensor):
    validate_state(g)
    mid = g.material_id.reshape(-1).long()
    u = material_u[mid]
    v = material_v[mid]
    xyz_def, F, Js = deformation(xyz=g.xyz, u=u, v=v)
    if xyz_def.shape != g.xyz.shape:
        raise ValueError(f"xyz_def shape mismatch")
    if F.shape != (g.n, 3, 3):
        raise ValueError(f"F shape {F.shape}")
    if Js.reshape(-1).shape[0] != g.n:
        raise ValueError("Js shape")
    Sigma_def = transport_covariance(g.scale, g.rotation, F)
    scale_def, rotation_def = covariance_to_scale_rotation(Sigma_def)
    out = GaussianState(xyz=xyz_def, scale=scale_def, rotation=rotation_def,
                        tau=g.tau.clone(), color=g.color.clone(), material_id=g.material_id.clone())
    validate_state(out)
    return out, F, Js
