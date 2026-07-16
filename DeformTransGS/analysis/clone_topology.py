"""Clone topology module for Stage 3.4A Gaussian density topology optical invariance tests."""
from __future__ import annotations
import torch
import numpy as np


class GaussianTensors:
    """Container for Gaussian representation tensors."""
    def __init__(self, xyz, scale, rotation, tau, color, material_id):
        self.xyz = xyz
        self.scale = scale
        self.rotation = rotation
        self.tau = tau
        self.color = color
        self.material_id = material_id

    @property
    def n(self):
        return self.xyz.shape[0]

    def clone(self):
        return GaussianTensors(
            xyz=self.xyz.clone(), scale=self.scale.clone(),
            rotation=self.rotation.clone(), tau=self.tau.clone(),
            color=self.color.clone(), material_id=self.material_id.clone(),
        )

    def to(self, device):
        return GaussianTensors(
            xyz=self.xyz.to(device), scale=self.scale.to(device),
            rotation=self.rotation.to(device), tau=self.tau.to(device),
            color=self.color.to(device), material_id=self.material_id.to(device),
        )


def validate_gaussians(g: GaussianTensors):
    n = g.xyz.shape[0]
    checks = [
        (g.xyz.shape, (n, 3), "xyz"),
        (g.scale.shape, (n, 3), "scale"),
        (g.rotation.shape, (n, 4), "rotation"),
        (g.tau.reshape(-1).shape[0], n, "tau"),
        (g.color.shape, (n, 3), "color"),
        (g.material_id.reshape(-1).shape[0], n, "material_id"),
    ]
    for actual, expected, name in checks:
        if actual != expected:
            raise ValueError(f"{name} shape {actual} != {expected}")
    for name in ["xyz", "scale", "rotation", "tau", "color"]:
        t = getattr(g, name)
        if not torch.isfinite(t).all():
            raise ValueError(f"{name} contains non-finite values")


def clone_gaussians(reference: GaussianTensors, parent_indices: torch.Tensor, mode: str) -> GaussianTensors:
    """Clone selected parents. mode: naive | opacity_corrected"""
    validate_gaussians(reference)
    parent_indices = parent_indices.to(device=reference.xyz.device, dtype=torch.long).reshape(-1)
    if parent_indices.numel() == 0:
        raise ValueError("No clone parents selected")
    if parent_indices.min() < 0 or parent_indices.max() >= reference.xyz.shape[0]:
        raise ValueError("Parent index out of range")

    # Clone copies
    clone_xyz = reference.xyz[parent_indices].clone()
    clone_scale = reference.scale[parent_indices].clone()
    clone_rot = reference.rotation[parent_indices].clone()
    clone_tau = reference.tau[parent_indices].clone()
    clone_col = reference.color[parent_indices].clone()
    clone_mid = reference.material_id[parent_indices].clone()

    out_tau = reference.tau.clone()
    clone_tau_out = clone_tau.clone()

    if mode == "opacity_corrected":
        # (1-o_new)^2 = 1-o_old → tau_new = tau_old / 2
        out_tau[parent_indices] = out_tau[parent_indices] / 2.0
        clone_tau_out = out_tau[parent_indices].clone()
    elif mode != "naive":
        raise ValueError(f"Unknown mode: {mode}")

    out = GaussianTensors(
        xyz=torch.cat([reference.xyz, clone_xyz], dim=0),
        scale=torch.cat([reference.scale, clone_scale], dim=0),
        rotation=torch.cat([reference.rotation, clone_rot], dim=0),
        tau=torch.cat([out_tau, clone_tau_out], dim=0),
        color=torch.cat([reference.color, clone_col], dim=0),
        material_id=torch.cat([reference.material_id, clone_mid], dim=0),
    )
    validate_gaussians(out)
    return out


def tau_to_opacity(tau):
    return 1.0 - np.exp(-np.asarray(tau, dtype=np.float64))


def clone_unit_test():
    """Verify Revising Densification opacity correction formula."""
    tau_old = np.linspace(1e-4, 10.0, 10000)
    o_old = tau_to_opacity(tau_old)
    tau_new = tau_old / 2.0
    o_new = tau_to_opacity(tau_new)
    lhs = (1.0 - o_new) ** 2
    rhs = 1.0 - o_old
    max_err = np.max(np.abs(lhs - rhs))
    return max_err
