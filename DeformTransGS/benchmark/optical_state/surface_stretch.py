"""
surface_stretch.py
Computes local surface area stretch ratio Js from deformation gradient F and normal n.

Js = |det(F)| * ||F^{-T} n||

For an incompressible thin material:
h' / h0 = 1 / Js
"""
import torch
import numpy as np

def compute_Js(F, normal):
    """
    Args:
        F: (N, 3, 3) deformation gradient
        normal: (N, 3) unit normal vectors
    Returns:
        Js: (N,) surface area stretch ratio
    """
    det_F = torch.linalg.det(F)  # (N,)
    F_inv_T = torch.linalg.inv(F).transpose(1, 2)  # (N, 3, 3)
    n_transformed = F_inv_T @ normal.unsqueeze(-1)  # (N, 3, 1)
    n_transformed_norm = n_transformed.squeeze(-1).norm(dim=1)  # (N,)
    Js = det_F.abs() * n_transformed_norm
    return Js

def h_ratio(Js):
    """Thickness ratio for incompressible thin material: h'/h0 = 1/Js"""
    return 1.0 / Js.clamp(min=1e-8)


# === Validation functions ===

def validate_rotation():
    """Pure rotation should give Js ≈ 1"""
    N = 1000
    torch.manual_seed(20260712)
    normals = torch.randn(N, 3)
    normals = normals / normals.norm(dim=1, keepdim=True)
    
    # Random rotation matrices
    angles = torch.rand(N) * 2 * np.pi
    axis = torch.randn(N, 3)
    axis = axis / axis.norm(dim=1, keepdim=True)
    # Build rotation matrices (Rodrigues)
    R = torch.zeros(N, 3, 3)
    for i in range(N):
        a = angles[i]
        k = axis[i]
        K = torch.zeros(3, 3)
        K[0, 1] = -k[2]; K[0, 2] = k[1]
        K[1, 0] = k[2]; K[1, 2] = -k[0]
        K[2, 0] = -k[1]; K[2, 1] = k[0]
        R[i] = torch.eye(3) + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)
    
    Js = compute_Js(R, normals)
    return (Js - 1.0).abs().max().item()

def validate_uniform_scale(s):
    """Uniform scaling F = sI should give Js = s^2"""
    N = 1000
    normals = torch.randn(N, 3)
    normals = normals / normals.norm(dim=1, keepdim=True)
    F = torch.eye(3).unsqueeze(0).expand(N, -1, -1).clone() * s
    Js = compute_Js(F, normals)
    expected = s ** 2
    return (Js - expected).abs().max().item()

def validate_anisotropic(sx, sy, sz):
    """For normal = [0, 0, 1], Js should be sx * sy"""
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    F = torch.diag(torch.tensor([sx, sy, sz])).unsqueeze(0)
    Js = compute_Js(F, normal)
    expected = sx * sy
    return (Js.item() - expected)

def validate_shear(gamma):
    """For normal = [0, 0, 1] and F = [[1, gamma, 0], [0, 1, 0], [0, 0, 1]], Js should be 1"""
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    F = torch.eye(3).unsqueeze(0).clone()
    F[0, 0, 1] = gamma
    Js = compute_Js(F, normal)
    return abs(Js.item() - 1.0)


if __name__ == "__main__":
    import argparse
    print("=== Surface Area Stretch Validation ===")
    print(f"Rotation max error: {validate_rotation():.6e}")
    for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        print(f"Uniform scale s={s:.2f}: max error={validate_uniform_scale(s):.6e}")
    for sx in [1.0, 1.25, 1.5, 2.0]:
        err = validate_anisotropic(sx, 1.0, 1.0)
        print(f"Anisotropic sx={sx:.2f}, sy=1.0: error={err:.6e}")
    for gamma in [0.1, 0.25, 0.5, 1.0]:
        err = validate_shear(gamma)
        print(f"Shear gamma={gamma:.2f}: error={err:.6e}")
