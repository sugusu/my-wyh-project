"""
shear.py - Shear deformation
x' = x + gamma * y
y' = y
z' = z

F = [[1, gamma, 0], [0, 1, 0], [0, 0, 1]]
"""
import torch
import numpy as np

def deform_points(points, gamma):
    if isinstance(points, np.ndarray):
        out = points.copy()
        out[:, 0] = points[:, 0] + gamma * points[:, 1]
        return out
    out = points.clone()
    out[:, 0] = points[:, 0] + gamma * points[:, 1]
    return out

def jacobian(points, gamma):
    N = points.shape[0]
    device = points.device if torch.is_tensor(points) else "cpu"
    F = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).clone()
    F[:, 0, 1] = gamma
    return F

def validate_jacobian(n_points=1000, gamma=0.5, seed=20260712):
    torch.manual_seed(seed)
    pts = torch.randn(n_points, 3)
    pts.requires_grad_(True)
    def_pts = deform_points(pts, gamma)
    J_autograd = []
    for i in range(3):
        grad = torch.autograd.grad(def_pts[:, i].sum(), pts, create_graph=True)[0]
        J_autograd.append(grad.detach())
    J_analytical = jacobian(pts, gamma)
    max_err = 0.0
    for i in range(3):
        err = (J_autograd[i] - J_analytical[:, i]).abs().max().item()
        max_err = max(max_err, err)
    return max_err
