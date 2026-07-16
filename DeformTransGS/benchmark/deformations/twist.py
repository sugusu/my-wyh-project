"""
twist.py - Twist deformation around z-axis
theta(z) = theta_max * z_norm
x' = cos(theta)*x - sin(theta)*y
y' = sin(theta)*x + cos(theta)*y
z' = z
"""
import torch
import numpy as np

def z_normalize(z, z_range=None):
    if z_range is None:
        z_min, z_max = z.min(), z.max()
    else:
        z_min, z_max = z_range
    z_norm = 2 * (z - z_min) / (z_max - z_min + 1e-8) - 1
    return z_norm.clamp(-1, 1)

def deform_points(points, theta_max_deg, z_range=None):
    theta_max = torch.tensor(theta_max_deg * np.pi / 180.0, dtype=points.dtype, device=points.device if torch.is_tensor(points) else 'cpu')
    z = points[:, 2]
    z_norm = z_normalize(z, z_range)
    theta = theta_max * z_norm
    ct = theta.cos(); st = theta.sin()
    if isinstance(points, np.ndarray):
        out = points.copy()
        x, y = points[:, 0], points[:, 1]
        out[:, 0] = ct.cpu().numpy() * x - st.cpu().numpy() * y
        out[:, 1] = st.cpu().numpy() * x + ct.cpu().numpy() * y
        return out
    out = points.clone()
    x, y = points[:, 0], points[:, 1]
    out[:, 0] = ct * x - st * y
    out[:, 1] = st * x + ct * y
    return out

def jacobian(points, theta_max_deg, z_range=None):
    N = points.shape[0]
    device = points.device if torch.is_tensor(points) else "cpu"
    is_np = isinstance(points, np.ndarray)
    if is_np:
        pts_t = torch.tensor(points, device=device)
    else:
        pts_t = points

    z = pts_t[:, 2]
    if z_range is None:
        z_min, z_max = z.min(), z.max()
    else:
        z_min, z_max = z_range
    eps = 1e-8
    z_norm = 2 * (z - z_min) / (z_max - z_min + eps) - 1
    z_norm = z_norm.clamp(-1, 1)
    theta_max_rad = theta_max_deg * np.pi / 180.0
    theta = torch.tensor(theta_max_rad, device=device) * z_norm
    ct = theta.cos(); st = theta.sin()
    x, y = pts_t[:, 0], pts_t[:, 1]

    # Deformed positions for Jacobian dz terms
    xp = ct * x - st * y
    yp = st * x + ct * y

    # d(theta)/dz = 2 * theta_max_rad / (z_max - z_min)
    a = 2.0 * theta_max_rad / (z_max - z_min + eps)

    F = torch.zeros(N, 3, 3, device=device)
    F[:, 0, 0] = ct; F[:, 0, 1] = -st; F[:, 0, 2] = -a * yp
    F[:, 1, 0] = st; F[:, 1, 1] =  ct; F[:, 1, 2] =  a * xp
    F[:, 2, 2] = 1.0
    return F

def validate_jacobian(n_points=1000, theta_max_deg=30, seed=20260712, z_range=None):
    torch.manual_seed(seed)
    pts = torch.randn(n_points, 3)
    pts.requires_grad_(True)
    def_pts = deform_points(pts, theta_max_deg, z_range=z_range)
    J_autograd = []
    for i in range(3):
        grad = torch.autograd.grad(def_pts[:, i].sum(), pts, create_graph=True)[0]
        J_autograd.append(grad.detach())
    J_analytical = jacobian(pts, theta_max_deg, z_range=z_range)
    max_err = 0.0
    for i in range(3):
        err = (J_autograd[i] - J_analytical[:, i]).abs().max().item()
        max_err = max(max_err, err)
    return max_err
