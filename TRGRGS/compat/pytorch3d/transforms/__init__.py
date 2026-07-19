import torch

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """PyTorch3D-compatible real-first quaternion conversion."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1-two_s*(j*j+k*k), two_s*(i*j-k*r), two_s*(i*k+j*r),
        two_s*(i*j+k*r), 1-two_s*(i*i+k*k), two_s*(j*k-i*r),
        two_s*(i*k-j*r), two_s*(j*k+i*r), 1-two_s*(i*i+j*j)), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))

