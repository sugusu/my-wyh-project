from __future__ import annotations

import math
import numpy as np
import torch

ALPHA_SKIP = 1.0 / 255.0
ALPHA_MAX = 0.99


def li2_unit_interval_np(x):
    x = np.asarray(x, dtype=np.float64)
    try:
        from scipy import special
        return special.spence(1.0 - x)
    except Exception:
        out = []
        for v in x.ravel():
            if v <= 0:
                out.append(0.0); continue
            if v >= 1:
                out.append(math.pi * math.pi / 6.0); continue
            if v <= 0.7:
                s = 0.0; term = v
                for k in range(1, 20000):
                    add = term / (k * k); s += add
                    if abs(add) < 1e-16: break
                    term *= v
                out.append(s)
            else:
                y = 1.0 - v; s = 0.0; term = y
                for k in range(1, 20000):
                    add = term / (k * k); s += add
                    if abs(add) < 1e-16: break
                    term *= y
                out.append(math.pi * math.pi / 6.0 - math.log(v) * math.log(y) - s)
        return np.asarray(out, dtype=np.float64).reshape(x.shape)


def phi_cont_np(opacity):
    return 2.0 * math.pi * li2_unit_interval_np(np.clip(opacity, 0.0, 1.0))


def phi_cuda_np(opacity):
    o = np.asarray(opacity, dtype=np.float64)
    oc = np.clip(o, 0.0, 1.0 - 1e-12)
    out = np.zeros_like(oc)
    li_skip = float(li2_unit_interval_np(ALPHA_SKIP))
    li_max = float(li2_unit_interval_np(ALPHA_MAX))
    mid = (oc > ALPHA_SKIP) & (oc <= ALPHA_MAX)
    high = oc > ALPHA_MAX
    out[mid] = 2.0 * math.pi * (li2_unit_interval_np(oc[mid]) - li_skip)
    out[high] = 2.0 * math.pi * ((li_max - li_skip) + (-math.log1p(-ALPHA_MAX)) * np.log(oc[high] / ALPHA_MAX))
    return out


def invert_phi_cont_np(target, n_iter: int = 100):
    target = np.asarray(target, dtype=np.float64)
    lo = np.zeros_like(target)
    hi = np.full_like(target, 1.0 - 1e-12)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        val = phi_cont_np(mid)
        lo = np.where(val < target, mid, lo)
        hi = np.where(val >= target, mid, hi)
    return 0.5 * (lo + hi)


def invert_phi_cuda_np(target, n_iter: int = 100):
    target = np.asarray(target, dtype=np.float64)
    lo = np.full_like(target, ALPHA_SKIP)
    hi = np.full_like(target, 1.0 - 1e-12)
    zero = target <= 0.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        val = phi_cuda_np(mid)
        lo = np.where(val < target, mid, lo)
        hi = np.where(val >= target, mid, hi)
    out = 0.5 * (lo + hi)
    out[zero] = 0.0
    return out


def kiot_cuda_identity_safe_np(opacity, q, plateau_eps: float = 0.0):
    opacity, q = np.broadcast_arrays(np.asarray(opacity, dtype=np.float64), np.asarray(q, dtype=np.float64))
    out = np.empty_like(opacity)
    identity = np.abs(q - 1.0) <= 1e-12
    out[identity] = opacity[identity]
    active = ~identity
    if not np.any(active):
        return out
    sub_o = opacity[active]
    sub_q = q[active]
    phi_old = phi_cuda_np(sub_o)
    active_out = np.empty_like(sub_o)
    plateau = phi_old <= plateau_eps
    if np.any(plateau):
        active_out[plateau] = invert_phi_cont_np(sub_q[plateau] * phi_cont_np(sub_o[plateau]))
    normal = ~plateau
    if np.any(normal):
        active_out[normal] = invert_phi_cuda_np(sub_q[normal] * phi_old[normal])
    out[active] = active_out
    return out


class KiotCudaLUT:
    def __init__(self, size: int = 262144, device: str | torch.device = 'cuda'):
        self.size = int(size)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        opacity = np.linspace(ALPHA_SKIP, 1.0 - 1e-8, self.size, dtype=np.float64)
        phi = phi_cuda_np(opacity)
        self.opacity_lut = torch.as_tensor(opacity, dtype=torch.float64, device=self.device)
        self.phi_lut = torch.as_tensor(phi, dtype=torch.float64, device=self.device)

    def invert_phi_cuda(self, target: torch.Tensor) -> torch.Tensor:
        target = target.to(device=self.device, dtype=torch.float64)
        idx = torch.searchsorted(self.phi_lut, target).clamp(1, self.size - 1)
        lo = idx - 1
        hi = idx
        p0 = self.phi_lut[lo]
        p1 = self.phi_lut[hi]
        o0 = self.opacity_lut[lo]
        o1 = self.opacity_lut[hi]
        w = ((target - p0) / (p1 - p0).clamp_min(1e-30)).clamp(0.0, 1.0)
        out = o0 + w * (o1 - o0)
        return torch.where(target <= 0, torch.zeros_like(out), out)

    def transform(self, opacity: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        opacity = opacity.to(device=self.device, dtype=torch.float64)
        q = q.to(device=self.device, dtype=torch.float64)
        opacity, q = torch.broadcast_tensors(opacity, q)
        out = torch.empty_like(opacity)
        identity = torch.abs(q - 1.0) <= 1e-12
        out[identity] = opacity[identity]
        active = ~identity
        if bool(active.any()):
            sub_o = opacity[active]
            sub_q = q[active]
            # Reuse numpy exact phi formula values moved to torch; this keeps CUDA search/interp as the inverse path.
            phi_old = torch.as_tensor(phi_cuda_np(sub_o.detach().cpu().numpy()), dtype=torch.float64, device=self.device)
            phi_cont_old = torch.as_tensor(phi_cont_np(sub_o.detach().cpu().numpy()), dtype=torch.float64, device=self.device)
            plateau = phi_old <= 0
            active_out = torch.empty_like(sub_o)
            if bool(plateau.any()):
                cont_np = invert_phi_cont_np((sub_q[plateau] * phi_cont_old[plateau]).detach().cpu().numpy())
                active_out[plateau] = torch.as_tensor(cont_np, dtype=torch.float64, device=self.device)
            normal = ~plateau
            if bool(normal.any()):
                active_out[normal] = self.invert_phi_cuda(sub_q[normal] * phi_old[normal])
            out[active] = active_out
        return out
