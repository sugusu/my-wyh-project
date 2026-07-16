from __future__ import annotations

import numpy as np


def opacity_from_tau(tau):
    tau = np.asarray(tau, dtype=np.float64)
    return 1.0 - np.exp(-tau)


def psi_continuous(opacity, g):
    opacity = np.asarray(opacity, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    alpha = opacity * g
    if np.any(alpha < 0):
        raise ValueError("negative alpha")
    if np.any(alpha >= 1):
        raise ValueError("alpha >=1")
    return -np.log1p(-alpha)


def required_opacity_for_scaled_psi(opacity, g, q):
    opacity = np.asarray(opacity, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    q = float(q)
    if np.any(g <= 0):
        raise ValueError("g must be positive")
    return (1.0 - np.power(1.0 - opacity * g, q)) / g
