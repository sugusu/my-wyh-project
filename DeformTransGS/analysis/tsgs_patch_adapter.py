from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class TSGSCheckpoint:
    root: Path
    ply_path: Path
    gaussian_count: int
    properties: tuple[str, ...]
    xyz: np.ndarray
    raw_opacity: np.ndarray
    raw_transparency: np.ndarray
    activated_opacity: np.ndarray
    activated_transparency: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray


class TSGSPatchAdapter:
    """Read-only bridge for the official TSGS checkpoint format."""

    def __init__(self, checkpoint_root: str | Path, iteration: int = 30000) -> None:
        self.root = Path(checkpoint_root)
        self.iteration = int(iteration)
        self.ply_path = self.root / "point_cloud" / f"iteration_{self.iteration}" / "point_cloud.ply"

    def load(self) -> TSGSCheckpoint:
        if not self.ply_path.exists():
            raise FileNotFoundError(self.ply_path)

        ply = PlyData.read(str(self.ply_path))
        vertex = ply["vertex"]
        names = tuple(vertex.data.dtype.names or ())
        required = {"x", "y", "z", "opacity", "transparency", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
        missing = sorted(required.difference(names))
        if missing:
            raise RuntimeError(f"TSGS checkpoint missing required fields: {missing}")

        xyz = np.stack([np.asarray(vertex[c], dtype=np.float64) for c in ("x", "y", "z")], axis=1)
        raw_opacity = np.asarray(vertex["opacity"], dtype=np.float64)
        raw_transparency = np.asarray(vertex["transparency"], dtype=np.float64)
        scale = np.stack([np.asarray(vertex[f"scale_{i}"], dtype=np.float64) for i in range(3)], axis=1)
        rotation = np.stack([np.asarray(vertex[f"rot_{i}"], dtype=np.float64) for i in range(4)], axis=1)

        return TSGSCheckpoint(
            root=self.root,
            ply_path=self.ply_path,
            gaussian_count=len(vertex),
            properties=names,
            xyz=xyz,
            raw_opacity=raw_opacity,
            raw_transparency=raw_transparency,
            activated_opacity=sigmoid(raw_opacity),
            activated_transparency=sigmoid(raw_transparency),
            scale=scale,
            rotation=rotation,
        )


def opacity_to_tau(alpha: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0 - eps)
    return -np.log1p(-alpha)


def tau_to_opacity(tau: np.ndarray) -> np.ndarray:
    return -np.expm1(-np.asarray(tau, dtype=np.float64))
