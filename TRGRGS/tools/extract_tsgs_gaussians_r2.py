#!/usr/bin/env python3
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'compat'),str(ROOT/'third_party/TSGS')]
from scene.gaussian_model import GaussianModel
g=GaussianModel(3,24);g.load_ply(str(ROOT/'checkpoints/baseline/scene_01/point_cloud/iteration_30000/point_cloud.ply'))
def n(x):return x.detach().cpu().numpy()
np.savez(ROOT/'outputs/scene_01/foreground_audit_r2/activated_gaussians_from_tsgs.npz',xyz=n(g.get_xyz),scales=n(g.get_scaling),opacity=n(g.get_opacity),rotation=n(g.get_rotation))

