"""
common.py - 共享 deformation API
deform_points(points, deformation_type, strength)
deformation_jacobian(points, deformation_type, strength)
"""
import torch
import numpy as np

def deform_points(points, deformation_type, strength):
    if deformation_type == "shear":
        from .shear import deform_points as dp
        return dp(points, strength)
    elif deformation_type == "twist":
        from .twist import deform_points as dp
        return dp(points, strength, z_range=None)
    raise ValueError(f"Unknown deformation: {deformation_type}")

def deformation_jacobian(points, deformation_type, strength):
    if deformation_type == "shear":
        from .shear import jacobian as jac
        return jac(points, strength)
    elif deformation_type == "twist":
        from .twist import jacobian as jac
        return jac(points, strength, z_range=None)
    raise ValueError(f"Unknown deformation: {deformation_type}")
