#!/usr/bin/env python3
"""Test StretchableFilm-v0 material and render smoke test"""
import sys, os, json, numpy as np
sys.path.insert(0, "/data/wyh/DeformTransGS/benchmark")
import mitsuba as mi
mi.set_variant("llvm_ad_rgb")

from deformations.shear import deform_points as shear_def
from deformations.twist import deform_points as twist_def
import trimesh, torch

# Create rectangular sheet mesh
W, H = 1.5, 1.5  # sheet dimensions
divs = 40  # subdivisions
verts = []
faces = []
for i in range(divs + 1):
    for j in range(divs + 1):
        x = -W/2 + W * i / divs
        y = -H/2 + H * j / divs
        verts.append([x, y, 0.0])

for i in range(divs):
    for j in range(divs):
        idx = i * (divs + 1) + j
        faces.append([idx, idx + 1, idx + divs + 2])
        faces.append([idx, idx + divs + 2, idx + divs + 1])

mesh = trimesh.Trimesh(vertices=np.float32(verts), faces=np.int32(faces), process=False)
mesh.export("/tmp/canonical_sheet.obj")
print(f"Sheet: {len(verts)} vertices, {len(faces)} faces")

# Compute deformed meshes
verts_t = torch.tensor(verts, dtype=torch.float32)
z_range = (verts_t[:, 2].min().item(), verts_t[:, 2].max().item())

# Deformations
def save_deformed(name, deformed_verts):
    m = trimesh.Trimesh(vertices=deformed_verts.numpy(), faces=mesh.faces, process=False)
    m.export(f"/tmp/{name}.obj")
    print(f"  {name}: saved")

save_deformed("canonical", verts_t)

# Uniaxial stretch: F = diag(s, 1, 1) -> x' = s*x, y' = y
for s in [1.10, 1.25, 1.50, 2.00]:
    def_verts = verts_t.clone()
    def_verts[:, 0] *= s
    save_deformed(f"stretch_x_{s:.2f}".replace(".", "_"), def_verts)

# Biaxial stretch: F = diag(s, s, 1)
for s in [1.10, 1.25, 1.50]:
    def_verts = verts_t.clone()
    def_verts[:, 0] *= s
    def_verts[:, 1] *= s
    save_deformed(f"biaxial_{s:.2f}".replace(".", "_"), def_verts)

# Twist control
for deg in [30, 60]:
    def_verts = twist_def(verts_t, deg, z_range)
    save_deformed(f"twist_{deg}", def_verts)

# Render test with null BSDF (straight transmission)
# Using blendbsdf: weight controls transmission
def render_sheet(mesh_path, opacity_val, spp=128):
    scene = mi.load_dict({
        "type": "scene", "integrator": {"type": "path", "max_depth": 12},
        "sensor": {"type": "perspective", "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                mi.ScalarPoint3f(0, -3.5, 1.5), mi.ScalarPoint3f(0, 0, 0), mi.ScalarPoint3f(0, 0, 1)),
            "film": {"type": "hdrfilm", "width": 256, "height": 256, "rfilter": {"type": "gaussian"}},
            "sampler": {"type": "independent", "sample_count": spp}},
        "sheet": {"type": "obj", "filename": mesh_path,
            "bsdf": {
                "type": "blendbsdf",
                "weight": 1.0 - opacity_val,
                "bsdf1": {"type": "null"},
                "bsdf2": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.3, 0.3, 0.8]}}
            }},
        "bg": {"type": "rectangle", "to_world": mi.ScalarTransform4f.translate([0, 0, -2]) @ mi.ScalarTransform4f.scale([4, 4, 1]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "checkerboard",
                "color0": {"type": "rgb", "value": [0.9, 0.9, 0.9]},
                "color1": {"type": "rgb", "value": [0.1, 0.1, 0.1]}}}},
        "ground": {"type": "disk", "to_world": mi.ScalarTransform4f.translate([0, 0, -2.5]).scale(10),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]}}},
        "emitter": {"type": "constant", "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]}},
    })
    return mi.render(scene, spp=spp)

# Smoke test: render canonical with different opacities
print("\nSmoke render test:")
for op_val in [0.0, 0.5, 0.8, 1.0]:
    img = render_sheet("/tmp/canonical_sheet.obj", op_val, spp=32)
    print(f"  opacity={op_val:.1f}: render OK, shape={img.shape}")

print("\nAll tests passed!")
