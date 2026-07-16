#!/usr/bin/env python3
"""Stage 2.2: ASG-active benchmark construction"""
import sys, os, json, csv, numpy as np, torch

BASE = "/data/wyh/DeformTransGS"
MESH_DIR = f"{BASE}/experiments/stage1_minimal_gt/meshes"
OUTPUT = f"{BASE}/data/stage2_2_asg_active"
SCRIPT_DIR = f"{BASE}/experiments/stage2_2_asg_active_benchmark"
os.makedirs(OUTPUT, exist_ok=True)

log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

import mitsuba as mi
mi.set_variant("llvm_ad_rgb")

# ═══════════════════════════════════════════════════════════
# 1. 48-camera rig
# ═══════════════════════════════════════════════════════════
hdr("1. Creating 48-camera rig")
azimuths = list(range(0, 360, 30))  # 12
elevations = [-15, 0, 15, 30]  # 4

cams = []
for ei, elev in enumerate(elevations):
    for ai, az in enumerate(azimuths):
        az_rad = np.deg2rad(az)
        el_rad = np.deg2rad(elev)
        origin = [5*np.cos(az_rad)*np.cos(el_rad),
                  5*np.sin(az_rad)*np.cos(el_rad),
                  5*np.sin(el_rad)]
        cams.append({
            "id": ei * 12 + ai,
            "azimuth": az, "elevation": elev,
            "origin": origin, "target": [0,0,0], "up": [0,0,1]
        })

# Split into train (8 per elev) and test (4 per elev)
train_cams = []
test_cams = []
test_az = {60, 150, 240, 330}
for ei in range(4):
    for ai in range(12):
        az = azimuths[ai]
        if az in test_az:
            test_cams.append(cams[ei*12+ai])
        else:
            train_cams.append(cams[ei*12+ai])

log(f"Total: {len(cams)} cameras")
log(f"Train: {len(train_cams)}, Test: {len(test_cams)}")
json.dump(cams, open(f"{OUTPUT}/cameras_48.json","w"), indent=2)
json.dump({"train": [c["id"] for c in train_cams],
            "test": [c["id"] for c in test_cams]}, open(f"{OUTPUT}/split_32_16.json","w"), indent=2)

# ═══════════════════════════════════════════════════════════
# 2. Scene builders
# ═══════════════════════════════════════════════════════════
hdr("2. Scene configurations")

def make_cam_tf(cam):
    return mi.ScalarTransform4f.look_at(
        mi.ScalarVector3f(*cam["origin"]),
        mi.ScalarPoint3f(0,0,0),
        mi.ScalarPoint3f(0,0,1))

def make_scene_constant(mesh_path, cam, spp, bg_only=False):
    d = {
        "type":"scene","integrator":{"type":"path","max_depth":12},
        "sensor":{"type":"perspective","fov":45,"to_world":make_cam_tf(cam),
            "film":{"type":"hdrfilm","width":512,"height":512,"rfilter":{"type":"gaussian"}},
            "sampler":{"type":"independent","sample_count":spp}},
        "bg":{"type":"rectangle","to_world":mi.ScalarTransform4f.translate([0,0,-2])@mi.ScalarTransform4f.scale([4,4,1]),
            "bsdf":{"type":"diffuse","reflectance":{"type":"checkerboard",
                "color0":{"type":"rgb","value":[0.9,0.9,0.9]},
                "color1":{"type":"rgb","value":[0.1,0.1,0.1]}}}},
        "ground":{"type":"disk","to_world":mi.ScalarTransform4f.translate([0,0,-2.5])@mi.ScalarTransform4f.scale(10),
            "bsdf":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.5,0.5,0.5]}}},
        "emitter":{"type":"constant","radiance":{"type":"rgb","value":[1.0,1.0,1.0]}}}
    if not bg_only and mesh_path:
        d["object"] = {"type":"obj","filename":mesh_path,
            "bsdf":{"type":"dielectric","int_ior":1.49,"ext_ior":1.0}}
    return mi.load_dict(d)

def make_scene_multilobe(mesh_path, cam, spp, bg_only=False, rough=False):
    # 6 area emitters with RGB colors
    lights = [
        ("L1", [3,0,2], [8.0, 5.0, 3.0]),   # warm
        ("L2", [-3,0,1], [2.0, 3.0, 6.0]),   # cool
        ("L3", [0,3,3], [6.0, 6.0, 6.0]),    # neutral
        ("L4", [0,-3,2], [2.0, 4.0, 5.0]),   # cool
        ("L5", [2,2,-1], [5.0, 3.0, 2.0]),   # warm
        ("L6", [-2,-2,1], [3.0, 3.0, 3.0]),  # neutral
    ]
    d = {
        "type":"scene","integrator":{"type":"path","max_depth":12},
        "sensor":{"type":"perspective","fov":45,"to_world":make_cam_tf(cam),
            "film":{"type":"hdrfilm","width":512,"height":512,"rfilter":{"type":"gaussian"}},
            "sampler":{"type":"independent","sample_count":spp}},
        "bg":{"type":"rectangle","to_world":mi.ScalarTransform4f.translate([0,0,-2])@mi.ScalarTransform4f.scale([4,4,1]),
            "bsdf":{"type":"diffuse","reflectance":{"type":"checkerboard",
                "color0":{"type":"rgb","value":[0.9,0.9,0.9]},
                "color1":{"type":"rgb","value":[0.1,0.1,0.1]}}}},
        "ground":{"type":"disk","to_world":mi.ScalarTransform4f.translate([0,0,-2.5])@mi.ScalarTransform4f.scale(10),
            "bsdf":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.5,0.5,0.5]}}},
        "emitter_ambient":{"type":"constant","radiance":{"type":"rgb","value":[0.05,0.05,0.05]}},
    }
    for li, (name, pos, radiance) in enumerate(lights):
        d[f"light_{li}"] = {
            "type":"rectangle",
            "to_world":mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(*pos), mi.ScalarPoint3f(0,0,0), mi.ScalarPoint3f(0,0,1)),
            "emitter":{"type":"area","radiance":{"type":"rgb","value":radiance}}
        }
    if not bg_only and mesh_path:
        bsdf = {"type":"roughdielectric","int_ior":1.49,"ext_ior":1.0,
                "distribution":"ggx","alpha":0.03} if rough else {"type":"dielectric","int_ior":1.49,"ext_ior":1.0}
        d["object"] = {"type":"obj","filename":mesh_path, "bsdf":bsdf}
    return mi.load_dict(d)

# ═══════════════════════════════════════════════════════════
# 3. Render
# ═══════════════════════════════════════════════════════════
hdr("3. Rendering")
SPP = 256

variants = [
    ("B2_clear_multilobe", False, True),
    ("B3_rough_constant", True, False),
    ("B4_rough_multilobe", True, True),
]

for var_name, rough, multilobe in variants:
    hdr(f"  Rendering {var_name}")
    var_dir = f"{OUTPUT}/{var_name}"
    os.makedirs(f"{var_dir}/canonical", exist_ok=True)
    os.makedirs(f"{var_dir}/background_only", exist_ok=True)

    def scene_fn(mesh_path, cam, spp, bg_only):
        if multilobe:
            return make_scene_multilobe(mesh_path, cam, spp, bg_only, rough)
        else:
            return make_scene_constant(mesh_path, cam, spp, bg_only)

    for cam in cams:
        # Canonical
        s = scene_fn(f"{MESH_DIR}/canonical.obj", cam, SPP, bg_only=False)
        img = mi.render(s, spp=SPP)
        mi.util.write_bitmap(f"{var_dir}/canonical/cam_{cam['id']:03d}.png", img)

        # Background only
        s_bg = scene_fn(None, cam, SPP, bg_only=True)
        img_bg = mi.render(s_bg, spp=SPP)
        mi.util.write_bitmap(f"{var_dir}/background_only/cam_{cam['id']:03d}.png", img_bg)

    log(f"  {var_name}: {len(cams)} views done")

log("\n=== Stage 2.2 rendering complete ===")
