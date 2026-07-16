#!/usr/bin/env python3
"""Render StretchableFilm-v0 GT with dynamic vs fixed optical state"""
import sys, os, json, csv, numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, "/data/wyh/DeformTransGS/benchmark")
import mitsuba as mi
import trimesh, torch
mi.set_variant("llvm_ad_rgb")
from deformations.twist import deform_points as twist_def
from surface_stretch import compute_Js, h_ratio

OUTPUT = "/data/wyh/DeformTransGS/experiments/stage3_1_runtime_optical_benchmark"
os.makedirs(f"{OUTPUT}/renders", exist_ok=True)
os.makedirs(f"{OUTPUT}/meshes", exist_ok=True)

log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

# Create sheet mesh
W, H, divs = 1.5, 1.5, 40
verts = [[-W/2+W*i/divs, -H/2+H*j/divs, 0.0] for i in range(divs+1) for j in range(divs+1)]
faces = []
for i in range(divs):
    for j in range(divs):
        idx = i*(divs+1)+j
        faces.extend([[idx, idx+1, idx+divs+2], [idx, idx+divs+2, idx+divs+1]])
mesh = trimesh.Trimesh(vertices=np.float32(verts), faces=np.int32(faces), process=False)

# Generate deformed meshes
verts_t = torch.tensor(verts, dtype=torch.float32)
z_range = (verts_t[:,2].min().item(), verts_t[:,2].max().item())

states = {}
states["canonical"] = verts_t
for s in [1.10, 1.25, 1.50, 2.00]:
    dv = verts_t.clone(); dv[:,0] *= s; states[f"stretch_{s:.2f}".replace(".","_")] = dv
for s in [1.10, 1.25, 1.50]:
    dv = verts_t.clone(); dv[:,0] *= s; dv[:,1] *= s; states[f"biaxial_{s:.2f}".replace(".","_")] = dv
for deg in [30, 60]:
    states[f"twist_{deg}"] = twist_def(verts_t, deg, z_range)

# Compute Js and h_ratio for each state
# For stretch: F = diag(s, 1, 1), normal = [0,0,1] -> Js = s*1 = s
# For biaxial: F = diag(s, s, 1) -> Js = s*s
# For twist: use analytical Js (approximation: ~1 for our sheet)
hdr("Computing Js and h_ratio")
Js_vals = {}
h_vals = {}
for name, pts in states.items():
    if name == "canonical":
        Js, h_val = 1.0, 1.0
    elif name.startswith("stretch"):
        s = float(name.split("_")[1])
        Js = s  # uniaxial stretch with normal [0,0,1]
        h_val = 1.0 / Js
    elif name.startswith("biaxial"):
        s = float(name.split("_")[1])
        Js = s * s
        h_val = 1.0 / Js
    else:  # twist - approximate as area-preserving
        Js = 1.0
        h_val = 1.0
    Js_vals[name] = Js
    h_vals[name] = h_val
    log(f"  {name:20s}: Js={Js:.4f}, h_ratio={h_val:.4f}")

# Save meshes
for name, pts in states.items():
    trimesh.Trimesh(vertices=pts.numpy(), faces=mesh.faces, process=False).export(f"{OUTPUT}/meshes/{name}.obj")

# Render with dynamic vs fixed optical state
hdr("Rendering GT (dynamic vs fixed)")
cameras = [
    {"id": 0, "pos": [0, -3.5, 1.5], "target": [0,0,0], "up": [0,0,1]},
    {"id": 4, "pos": [3.0, 0, 2.0], "target": [0,0,0], "up": [0,0,1]},
    {"id": 8, "pos": [0, 3.5, 1.5], "target": [0,0,0], "up": [0,0,-1]},
]

def render_state(mesh_path, opacity, spp=128):
    scene = mi.load_dict({
        "type":"scene","integrator":{"type":"path","max_depth":12},
        "sensor":{"type":"perspective","fov":45,
            "to_world":mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(*cam["pos"]),mi.ScalarPoint3f(0,0,0),mi.ScalarPoint3f(*cam["up"])),
            "film":{"type":"hdrfilm","width":256,"height":256,"rfilter":{"type":"gaussian"}},
            "sampler":{"type":"independent","sample_count":spp}},
        "sheet":{"type":"obj","filename":mesh_path,"bsdf":{"type":"blendbsdf","weight":opacity,
            "bsdf1":{"type":"null"},"bsdf2":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.3,0.3,0.8]}}}},
        "bg":{"type":"rectangle","to_world":mi.ScalarTransform4f.translate([0,0,-2])@mi.ScalarTransform4f.scale([4,4,1]),
            "bsdf":{"type":"diffuse","reflectance":{"type":"checkerboard",
                "color0":{"type":"rgb","value":[0.9,0.9,0.9]},"color1":{"type":"rgb","value":[0.1,0.1,0.1]}}}},
        "ground":{"type":"disk","to_world":mi.ScalarTransform4f.translate([0,0,-2.5]).scale(10),
            "bsdf":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.5,0.5,0.5]}}},
        "emitter":{"type":"constant","radiance":{"type":"rgb","value":[1.0,1.0,1.0]}},
    })
    return mi.render(scene, spp=spp)

# Beer-Lambert: T = exp(-tau), opacity_weight = 1 - T
for tau0 in [0.5, 1.0]:
    log(f"\ntau0={tau0}:")
    for variant, suffix in [(True, "dynamic"), (False, "fixed")]:
        out_dir = f"{OUTPUT}/renders/tau{tau0}_{suffix}"
        os.makedirs(out_dir, exist_ok=True)
        for name, pts in states.items():
            mesh_path = f"{OUTPUT}/meshes/{name}.obj"
            if variant:
                h = h_vals[name]
                tau = tau0 * h
            else:
                tau = tau0  # fixed canonical
            T = np.exp(-tau)
            opacity = 1.0 - T  # blend weight for null BSDF
            
            for cam in cameras:
                T_rend = render_state(mesh_path, opacity, spp=128)
                mi.util.write_bitmap(f"{out_dir}/{name}_cam{cam['id']:03d}.png", T_rend)
        log(f"  {suffix}: {len(states)} states × {len(cameras)} cameras done")

# Compute metrics
hdr("Computing metrics")
import lpips
lpips_fn = lpips.LPIPS(net='alex').eval()

metric_rows = []
for tau0 in [0.5, 1.0]:
    dynamic_dir = f"{OUTPUT}/renders/tau{tau0}_dynamic"
    fixed_dir = f"{OUTPUT}/renders/tau{tau0}_fixed"
    
    for name in states.keys():
        for cam in cameras:
            # Load dynamic and fixed
            dyn = np.array(Image.open(f"{dynamic_dir}/{name}_cam{cam['id']:03d}.png")).astype(np.float32)/255.0
            fix = np.array(Image.open(f"{fixed_dir}/{name}_cam{cam['id']:03d}.png")).astype(np.float32)/255.0
            # Use canonical dynamic as reference
            can = np.array(Image.open(f"{dynamic_dir}/canonical_cam{cam['id']:03d}.png")).astype(np.float32)/255.0
            
            # Compute optical effect mask from canonical
            diff_can = np.max(np.abs(can), axis=2)
            # Actually we need bg for proper mask. Use BG from external
            # For simplicity, use pixels where avg_abs > 0.01 as mask
            mask = binary_dilation(np.abs(can).mean(axis=2) > 0.02, iterations=2)
            if mask.sum() == 0:
                mask = np.ones((256,256), dtype=bool)
            
            diff = np.abs(dyn - fix)
            mae_full = diff.mean()
            mae_masked = diff[mask].mean() if mask.sum()>0 else 1.0
            
            metric_rows.append({
                "tau0": tau0, "state": name, "cam": cam["id"],
                "Js": Js_vals[name], "h_ratio": h_vals[name],
                "mae_full": float(mae_full), "mae_masked": float(mae_masked),
            })

csv.DictWriter(open(f"{OUTPUT}/gt_state_signal_metrics.csv","w",newline=""),
    fieldnames=["tau0","state","cam","Js","h_ratio","mae_full","mae_masked"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/gt_state_signal_metrics.csv","a",newline=""),
    fieldnames=["tau0","state","cam","Js","h_ratio","mae_full","mae_masked"]).writerows(metric_rows)

# Summary
for tau0 in [0.5, 1.0]:
    log(f"\ntau0={tau0}:")
    for name in states.keys():
        vals = [r["mae_masked"] for r in metric_rows if r["tau0"]==tau0 and r["state"]==name]
        if vals:
            log(f"  {name:20s}: masked MAE={np.mean(vals):.6f}")
    
    # Stretch signal: stretch_2_00
    s_mae = np.mean([r["mae_masked"] for r in metric_rows if r["tau0"]==tau0 and r["state"]=="stretch_2_00"])
    t_mae = np.mean([r["mae_masked"] for r in metric_rows if r["tau0"]==tau0 and r["state"]=="twist_60"])
    log(f"  stretch_2.00 vs twist_60: {s_mae:.6f} vs {t_mae:.6f} (twist ratio={t_mae/(s_mae+1e-8)*100:.1f}%)")

log("\n=== Done ===")
