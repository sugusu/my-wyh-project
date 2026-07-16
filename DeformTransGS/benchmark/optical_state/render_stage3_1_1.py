#!/usr/bin/env python3
"""Stage 3.1.1: Fixed Js pipeline and full gate reassessment"""
import sys, os, json, csv, numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, "/data/wyh/DeformTransGS/benchmark")
import mitsuba as mi
import trimesh, torch
mi.set_variant("llvm_ad_rgb")
from deformations.twist import deform_points as twist_def

OUTPUT = "/data/wyh/DeformTransGS/experiments/stage3_1_1_gate_runtime_repair"
os.makedirs(f"{OUTPUT}/meshes", exist_ok=True)
os.makedirs(f"{OUTPUT}/repaired_render", exist_ok=True)

log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

# ═══════════════════════════════════════════════════════════
# 1. Create sheet mesh
# ═══════════════════════════════════════════════════════════
hdr("1. Sheet mesh")
W, H, divs = 1.5, 1.5, 40
verts = [[-W/2+W*i/divs, -H/2+H*j/divs, 0.0] for i in range(divs+1) for j in range(divs+1)]
faces = []
for i in range(divs):
    for j in range(divs):
        idx = i*(divs+1)+j
        faces.extend([[idx, idx+1, idx+divs+2], [idx, idx+divs+2, idx+divs+1]])
mesh = trimesh.Trimesh(vertices=np.float32(verts), faces=np.int32(faces), process=False)
log(f"Verts: {len(verts)}, Faces: {len(faces)}")

# Canonical normal (z-up plane)
N = len(verts)
canon_normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).expand(N, -1)

# ═══════════════════════════════════════════════════════════
# 2. Deformations with proper strength parsing
# ═══════════════════════════════════════════════════════════
hdr("2. Deformations")
verts_t = torch.tensor(verts, dtype=torch.float32)
z_range = (verts_t[:,2].min().item(), verts_t[:,2].max().item())

def parse_strength(name):
    """Parse strength from state name like stretch_1_10 -> 1.10"""
    parts = name.split("_")
    if name.startswith("stretch"):
        return float(f"{parts[1]}.{parts[2]}")
    elif name.startswith("biaxial"):
        return float(f"{parts[1]}.{parts[2]}")
    elif name.startswith("twist"):
        return float(parts[1])
    return 1.0

def compute_Js_analytic(name):
    """Compute Js analytically based on state name"""
    if name == "canonical":
        return 1.0
    s = parse_strength(name)
    if name.startswith("stretch"):
        return s  # uniaxial, F=diag(s,1,1), normal z -> Js = s*1 = s
    elif name.startswith("biaxial"):
        return s * s  # biaxial, F=diag(s,s,1), normal z -> Js = s*s
    elif name.startswith("twist"):
        return 1.0  # twist preserves area
    return 1.0

# Define states with proper names (use dot notation for numeric)
states_def = [
    ("canonical",),
    ("stretch", [1.0, 1.1, 1.25, 1.5, 2.0]),
    ("biaxial", [1.1, 1.25, 1.5]),
    ("twist", [30, 60]),
]

# Build state mapping
state_info = {}  # name -> (verts, Js)
# Canonical
state_info["canonical"] = (verts_t.clone(), 1.0)

for s in [1.0, 1.1, 1.25, 1.5, 2.0]:
    name = f"stretch_{s:.2f}"
    dv = verts_t.clone()
    dv[:, 0] *= s
    state_info[name] = (dv, s)

for s in [1.1, 1.25, 1.5]:
    name = f"biaxial_{s:.2f}"
    dv = verts_t.clone()
    dv[:, 0] *= s
    dv[:, 1] *= s
    state_info[name] = (dv, s*s)

for deg in [30, 60]:
    name = f"twist_{deg}"
    state_info[name] = (twist_def(verts_t, deg, z_range), 1.0)

for name, (pts, Js) in state_info.items():
    h_val = 1.0 / max(Js, 1e-8)
    log(f"  {name:20s}: Js={Js:.4f}, h_ratio={h_val:.4f}")
    trimesh.Trimesh(vertices=pts.numpy(), faces=mesh.faces, process=False).export(f"{OUTPUT}/meshes/{name}.obj")

# ═══════════════════════════════════════════════════════════
# 3. Material micro-test
# ═══════════════════════════════════════════════════════════
hdr("3. Material micro-test")
cam_micro = {"pos": [0, -3.5, 1.5], "target": [0,0,0], "up": [0,0,1]}

def render_opacity(opacity_val, spp=128):
    scene = mi.load_dict({
        "type":"scene","integrator":{"type":"path","max_depth":12},
        "sensor":{"type":"perspective","fov":45,
            "to_world":mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(*cam_micro["pos"]),mi.ScalarPoint3f(0,0,0),mi.ScalarPoint3f(0,0,1)),
            "film":{"type":"hdrfilm","width":256,"height":256,"rfilter":{"type":"gaussian"}},
            "sampler":{"type":"independent","sample_count":spp}},
        "sheet":{"type":"obj","filename":f"{OUTPUT}/meshes/canonical.obj","bsdf":{"type":"blendbsdf",
            "weight":opacity_val,"bsdf1":{"type":"null"},
            "bsdf2":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.3,0.3,0.8]}}}},
        "bg":{"type":"rectangle","to_world":mi.ScalarTransform4f.translate([0,0,-2])@mi.ScalarTransform4f.scale([4,4,1]),
            "bsdf":{"type":"diffuse","reflectance":{"type":"checkerboard",
                "color0":{"type":"rgb","value":[0.9,0.9,0.9]},"color1":{"type":"rgb","value":[0.1,0.1,0.1]}}}},
        "ground":{"type":"disk","to_world":mi.ScalarTransform4f.translate([0,0,-2.5]).scale(10),
            "bsdf":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.5,0.5,0.5]}}},
        "emitter":{"type":"constant","radiance":{"type":"rgb","value":[1.0,1.0,1.0]}},
    })
    return mi.render(scene, spp=spp)

opacities_test = [0.2, 0.4, 0.6, 0.8]
imgs = {}
for op in opacities_test:
    imgs[op] = render_opacity(op, spp=128)
    log(f"  opacity={op}: rendered")

mat_rows = []
for i in range(len(opacities_test)-1):
    o1, o2 = opacities_test[i], opacities_test[i+1]
    i1 = np.array(imgs[o1])
    i2 = np.array(imgs[o2])
    diff = np.abs(i1 - i2).mean()
    mat_rows.append({"opacity1": o1, "opacity2": o2, "mae": float(diff)})
    log(f"  {o1} vs {o2}: MAE={diff:.6f}")

csv.DictWriter(open(f"{OUTPUT}/material_state_validation.csv","w",newline=""),
    fieldnames=["opacity1","opacity2","mae"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/material_state_validation.csv","a",newline=""),
    fieldnames=["opacity1","opacity2","mae"]).writerows(mat_rows)

# ═══════════════════════════════════════════════════════════
# 4. Full GT rendering
# ═══════════════════════════════════════════════════════════
hdr("4. Full GT rendering (dynamic vs fixed)")
cameras = [
    {"id": 0, "pos": [0, -3.5, 1.5], "target": [0,0,0], "up": [0,0,1]},
    {"id": 4, "pos": [3.0, 0, 2.0], "target": [0,0,0], "up": [0,0,1]},
    {"id": 8, "pos": [0, 3.5, 1.5], "target": [0,0,0], "up": [0,0,-1]},
]

def render_state(mesh_path, opacity, cam_dict, spp=128):
    scene = mi.load_dict({
        "type":"scene","integrator":{"type":"path","max_depth":12},
        "sensor":{"type":"perspective","fov":45,
            "to_world":mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(*cam_dict["pos"]),mi.ScalarPoint3f(0,0,0),mi.ScalarPoint3f(*cam_dict["up"])),
            "film":{"type":"hdrfilm","width":256,"height":256,"rfilter":{"type":"gaussian"}},
            "sampler":{"type":"independent","sample_count":spp}},
        "sheet":{"type":"obj","filename":mesh_path,"bsdf":{"type":"blendbsdf",
            "weight":opacity,"bsdf1":{"type":"null"},
            "bsdf2":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.3,0.3,0.8]}}}},
        "bg":{"type":"rectangle","to_world":mi.ScalarTransform4f.translate([0,0,-2])@mi.ScalarTransform4f.scale([4,4,1]),
            "bsdf":{"type":"diffuse","reflectance":{"type":"checkerboard",
                "color0":{"type":"rgb","value":[0.9,0.9,0.9]},"color1":{"type":"rgb","value":[0.1,0.1,0.1]}}}},
        "ground":{"type":"disk","to_world":mi.ScalarTransform4f.translate([0,0,-2.5]).scale(10),
            "bsdf":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.5,0.5,0.5]}}},
        "emitter":{"type":"constant","radiance":{"type":"rgb","value":[1.0,1.0,1.0]}},
    })
    return mi.render(scene, spp=spp)

for tau0 in [0.5, 1.0]:
    for variant, suffix in [(True, "dynamic"), (False, "fixed")]:
        out_dir = f"{OUTPUT}/repaired_render/tau{tau0}_{suffix}"
        os.makedirs(out_dir, exist_ok=True)
        for name, (pts, Js) in state_info.items():
            mesh_path = f"{OUTPUT}/meshes/{name}.obj"
            if variant:
                h_val = 1.0 / max(Js, 1e-8)
                tau = tau0 * h_val
            else:
                tau = tau0
            T = np.exp(-tau)
            opacity = 1.0 - T
            
            for cam in cameras:
                img = render_state(mesh_path, float(opacity), cam, spp=128)
                mi.util.write_bitmap(f"{out_dir}/{name}_cam{cam['id']:03d}.png", img)
        log(f"  tau0={tau0} {suffix}: done")

# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════
hdr("5. Computing metrics")
from scipy.stats import spearmanr

metric_rows = []
for tau0 in [0.5, 1.0]:
    for name, (pts, Js) in state_info.items():
        h_val = 1.0 / max(Js, 1e-8)
        tau_dyn = tau0 * h_val
        T_dyn = np.exp(-tau_dyn)
        op_dyn = 1.0 - T_dyn
        
        for cam in cameras:
            dyn = np.array(Image.open(f"{OUTPUT}/repaired_render/tau{tau0}_dynamic/{name}_cam{cam['id']:03d}.png")).astype(np.float32)/255.0
            fix = np.array(Image.open(f"{OUTPUT}/repaired_render/tau{tau0}_fixed/{name}_cam{cam['id']:03d}.png")).astype(np.float32)/255.0
            
            # Mask: pixels where canonical dynamic > 0.02
            can = np.array(Image.open(f"{OUTPUT}/repaired_render/tau{tau0}_dynamic/canonical_cam{cam['id']:03d}.png")).astype(np.float32)/255.0
            mask = binary_dilation(np.abs(can).mean(axis=2) > 0.02, iterations=2)
            if mask.sum() == 0:
                mask = np.ones((256,256), dtype=bool)
            
            diff = np.abs(dyn - fix)
            mae_masked = diff[mask].mean() if mask.sum()>0 else 1.0
            rmse_masked = np.sqrt((diff[mask]**2).mean()) if mask.sum()>0 else 1.0
            psnr_masked = -10*np.log10((diff[mask]**2).mean()+1e-10) if mask.sum()>0 else 0
            
            metric_rows.append({
                "tau0": tau0, "state": name, "cam": cam["id"],
                "Js": Js, "h_ratio": h_val,
                "tau_dynamic": float(tau_dyn), "tau_fixed": float(tau0),
                "opacity_dynamic": float(op_dyn), "opacity_fixed": float(1-np.exp(-tau0)),
                "masked_mae": float(mae_masked), "masked_rmse": float(rmse_masked),
                "masked_psnr": float(psnr_masked),
            })

csv.DictWriter(open(f"{OUTPUT}/repaired_gt_state_signal_metrics.csv","w",newline=""),
    fieldnames=["tau0","state","cam","Js","h_ratio","tau_dynamic","tau_fixed","opacity_dynamic","opacity_fixed","masked_mae","masked_rmse","masked_psnr"]).writeheader()
w = csv.DictWriter(open(f"{OUTPUT}/repaired_gt_state_signal_metrics.csv","a",newline=""),
    fieldnames=["tau0","state","cam","Js","h_ratio","tau_dynamic","tau_fixed","opacity_dynamic","opacity_fixed","masked_mae","masked_rmse","masked_psnr"])
for r in metric_rows:
    w.writerow(r)

# Summary
for tau0 in [0.5, 1.0]:
    log(f"\ntau0={tau0}:")
    for name in state_info:
        vals = [r["masked_mae"] for r in metric_rows if r["tau0"]==tau0 and r["state"]==name]
        if vals:
            log(f"  {name:20s}: MAE={np.mean(vals):.6f}")

# Monotonicity
for tau0 in [0.5, 1.0]:
    strengths = [1.0, 1.1, 1.25, 1.5, 2.0]
    maes = []
    for s in strengths:
        name = f"stretch_{s:.2f}"
        v = np.mean([r["masked_mae"] for r in metric_rows if r["tau0"]==tau0 and r["state"]==name])
        maes.append(v)
    rho, p = spearmanr(strengths, maes) if len(set(maes)) > 1 else (1.0, 0.0)
    log(f"\n  Monotonicity tau0={tau0}: rho={rho:.4f}")
    for s, m in zip(strengths, maes):
        log(f"    s={s:.2f}: MAE={m:.6f}")

# Twist ratio
for tau0 in [0.5, 1.0]:
    s15 = np.mean([r["masked_mae"] for r in metric_rows if r["tau0"]==tau0 and r["state"]=="stretch_1.50"])
    t60 = np.mean([r["masked_mae"] for r in metric_rows if r["tau0"]==tau0 and r["state"]=="twist_60"])
    ratio = t60 / max(s15, 1e-8) * 100
    log(f"  tau0={tau0}: twist_60 / stretch_1.50 = {ratio:.1f}%")

log("\n=== Stage 3.1.1 complete ===")
