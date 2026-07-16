#!/usr/bin/env python3
"""Stage 1: Minimal Transparent Deformation GT Benchmark"""
import sys, os, json, csv, numpy as np
BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage1_minimal_gt"
MESH_DIR = f"{OUTPUT}/meshes"
RENDER_DIR = f"{OUTPUT}/render_gt"
MESH_SRC = "/data/wyh/RecycleGS/data/translab_full/scene_01/meshes/scene_mesh.obj"
for d in [OUTPUT, MESH_DIR, RENDER_DIR]: os.makedirs(d, exist_ok=True)
sys.path.insert(0, f"{BASE}/benchmark")

log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

import trimesh

hdr("1. Load & normalize mesh")
scene = trimesh.load(MESH_SRC)
geom = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
mesh = trimesh.util.concatenate(geom) if len(geom) > 1 else geom[0]
log(f"Verts: {mesh.vertices.shape[0]}, Faces: {mesh.faces.shape[0]}, WT: {mesh.is_watertight}")

import torch
verts = torch.tensor(mesh.vertices, dtype=torch.float32)
center = verts.mean(dim=0)
scale = 2.0 / (verts.max(0).values - verts.min(0).values).max().item()
verts_n = (verts - center) * scale
mesh_norm = trimesh.Trimesh(vertices=verts_n.numpy(), faces=mesh.faces, process=False)
mesh_norm.export(f"{MESH_DIR}/canonical.obj")

norm_info = {"center": center.tolist(), "scale": scale,
             "extent": (verts.max(0).values - verts.min(0).values).tolist(),
             "forward": (np.eye(4)*scale).tolist(), "inverse": (np.eye(4)/scale).tolist()}
json.dump(norm_info, open(f"{OUTPUT}/normalization.json","w"), indent=2)
log(f"Normalized. Scale={scale:.4f}")



hdr("2. Jacobian validation")
from deformations.shear import validate_jacobian as check_shear
from deformations.twist import validate_jacobian as check_twist
shear_jerr = check_shear(1000, 0.5, 20260712)
twist_jerr = check_twist(1000, 30, 20260712)
log(f"Shear Jacobian err: {shear_jerr:.2e}")
log(f"Twist Jacobian err: {twist_jerr:.2e}")

hdr("3. Generate deformed meshes")
from deformations.shear import deform_points as d_shear
from deformations.twist import deform_points as d_twist
cv = torch.tensor(mesh_norm.vertices, dtype=torch.float32)
zmin, zmax = cv[:,2].min().item(), cv[:,2].max().item()

configs = [("shear",0.25,"shear_0.25"),("shear",0.50,"shear_0.50"),("shear",1.00,"shear_1.00"),
           ("twist",15,"twist_15"),("twist",30,"twist_30"),("twist",60,"twist_60")]
for dt, st, nm in configs:
    dv = d_shear(cv,st) if dt=="shear" else d_twist(cv,st,(zmin,zmax))
    trimesh.Trimesh(vertices=dv.numpy(), faces=mesh.faces, process=False).export(f"{MESH_DIR}/{nm}.obj")
    log(f"  {nm}.obj")

hdr("4. Mitsuba rendering")
import mitsuba as mi; mi.set_variant("llvm_ad_rgb")

azimuths = list(range(0,360,30)); elev = np.deg2rad(15); dist = 5.0
cams = [{"id":i//30,"origin":[dist*np.cos(np.deg2rad(i))*np.cos(elev),
         dist*np.sin(np.deg2rad(i))*np.cos(elev),dist*np.sin(elev)],
         "target":[0,0,0],"up":[0,0,1]} for i in azimuths]
json.dump(cams, open(f"{OUTPUT}/cameras.json","w"), indent=2)
log(f"{len(cams)} cameras")



def make_scene_dict(cam_idx, spp, mesh_path=None):
    c = cams[cam_idx]
    d = {
        "type":"scene","integrator":{"type":"path","max_depth":12},
        "sensor":{"type":"perspective","fov":45,
            "to_world":mi.ScalarTransform4f().look_at(
                mi.ScalarVector3f(*c["origin"]),mi.ScalarVector3f(*c["target"]),mi.ScalarVector3f(*c["up"])),
            "film":{"type":"hdrfilm","width":512,"height":512,"rfilter":{"type":"gaussian"}},
            "sampler":{"type":"independent","sample_count":spp}},
        "bg_plane":{"type":"rectangle",
            "to_world":mi.ScalarTransform4f().translate([0,0,-2]).scale([4,4,1]),
            "bsdf":{"type":"diffuse","reflectance":{"type":"checkerboard",
                "color0":{"type":"rgb","value":[0.9,0.9,0.9]},
                "color1":{"type":"rgb","value":[0.1,0.1,0.1]}}}},
        "ground":{"type":"disk","to_world":mi.ScalarTransform4f().translate([0,0,-2.5]).scale(10),
            "bsdf":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.5,0.5,0.5]}}},
        "emitter":{"type":"constant","radiance":{"type":"rgb","value":[1.0,1.0,1.0]}}}
    if mesh_path:
        d["object"] = {"type":"obj","filename":mesh_path,
            "bsdf":{"type":"dielectric","int_ior":1.49,"ext_ior":1.0}}
    return d

def render_one(mesh_path, cam, spp):
    return mi.render(mi.load_dict(make_scene_dict(cam, spp, mesh_path)), spp=spp)

def render_bg(cam, spp):
    return mi.render(mi.load_dict(make_scene_dict(cam, spp)), spp=spp)

states = ["canonical"]+[c[2] for c in configs]
spp_val, spp_final = 64, 256

hdr(f"5a. Validation render ({spp_val} spp)")
for st in states:
    mp = f"{MESH_DIR}/{st}.obj"
    os.makedirs(f"{RENDER_DIR}/{st}", exist_ok=True)
    img = render_one(mp, 0, spp_val)
    mi.util.write_bitmap(f"{RENDER_DIR}/{st}/cam_000.png", img)
    log(f"  {st} OK")

os.makedirs(f"{RENDER_DIR}/background_only", exist_ok=True)
img = render_bg(0, spp_val)
mi.util.write_bitmap(f"{RENDER_DIR}/background_only/cam_000.png", img)
log("  bg OK")

hdr(f"5b. Full render ({spp_final} spp)")
for st in states:
    mp = f"{MESH_DIR}/{st}.obj"
    od = f"{RENDER_DIR}/{st}"; os.makedirs(od, exist_ok=True)
    for ci in range(12):
        img = render_one(mp, ci, spp_final)
        mi.util.write_bitmap(f"{od}/cam_{ci:03d}.png", img)
    log(f"  {st}: 12 views done")

od = f"{RENDER_DIR}/background_only"; os.makedirs(od, exist_ok=True)
for ci in range(12):
    img = render_bg(ci, spp_final)
    mi.util.write_bitmap(f"{od}/cam_{ci:03d}.png", img)
log("  bg: 12 views done")

hdr("6. GT validation")
from PIL import Image
def load_img(p): return np.array(Image.open(p)).astype(np.float32)/255.0

# Identity check
for cam in [0,3,6,9]:
    ref = load_img(f"{RENDER_DIR}/canonical/cam_{cam:03d}.png")
    tst = load_img(f"{RENDER_DIR}/canonical/cam_{cam:03d}.png")
    mae = np.abs(ref-tst).mean()
    psnr = -10*np.log10(((ref-tst)**2).mean()+1e-10)
    log(f"  canonical self cam_{cam}: MAE={mae:.6f} PSNR={psnr:.1f}")

# Deformation difference
rows = []
for st in [c[2] for c in configs]:
    od = f"{RENDER_DIR}/{st}"
    for ci in range(12):
        if not os.path.exists(f"{od}/cam_{ci:03d}.png"): continue
        ref = load_img(f"{RENDER_DIR}/canonical/cam_{ci:03d}.png")
        tst = load_img(f"{od}/cam_{ci:03d}.png")
        rows.append({"state":st,"cam":ci,"mae":float(np.abs(ref-tst).mean()),"rmse":float(np.sqrt(((ref-tst)**2).mean()))})
csv.DictWriter(open(f"{OUTPUT}/gt_deformation_difference.csv","w",newline=""),
    fieldnames=["state","cam","mae","rmse"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/gt_deformation_difference.csv","a",newline=""),
    fieldnames=["state","cam","mae","rmse"]).writerows(rows)

for st in [c[2] for c in configs]:
    v=[r["mae"] for r in rows if r["state"]==st]
    if v: log(f"  {st}: MAE mean={np.mean(v):.4f} median={np.median(v):.4f} p95={np.percentile(v,95):.4f}")

hdr("7. Overview")
from PIL import Image as PImage
ov_states=["canonical","shear_0.50","shear_1.00","twist_30","twist_60"]
ov_cams=[0,3,6,9]
ims=[]
for st in ov_states:
    for ci in ov_cams:
        ims.append(np.array(PImage.open(f"{RENDER_DIR}/{st}/cam_{ci:03d}.png").resize((256,256))))
g=np.zeros((len(ov_states)*256,len(ov_cams)*256,3),dtype=np.uint8)
for ri in range(len(ov_states)):
    for ci in range(len(ov_cams)):
        g[ri*256:(ri+1)*256,ci*256:(ci+1)*256]=ims[ri*len(ov_cams)+ci]
PImage.fromarray(g).save(f"{OUTPUT}/gt_overview.png")
log("Overview saved")

with open(f"{OUTPUT}/stage1_log.txt","w") as f: f.write("\n".join(log_lines))
log("\n=== Stage 1 complete ===")
