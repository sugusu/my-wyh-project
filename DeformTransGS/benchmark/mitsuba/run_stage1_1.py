#!/usr/bin/env python3
"""Stage 1.1: Benchmark Repair and Stress Validation"""
import sys, os, json, csv, numpy as np, torch
BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage1_1_benchmark_repair"
MESH_DIR = f"{BASE}/experiments/stage1_minimal_gt/meshes"
GT_DIR = f"{BASE}/experiments/stage1_minimal_gt/render_gt"
os.makedirs(OUTPUT, exist_ok=True)
sys.path.insert(0, f"{BASE}/benchmark")

log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

import mitsuba as mi
mi.set_variant("llvm_ad_rgb")

# ═══════════════════════════════════════════════════════════
# 1. Twist Jacobian Triple Validation
# ═══════════════════════════════════════════════════════════
hdr("1. Twist Jacobian Triple Validation")
from deformations.twist import jacobian as twist_jac, deform_points as twist_def

# Load canonical mesh for z_range
import trimesh
mesh = trimesh.load(f"{MESH_DIR}/canonical.obj")
cv = torch.tensor(mesh.vertices, dtype=torch.float32)
z_min, z_max = cv[:, 2].min().item(), cv[:, 2].max().item()
log(f"Canonical z range: [{z_min:.4f}, {z_max:.4f}]")

torch.manual_seed(20260712)
test_pts = torch.randn(10000, 3)
test_verts = cv[torch.randperm(len(cv))[:10000]]

def finite_diff_jac(pts, theta_max_deg, eps=1e-5):
    J = torch.zeros(len(pts), 3, 3)
    for d in range(3):
        fwd = pts.clone(); fwd[:, d] += eps
        bwd = pts.clone(); bwd[:, d] -= eps
        J[:, :, d] = (twist_def(fwd, theta_max_deg, (z_min,z_max)) -
                      twist_def(bwd, theta_max_deg, (z_min,z_max))) / (2*eps)
    return J

jac_results = []
for theta_deg in [15, 30, 60]:
    ja = twist_jac(test_pts, theta_deg, (z_min, z_max))
    jf_4 = finite_diff_jac(test_pts, theta_deg, 1e-4)
    jf_5 = finite_diff_jac(test_pts, theta_deg, 1e-5)
    jf_6 = finite_diff_jac(test_pts, theta_deg, 1e-6)

    # Autograd
    test_pts_g = test_pts.clone().requires_grad_(True)
    def_pts = twist_def(test_pts_g, theta_deg, (z_min, z_max))
    ja_ag = torch.stack([torch.autograd.grad(def_pts[:, i].sum(), test_pts_g, retain_graph=True)[0].detach() for i in range(3)], dim=1)

    err_ag = (ja - ja_ag).abs().max().item()
    err_fd4 = (ja - jf_4).abs().max().item()
    err_fd5 = (ja - jf_5).abs().max().item()
    err_fd6 = (ja - jf_6).abs().max().item()

    for pts_name, pts in [("random", test_pts), ("vertices", test_verts)]:
        ja_p = twist_jac(pts, theta_deg, (z_min, z_max))
        pts_g = pts.clone().requires_grad_(True)
        dp = twist_def(pts_g, theta_deg, (z_min, z_max))
        jag = torch.stack([torch.autograd.grad(dp[:, i].sum(), pts_g, retain_graph=True)[0].detach() for i in range(3)], dim=1)
        e_ag = (ja_p - jag).abs().max().item()
        log(f"  twist_{theta_deg:2d}° {pts_name:10s}: analytic vs autograd max_err={e_ag:.6e}")
        jac_results.append({"theta_deg":theta_deg,"points":pts_name,"type":"analytic_vs_autograd","max_err":e_ag})

    log(f"  twist_{theta_deg:2d}° random: analytic vs FD(1e-4)={err_fd4:.6e} FD(1e-5)={err_fd5:.6e} FD(1e-6)={err_fd6:.6e}")

# Save twist validation CSV
csv.DictWriter(open(f"{OUTPUT}/twist_jacobian_validation.csv","w",newline=""),
    fieldnames=["theta_deg","points","type","max_err"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/twist_jacobian_validation.csv","a",newline=""),
    fieldnames=["theta_deg","points","type","max_err"]).writerows(jac_results)

# ═══════════════════════════════════════════════════════════
# 2. Load GT images (PNG)
# ═══════════════════════════════════════════════════════════
hdr("2. Loading GT images")
from PIL import Image
def load_img(p): return np.array(Image.open(p)).astype(np.float32)/255.0

states = ["canonical","shear_0.25","shear_0.50","shear_1.00","twist_15","twist_30","twist_60"]
n_cam = 12

gt = {}
for st in states:
    gt[st] = [load_img(f"{GT_DIR}/{st}/cam_{c:03d}.png") for c in range(n_cam)]
bg = [load_img(f"{GT_DIR}/background_only/cam_{c:03d}.png") for c in range(n_cam)]

# ═══════════════════════════════════════════════════════════
# 3. Optical Effect Mask
# ═══════════════════════════════════════════════════════════
hdr("3. Optical Effect Mask analysis")
from scipy.ndimage import binary_dilation

def make_mask(img, bg_img, eps):
    diff = np.max(np.abs(img.astype(np.float32) - bg_img.astype(np.float32)), axis=2)
    return diff > eps

epsilons = [0.005, 0.01, 0.02]
mask_data = []
for st in states:
    for c in range(n_cam):
        for eps in epsilons:
            m = make_mask(gt[st][c], bg[c], eps)
            md = binary_dilation(m, iterations=1) if eps == 0.01 else m
            cnt = m.sum()
            rat = cnt / (512*512)
            mask_data.append({"state":st,"cam":c,"eps":eps,"pixels":int(cnt),"ratio":float(rat)})

csv.DictWriter(open(f"{OUTPUT}/optical_effect_mask_metrics.csv","w",newline=""),
    fieldnames=["state","cam","eps","pixels","ratio"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/optical_effect_mask_metrics.csv","a",newline=""),
    fieldnames=["state","cam","eps","pixels","ratio"]).writerows(mask_data)

for eps in epsilons:
    ratios = [d["ratio"] for d in mask_data if d["eps"]==eps]
    log(f"  eps={eps:.3f}: mean ratio={np.mean(ratios):.4f}, min={np.min(ratios):.4f}, max={np.max(ratios):.4f}")

# Generate masks with dilation for eps=0.01
masks = {}
for st in states:
    masks[st] = [binary_dilation(make_mask(gt[st][c], bg[c], 0.01), iterations=1) for c in range(n_cam)]
mask_canon = masks["canonical"]

# ═══════════════════════════════════════════════════════════
# 4. Masked Deformation Difference
# ═══════════════════════════════════════════════════════════
hdr("4. Masked deformation difference")
diff_rows = []
for st in ["shear_0.25","shear_0.50","shear_1.00","twist_15","twist_30","twist_60"]:
    for c in range(n_cam):
        ref = gt["canonical"][c]; tst = gt[st][c]
        # Full image
        diff = ref - tst
        full_mae = np.abs(diff).mean()
        full_rmse = np.sqrt((diff**2).mean())
        full_psnr = -10*np.log10(((diff)**2).mean()+1e-10)
        diff_rows.append({"state":st,"cam":c,"region":"full","pixels":512*512,"ratio":1.0,
            "mae":full_mae,"rmse":full_rmse,"psnr":full_psnr})
        # Canonical mask
        m = masks["canonical"][c]
        if m.sum()>0:
            dm = diff[m]
            diff_rows.append({"state":st,"cam":c,"region":"mask_canon","pixels":int(m.sum()),"ratio":float(m.mean()),
                "mae":float(np.abs(dm).mean()),"rmse":float(np.sqrt((dm**2).mean())),
                "psnr":float(-10*np.log10((dm**2).mean()+1e-10))})
        # Union mask
        mu = masks["canonical"][c] | masks[st][c]
        if mu.sum()>0:
            du = diff[mu]
            diff_rows.append({"state":st,"cam":c,"region":"mask_union","pixels":int(mu.sum()),"ratio":float(mu.mean()),
                "mae":float(np.abs(du).mean()),"rmse":float(np.sqrt((du**2).mean())),
                "psnr":float(-10*np.log10((du**2).mean()+1e-10))})

csv.DictWriter(open(f"{OUTPUT}/masked_gt_deformation_difference.csv","w",newline=""),
    fieldnames=["state","cam","region","pixels","ratio","mae","rmse","psnr"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/masked_gt_deformation_difference.csv","a",newline=""),
    fieldnames=["state","cam","region","pixels","ratio","mae","rmse","psnr"]).writerows(diff_rows)

# Summary by region type
for region in ["full","mask_canon","mask_union"]:
    log(f"\n  Region: {region}")
    for st in ["shear_0.25","shear_0.50","shear_1.00","twist_15","twist_30","twist_60"]:
        vals = [r["mae"] for r in diff_rows if r["state"]==st and r["region"]==region]
        if vals:
            log(f"    {st:15s}: mean={np.mean(vals):.4f} median={np.median(vals):.4f} p95={np.percentile(vals,95):.4f}")

# ═══════════════════════════════════════════════════════════
# 5. Camera Background Coverage
# ═══════════════════════════════════════════════════════════
hdr("5. Camera background coverage")
cam_rows = []
for c in range(n_cam):
    m = masks["canonical"][c]
    mag = np.max(np.abs(gt["canonical"][c].astype(np.float32) - bg[c].astype(np.float32)), axis=2)
    cov = {"cam":c,"azimuth":c*30,"effect_ratio":float(m.mean()),
           "effect_mag_mean":float(mag[m].mean()) if m.sum()>0 else 0,
           "effect_mag_p95":float(np.percentile(mag[m],95)) if m.sum()>0 else 0}
    cam_rows.append(cov)
    log(f"  cam_{c:02d} az={c*30:3d}°: effect_ratio={cov['effect_ratio']:.4f} mean_mag={cov['effect_mag_mean']:.4f}")

csv.DictWriter(open(f"{OUTPUT}/camera_background_coverage.csv","w",newline=""),
    fieldnames=["cam","azimuth","effect_ratio","effect_mag_mean","effect_mag_p95"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/camera_background_coverage.csv","a",newline=""),
    fieldnames=["cam","azimuth","effect_ratio","effect_mag_mean","effect_mag_p95"]).writerows(cam_rows)

# ═══════════════════════════════════════════════════════════
# 6. Directional Lighting Scene + Render
# ═══════════════════════════════════════════════════════════
hdr("6. Directional lighting rendering")
W, H, SPP = 512, 512, 256
render_dir = f"{OUTPUT}/render_directional"
cams = json.load(open(f"{BASE}/experiments/stage1_minimal_gt/cameras.json"))

def make_dir_scene(cam_idx, mesh_path=None):
    c = cams[cam_idx]
    sensor_tf = mi.ScalarTransform4f.look_at(
        mi.ScalarVector3f(*c["origin"]), mi.ScalarVector3f(*c["target"]), mi.ScalarVector3f(*c["up"]))
    key_tf = mi.ScalarTransform4f.look_at(
        mi.ScalarPoint3f(3,-2,4), mi.ScalarPoint3f(0,0,0), mi.ScalarPoint3f(0,0,1))
    fill_tf = mi.ScalarTransform4f.look_at(
        mi.ScalarPoint3f(-2,3,1), mi.ScalarPoint3f(0,0,0), mi.ScalarPoint3f(0,0,1))
    d = {
        "type":"scene","integrator":{"type":"path","max_depth":12},
        "sensor":{"type":"perspective","fov":45,"to_world":sensor_tf,
            "film":{"type":"hdrfilm","width":W,"height":H,"rfilter":{"type":"gaussian"}},
            "sampler":{"type":"independent","sample_count":SPP}},
        "bg_plane":{"type":"rectangle","to_world":mi.ScalarTransform4f.translate([0,0,-2]) @ mi.ScalarTransform4f.scale([4,4,1]),
            "bsdf":{"type":"diffuse","reflectance":{"type":"checkerboard",
                "color0":{"type":"rgb","value":[0.9,0.9,0.9]},
                "color1":{"type":"rgb","value":[0.1,0.1,0.1]}}}},
        "ground":{"type":"disk","to_world":mi.ScalarTransform4f.translate([0,0,-2.5]) @ mi.ScalarTransform4f.scale(10),
            "bsdf":{"type":"diffuse","reflectance":{"type":"rgb","value":[0.5,0.5,0.5]}}},
        "emitter_constant":{"type":"constant","radiance":{"type":"rgb","value":[0.2,0.2,0.2]}},
        "emitter_key":{"type":"rectangle","to_world":key_tf,
            "emitter":{"type":"area","radiance":{"type":"rgb","value":[5.0,5.0,5.0]}}},
        "emitter_fill":{"type":"rectangle","to_world":fill_tf,
            "emitter":{"type":"area","radiance":{"type":"rgb","value":[1.0,1.0,1.0]}}},
    }
    if mesh_path:
        d["object"] = {"type":"obj","filename":mesh_path,
            "bsdf":{"type":"dielectric","int_ior":1.49,"ext_ior":1.0}}
    return mi.load_dict(d)

dir_states = ["canonical","shear_1.00","twist_60"]
for st in dir_states + ["background_only"]:
    od = f"{render_dir}/{st}"; os.makedirs(od, exist_ok=True)
    mp = f"{MESH_DIR}/{st}.obj" if st != "background_only" else None
    for ci in range(n_cam):
        sc = make_dir_scene(ci, mp)
        img = mi.render(sc, spp=SPP)
        mi.util.write_bitmap(f"{od}/cam_{ci:03d}.png", img)
    log(f"  {st}: 12 views done")

# ═══════════════════════════════════════════════════════════
# 7. Lighting Stress Metrics
# ═══════════════════════════════════════════════════════════
hdr("7. Lighting stress metrics")
gt_dir = {st: [load_img(f"{GT_DIR}/{st}/cam_{c:03d}.png") for c in range(n_cam)] for st in dir_states}
bg_gt = [load_img(f"{GT_DIR}/background_only/cam_{c:03d}.png") for c in range(n_cam)]
gt_dir_render = {}
for st in dir_states:
    gt_dir_render[st] = [load_img(f"{render_dir}/{st}/cam_{c:03d}.png") for c in range(n_cam)]
bg_dir = [load_img(f"{render_dir}/background_only/cam_{c:03d}.png") for c in range(n_cam)]

light_rows = []
for lighting, img_dict, bg_arr in [("constant", gt_dir, bg_gt), ("directional", gt_dir_render, bg_dir)]:
    for st in ["shear_1.00","twist_60"]:
        for c in range(n_cam):
            ref = img_dict["canonical"][c]; tst = img_dict[st][c]
            diff = ref - tst
            full_mae = np.abs(diff).mean()
            full_rmse = np.sqrt((diff**2).mean())
            full_psnr = -10*np.log10((diff**2).mean()+1e-10)

            # Union mask for this lighting
            m_c = binary_dilation(make_mask(img_dict["canonical"][c], bg_arr[c], 0.01), iterations=1)
            m_d = binary_dilation(make_mask(img_dict[st][c], bg_arr[c], 0.01), iterations=1)
            mu = m_c | m_d
            du = diff[mu] if mu.sum()>0 else diff
            um_mae = np.abs(du).mean()
            um_rmse = np.sqrt((du**2).mean())
            um_psnr = -10*np.log10((du**2).mean()+1e-10)

            light_rows.append({"lighting":lighting,"state":st,"cam":c,
                "full_mae":full_mae,"full_rmse":full_rmse,"full_psnr":full_psnr,
                "union_mae":um_mae,"union_rmse":um_rmse,"union_psnr":um_psnr})

            # Canonical object effect magnitude
            eff = np.max(np.abs(img_dict["canonical"][c].astype(np.float32) - bg_arr[c].astype(np.float32)), axis=2)
            light_rows.append({"lighting":lighting,"state":"canonical_effect","cam":c,
                "full_mae":eff.mean(),"full_rmse":eff.std(),"full_psnr":0,
                "union_mae":float(np.percentile(eff,95)),"union_rmse":0,"union_psnr":0})

csv.DictWriter(open(f"{OUTPUT}/lighting_stress_metrics.csv","w",newline=""),
    fieldnames=["lighting","state","cam","full_mae","full_rmse","full_psnr","union_mae","union_rmse","union_psnr"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/lighting_stress_metrics.csv","a",newline=""),
    fieldnames=["lighting","state","cam","full_mae","full_rmse","full_psnr","union_mae","union_rmse","union_psnr"]).writerows(light_rows)

for lighting in ["constant","directional"]:
    log(f"\n  Lighting: {lighting}")
    for st in ["shear_1.00","twist_60"]:
        fv = [r["full_mae"] for r in light_rows if r["lighting"]==lighting and r["state"]==st]
        uv = [r["union_mae"] for r in light_rows if r["lighting"]==lighting and r["state"]==st]
        if fv:
            log(f"    {st:15s}: full_MAE={np.mean(fv):.4f} union_MAE={np.mean(uv):.4f}")

# ═══════════════════════════════════════════════════════════
# 8. Save log
# ═══════════════════════════════════════════════════════════
with open(f"{OUTPUT}/stage1_1_log.txt","w") as f: f.write("\n".join(log_lines))
log("\n=== Stage 1.1 complete ===")
