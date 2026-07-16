#!/usr/bin/env python3
"""Stage 2.1: ASG contribution gate - render and evaluate"""
import sys, os, json, csv, numpy as np
from PIL import Image
from pathlib import Path

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage2_1_asg_contribution"
TSGS = "/data/wyh/repos/TSGS"
MODEL_A = f"{BASE}/baselines/tsgs_synth_canonical_asg"
MODEL_B = f"{BASE}/baselines/tsgs_synth_canonical_sh"
GT_DIR = f"{BASE}/experiments/stage1_minimal_gt/render_gt"
DS_DIR = f"{BASE}/data/stage2_canonical_tsgs/nerf_format"

os.makedirs(f"{OUTPUT}/renders/asg_full", exist_ok=True)
os.makedirs(f"{OUTPUT}/renders/asg_no_spec", exist_ok=True)
os.makedirs(f"{OUTPUT}/renders/sh_retrained", exist_ok=True)
os.makedirs(f"{OUTPUT}/contribution_maps", exist_ok=True)

sys.path.insert(0, TSGS)
sys.path.insert(0, f"{TSGS}/pytorch3d_stub")

log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

import argparse, torch
torch.set_grad_enabled(False)

from scene.gaussian_model import GaussianModel
from scene.specular_model import SpecularModel
from scene import Scene
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.graphics_utils import focal2fov, fov2focal

device = "cuda"

# ═══════════════════════════════════════════════════════════
# 1. Load dataset cameras
# ═══════════════════════════════════════════════════════════
hdr("1. Loading cameras")
from scene.dataset_readers import readNerfSyntheticInfo
scene_info = readNerfSyntheticInfo(DS_DIR, white_background=False, eval=False)
from utils.camera_utils import cameraList_from_camInfos, loadCam

class Args:
    def __init__(self):
        self.resolution = 2
        self.preload_img = False
        self.ncc_scale = 1.0
        self.data_device = "cuda"

args = Args()
cam_infos = scene_info.train_cameras + scene_info.test_cameras
cameras = cameraList_from_camInfos(cam_infos, 1.0, args)
log(f"Loaded {len(cameras)} cameras")

bg = torch.tensor([0, 0, 0], device=device)

# ═══════════════════════════════════════════════════════════
# 2. Load Model A (ASG) and Model B (SH)
# ═══════════════════════════════════════════════════════════
hdr("2. Loading models")

def load_model(model_path, use_asg=False):
    ckpt_path = f"{model_path}/point_cloud/iteration_30000/point_cloud.ply"
    spec_path = f"{model_path}/specular/iteration_30000/specular.pth"

    gm = GaussianModel(sh_degree=3, asg_degree=24 if use_asg else None)
    gm.load_ply(ckpt_path)
    gm.active_sh_degree = gm.max_sh_degree
    log(f"  Loaded {gm.get_xyz.shape[0]} Gaussians from {ckpt_path}")
    log(f"  ASG: {gm.max_asg_degree}, SH deg: {gm.active_sh_degree}")

    sm = None
    if use_asg and os.path.exists(spec_path):
        sm = SpecularModel(is_real=False)
        sm.specular.load_state_dict(torch.load(spec_path, map_location=device, weights_only=True))
        sm.specular.eval().to(device)
        log(f"  Loaded SpecularModel from {spec_path}")

    return gm, sm

gm_a, sm_a = load_model(MODEL_A, use_asg=True)
gm_b, _ = load_model(MODEL_B, use_asg=False)

# ═══════════════════════════════════════════════════════════
# 3. Same-checkpoint validation
# ═══════════════════════════════════════════════════════════
hdr("3. Same-checkpoint validation (ASG_FULL vs ASG_NO_SPEC)")
# Since both use the same gm_a, there's no difference
log("  ASG_FULL and ASG_NO_SPEC use SAME GaussianModel object")
log("  All Gaussian attributes are identical (confirmed by using same object)")

# ═══════════════════════════════════════════════════════════
# 4. Render functions
# ═══════════════════════════════════════════════════════════
hdr("4. Rendering")

def render_view(cam, gm, sm, use_spec=True):
    dir_pp = gm.get_xyz - cam.camera_center.repeat(gm.get_xyz.shape[0], 1)
    dir_pp_norm = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normal = gm.get_normal_axis(dir_pp_normalized=dir_pp_norm, return_delta=True)

    if sm is not None and use_spec:
        mlp_color = sm.step(gm.get_asg_features, dir_pp_norm, normal.detach())
    else:
        mlp_color = 0

    render_pkg = render(cam, gm, pipe, bg, app_model=None, mlp_color=mlp_color,
                        return_plane=False, return_depth_normal=False)
    return render_pkg["render"].clamp(0, 1)

pipe = argparse.Namespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)


class FakeArgs:
    def __init__(self):
        self.detach_before = False
        self.detach_after = False
        self.SH_detach = False
        self.detach_xyz = False

for mode, gm, sm, use_spec, out_dir in [
    ("asg_full", gm_a, sm_a, True, f"{OUTPUT}/renders/asg_full"),
    ("asg_no_spec", gm_a, sm_a, False, f"{OUTPUT}/renders/asg_no_spec"),
    ("sh_retrained", gm_b, None, False, f"{OUTPUT}/renders/sh_retrained"),
]:
    hdr(f"  Rendering {mode}")
    for ci, cam in enumerate(cameras):
        img = render_view(cam, gm, sm, use_spec)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(f"{out_dir}/cam_{ci:03d}.png")
    log(f"  {mode}: 12 views done")

# ═══════════════════════════════════════════════════════════
# 5. Load GT and compute metrics
# ═══════════════════════════════════════════════════════════
hdr("5. Computing metrics")
from skimage.metrics import structural_similarity as ssim_sk
from scipy.ndimage import binary_dilation

def load_img(p): return np.array(Image.open(p)).astype(np.float32) / 255.0
gt_imgs = [load_img(f"{GT_DIR}/canonical/cam_{c:03d}.png") for c in range(12)]
bg_imgs = [load_img(f"{GT_DIR}/background_only/cam_{c:03d}.png") for c in range(12)]

# Optical effect mask
masks = []
for c in range(12):
    diff = np.max(np.abs(gt_imgs[c] - bg_imgs[c]), axis=2)
    m = binary_dilation(diff > 0.01, iterations=1)
    masks.append(m)
    log(f"  Mask cam_{c}: {m.sum()}/{512*512} pixels ({m.mean()*100:.2f}%)")

methods = {
    "ASG_FULL": [load_img(f"{OUTPUT}/renders/asg_full/cam_{c:03d}.png") for c in range(12)],
    "ASG_NO_SPEC": [load_img(f"{OUTPUT}/renders/asg_no_spec/cam_{c:03d}.png") for c in range(12)],
    "SH_RETRAINED": [load_img(f"{OUTPUT}/renders/sh_retrained/cam_{c:03d}.png") for c in range(12)],
}

csv_rows = []
for method_name, imgs in methods.items():
    for c in range(12):
        pred = imgs[c]
        gt = gt_imgs[c]
        for region_name, mask in [("full", None), ("mask", masks[c])]:
            if mask is not None:
                pix = mask.sum()
                if pix == 0: continue
                p = pred[mask]; g = gt[mask]
            else:
                pix = 512*512; p = pred; g = gt

            mse = ((p - g)**2).mean()
            psnr = -10*np.log10(mse + 1e-10)
            mae = np.abs(p - g).mean()
            rmse = np.sqrt(mse)

            row = {"method": method_name, "camera": c, "region": region_name,
                   "pixels": int(pix), "mae": float(mae), "rmse": float(rmse),
                   "psnr": float(psnr)}
            csv_rows.append(row)

csv.DictWriter(open(f"{OUTPUT}/canonical_metrics.csv","w",newline=""),
    fieldnames=["method","camera","region","pixels","mae","rmse","psnr"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/canonical_metrics.csv","a",newline=""),
    fieldnames=["method","camera","region","pixels","mae","rmse","psnr"]).writerows(csv_rows)

for method_name in ["ASG_FULL", "ASG_NO_SPEC", "SH_RETRAINED"]:
    vals = [r["psnr"] for r in csv_rows if r["method"]==method_name and r["region"]=="mask"]
    log(f"  {method_name:15s} masked PSNR: mean={np.mean(vals):.2f} median={np.median(vals):.2f}")

# ═══════════════════════════════════════════════════════════
# 6. ORE / NORE
# ═══════════════════════════════════════════════════════════
hdr("6. Optical Residual")
ore_rows = []
for method_name, imgs in methods.items():
    for c in range(12):
        o_pred = imgs[c] - bg_imgs[c]
        o_gt = gt_imgs[c] - bg_imgs[c]
        m = masks[c]
        diff = np.abs(o_pred - o_gt)[m]
        gt_mag = np.abs(o_gt)[m]
        ore = diff.mean()
        nore = ore / (gt_mag.mean() + 1e-8)
        ore_rows.append({"method": method_name, "camera": c, "ore": float(ore), "nore": float(nore)})

csv.DictWriter(open(f"{OUTPUT}/optical_residual_metrics.csv","w",newline=""),
    fieldnames=["method","camera","ore","nore"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/optical_residual_metrics.csv","a",newline=""),
    fieldnames=["method","camera","ore","nore"]).writerows(ore_rows)

for method_name in ["ASG_FULL", "ASG_NO_SPEC", "SH_RETRAINED"]:
    v = [r["nore"] for r in ore_rows if r["method"]==method_name]
    log(f"  {method_name:15s} mean NORE={np.mean(v):.4f}")

# ═══════════════════════════════════════════════════════════
# 7. ASG Contribution Metrics
# ═══════════════════════════════════════════════════════════
hdr("7. ASG contribution metrics")
full_imgs = methods["ASG_FULL"]
no_spec_imgs = methods["ASG_NO_SPEC"]

contrib_rows = []
for c in range(12):
    contrib = np.abs(full_imgs[c].astype(np.float32) - no_spec_imgs[c].astype(np.float32))
    contrib_mag = contrib.mean(axis=2)  # mean across channels

    inside = contrib_mag[masks[c]]
    outside = contrib_mag[~masks[c]]
    gt_mag = np.abs(gt_imgs[c].astype(np.float32) - bg_imgs[c].astype(np.float32))[masks[c]].mean()

    c_mean = contrib_mag.mean()
    c_inside = inside.mean() if inside.size > 0 else 0
    c_outside = outside.mean() if outside.size > 0 else 0
    ratio = c_inside / (gt_mag + 1e-8)
    concentration = c_inside / (c_outside + 1e-8)

    contrib_rows.append({"camera": c, "contrib_mean": float(c_mean),
        "contrib_inside": float(c_inside), "contrib_outside": float(c_outside),
        "contrib_ratio": float(ratio), "concentration": float(concentration)})
    log(f"  cam_{c}: inside={c_inside:.6f} outside={c_outside:.6f} ratio={ratio:.4f} conc={concentration:.2f}")

csv.DictWriter(open(f"{OUTPUT}/asg_contribution_metrics.csv","w",newline=""),
    fieldnames=["camera","contrib_mean","contrib_inside","contrib_outside","contrib_ratio","concentration"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/asg_contribution_metrics.csv","a",newline=""),
    fieldnames=["camera","contrib_mean","contrib_inside","contrib_outside","contrib_ratio","concentration"]).writerows(contrib_rows)

mean_ratio = np.mean([r["contrib_ratio"] for r in contrib_rows])
log(f"\n  Mean ASG Contribution Ratio: {mean_ratio:.4f}")

# ═══════════════════════════════════════════════════════════
# 8. Camera Gain Stability
# ═══════════════════════════════════════════════════════════
hdr("8. Camera gain stability")
gain_rows = []
for c in range(12):
    psnr_full = [r["psnr"] for r in csv_rows if r["method"]=="ASG_FULL" and r["camera"]==c and r["region"]=="mask"][0]
    psnr_nospec = [r["psnr"] for r in csv_rows if r["method"]=="ASG_NO_SPEC" and r["camera"]==c and r["region"]=="mask"][0]
    psnr_sh = [r["psnr"] for r in csv_rows if r["method"]=="SH_RETRAINED" and r["camera"]==c and r["region"]=="mask"][0]
    gain_rows.append({"camera": c, "full_psnr": psnr_full, "no_spec_psnr": psnr_nospec,
        "gain_vs_no_spec": psnr_full - psnr_nospec, "sh_psnr": psnr_sh,
        "gain_vs_sh": psnr_full - psnr_sh})

csv.DictWriter(open(f"{OUTPUT}/camera_gain_stability.csv","w",newline=""),
    fieldnames=["camera","full_psnr","no_spec_psnr","gain_vs_no_spec","sh_psnr","gain_vs_sh"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/camera_gain_stability.csv","a",newline=""),
    fieldnames=["camera","full_psnr","no_spec_psnr","gain_vs_no_spec","sh_psnr","gain_vs_sh"]).writerows(gain_rows)

pos = sum(1 for r in gain_rows if r["gain_vs_no_spec"] > 0)
pos_05 = sum(1 for r in gain_rows if r["gain_vs_no_spec"] >= 0.5)
log(f"  PSNR gain > 0: {pos}/12 cameras")
log(f"  PSNR gain >= 0.5dB: {pos_05}/12 cameras")

# ═══════════════════════════════════════════════════════════
# 9. Overview
# ═══════════════════════════════════════════════════════════
hdr("9. Overview")
from PIL import Image as PImage
ov_cams = [0, 3, 6, 9]
ov_methods = ["ASG_FULL", "ASG_NO_SPEC", "SH_RETRAINED"]
ims = {}
for m in ov_methods:
    ims[m] = [load_img(f"{OUTPUT}/renders/{m.lower()}/cam_{c:03d}.png") for c in ov_cams]
gt_ov = [gt_imgs[c] for c in ov_cams]

rows = len(ov_methods) + 1  # GT + methods
cols = len(ov_cams)
grid = np.zeros((rows*256, cols*256, 3), dtype=np.uint8)
for ci, c in enumerate(ov_cams):
    grid[0*256:1*256, ci*256:(ci+1)*256] = (gt_imgs[c]*255).astype(np.uint8)
for ri, m in enumerate(ov_methods):
    for ci, c in enumerate(ov_cams):
        grid[(ri+1)*256:(ri+2)*256, ci*256:(ci+1)*256] = (ims[m][ci]*255).astype(np.uint8)
PImage.fromarray(grid).save(f"{OUTPUT}/canonical_asg_contribution_overview.png")
log("Overview saved")

# ═══════════════════════════════════════════════════════════
# 10. Log
# ═══════════════════════════════════════════════════════════
with open(f"{OUTPUT}/stage2_1_log.txt","w") as f: f.write("\n".join(log_lines))
log("\n=== Stage 2.1 complete ===")
