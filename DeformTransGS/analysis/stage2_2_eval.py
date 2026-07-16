#!/usr/bin/env python3
"""Stage 2.2 evaluation: test-view rendering + ASG contribution gate"""
import sys, os, json, csv, numpy as np, torch
from PIL import Image
from scipy.ndimage import binary_dilation

BASE = "/data/wyh/DeformTransGS"
DATA_DIR = f"{BASE}/data/stage2_2_asg_active"
BASELINE = f"{BASE}/baselines/stage2_2"
OUTPUT = f"{BASE}/experiments/stage2_2_asg_active_benchmark"
os.makedirs(OUTPUT, exist_ok=True)

sys.path.insert(0, '/data/wyh/repos/TSGS'); sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
from scene.gaussian_model import GaussianModel
from scene.specular_model import SpecularModel
from scene import Scene
from gaussian_renderer import render
from argparse import Namespace

device = "cuda"

cams = json.load(open(f"{DATA_DIR}/cameras_48.json"))
split = json.load(open(f"{DATA_DIR}/split_32_16.json"))
test_ids = split["test"]

log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

def load_model(model_path, use_asg=False):
    gm = GaussianModel(sh_degree=3, asg_degree=24 if use_asg else None)
    gm.load_ply(f"{model_path}/point_cloud/iteration_15000/point_cloud.ply")
    gm.active_sh_degree = gm.max_sh_degree
    sm = None
    if use_asg:
        spec_path = f"{model_path}/specular/iteration_15000/specular.pth"
        if os.path.exists(spec_path):
            sm = SpecularModel(is_real=False)
            sm.specular.load_state_dict(torch.load(spec_path, map_location=device))
            sm.specular.eval().to(device)
    return gm, sm

class Args:
    def __init__(self, var_dir, model_path):
        self.source_path = var_dir; self.model_path = model_path
        self.images = "images"; self.resolution = 1; self.white_background = False
        self.data_device = "cuda"; self.eval = True; self.preload_img = False
        self.ncc_scale = 1.0; self.sh_degree = 3; self.asg_degree = 24
        self.delight = False; self.normal = False; self.normal_folder = "normals"
        self.mask_background = False; self.use_delighted_normal = False
        self.use_transparencies_map = False; self.not_delight_only_transparent = False
        self.load2gpu_on_the_fly = False; self.is_real = False; self.is_indoor = False
        self.add_val = False; self.multi_view_num = 8; self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01; self.multi_view_max_dis = 1.5

# Evaluate each variant
results = []

for variant in ["B2_clear_multilobe", "B3_rough_constant", "B4_rough_multilobe"]:
    hdr(f"Evaluating {variant}")
    var_dir = f"{DATA_DIR}/{variant}"
    
    # Load GT and background for test views
    gt_imgs = []
    bg_imgs = []
    for tid in test_ids:
        gt_imgs.append(np.array(Image.open(f"{var_dir}/canonical/cam_{tid:03d}.png").convert("RGB")).astype(np.float32)/255.0)
        bg_imgs.append(np.array(Image.open(f"{var_dir}/background_only/cam_{tid:03d}.png").convert("RGB")).astype(np.float32)/255.0)
    masks = [binary_dilation(np.max(np.abs(g-b),axis=2)>0.01,iterations=1) for g,b in zip(gt_imgs,bg_imgs)]
    
    # Load models
    for model_type in ["asg", "sh"]:
        model_path = f"{BASELINE}/{variant}_{model_type}_15k"
        use_asg = (model_type == "asg")
        gm, sm = load_model(model_path, use_asg)
        
        # Create Scene for camera loading
        scene = Scene(Args(var_dir, model_path), gm, load_iteration=15000, shuffle=False)
        train_cams = scene.getTrainCameras()
        test_cams_list = scene.getTestCameras()
        
        # Get only test cameras (the loader returns all, we filter by id)
        # Actually the test cameras are loaded as test set
        all_test_cams = [c for c in test_cams_list] if test_cams_list else []
        
        # If eval=False, test cameras are empty. We need to get cameras manually.
        if not all_test_cams:
            all_cams = scene.getTrainCameras()
            all_test_cams = [c for i, c in enumerate(all_cams) if c.colmap_id in test_ids]
        
        bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)
        pipe = Namespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
        
        renders_full = []
        renders_nospec = []
        
        for ci, cam in enumerate(all_test_cams[:16]):
            dir_pp = gm.get_xyz - cam.camera_center.repeat(gm.get_xyz.shape[0], 1)
            dir_pp_norm = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normal = gm.get_normal_axis(dir_pp_normalized=dir_pp_norm, return_delta=True)
            
            if sm is not None:
                mlp = sm.step(gm.get_asg_features, dir_pp_norm, normal.detach())
            else:
                mlp = None
            
            r1 = render(cam, gm, pipe, bg_color, app_model=None, mlp_color=mlp, return_plane=False, return_depth_normal=False)
            renders_full.append(r1["render"].detach().clamp(0,1).permute(1,2,0).cpu().numpy())
            
            if sm is not None:
                r2 = render(cam, gm, pipe, bg_color, app_model=None, mlp_color=0, return_plane=False, return_depth_normal=False)
                renders_nospec.append(r2["render"].detach().clamp(0,1).permute(1,2,0).cpu().numpy())
            else:
                renders_nospec.append(renders_full[-1])
        
        # Compute metrics
        for ti in range(16):
            for region_name, mask in [("full", None), ("mask", masks[ti])]:
                if mask is not None and mask.sum() == 0: continue
                p = renders_full[ti][mask] if mask is not None else renders_full[ti]
                g = gt_imgs[ti][mask] if mask is not None else gt_imgs[ti]
                mse = ((p-g)**2).mean()
                psnr = -10*np.log10(mse+1e-10)
                results.append({"variant": variant, "model": f"{model_type}_full", "cam": test_ids[ti], "region": region_name, "psnr": float(psnr)})
                
                if sm is not None:
                    p_ns = renders_nospec[ti][mask] if mask is not None else renders_nospec[ti]
                    mse_ns = ((p_ns-g)**2).mean()
                    psnr_ns = -10*np.log10(mse_ns+1e-10)
                    results.append({"variant": variant, "model": f"{model_type}_nospec", "cam": test_ids[ti], "region": region_name, "psnr": float(psnr_ns)})
                    
                    # Contribution
                    diff = np.abs(renders_full[ti]-renders_nospec[ti]).mean(axis=2)
                    gt_mag = np.abs(gt_imgs[ti]-bg_imgs[ti])[mask].mean() if mask is not None else 0
                    inside = diff[mask].mean() if mask is not None else diff.mean()
                    ratio = inside/(gt_mag+1e-8)
                    results.append({"variant": variant, "model": "asg_contrib", "cam": test_ids[ti], "region": "mask", "ratio": float(ratio)})
        
        log(f"  {model_type}: done")
    
    # Summary
    for model_filter in ["asg_full", "asg_nospec", "sh_full"]:
        vals = [r.get("psnr",0) for r in results if r.get("variant")==variant and r.get("model")==model_filter and r.get("region")=="mask"]
        if vals:
            log(f"  {model_filter:12s} test masked PSNR: {np.mean(vals):.2f}")

# Save CSV
csv_rows = []
for r in results:
    row = {k: r.get(k, "") for k in ["variant","model","cam","region","psnr","ratio"]}
    csv_rows.append(row)
csv.DictWriter(open(f"{OUTPUT}/screening_metrics.csv","w",newline=""),
    fieldnames=["variant","model","cam","region","psnr","ratio"]).writeheader()
csv.DictWriter(open(f"{OUTPUT}/screening_metrics.csv","a",newline=""),
    fieldnames=["variant","model","cam","region","psnr","ratio"]).writerows(csv_rows)

log("\n=== Stage 2.2 evaluation complete ===")
