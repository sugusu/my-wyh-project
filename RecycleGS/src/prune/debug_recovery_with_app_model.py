#!/usr/bin/env python3
"""Debug recovery with AppModel: evaluate recovery PLY with/without AppModel.
Hypothesis: if PSNR recovers from ~8.6 to ~22 when AppModel is loaded, then the issue is confirmed."""
import os, sys, json, torch, numpy as np, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

SCENE_01_CONFIG = "/data/wyh/RecycleGS/configs/scene_01.yaml"
BASELINE_DIR = "/data/wyh/RecycleGS/baselines/tsgs_scene01_full"
RECOVERY_PLY = "/data/wyh/RecycleGS/outputs/prune_only/scene_01/ratio_005/schedule_control/recovery_500/point_cloud/iteration_15500/point_cloud.ply"
OUTPUT_DIR = "/data/wyh/RecycleGS/outputs/debug/stage2b_bundle_audit"

def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse < 1e-10:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_app_model(model_path, iteration=15000, device='cuda:0'):
    from scene.app_model import AppModel
    app_model = AppModel()
    weights_path = os.path.join(model_path, "app_model", f"iteration_{iteration}", "app.pth")
    if not os.path.exists(weights_path):
        print(f"  AppModel weights NOT found: {weights_path}")
        return None
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    app_model.load_state_dict(state_dict)
    app_model.cuda()
    app_model.eval()
    print(f"  AppModel loaded from: {weights_path}")
    print(f"  appear_ab shape: {app_model.appear_ab.shape}")
    return app_model

def evaluate_without_app_model(gaussians, cameras, pipe, bg_color, device, n_views=64):
    from gaussian_renderer import render
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    psnrs = []
    for ci in range(min(n_views, len(cameras))):
        cam = cameras[ci]
        rendered = render(cam, gaussians, pipe, bg)
        render_img = rendered['render'].clamp(0.0, 1.0)
        gt_img = getattr(cam, 'original_image', None)
        if gt_img is None:
            gt_img = getattr(cam, 'image', None)
        if gt_img is not None:
            gt_img = gt_img.to(device).clamp(0.0, 1.0)
            psnr = compute_psnr(render_img, gt_img[:3]).item()
            psnrs.append(psnr)
    return np.mean(psnrs) if psnrs else 0.0

def evaluate_with_app_model(gaussians, cameras, pipe, bg_color, device, app_model, n_views=64):
    from gaussian_renderer import render
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    psnrs = []
    for ci in range(min(n_views, len(cameras))):
        cam = cameras[ci]
        rendered = render(cam, gaussians, pipe, bg, app_model=app_model)
        # Use app_image if available, otherwise fall back to render
        if 'app_image' in rendered:
            render_img = rendered['app_image'].clamp(0.0, 1.0)
        else:
            render_img = rendered['render'].clamp(0.0, 1.0)
        gt_img = getattr(cam, 'original_image', None)
        if gt_img is None:
            gt_img = getattr(cam, 'image', None)
        if gt_img is not None:
            gt_img = gt_img.to(device).clamp(0.0, 1.0)
            psnr = compute_psnr(render_img, gt_img[:3]).item()
            psnrs.append(psnr)
    return np.mean(psnrs) if psnrs else 0.0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda:0'
    bg_color = [0.0, 0.0, 0.0]

    with open(SCENE_01_CONFIG) as f:
        cfg = yaml.safe_load(f)

    baseline_dir = cfg.get('baseline_model_dir', BASELINE_DIR)

    print("=" * 60)
    print("Stage 2B-AA: Debug Recovery with AppModel")
    print("=" * 60)

    # ── Setup: load scene, gaussians, pipe ──
    from recyclegs.tsgs_loader import load_scene
    from recyclegs.tsgs_loader import init_tsgs_env

    init_tsgs_env('/data/wyh/repos/TSGS')

    results = {}

    # ── Test 1: Baseline PLY without AppModel ──
    print("\n[Test 1] Baseline PLY WITHOUT AppModel")
    scene_cfg = {
        'scene_dir': cfg['scene_dir'],
        'model_dir': baseline_dir,
        'checkpoint_path': cfg['baseline_gaussian_checkpoint'],
        'checkpoint_iteration': 15000,
    }
    scene, gaussians, pipe = load_scene(scene_cfg, device)
    psnr_no_app_baseline = evaluate_without_app_model(gaussians, scene.getTestCameras(), pipe, bg_color, device)
    print(f"  PSNR (no AppModel): {psnr_no_app_baseline:.4f}")
    results['baseline_no_app'] = round(psnr_no_app_baseline, 4)

    # ── Test 2: Baseline PLY with AppModel ──
    print("\n[Test 2] Baseline PLY WITH AppModel")
    scene, gaussians, pipe = load_scene(scene_cfg, device)
    app_model = load_app_model(baseline_dir, iteration=15000, device=device)
    psnr_with_app_baseline = 0.0
    if app_model is not None:
        gaussians.use_app = True  # Must set this flag!
        psnr_with_app_baseline = evaluate_with_app_model(gaussians, scene.getTestCameras(), pipe, bg_color, device, app_model)
        print(f"  PSNR (with AppModel, use_app=True): {psnr_with_app_baseline:.4f}")
        results['baseline_with_app'] = round(psnr_with_app_baseline, 4)
    else:
        print("  SKIPPED: AppModel not found")

    # ── Test 3: Baseline PLY with AppModel but use_app=False ──
    print("\n[Test 3] Baseline PLY WITH AppModel, use_app=False")
    scene, gaussians, pipe = load_scene(scene_cfg, device)
    if app_model is not None:
        gaussians.use_app = False
        psnr_with_app_noflag = evaluate_with_app_model(gaussians, scene.getTestCameras(), pipe, bg_color, device, app_model)
        print(f"  PSNR (with AppModel, use_app=False): {psnr_with_app_noflag:.4f}")
        results['baseline_app_noflag'] = round(psnr_with_app_noflag, 4)

    # ── Test 4: Recovery PLY without AppModel ──
    print("\n[Test 4] Recovery PLY WITHOUT AppModel")
    rec_cfg = {
        'scene_dir': cfg['scene_dir'],
        'model_dir': baseline_dir,
        'checkpoint_path': RECOVERY_PLY,
        'checkpoint_iteration': -1,
    }
    scene_rec, gaussians_rec, pipe_rec = load_scene(rec_cfg, device)
    psnr_no_app_recovery = evaluate_without_app_model(gaussians_rec, scene_rec.getTestCameras(), pipe_rec, bg_color, device)
    print(f"  PSNR (no AppModel): {psnr_no_app_recovery:.4f}")
    results['recovery_no_app'] = round(psnr_no_app_recovery, 4)

    # ── Test 5: Recovery PLY with AppModel ──
    print("\n[Test 5] Recovery PLY WITH AppModel (baseline weights)")
    scene_rec, gaussians_rec, pipe_rec = load_scene(rec_cfg, device)
    if app_model is not None:
        gaussians_rec.use_app = True
        psnr_with_app_recovery = evaluate_with_app_model(gaussians_rec, scene_rec.getTestCameras(), pipe_rec, bg_color, device, app_model)
        print(f"  PSNR (with AppModel, use_app=True): {psnr_with_app_recovery:.4f}")
        results['recovery_with_app'] = round(psnr_with_app_recovery, 4)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    if 'baseline_no_app' in results and 'baseline_with_app' in results:
        print(f"\n  AppModel uplift (baseline): {results['baseline_with_app'] - results['baseline_no_app']:.4f} dB")
    if 'recovery_no_app' in results and 'recovery_with_app' in results:
        print(f"  AppModel uplift (recovery): {results['recovery_with_app'] - results['recovery_no_app']:.4f} dB")
    if 'baseline_no_app' in results and 'recovery_no_app' in results:
        print(f"  Recovery gap (no AppModel): {results['baseline_no_app'] - results['recovery_no_app']:.4f} dB")
    if 'baseline_with_app' in results and 'recovery_with_app' in results:
        print(f"  Recovery gap (with AppModel): {results['baseline_with_app'] - results['recovery_with_app']:.4f} dB")

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "debug_recovery_app_model_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

if __name__ == '__main__':
    main()
