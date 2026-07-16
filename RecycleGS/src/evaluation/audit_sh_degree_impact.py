#!/usr/bin/env python3
"""Audit SH degree effect on rendering PSNR.
Renders with active_sh_degree=0 vs active_sh_degree=3 for:
- Baseline 15k PLY
- Recovery PLY (any method)

And computes the delta."""
import argparse, json, os, sys, torch, numpy as np, yaml
from pathlib import Path

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import PipelineParams, GroupParams

SCENE_CONFIG = '/data/wyh/RecycleGS/configs/stage1/reliability_scene01.yaml'
OUT_DIR = Path('/data/wyh/RecycleGS/outputs/debug/stage2bab')

def make_pipe():
    pipe = GroupParams()
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    pipe.debug = False
    return pipe

def make_dataset(scene_dir, model_dir):
    d = GroupParams()
    d.source_path = scene_dir
    d.model_path = model_dir
    d.images = "images"
    d.resolution = 2
    d.sh_degree = 3
    d.asg_degree = 24
    d.eval = True
    d.preload_img = True
    d.white_background = False
    d.data_device = "cuda"
    d.delight = False
    d.normal = False
    d.normal_folder = "normals"
    d.mask_background = False
    d.use_delighted_normal = False
    d.use_transparencies_map = False
    d.not_delight_only_transparent = False
    d.load2gpu_on_the_fly = False
    d.is_real = False
    d.is_indoor = False
    d.add_val = False
    d.multi_view_num = 8
    d.multi_view_max_angle = 30
    d.multi_view_min_dis = 0.01
    d.multi_view_max_dis = 1.5
    d.ncc_scale = 1.0
    return d

def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

@torch.no_grad()
def evaluate_ply_with_sh(ply_path, scene_dir, model_dir, active_sh_degree, device='cuda:0'):
    dataset = make_dataset(scene_dir, model_dir)
    pipe = make_pipe()
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    gaussians.load_ply(str(ply_path))
    gaussians.active_sh_degree = active_sh_degree
    print(f"  active_sh_degree={gaussians.active_sh_degree}, max_sh_degree={gaussians.max_sh_degree}")

    scene = Scene(dataset, gaussians, load_iteration=15000, shuffle=False)
    gaussians.load_ply(str(ply_path))
    gaussians.active_sh_degree = active_sh_degree

    test_cams = scene.getTestCameras()
    if len(test_cams) == 0:
        test_cams = scene.getTrainCameras()

    n_eval = min(64, len(test_cams))
    print(f"  Evaluating on {n_eval} views")

    psnrs = []
    for ci in range(n_eval):
        cam = test_cams[ci]
        gt = cam.original_image
        if gt is None:
            gt, _, _, _, _ = cam.get_image()
        gt = gt.to(device)

        rendered = render(cam, gaussians, pipe, bg, app_model=None,
                          return_plane=False, return_depth_normal=False)
        image = rendered['render'].clamp(0.0, 1.0)

        r = image[:3].detach().cpu().numpy().transpose(1, 2, 0)
        g = gt[:3].detach().cpu().numpy().transpose(1, 2, 0)
        psnr = compute_psnr(r, g)
        psnrs.append(psnr)

    mean_psnr = float(np.mean(psnrs))
    std_psnr = float(np.std(psnrs))
    print(f"  PSNR: {mean_psnr:.4f} +- {std_psnr:.4f} (active_sh_degree={active_sh_degree})")
    return mean_psnr, std_psnr, gaussians.get_xyz.shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ply', type=str,
                        default='/data/wyh/RecycleGS/baselines/tsgs_scene01_full/point_cloud/iteration_15000/point_cloud.ply')
    parser.add_argument('--recovery-ply', type=str,
                        default='/data/wyh/RecycleGS/outputs/prune_only/scene_01/ratio_005/schedule_control/recovery_500/point_cloud/iteration_15500/point_cloud.ply')
    parser.add_argument('--tag', type=str, default='scene_01')
    parser.add_argument('--output', type=str, default=str(OUT_DIR / 'sh_degree_audit.json'))
    args = parser.parse_args()

    with open(SCENE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    scene_dir = cfg['scene_dir']
    model_dir = cfg['model_dir']

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    results = {}

    # --- Baseline PLY ---
    print("\n" + "="*60)
    print("BASELINE 15k PLY (full SH)")
    print("="*60)
    baseline = {}
    for sh in [0, 3]:
        psnr, psnr_std, N = evaluate_ply_with_sh(
            args.baseline_ply, scene_dir, model_dir, sh, device)
        baseline[f'active_sh_degree_{sh}'] = {
            'psnr_mean': psnr,
            'psnr_std': psnr_std,
            'N': int(N),
            'ply_path': args.baseline_ply,
        }
    baseline['psnr_delta_sh0_vs_sh3'] = round(
        baseline['active_sh_degree_0']['psnr_mean'] - baseline['active_sh_degree_3']['psnr_mean'], 4)
    results['baseline'] = baseline

    # --- Recovery PLY ---
    print("\n" + "="*60)
    print("RECOVERY PLY (after 500 steps of recovery)")
    print("="*60)
    recovery = {}
    for sh in [0, 3]:
        psnr, psnr_std, N = evaluate_ply_with_sh(
            args.recovery_ply, scene_dir, model_dir, sh, device)
        recovery[f'active_sh_degree_{sh}'] = {
            'psnr_mean': psnr,
            'psnr_std': psnr_std,
            'N': int(N),
            'ply_path': args.recovery_ply,
        }
    recovery['psnr_delta_sh0_vs_sh3'] = round(
        recovery['active_sh_degree_0']['psnr_mean'] - recovery['active_sh_degree_3']['psnr_mean'], 4)
    results['recovery'] = recovery

    # --- Cross comparison ---
    results['analysis'] = {
        'baseline_psnr_at_sh0': baseline['active_sh_degree_0']['psnr_mean'],
        'baseline_psnr_at_sh3': baseline['active_sh_degree_3']['psnr_mean'],
        'recovery_psnr_at_sh0': recovery['active_sh_degree_0']['psnr_mean'],
        'recovery_psnr_at_sh3': recovery['active_sh_degree_3']['psnr_mean'],
        'baseline_delta_sh0vssh3': baseline['psnr_delta_sh0_vs_sh3'],
        'recovery_delta_sh0vssh3': recovery['psnr_delta_sh0_vs_sh3'],
    }

    # Determine root cause
    bl_delta = baseline['psnr_delta_sh0_vs_sh3']
    rec_delta = recovery['psnr_delta_sh0_vs_sh3']
    bl_sh0 = baseline['active_sh_degree_0']['psnr_mean']
    bl_sh3 = baseline['active_sh_degree_3']['psnr_mean']
    rec_sh0 = recovery['active_sh_degree_0']['psnr_mean']
    rec_sh3 = recovery['active_sh_degree_3']['psnr_mean']

    sh_causes_same_drop = abs(bl_delta - rec_delta) < 1.0
    if sh_causes_same_drop:
        sh_root_cause = "NO - SH degree=0 causes approximately EQUAL PSNR drop for both baseline and recovery. This is NOT the root cause of the recovery gap."
    else:
        sh_root_cause = "YES - SH degree=0 causes DIFFERENT PSNR drops for baseline vs recovery, indicating this IS the root cause."

    if abs(bl_delta) > 3:
        sh_severity = f"SH degree=0 causes significant PSNR loss ({abs(bl_delta):.2f} dB) even for the baseline model"
    else:
        sh_severity = f"SH degree=0 causes minimal PSNR loss ({abs(bl_delta):.2f} dB) for the baseline model"

    results['root_cause_analysis'] = {
        'sh_degree_causes_same_drop': sh_causes_same_drop,
        'sh_root_cause_verdict': sh_root_cause,
        'sh_severity_note': sh_severity,
        'baseline_psnr_sh0_vs_sh3_delta': bl_delta,
        'recovery_psnr_sh0_vs_sh3_delta': rec_delta,
        'note': (
            f"Baseline PSNR: sh=0 -> {bl_sh0:.4f}, sh=3 -> {bl_sh3:.4f}, delta={bl_delta:.4f}\n"
            f"Recovery PSNR: sh=0 -> {rec_sh0:.4f}, sh=3 -> {rec_sh3:.4f}, delta={rec_delta:.4f}\n"
            f"SH degree effect: baseline delta={bl_delta:.4f}, recovery delta={rec_delta:.4f}\n"
            f"These deltas are {'similar' if sh_causes_same_drop else 'different'} -> "
            f"SH degree is {'NOT' if sh_causes_same_drop else 'IS'} the root cause of the 17.54 vs 22.39 gap."
        )
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline PSNR (sh=0): {bl_sh0:.4f}")
    print(f"Baseline PSNR (sh=3): {bl_sh3:.4f}")
    print(f"Baseline delta (sh0-sh3): {bl_delta:.4f}")
    print(f"Recovery PSNR (sh=0): {rec_sh0:.4f}")
    print(f"Recovery PSNR (sh=3): {rec_sh3:.4f}")
    print(f"Recovery delta (sh0-sh3): {rec_delta:.4f}")
    print(f"\n{sh_root_cause}")
    print(f"\n{sh_severity}")

if __name__ == '__main__':
    main()
