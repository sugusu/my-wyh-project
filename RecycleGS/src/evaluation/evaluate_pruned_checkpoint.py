#!/usr/bin/env python3
"""Evaluate pruned checkpoint: rendering and metrics.
Fixed to ensure correct PLY is loaded (TSGS Scene reloads from model_path, so we re-load after)."""
import argparse, json, os, sys, numpy as np, yaml, torch
from pathlib import Path

sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.tsgs_loader import load_scene, get_train_cameras

def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

def tsgs_render(gaussians, cam, pipe, bg_color, device='cuda:0'):
    from gaussian_renderer import render
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    return render(cam, gaussians, pipe, bg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--locked-config', type=str, default='configs/stage2/type_a_prune_locked.yaml')
    parser.add_argument('--all-methods', action='store_true')
    parser.add_argument('--skip-rendering', action='store_true', help='Skip rendering, only compute removed set metrics')
    parser.add_argument('--force-render', action='store_true', help='Clear render cache and re-render')
    parser.add_argument('--precision', type=int, default=8, help='Decimal places for metrics')
    args = parser.parse_args()
    prec = args.precision

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.locked_config) as f:
        locked_cfg = yaml.safe_load(f)

    scene_name = cfg.get('scene_name', 'scene_01')
    rel_dir = Path(cfg['reliability_output_dir'])
    iter_dir = rel_dir / 'iter_15000'
    ratio = locked_cfg.get('prune_ratio', 0.005)
    ratio_str = f"ratio_{int(ratio*1000):03d}"
    device = locked_cfg.get('device', 'cuda:0')

    out_base = Path(locked_cfg['project_root']) / 'outputs' / 'prune_only' / scene_name / ratio_str

    with open(out_base / 'prune_metadata.json') as f:
        metadata = json.load(f)
    methods = ['baseline'] + list(metadata['methods'].keys()) if args.all_methods else ['baseline']

    # Load original checkpoint for removed set analysis
    ckpt_path = cfg['checkpoint_path']
    from plyfile import PlyData
    ply_orig = PlyData.read(ckpt_path)
    orig_xyz = np.stack([np.asarray(ply_orig['vertex'][p]) for p in ['x', 'y', 'z']], axis=1)
    N_before = len(orig_xyz)

    base_path = rel_dir / 'gaussian_base_features.npz'
    base = np.load(base_path)
    opacity_sigmoid = base['opacity_sigmoid']

    err = np.load(iter_dir / 'geometry_errors.npz')
    d_center_norm_full = np.full(N_before, np.nan, dtype=np.float32)
    cgi = np.load(iter_dir / 'candidate_global_indices.npy')
    if len(cgi) == len(err['d_center_norm']):
        d_center_norm_full[cgi] = err['d_center_norm']

    mask_risk_mean_full = np.full(N_before, np.nan, dtype=np.float32)
    mr = np.load(iter_dir / 'mask_risk_mean.npy')
    if len(cgi) == len(mr):
        mask_risk_mean_full[cgi] = mr

    contribution = None
    contrib_path = rel_dir / 'contribution.npy'
    if contrib_path.exists():
        contribution = np.load(contrib_path)

    results = {}

    for method in methods:
        if method == 'baseline':
            ply_path = ckpt_path
            out_dir = out_base / method
        else:
            out_dir = out_base / method
            ply_path = out_dir / 'retained.ply'
            if not ply_path.exists():
                print(f"  SKIP {method}: retained.ply not found")
                continue

        os.makedirs(out_dir, exist_ok=True)

        # Re-check existing metrics unless force-render
        metrics_path = out_dir / 'render_metrics.json'
        if not args.force_render and metrics_path.exists() and not args.skip_rendering:
            print(f"  {method}: render cache exists, loading...")
            with open(metrics_path) as f:
                render_metrics = json.load(f)
        else:
            render_metrics = {'error': 'skipped'}

        print(f"\n=== Evaluating {method} ===")

        # Load pruned PLY
        ply = PlyData.read(ply_path)
        N_after = ply['vertex'].count

        if method != 'baseline':
            prune_idx = np.load(out_dir / 'prune_indices.npy')
        else:
            prune_idx = np.array([], dtype=np.int64)

        K = len(prune_idx)

        # --- Removed set statistics ---
        removed_metrics = {}
        if method != 'baseline' and K > 0:
            rem_d_center = d_center_norm_full[prune_idx]
            rem_mask_risk = mask_risk_mean_full[prune_idx]
            rem_opacity = opacity_sigmoid[prune_idx]
            valid_dc = np.isfinite(rem_d_center)
            valid_mr = np.isfinite(rem_mask_risk)

            removed_metrics = {
                'K': int(K),
                'd_center_norm_mean': float(np.nanmean(rem_d_center)) if valid_dc.any() else None,
                'd_center_norm_median': float(np.nanmedian(rem_d_center)) if valid_dc.any() else None,
                'd_center_norm_p90': float(np.nanpercentile(rem_d_center, 90)) if valid_dc.any() else None,
                'mask_risk_mean': float(np.nanmean(rem_mask_risk)) if valid_mr.any() else None,
                'opacity_mean': float(rem_opacity.mean()),
                'opacity_median': float(np.median(rem_opacity)),
            }

        if method != 'baseline':
            remaining_indices = np.setdiff1d(np.arange(N_before), prune_idx, assume_unique=True)
            rem_risk = mask_risk_mean_full[remaining_indices]
            valid_rr = np.isfinite(rem_risk)
            if valid_rr.any():
                surface_prox_ratio = float(np.mean(rem_risk[valid_rr] < 0.5))
                removed_metrics['surface_proximity_ratio'] = surface_prox_ratio
                removed_metrics['remaining_mask_risk_mean'] = float(np.nanmean(rem_risk))

        # --- Rendering (skip if cached) ---
        if not args.skip_rendering and (args.force_render or not metrics_path.exists()):
            try:
                with torch.no_grad():
                    scene, gaussians, pipe = load_scene(cfg, device)
                    # TSGS Scene reloads from model_path, re-load our PLY
                    gaussians.load_ply(str(ply_path))
                    gaussians.active_sh_degree = 0

                    cameras = get_train_cameras(scene)
                    total_cams = len(cameras)
                    n_eval = min(64, total_cams)

                    psnrs = []
                    ssims = []
                    lpipss = []
                    from skimage.metrics import structural_similarity
                    import lpips
                    lpips_fn = lpips.LPIPS(net='alex').to(device)

                    baseline_render = None
                    first_diff = None

                    for ci in range(n_eval):
                        cam = cameras[ci]
                        rendered = tsgs_render(gaussians, cam, pipe, [0, 0, 0], device)
                        if 'render' in rendered:
                            render_img = rendered['render']
                        elif 'rgb' in rendered:
                            render_img = rendered['rgb']
                        else:
                            render_img = list(rendered.values())[0]

                        gt_img = getattr(cam, 'original_image', None)
                        if gt_img is None:
                            gt_img = getattr(cam, 'image', None)
                        if gt_img is not None and not torch.is_tensor(gt_img):
                            gt_img = torch.tensor(gt_img, device=device)
                        if gt_img is not None:
                            gt_img = gt_img.to(device)

                        if gt_img is not None:
                            r = render_img[:3].detach().cpu().numpy().transpose(1, 2, 0)
                            g = gt_img[:3].detach().cpu().numpy().transpose(1, 2, 0)
                            r = r.clip(0, 1)
                            g = g.clip(0, 1)
                            psnr = compute_psnr(r, g)
                            ssim = structural_similarity(r, g, channel_axis=2, data_range=1.0)
                            psnrs.append(psnr)
                            ssims.append(ssim)

                            r_t = torch.tensor(r).permute(2, 0, 1).unsqueeze(0).to(device)
                            g_t = torch.tensor(g).permute(2, 0, 1).unsqueeze(0).to(device)
                            with torch.no_grad():
                                lpips_val = lpips_fn(r_t, g_t).item()
                            lpipss.append(lpips_val)

                        if method != 'baseline' and baseline_render is not None and ci == 0:
                            first_diff = np.abs(r - g).mean()

                    if psnrs:
                        render_metrics = {
                            'n_views': int(n_eval),
                            'psnr_mean': round(float(np.mean(psnrs)), prec),
                            'psnr_std': round(float(np.std(psnrs)), prec),
                            'ssim_mean': round(float(np.mean(ssims)), prec),
                            'ssim_std': round(float(np.std(ssims)), prec),
                            'lpips_mean': round(float(np.mean(lpipss)), prec),
                            'lpips_std': round(float(np.std(lpipss)), prec),
                        }
                        print(f"    PSNR: {render_metrics['psnr_mean']:.{prec}f} +- {render_metrics['psnr_std']:.{prec}f}")
                        print(f"    SSIM: {render_metrics['ssim_mean']:.{prec}f} +- {render_metrics['ssim_std']:.{prec}f}")
                        print(f"    LPIPS: {render_metrics['lpips_mean']:.{prec}f} +- {render_metrics['lpips_std']:.{prec}f}")
                    else:
                        render_metrics = {'error': 'no valid renderings', 'n_views': int(n_eval)}

                    with open(out_dir / 'render_metrics.json', 'w') as f:
                        json.dump(render_metrics, f, indent=2)

            except Exception as e:
                render_metrics = {'error': str(e)}
                print(f"    Rendering failed: {e}")

        method_result = {
            'method': method,
            'N_before': int(N_before),
            'N_after': int(N_after),
            'K': int(K),
            'render_metrics': render_metrics,
            'removed_set_metrics': removed_metrics,
        }

        # Verify render difference is NOT exactly 0 for pruned methods
        if method != 'baseline' and render_metrics.get('error', '') == 'skipped' and not args.skip_rendering:
            baseline_metrics_path = out_base / 'baseline' / 'render_metrics.json'
            if baseline_metrics_path.exists():
                with open(baseline_metrics_path) as f:
                    bm = json.load(f)
                if bm.get('psnr_mean', 0) != render_metrics.get('psnr_mean', -1):
                    method_result['render_diff_from_baseline'] = round(
                        float(render_metrics.get('psnr_mean', 0)) - float(bm.get('psnr_mean', 0)), prec
                    )
                    method_result['render_diff_nonzero'] = True
                else:
                    method_result['render_diff_nonzero'] = False

        results[method] = method_result

        with open(out_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(method_result, f, indent=2)

    # Save cross-method summary
    summary_path = out_base / 'cross_method_evaluation.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved cross-method evaluation: {summary_path}")

if __name__ == '__main__':
    main()
