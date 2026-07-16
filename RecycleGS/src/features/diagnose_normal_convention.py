import argparse, sys, os, json, torch, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config
from recyclegs.tsgs_loader import load_scene, get_train_cameras, render_view
from arguments import PipelineParams

# Read TSGS source to understand normal PNG decode convention
def read_tsgs_normal_decode_convention():
    """Check TSGS normal PNG decode convention from source."""
    conventions = {
        'rgb_to_normal': 'read as RGB uint8, decode = (val / 255.0) * 2 - 1, range [-1,1]',
        'channel_order': 'RGB order (no BGR swap)',
        'world_camera': 'normals stored in camera space',
    }
    return conventions

CANDIDATE_FLIPS = {
    'none': np.array([[1,0,0],[0,1,0],[0,0,1]]),
    '-x': np.array([[-1,0,0],[0,1,0],[0,0,1]]),
    '-y': np.array([[1,0,0],[0,-1,0],[0,0,1]]),
    '-z': np.array([[1,0,0],[0,1,0],[0,0,-1]]),
    'xy_flip': np.array([[0,1,0],[1,0,0],[0,0,1]]),
    'xz_flip': np.array([[0,0,1],[0,1,0],[1,0,0]]),
    'yz_flip': np.array([[1,0,0],[0,0,1],[0,1,0]]),
    'full_flip': np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
}
RGB_BGR_SWAP = np.array([[0,0,1],[0,1,0],[1,0,0]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    debug_dir = Path(cfg['debug_output_dir'])
    out_dir = debug_dir / 'normal_convention'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get('device', 'cuda:0')

    decode_convention = read_tsgs_normal_decode_convention()

    print("[1/5] Loading model...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)

    n_views = 8
    rng = np.random.RandomState(42)
    selected_idx = rng.choice(total, min(n_views, total), replace=False)
    print(f"  Selected {len(selected_idx)} views: {selected_idx.tolist()}")
    print(f"  Normal decode convention: {decode_convention}")

    print("[2/5] Rendering Gaussian normals and loading normal priors...")
    pipe = PipelineParams(argparse.ArgumentParser())
    bg_color = [1, 1, 1]
    scene_dir = cfg['scene_dir']

    per_view = {}
    for vi, cam_idx in enumerate(selected_idx):
        cam = cameras[cam_idx]

        rendered = render_view(gaussians, cam, pipe, bg_color, device)
        if 'rendered_normal' in rendered:
            normal_render = rendered['rendered_normal'].detach().cpu().numpy()  # (3, H, W) in camera space
        else:
            print(f"  Warning: no rendered_normal for view {cam_idx}")
            continue
        normal_render = normal_render.transpose(1, 2, 0)  # (H, W, 3)
        normal_render_norm = np.linalg.norm(normal_render, axis=2, keepdims=True)
        normal_render = normal_render / np.clip(normal_render_norm, 1e-8, None)

        normal_prior_path = os.path.join(scene_dir, 'normals', f'frame_{cam_idx+1:04d}_normal.png')
        if not os.path.exists(normal_prior_path):
            print(f"  Warning: no normal prior for view {cam_idx} at {normal_prior_path}")
            continue
        nimg = Image.open(normal_prior_path).convert('RGB')
        n_arr = np.array(nimg).astype(float) / 255.0 * 2 - 1
        n_arr = n_arr / np.clip(np.linalg.norm(n_arr, axis=2, keepdims=True), 1e-8, None)

        if n_arr.shape[:2] != normal_render.shape[:2]:
            from PIL import Image as PILImage
            n_uint8 = ((n_arr * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
            n_pil = PILImage.fromarray(n_uint8).resize(
                (normal_render.shape[1], normal_render.shape[0]), PILImage.BILINEAR)
            n_arr = np.array(n_pil).astype(float) / 255.0 * 2 - 1
            n_arr = n_arr / np.clip(np.linalg.norm(n_arr, axis=2, keepdims=True), 1e-8, None)

        per_view[cam_idx] = {
            'normal_render': normal_render,
            'normal_prior': n_arr,
        }
        print(f"  [{vi+1}/{len(selected_idx)}] View {cam_idx}: normal_render.shape={normal_render.shape}, prior.shape={n_arr.shape}")

    if not per_view:
        print("No views with normals available. Exiting.")
        return

    print("[3/5] Trying candidate axis flips...")
    H, W, _ = next(iter(per_view.values()))['normal_render'].shape
    results = []
    for flip_name, flip_mat in CANDIDATE_FLIPS.items():
        for use_bgr in [False, True]:
            label = f"{flip_name}{'_bgr' if use_bgr else ''}"
            abs_cosine_errors = []
            for cam_idx, pv in per_view.items():
                render_n = pv['normal_render']
                prior_n = pv['normal_prior']
                if use_bgr:
                    prior_n = prior_n @ RGB_BGR_SWAP.T
                transformed = prior_n @ flip_mat.T
                transformed = transformed / np.clip(np.linalg.norm(transformed, axis=2, keepdims=True), 1e-8, None)
                dot = (render_n * transformed).sum(axis=2)
                abs_cos = np.abs(dot.clip(-1, 1))
                err = 1.0 - abs_cos
                abs_cosine_errors.append(err)
            mean_err = float(np.mean(abs_cosine_errors))
            results.append({'convention': label, 'mean_abs_cosine_error': mean_err})

    results.sort(key=lambda x: x['mean_abs_cosine_error'])
    print("  Ranked conventions:")
    for r in results[:5]:
        print(f"    {r['convention']}: {r['mean_abs_cosine_error']:.6f}")

    print("[4/5] Saving visualizations for best and default convention...")
    best_conv = results[0]['convention']
    parts = best_conv.split('_')
    best_flip_name = parts[0]
    best_use_bgr = 'bgr' in parts
    best_flip_mat = CANDIDATE_FLIPS.get(best_flip_name, np.eye(3))

    for cam_idx, pv in per_view.items():
        render_n = pv['normal_render']
        prior_n = pv['normal_prior']
        prior_best = prior_n @ best_flip_mat.T
        if best_use_bgr:
            prior_best = prior_best @ RGB_BGR_SWAP.T
        prior_best = prior_best / np.clip(np.linalg.norm(prior_best, axis=2, keepdims=True), 1e-8, None)

        dot_best = np.abs((render_n * prior_best).sum(axis=2).clip(-1, 1))
        err_best = 1.0 - dot_best

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        vis_render = ((render_n * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        vis_prior = ((prior_n * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        vis_prior_best = ((prior_best * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        axes[0].imshow(vis_render)
        axes[0].set_title('Gaussian Normal')
        axes[1].imshow(vis_prior)
        axes[1].set_title(f'Prior (orig)')
        axes[2].imshow(vis_prior_best)
        axes[2].set_title(f'Prior ({best_conv})')
        axes[3].imshow(err_best, cmap='hot', vmin=0, vmax=1)
        axes[3].set_title(f'Abs Cos Error ({best_conv})')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        fig.savefig(str(out_dir / f'view_{cam_idx:04d}_normal_convention.png'), dpi=150)
        plt.close(fig)
        print(f"    Saved view {cam_idx}")

    print("[5/5] Saving diagnosis report...")
    diagnosis = {
        'scene_name': cfg['scene_name'],
        'normal_decode_convention': decode_convention,
        'num_views_analyzed': len(per_view),
        'candidate_conventions_tried': len(results),
        'ranked_conventions': results,
        'best_convention': results[0]['convention'],
        'best_mean_abs_cosine_error': results[0]['mean_abs_cosine_error'],
        'default_convention': 'none',
        'default_mean_abs_cosine_error': [r['mean_abs_cosine_error'] for r in results if r['convention'] == 'none'],
        'note': 'Diagnosis only. Do NOT use results to select final normal convention.',
    }
    if diagnosis['default_mean_abs_cosine_error']:
        diagnosis['default_mean_abs_cosine_error'] = diagnosis['default_mean_abs_cosine_error'][0]
    else:
        diagnosis['default_mean_abs_cosine_error'] = None

    with open(debug_dir / 'normal_convention_diagnosis.json', 'w') as f:
        json.dump(diagnosis, f, indent=2)

    md = [
        f"# Normal Convention Diagnosis - {cfg['scene_name']}",
        f"",
        f"## TSGS Normal Decode Convention",
        f"- {decode_convention}",
        f"",
        f"## Views Analyzed",
        f"- Num views: {len(per_view)}",
        f"- Selected: {list(per_view.keys())}",
        f"",
        f"## Candidate Conventions (ranked by mean abs cosine error)",
        f"| Rank | Convention | Mean Abs Cosine Error |",
        f"|------|-----------|----------------------|",
    ]
    for i, r in enumerate(results):
        md.append(f"| {i+1} | {r['convention']} | {r['mean_abs_cosine_error']:.6f} |")
    md.append(f"")
    md.append(f"## Best Convention")
    md.append(f"- {results[0]['convention']} (error={results[0]['mean_abs_cosine_error']:.6f})")
    md.append(f"")
    md.append(f"## Default (no flip) Convention")
    if diagnosis['default_mean_abs_cosine_error'] is not None:
        md.append(f"- error={diagnosis['default_mean_abs_cosine_error']:.6f}")
    md.append(f"")
    md.append(f"**IMPORTANT**: This is diagnosis only. Do NOT use results to select final normal convention.")
    md.append(f"")
    md.append(f"## Files Saved")
    md.append(f"- normal_convention_diagnosis.json")
    md.append(f"- normal_convention_report.md")
    md.append(f"- Visualizations in normal_convention/")

    with open(debug_dir / 'normal_convention_report.md', 'w') as f:
        f.write('\n'.join(md))

    print("Done.")

if __name__ == '__main__':
    main()
