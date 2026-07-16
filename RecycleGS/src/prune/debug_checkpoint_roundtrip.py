#!/usr/bin/env python3
"""Zero-step roundtrip test - uses eval=True to match evaluator."""
import json, os, sys, torch, numpy as np
from pathlib import Path

sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
sys.path.insert(0, '/data/wyh/RecycleGS/src')

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import OptimizationParams, GroupParams
from argparse import ArgumentParser as ArgParser
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

CKPT = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full/chkpnt15000.pth'
PLY = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full/point_cloud/iteration_15000/point_cloud.ply'
MDIR = '/data/wyh/RecycleGS/baselines/tsgs_scene01_full'
SDIR = '/data/wyh/RecycleGS/data/translab_full/scene_01'
OUT = Path('/data/wyh/RecycleGS/outputs/debug/stage2b_recovery_collapse/scene_01')

def mk_dataset(**kw):
    d = GroupParams()
    d.source_path=SDIR; d.model_path=MDIR; d.images='images'; d.resolution=2
    d.sh_degree=3; d.asg_degree=24; d.eval=True; d.preload_img=True; d.white_background=False
    d.data_device='cuda'; d.delight=False; d.normal=False; d.mask_background=False
    d.use_delighted_normal=False; d.use_transparencies_map=False; d.not_delight_only_transparent=False
    d.load2gpu_on_the_fly=False; d.is_real=False; d.is_indoor=False; d.add_val=False
    d.multi_view_num=8; d.multi_view_max_angle=30; d.multi_view_min_dis=0.01
    d.multi_view_max_dis=1.5; d.ncc_scale=1.0; d.normal_folder='normals'
    for k,v in kw.items(): setattr(d,k,v)
    return d

def make_opt():
    o = OptimizationParams(ArgParser())
    for k,v in {'iterations':30000,'position_lr_init':0.00016,'position_lr_final':0.0000016,
        'position_lr_delay_mult':0.01,'position_lr_max_steps':30000,'feature_lr':0.0025,
        'opacity_lr':0.05,'scaling_lr':0.005,'rotation_lr':0.001,'percent_dense':0.001,
        'lambda_dssim':0.2,'densification_interval':100,'opacity_reset_interval':3000,
        'densify_from_iter':500,'densify_until_iter':15000,'densify_grad_threshold':0.0002,
        'scale_loss_weight':100.0,'opacity_cull_threshold':0.005,'densify_abs_grad_threshold':0.0008,
        'abs_split_radii2D_threshold':20,'max_abs_split_points':50000,'max_all_points':6000000,
        'random_background':False,'exposure_compensation':False,'wo_depth_normal_detach':False,
        'use_2dgsnormal_loss':False,'use_asg':False,'delight_iterations':15000,
        'sd_normal_until_iter':30000,'lambda_sd_normal':0.05,'normal_cos_threshold_iter':3000,
        'ncc_loss_from_iter':7000,'single_view_weight':0.015,'single_view_weight_from_iter':7000,
        'multi_view_ncc_weight':0.15,'multi_view_geo_weight':0.03,'multi_view_weight_from_iter':7000,
        'multi_view_patch_size':3,'multi_view_sample_num':102400,'multi_view_pixel_noise_th':1.0,
        'wo_use_geo_occ_aware':False,'use_multi_view_trim':True,'T_threshold':0.0001,
        'observe_T_threshold':0.5,'bg_T_threshold':0.98,'trans_binary_threshold':0.5,
        'nofix_position':False,'nofix_opacity':False,'nofix_param':False,
        'nofix_scaling':False,'nofix_rotation':False}.items(): setattr(o,k,v)
    return o

def cat_params(g):
    return {k: g._xyz.detach() if k=='xyz' else getattr(g,f'_{k}').detach()
            for k in ['xyz','features_dc','features_rest','opacity','scaling','rotation']}

def compare(a,b):
    return {k:{'same':bool((a[k]==b[k]).all()),'max_abs_diff':round(float((a[k]-b[k]).abs().max().cpu()),8),
               'mean_abs_diff':round(float((a[k]-b[k]).abs().mean().cpu()),8)} for k in a if k in b}

@torch.no_grad()
def measure(g, cams, pipe, bg):
    ps, ss, ls = [], [], []
    for cam in cams[:20]:
        gt = cam.original_image.cuda()
        out = render(cam, g, pipe, bg, app_model=None, return_plane=False, return_depth_normal=False)
        img = out['render'].clamp(0.0, 1.0)
        ps.append(psnr(img, gt).mean().item()); ss.append(ssim(img, gt).mean().item())
        ls.append(lpips(img, gt, net_type='vgg').mean().item())
    return {'psnr':round(float(np.mean(ps)),4),'psnr_std':round(float(np.std(ps)),4),
            'ssim':round(float(np.mean(ss)),4),'num_views':len(ps)}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    pipe = GroupParams(); pipe.convert_SHs_python=False; pipe.compute_cov3D_python=False; pipe.debug=False
    bg = torch.tensor([0,0,0], dtype=torch.float32, device='cuda:0')
    ckpt = torch.load(CKPT, map_location='cuda:0', weights_only=False)
    model_params, iteration = ckpt; print(f"Iteration: {iteration}")

    # Load scene + cameras once (eval=True)
    ds = mk_dataset()
    g_tmp = GaussianModel(3, 24)
    scene = Scene(ds, g_tmp, load_iteration=15000, shuffle=False)
    cams = scene.getTestCameras()
    print(f"Test cameras: {len(cams)}, using first 20")
    del g_tmp; torch.cuda.empty_cache()

    report = {'iteration': iteration, 'states': {}}

    # State A: Direct PLY (SH=0, eval config)
    print("\n=== A: Direct PLY (SH=0) ===")
    ga = GaussianModel(3, 24); ga.load_ply(PLY); ga.active_sh_degree = 0
    ma = measure(ga, cams, pipe, bg); print(f"  PSNR: {ma['psnr']}")
    report['states']['A_ply_direct'] = {**ma, 'count': int(ga.get_xyz.shape[0]), 'sh': ga.active_sh_degree}

    # State B: Checkpoint restore (SH=3, then set to 0 for eval)
    print("=== B: Checkpoint restore -> SH=0 ===")
    gb = GaussianModel(3, None); gb.restore(model_params, make_opt()); gb.active_sh_degree = 0
    mb = measure(gb, cams, pipe, bg); print(f"  PSNR: {mb['psnr']}")
    report['states']['B_ckpt_restore'] = {**mb, 'count': int(gb.get_xyz.shape[0]), 'sh': gb.active_sh_degree}
    report['states']['B_vs_A_diff'] = compare(cat_params(ga), cat_params(gb))

    # State C: save_ply + load_ply from B (SH=0)
    print("=== C: save+load PLY (SH=0) ===")
    tmp = OUT / 'roundtrip_tmp.ply'
    gb.save_ply(str(tmp))
    gc = GaussianModel(3, 24); gc.load_ply(str(tmp)); gc.active_sh_degree = 0
    mc = measure(gc, cams, pipe, bg); print(f"  PSNR: {mc['psnr']}")
    report['states']['C_ply_roundtrip'] = {**mc, 'count': int(gc.get_xyz.shape[0]), 'sh': gc.active_sh_degree}
    report['states']['C_vs_B_diff'] = compare(cat_params(gb), cat_params(gc))

    # State D: same as C but SH=3 (training config)
    print("=== D: Same PLY, SH=3 ===")
    gc.active_sh_degree = 3
    md = measure(gc, cams, pipe, bg); print(f"  PSNR: {md['psnr']}")
    report['states']['D_sh3_training'] = {**md, 'count': int(gc.get_xyz.shape[0]), 'sh': 3}

    if tmp.exists(): os.remove(str(tmp))

    json_path = OUT / 'roundtrip_report.json'
    with open(json_path, 'w') as f: json.dump(report, f, indent=2, default=str)

    a,b,c,d = [report['states'][s]['psnr'] for s in ['A_ply_direct','B_ckpt_restore','C_ply_roundtrip','D_sh3_training']]
    md_lines = ["# Checkpoint Roundtrip Report", "",
        f"- State A (PLY direct, SH=0): PSNR={a}",
        f"- State B (checkpoint restore, SH=0): PSNR={b}",
        f"- State C (save_ply+load_ply, SH=0): PSNR={c}",
        f"- State D (same PLY, SH=3): PSNR={d}", ""]
    for label, x, y in [("Checkpoint restore (A vs B)", a, b),
                         ("PLY roundtrip (B vs C)", b, c),
                         ("SH effect (C vs D, 0→3)", c, d)]:
        diff = round(abs(x - y), 2)
        md_lines.append(f"- {label}: delta={diff:.2f}dB [{'OK' if diff<1.0 else 'ISSUE'}]")
    md_lines.append("")
    if abs(c - d) > 1.0:
        md_lines.append("**SH degree mismatch**: PSNR differs significantly between SH=0 and SH=3 render modes.")
    if abs(a - b) > 1.0:
        md_lines.append("**Checkpoint restore issue**: PSNR differs from PLY direct load.")
    if abs(b - c) > 1.0:
        md_lines.append("**PLY roundtrip issue**: save_ply/load_ply corrupts parameters.")

    md_path = OUT / 'parameter_roundtrip_report.md'
    with open(md_path, 'w') as f: f.write('\n'.join(md_lines) + '\n')
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    main()
