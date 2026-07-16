#!/bin/bash
# Debug: Resume TSGS native training from 15k to 15020 with densification disabled
# Uses TSGS official train.py with checkpoint resume

set -e
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

MODEL_DIR=/data/wyh/RecycleGS/baselines/tsgs_scene01_full
OUTPUT_DIR=/data/wyh/RecycleGS/outputs/debug/stage2b_recovery_collapse/native_resume_test
SCENE_DIR=/data/wyh/RecycleGS/data/translab_full/scene_01
CHECKPOINT_PATH=${MODEL_DIR}/chkpnt15000.pth

mkdir -p ${OUTPUT_DIR}

# Copy TSGS train.py and modify it for our test
# We'll create a modified version that:
# 1. Disables densification (densify_from_iter=999999, densify_until_iter=0)
# 2. Resumes from checkpoint at 15000
# 3. Runs only 20 steps
# 4. Logs PSNR at each step

cat > ${OUTPUT_DIR}/native_resume_train.py << 'PYEOF'
import os, sys, torch, numpy as np, random, json
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from argparse import ArgumentParser

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def training(dataset, opt, pipe, checkpoint_path, output_dir, max_steps=20):
    device = 'cuda:0'
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    (model_params, first_iter) = torch.load(checkpoint_path, weights_only=False)
    print(f"Resuming from iteration {first_iter}")

    # Create model and scene (like TSGS train.py flow)
    asg_degree = None
    gaussians = GaussianModel(dataset.sh_degree, asg_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Restore from checkpoint
    gaussians.restore(model_params, opt)

    # Densification disabled
    opt.densify_from_iter = 999999
    opt.densify_until_iter = 0
    opt.opacity_reset_interval = 999999

    # AppModel
    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    app_model.load_weights(dataset.model_path)

    # Training cameras
    train_cams = scene.getTrainCameras()
    cam_indices = list(range(len(train_cams)))
    fixed_rng = np.random.RandomState(0)
    fixed_rng.shuffle(cam_indices)
    cam_queue = [train_cams[i] for i in cam_indices]

    # One fixed test camera for monitoring
    test_cams = scene.getTestCameras()
    if len(test_cams) == 0:
        test_cams = scene.getTrainCameras()
    test_cam = test_cams[0]

    trace = []
    end_iter = first_iter + max_steps

    for iteration in range(first_iter + 1, end_iter + 1):
        # Update LR like normal TSGS training
        gaussians.update_learning_rate(iteration)

        cam_idx = (iteration - first_iter - 1) % len(cam_queue)
        viewpoint_cam = cam_queue[cam_idx]

        gt_image, gt_image_gray, gt_image_delight, gt_image_normal, _ = viewpoint_cam.get_image()

        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background, app_model=app_model,
            return_plane=True, return_depth_normal=True,
            wo_depth_normal_detach=opt.wo_depth_normal_detach,
            mlp_color=None,
        )
        image = render_pkg['render']

        Ll1 = l1_loss(image, gt_image)
        ssim_val = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        if viewpoint_cam.mask is not None:
            alpha_loss = 0.1 * torch.nn.functional.binary_cross_entropy(
                render_pkg['rendered_alpha'], viewpoint_cam.mask.cuda())
            loss += alpha_loss

        loss.backward()

        gaussians.optimizer.step()
        app_model.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        app_model.optimizer.zero_grad(set_to_none=True)

        # Evaluate on test view (using same render settings)
        with torch.no_grad():
            gt_test = test_cam.original_image
            if gt_test is None:
                gt_test, _, _, _, _ = test_cam.get_image()
            gt_test = gt_test.cuda()

            test_render = render(test_cam, gaussians, pipe, background, app_model=None,
                                 return_plane=False, return_depth_normal=False)
            test_image = test_render['render'].clamp(0.0, 1.0)
            test_psnr = float(psnr(test_image, gt_test).mean().item())

        entry = {
            'iteration': iteration,
            'loss': round(float(loss.item()), 8),
            'Ll1': round(float(Ll1.item()), 8),
            'test_psnr': round(test_psnr, 4),
        }
        trace.append(entry)
        print(f"  iter {iteration}: loss={entry['loss']:.6f}, test_psnr={entry['test_psnr']:.2f}, lr={gaussians.optimizer.param_groups[0]['lr']:.12e}")

    # Save trace
    with open(os.path.join(output_dir, 'native_resume_trace.json'), 'w') as f:
        json.dump(trace, f, indent=2)
    print(f"\nFinal test PSNR: {trace[-1]['test_psnr']:.2f} (from {trace[0]['test_psnr']:.2f})")
    print(f"Native resume test complete.")

if __name__ == '__main__':
    parser = ArgumentParser()
    dataset = ModelParams(parser)
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)

    args = parser.parse_args()
    dataset = dataset.extract(args)
    opt = opt.extract(args)
    pipe = pipe.extract(args)

    training(dataset, opt, pipe, args.checkpoint_path, args.output_dir, args.max_steps)
PYEOF

# Run native resume test
echo "=== Native TSGS Resume Test ==="
echo "Output: ${OUTPUT_DIR}"

python3 ${OUTPUT_DIR}/native_resume_train.py \
    --source_path ${SCENE_DIR} \
    --model_path ${MODEL_DIR} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --max_steps 20 \
    --sh_degree 3 \
    --asg_degree 24 \
    --resolution 2 \
    --preload_img \
    --data_device cuda \
    --delight \
    --normal \
    --mask_background \
    --use_transparencies_map

echo "=== Native TSGS Resume Test Complete ==="
echo "Results in: ${OUTPUT_DIR}/native_resume_trace.json"
