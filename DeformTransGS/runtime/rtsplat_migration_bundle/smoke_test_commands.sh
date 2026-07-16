#!/bin/bash
# RT-Splatting container smoke test commands
# Run inside CUDA 11.8 devel container

set -euo pipefail

echo "=== 1. Environment Check ==="
python3 --version
python3 -c "import sysconfig; print(f'Include: {sysconfig.get_paths()[\"include\"]}')"
python3 -c "import sysconfig; import os; print(f'Python.h: {os.path.exists(os.path.join(sysconfig.get_paths()[\"include\"], \"Python.h\"))}')"
nvcc --version

echo "=== 2. PyTorch Check ==="
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python3 -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, available: {torch.cuda.is_available()}'); t=torch.zeros(1,device='cuda'); print(f'Tensor: {t}')"

echo "=== 3. Install Dependencies ==="
pip install 'numpy<2' setuptools wheel ninja plyfile open3d scikit-image scikit-learn scipy matplotlib opencv-python imageio imageio-ffmpeg kornia torchmetrics mediapy tqdm

echo "=== 4. Build nvdiffrast ==="
cd /workspace/nvdiffrast
TORCH_CUDA_ARCH_LIST=7.0 pip install . --no-build-isolation -v
python3 -c "import nvdiffrast.torch as dr; print('nvdiffrast OK')"

echo "=== 5. nvdiffrast CUDA Smoke ==="
python3 -c "
import torch, nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()
v_pos = torch.tensor([[[0,0,0],[1,0,0],[0,1,0]]], dtype=torch.float32, device='cuda')
pos_idx = torch.tensor([[0,1,2]], dtype=torch.int32, device='cuda')
rast, _ = dr.rasterize(glctx, v_pos, pos_idx, (256,256))
print(f'Triangle raster OK, covered: {(rast[...,0]!=-1).sum().item()} pixels')
"

echo "=== 6. Build RT-Splatting Extensions ==="
cd /workspace/RT-Splatting
for ext in submodules/diff-surfel-anych submodules/simple-knn; do
  if [ -d "$ext" ]; then
    cd "$ext"
    pip install . --no-build-isolation -v
    cd /workspace/RT-Splatting
  fi
done

echo "=== 7. RT-Splatting Imports ==="
python3 -c "
import torch
import nvdiffrast.torch as dr
from scene import GaussianModel
from gaussian_renderer import render
print('RT-Splatting imports OK')
"

echo "=== 8. Training Smoke (10 iter) ==="
cd /workspace/RT-Splatting
python3 train.py \
  -s /data/smoke \
  -m /data/output/rtsplat_smoke_10iter \
  --iterations 10 \
  --save_iterations 10 \
  --test_iterations 10 \
  --checkpoint_iterations 10 \
  --data_device cuda

echo "=== Done ==="
