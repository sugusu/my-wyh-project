# RT-Splatting Environment Specification

## Required Software
- Docker with NVIDIA Container Toolkit (nvidia-docker2) or Apptainer/Singularity with --nv support
- NVIDIA GPU driver >= 525 (compatible with CUDA 11.8)

## CUDA Toolkit
- **Version**: 11.8
- **Type**: devel (must include nvcc)
- Recommended NVIDIA base image: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`

## Python
- Version: 3.10
- Python.h must exist at `<prefix>/include/python3.10/Python.h`

## PyTorch
- torch==2.0.1
- torchvision==0.15.2
- torchaudio==2.0.2
- CUDA 11.8 wheel index: https://download.pytorch.org/whl/cu118
- numpy < 2

## Required Repositories (read-only)
- /data/wyh/repos/RT-Splatting (commit 3f45b3c)
- /data/wyh/repos/nvdiffrast (NVlabs/nvdiffrast, latest main)
- /data/wyh/DeformTransGS (project root)
