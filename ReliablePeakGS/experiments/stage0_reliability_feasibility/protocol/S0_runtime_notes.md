# S0 Runtime Notes

## Network

The inherited shell has:

- `HTTP_PROXY=http://127.0.0.1:9999`
- `HTTPS_PROXY=http://127.0.0.1:9999`

This proxy breaks GitHub TLS clone/fetch operations. GitHub, ArXiv, PyPI, and Ubuntu package download commands are run with proxy variables unset.

Google Drive direct access fails from this server, while the local proxy can reach the Drive confirmation endpoint. The AlphaSurf data download therefore uses the ambient proxy and the explicit Google Drive confirmation URL recorded in `scripts/download_alphasurf_google_drive.sh`.

## CUDA

`nvcc` exists at `/usr/local/cuda-12.4/bin/nvcc`, but that directory is not in the default PATH. Stage 0 commands source `scripts/stage0_env.sh`, which sets:

- `CUDA_VISIBLE_DEVICES=2,3`
- `CUDA_HOME=/usr/local/cuda-12.4`
- `TORCH_CUDA_ARCH_LIST=7.0`

## Python Development Headers

The system lacks `/usr/include/python3.10/Python.h` and sudo is unavailable. The Python 3.10 development packages were downloaded with `apt-get download` and unpacked under:

`/data/wyh/ReliablePeakGS/environment/deb_headers/extracted`

This include path is exposed through `CPATH` in `scripts/stage0_env.sh` for CUDA extension builds.
