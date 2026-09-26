#!/bin/bash
# Install microsoft/TRELLIS (the original) and the extras this arm uses, with pip, no conda.
# Known-good: torch 2.4.1+cu124, python 3.11, nvcc 12.4, one 48 GB A40 (24 GB is enough).
#
#   git clone --recurse-submodules https://github.com/microsoft/TRELLIS
#   TORCH_CUDA_ARCH_LIST=8.6 bash setup_trellis.sh /path/to/TRELLIS
#
# TORCH_CUDA_ARCH_LIST is your GPU's compute capability (8.6 = A40 / RTX 30xx, 8.9 = L40S / RTX 40xx).
# CUDA_HOME defaults to /usr/local/cuda.
set -x
TRELLIS=${1:?usage: setup_trellis.sh /path/to/TRELLIS}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export MAX_JOBS=${MAX_JOBS:-16}
cd "$TRELLIS" || exit 1
pip install -q --upgrade pip
# a distutils-owned blinker makes every later install abort
pip install --ignore-installed blinker
# basic (setup.sh --basic) + extras we need downstream
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime-gpu trimesh open3d xatlas pyvista pymeshfix igraph transformers pillow-heif usd-core
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
# attention backend: xformers built for torch 2.4.1 / cu124 (setup.sh has no 2.4.1+cu124 row)
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
# kaolin for flexicubes, wheel index for torch-2.4.1_cu124 exists
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu124.html
# sparse conv
pip install spconv-cu120
# nvdiffrast + mip-splatting rasterizer (compiled)
bash setup.sh --nvdiffrast --mipgaussian
python - <<"PY"
import torch, importlib
for m in ["xformers","kaolin","spconv","nvdiffrast","diff_gaussian_rasterization","rembg","trimesh","xatlas","pymeshfix","igraph","utils3d","pillow_heif","pxr"]:
    try:
        importlib.import_module(m); print("OK  ", m)
    except Exception as e:
        print("FAIL", m, repr(e)[:120])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
echo SETUP_DONE
