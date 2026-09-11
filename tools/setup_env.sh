#!/usr/bin/env bash
# One-time local environment setup for RTMDet-tiny fine-tuning.
# Run from the project root: bash tools/setup_env.sh
set -euo pipefail

ENV_NAME="panel-rtmdet"
PYTHON_VERSION="3.10"

echo "== Checking for conda =="
if ! command -v conda &> /dev/null; then
    echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "== Checking for local GPU =="
CPU_ONLY=false
if ! command -v nvidia-smi &> /dev/null; then
    echo "No NVIDIA GPU detected. Proceeding CPU-only."
    echo "Fine for a small dataset (a handful to a few dozen images) -- expect"
    echo "minutes rather than hours for a short fine-tune. This will become"
    echo "the bottleneck if you later scale up to a large annotated set;"
    echo "revisit GPU access before a production-scale fine-tune."
    CPU_ONLY=true
else
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

echo "== Creating conda environment: ${ENV_NAME} =="
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "== Installing PyTorch =="
if [ "${CPU_ONLY}" = true ]; then
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
else
    # Edit the cuda version below to match your local driver if needed.
    # Check with: nvidia-smi (top right corner shows max supported CUDA version)
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
fi

echo "== Pinning numpy < 2 =="
# torch==2.1.0 predates numpy 2.0's release and isn't fully ABI-compatible
# with it -- left unpinned, some environments end up with numpy 2.x pulled
# in as a dependency of something else, producing "Failed to initialize
# NumPy: _ARRAY_API not found" warnings from torch, and in some code paths
# a harder failure than a warning. Pin explicitly rather than rely on
# whatever numpy version happens to land last.
pip install "numpy<2"

echo "== Installing MMEngine / MMCV / MMDetection =="
pip install -U openmim
mim install mmengine
# Upper-bounded at <2.2.0: mmdet 3.3.0 has a hard compatibility assertion
# against mmcv>=2.2.0 and raises at import time, not install time, so an
# unbounded install here looks fine until the first `import mmdet` fails.
mim install "mmcv>=2.0.0,<2.2.0"

if [ ! -d "mmdetection" ]; then
    git clone https://github.com/open-mmlab/mmdetection.git
fi
cd mmdetection
# NOT editable (-e). Modern pip requires the build_editable PEP 660 hook for
# editable installs, which mmdetection's old-style setup.py packaging does
# not implement -- `pip install -e .` fails with "build backend ... missing
# the 'build_editable' hook". A regular (non-editable) install works fine
# here: this project only needs mmdetection's config files and tools/
# scripts on disk (referenced by path), not live-editable source.
#
# --no-build-isolation is still required: pip's default isolated build
# environment doesn't see the torch already installed in this conda env,
# and mmdetection's setup.py needs torch importable at build time to check
# version compatibility.
pip install -v . --no-build-isolation
cd ..

echo "== Installing ONNX export + verification deps =="
pip install onnx onnxruntime onnxsim pyyaml

echo "== Re-pinning numpy < 2 =="
# CONFIRMED during real setup: mmcv, mmdetection's own dependency
# resolution, and/or onnx can each independently pull numpy back up to 2.x
# as a transitive dependency, even after the earlier pin. This has happened
# more than once in practice. Re-pin as the LAST install step so nothing
# after this point can silently undo it. If you add any further pip
# installs to this script, add them BEFORE this line, not after.
pip install "numpy<2" --force-reinstall

echo "== Verifying no numpy/torch ABI warning =="
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import mmdet, mmcv, mmengine; print('mmdet', mmdet.__version__)"

echo "== Downloading COCO-pretrained RTMDet-tiny checkpoint =="
mkdir -p checkpoints
CKPT_URL="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
CKPT_PATH="checkpoints/rtmdet_tiny_8xb32-300e_coco_pretrained.pth"
if [ ! -f "${CKPT_PATH}" ]; then
    curl -fL "${CKPT_URL}" -o "${CKPT_PATH}"
else
    echo "Checkpoint already present, skipping download."
fi

echo ""
echo "== Setup complete =="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Next: edit configs/classes.yaml, place your COCO data in data/raw_coco/,"
echo "then run tools/split_coco.py"