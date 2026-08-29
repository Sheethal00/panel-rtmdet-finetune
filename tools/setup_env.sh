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

echo "== Installing MMEngine / MMCV / MMDetection =="
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

if [ ! -d "mmdetection" ]; then
    git clone https://github.com/open-mmlab/mmdetection.git
fi
cd mmdetection
pip install -v -e .
cd ..

echo "== Installing ONNX export + verification deps =="
pip install onnx onnxruntime onnxsim pyyaml

echo "== Downloading COCO-pretrained RTMDet-tiny checkpoint =="
mkdir -p checkpoints
CKPT_URL="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
CKPT_PATH="checkpoints/rtmdet_tiny_8xb32-300e_coco_pretrained.pth"
if [ ! -f "${CKPT_PATH}" ]; then
    curl -L "${CKPT_URL}" -o "${CKPT_PATH}"
else
    echo "Checkpoint already present, skipping download."
fi

echo ""
echo "== Setup complete =="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Next: edit configs/classes.yaml, place your COCO data in data/raw_coco/,"
echo "then run tools/split_coco.py"
