#!/usr/bin/env bash
# Fine-tuning entry point.
# Usage: bash tools/train.sh --config configs/rtmdet_tiny_panel.py
set -euo pipefail

CONFIG="configs/rtmdet_tiny_panel.py"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "data/raw_coco/splits/train.json" ]; then
    echo "No split found at data/raw_coco/splits/train.json"
    echo "Run tools/split_coco.py first."
    exit 1
fi

TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
RUN_DIR="logs/run_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# Capture provenance before training starts, not after — if training crashes
# partway, you still know exactly what was attempted.
{
    echo "{"
    echo "  \"timestamp_utc\": \"${TIMESTAMP}\","
    echo "  \"config\": \"${CONFIG}\","
    echo "  \"config_hash\": \"$(sha256sum "${CONFIG}" | cut -d' ' -f1)\","
    echo "  \"split_manifest\": $(cat data/raw_coco/splits/split_manifest.json),"
    echo "  \"git_commit\": \"$(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')\","
    echo "  \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none-detected')\""
    echo "}"
} > "${RUN_DIR}/run_manifest.json"

echo "== Provenance written to ${RUN_DIR}/run_manifest.json =="
cat "${RUN_DIR}/run_manifest.json"
echo ""

# MMEngine auto-detects device (cpu vs cuda) based on torch.cuda.is_available(),
# so no explicit flag is needed either way -- but log which one it'll use.
python -c "import torch; print('== Training device:', 'cuda' if torch.cuda.is_available() else 'cpu', '==')"

echo "== Starting training =="

python mmdetection/tools/train.py "${CONFIG}" \
    --work-dir "checkpoints/" \
    2>&1 | tee "${RUN_DIR}/train.log"

echo ""
echo "== Training complete. Checkpoints in checkpoints/, log + manifest in ${RUN_DIR}/ =="
