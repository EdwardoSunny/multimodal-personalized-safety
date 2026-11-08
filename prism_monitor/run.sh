#!/bin/bash
# Train and evaluate the four PRISM configs from dataset/splits/{train,val,test}.json.
#
#   bash run.sh            # configs that already have checkpoints/<config>/model.safetensors (e.g. the
#                          # released weights from scripts/download_checkpoints.py) are not retrained
#   bash run.sh --retrain  # retrain every config, replacing existing checkpoints
set -e
cd "$(dirname "$0")"

case "${1:-}" in
    "") train_mode=--skip_existing ;;
    --retrain) train_mode=--overwrite ;;
    *) echo "Usage: bash run.sh [--retrain]" >&2; exit 2 ;;
esac

# Step 1: cache frozen encoder embeddings (once per encoder)
uv run python cache_embeddings.py

# Step 2: train the four PRISM configs (SigLIP2, CLIP, EVA-CLIP, Qwen3-VL)
uv run python train_models.py "$train_mode"

# Step 3: evaluate on the test split
uv run python evaluate.py
