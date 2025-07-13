#!/bin/bash
set -e

IMAGE_NAME="sleep-analysis"
DEFAULT_DATASET_PATH="data/polysomnography"

# Check if --dataset_dir is passed by the user
DATASET_FLAG_FOUND=false
for arg in "$@"; do
  if [[ "$arg" == "--dataset_dir" ]]; then
    DATASET_FLAG_FOUND=true
    break
  fi
done

# Build the base command
CMD=(python train.py)

# Inject default dataset path if not provided
if [ "$DATASET_FLAG_FOUND" = false ]; then
  CMD+=(--dataset_dir "/app/$DEFAULT_DATASET_PATH")
fi

# Add all remaining args (including possible overrides)
CMD+=("$@")

# Final docker run
docker run \
  --gpus all \
  --user $(id -u):$(id -g) \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/data:/app/data \
  "$IMAGE_NAME" \
  "${CMD[@]}"
