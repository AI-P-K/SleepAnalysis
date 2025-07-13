#!/bin/bash
set -e

IMAGE_NAME="sleep-analysis"

docker run \
  --gpus all \
  --user $(id -u):$(id -g) \
  -v $(pwd)/inference:/app/inference \
  -v $(pwd)/engine:/app/engine \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  "$IMAGE_NAME"