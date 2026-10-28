#!/bin/bash
# Reproduction script for Phase 10: MediaPipe SqueezeNet Classifier
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1. Verifying Environment ==="
PYTHON_EXEC="/Users/pavanaksshay/venv/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi
$PYTHON_EXEC --version

echo "=== 2. Auditing Dataset and Generating Leakage-Resistant Splits ==="
$PYTHON_EXEC phase10_prepare_splits.py

echo "=== 3. Training MediaPipe-Compatible SqueezeNet Classifier ==="
$PYTHON_EXEC Model_Training/train_squeezenet_mediapipe.py --epochs 25 --batch-size 32 --lr 0.0003

echo "=== 4. Evaluating Experimental Model Against Baseline ==="
$PYTHON_EXEC evaluate_mediapipe_squeezenet.py

echo "=== 5. Running Regression Tests ==="
$PYTHON_EXEC -m unittest discover tests/

echo "=== Reproduction Complete ==="
