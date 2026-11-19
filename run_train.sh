#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR"

echo "=== Launching Overnight Distillation on GTX 1050 ==="
echo "Python: $(which python) ($(python --version))"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"

nohup python train_overnight.py > train_overnight.log 2>&1 &
TRAIN_PID=$!

echo "Training successfully started in background!"
echo "PID: $TRAIN_PID"
echo "To watch live progress, run:"
echo "  tail -f train_overnight.log"
