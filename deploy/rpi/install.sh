#!/bin/bash
# ==============================================================================
# 1-Click Installer for GTCRN Speech Enhancement on Raspberry Pi OS (64-bit)
# ==============================================================================

set -e

echo "=== [1/5] Updating Apt & Installing System Libraries ==="
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv libportaudio2 libsndfile1

echo "=== [2/5] Optimizing Raspberry Pi CPU Governor for Real-Time DSP ==="
# Prevent frequency scaling jitter during streaming inference
if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
    echo "Setting CPU governor to performance mode..."
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true
fi

echo "=== [3/5] Setting User Permissions for Serial & Audio ==="
sudo usermod -a -G dialout,audio $USER

echo "=== [4/5] Creating Isolated Python Virtual Environment ==="
VENV_DIR="$HOME/.venv-gtcrn"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo "=== [5/5] Running Self-Test on GTCRN Streaming Engine ==="
python3 -c "
import numpy as np
from engine import GTCRNEngine

engine = GTCRNEngine('gtcrn_stream.onnx')
dummy_chunk = np.zeros(256, dtype=np.float32)
enh, lat, is_late = engine.process_frame(dummy_chunk)
print(f'Self-test passed! Engine initialized successfully. Benchmark latency: {lat:.2f} ms (Late: {is_late})')
"

echo "=============================================================================="
echo " Installation Complete! To run:"
echo "   source $VENV_DIR/bin/activate"
echo "   python main.py --mode duplex"
echo "   python main.py --mode serial --port /dev/ttyUSB0"
echo "=============================================================================="
