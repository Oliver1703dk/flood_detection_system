#!/bin/bash
# setup_jetson_pytorch.sh

echo "=== Jetson PyTorch CUDA Setup ==="

# Check current PyTorch installation
echo "Checking current PyTorch installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "PyTorch not installed or import failed"
fi

# Check JetPack version
echo -e "\n=== JetPack Version ==="
dpkg -l | grep nvidia-jetpack

# Check CUDA installation
echo -e "\n=== CUDA Installation ==="
nvcc --version 2>/dev/null || echo "CUDA compiler not found"

# Check GPU
echo -e "\n=== GPU Information ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available (normal for Jetson, use tegrastats instead)"

# For Jetson, use tegrastats
echo -e "\n=== Tegra Stats (5 seconds) ==="
timeout 5s tegrastats 2>/dev/null || echo "tegrastats not available"

echo -e "\n=== Recommendations ==="
echo "1. Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo "2. Download the appropriate PyTorch wheel for your JetPack version"
echo "3. Install with: pip3 install <downloaded-wheel>.whl"
echo "4. Verify with: python3 -c 'import torch; print(torch.cuda.is_available())'"