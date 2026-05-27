#!/bin/bash
# setup_a6000.sh — BirdCLEF 2026 VM Setup
# Run once when VM is first created.
set -e

echo "=== BirdCLEF 2026 — A6000 VM Setup ==="
echo "Date: $(date)"

# ── System ──
sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop git-lfs rclone unzip

# ── Conda ──
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
fi

source ~/.bashrc 2>/dev/null || true
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# ── Python env ──
conda create -n birdclef python=3.10 -y
conda activate birdclef

# ── PyTorch (CUDA 12.1) ──
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Dependencies ──
pip install timm librosa scikit-learn pandas onnx onnxruntime \
           iterative-stratification scipy pyyaml tqdm soundfile

# ── Clone repo ──
cd $HOME
if [ ! -d "__birdclef-v3" ]; then
    git clone https://github.com/<USER>/__birdclef-v3.git
fi
cd __birdclef-v3
git checkout sprint-1-preprocessing

# ── Data (download from Kaggle or GDrive) ──
mkdir -p data
# If you have a kaggle.json:
# pip install kaggle
# mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/
# kaggle competitions download -c birdclef-2026 -p data/

# ── Verify GPU ──
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"

echo ""
echo "=== Setup complete ==="
echo "Activate: conda activate birdclef && cd ~/__birdclef-v3"
echo "Next: bash scripts/launch_a6000_jobs.sh"
