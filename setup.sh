#!/usr/bin/env bash
set -e
export TERM=xterm
alias sudo=''

echo "[*] Updating system packages..."
apt update -y

echo "[*] Installing base system packages..."
apt install -y neovim cmake jq bc python3-pip unzip awscli

# -------------------------
# AWS CLI installation
# -------------------------
if ! command -v aws &>/dev/null; then
    echo "[*] Installing AWS CLI..."
    if [ -d ./aws ]; then
        ./aws/install || true
    else
        curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
        unzip -q awscliv2.zip
        ./aws/install
        rm -rf aws awscliv2.zip
    fi
else
    echo "[✓] AWS CLI already installed."
fi

# -------------------------
# Conda installation
# -------------------------
if [ ! -d ~/miniconda3 ]; then
    echo "[*] Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
else
    echo "[✓] Miniconda already installed."
fi

source ~/miniconda3/bin/activate
conda init --all >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# -------------------------
# Conda environments
# -------------------------
if ! conda info --envs | grep -q "^train "; then
    echo "[*] Creating train environment..."
    printf "a\na\ny\n" | conda create -n train python=3.11 -y
else
    echo "[✓] train environment already exists."
fi

if ! conda info --envs | grep -q "^eval "; then
    echo "[*] Creating eval environment..."
    conda create -n eval python=3.11 -y
else
    echo "[✓] eval environment already exists."
fi

# -------------------------
# Training libs
# -------------------------
if ! conda run -n train python -c "import torch" &>/dev/null; then
    echo "[*] Installing training libraries..."
    conda activate train
    pip install --upgrade pip
    pip install torch torchvision torchaudio torchao torchtune transformers datasets wandb boto3 accelerate -U
    conda deactivate
else
    echo "[✓] Training libraries already installed."
fi

# -------------------------
# Evaluation libs
# -------------------------
if ! conda run -n eval python -c "import torch" &>/dev/null; then
    echo "[*] Setting up EncodEval..."
    conda activate eval
    cd EncodEval
    pip install --upgrade pip
    pip install .
    pip install adjustText

    if [ ! -d avey1-tokenizer-base ]; then
        git clone https://huggingface.co/avey-ai/avey1-tokenizer-base
    else
        echo "[✓] Tokenizer repo already cloned."
    fi

    conda deactivate
else
    echo "[✓] Eval libraries already installed."
fi
