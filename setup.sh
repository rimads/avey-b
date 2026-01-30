#!/usr/bin/env bash
set -e

# Non-interactive mode for apt
export DEBIAN_FRONTEND=noninteractive
export TERM=xterm

# Mock sudo if running as root
if [ "$(id -u)" -eq 0 ]; then
    alias sudo=''
fi

# -------------------------
# System Packages
# -------------------------
echo "[*] Updating system packages..."
apt-get update -y

echo "[*] Installing base system packages..."
apt-get install -y --no-install-recommends \
    neovim cmake jq bc unzip curl git ca-certificates

# -------------------------
# AWS CLI installation
# -------------------------
if ! command -v aws &>/dev/null; then
    echo "[*] Installing AWS CLI v2..."
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
    unzip -q awscliv2.zip
    ./aws/install
    rm -rf aws awscliv2.zip
else
    echo "[✓] AWS CLI already installed."
fi

# -------------------------
# uv Installation
# -------------------------
if ! command -v uv &>/dev/null; then
    echo "[*] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    else
        echo "[!] Warning: uv env file not found, adding paths manually."
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
else
    echo "[✓] uv already installed."
fi

# -------------------------
# Unified Environment Setup
# -------------------------
VENV_DIR="$HOME/venvs/main"
mkdir -p "$(dirname "$VENV_DIR")"

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating unified environment (Python 3.11)..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "[✓] Environment already exists."
fi

# Check if a core library is installed to verify state
if ! "$VENV_DIR/bin/python" -c "import torch" &>/dev/null; then
    echo "[*] Installing all dependencies (Train + Eval)..."

    source "$VENV_DIR/bin/activate"

    # Installing all packages in one go allows uv to resolve
    # compatible versions for everything simultaneously.
    uv pip install --upgrade \
        torch torchvision torchaudio torchao torchtune \
        transformers datasets wandb boto3 accelerate \
        configue fire numpy pandas protobuf \
        sentence-transformers sentencepiece tensorboard \
        tiktoken adjustText
else
    echo "[✓] Dependencies already installed."
fi

# -------------------------
# Eval Tokenizer Setup
# -------------------------
# to ensure the tokenizer repo is present if the folder exists.
if [ -d "EncodEval" ]; then
    echo "[*] Checking EncodEval auxiliary files..."
    (
        cd EncodEval
        if [ ! -d avey1-tokenizer-base ]; then
            echo "[*] Cloning tokenizer repo..."
            git clone https://huggingface.co/avey-ai/avey1-tokenizer-base
        else
            echo "[✓] Tokenizer repo already cloned."
        fi
    )
else
    echo "[!] Note: 'EncodEval' directory not found in $(pwd). Skipping tokenizer clone."
fi

echo "------------------------------------------------"
echo "[✓] Setup complete."
echo "Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
echo "------------------------------------------------"
