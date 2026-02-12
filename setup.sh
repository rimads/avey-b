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
uv sync
source .venv/bin/activate

echo "------------------------------------------------"
echo "[✓] Setup complete."
echo "------------------------------------------------"
