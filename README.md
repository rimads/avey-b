# Avey-B

[![Paper](https://img.shields.io/badge/Paper-ICLR_2026-b31b1b)](placeholder_link)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](placeholder_link)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

This repository contains the official implementation, pretraining code, and evaluation scripts for **Avey-B**, as presented in the paper **"Avey-B"** (ICLR 2026).

> **Abstract:** Compact pretrained bidirectional encoders remain the backbone of industrial NLP under tight compute and memory budgets. Their effectiveness stems from self-attention’s ability to deliver high-quality bidirectional contextualization with sequence-level parallelism, as popularized by BERT-style architectures. Recently, Avey was introduced as an autoregressive, attention-free alternative that naturally admits an encoder-only adaptation. In this paper, we reformulate Avey for the encoder-only paradigm and propose several innovations to its architecture, including decoupled static and dynamic parameterizations, stability-oriented normalization, and neural compression. Results show that this reformulated architecture compares favorably to four widely used Transformer-based encoders, consistently outperforming them on standard token-classification and information-retrieval benchmarks while scaling more efficiently to long contexts.

---

## Repository Structure

```text
.
├── avey_b/              # Core implementation of the Avey-B model architecture
├── EncodEval/           # Evaluation framework (SC, TC, QA, IR benchmarks)
├── EncodEval/neobert/             # Custom implementations for NeoBERT baseline comparisons
├── bench_latency.py     # Script for benchmarking inference latency
├── bench_throughput.py  # Script for benchmarking training/inference throughput
├── setup.sh             # Environment setup script
├── train.sh             # Training launcher script
├── train_mlm.py         # Masked Language Modeling (MLM) pretraining script
└── pyproject.toml       # Dependency management via uv
```

## Setup & Installation

The codebase is tested on Ubuntu 22.04 using NVIDIA A100 and H100 GPUs. Python environments are managed using `uv` for strict reproducibility.

1.  **Initialize Environment:**
    The provided `setup.sh` script installs system dependencies (including `awscli`), installs `uv`, and syncs the Python environment defined in `pyproject.toml`.
    ```bash
    bash setup.sh
    ```

2.  **Activate Environment:**
    ```bash
    source .venv/bin/activate
    ```

---

## Pre-training

We provide scripts to pretrain Avey-B from scratch using the Masked Language Modeling (MLM) objective.

1.  **Configuration:** Adjust model hyperparameters inside `train_mlm.py` (approx. line 242) if needed.
2.  **Launch Training:**
    Use `train.sh` to automatically detect available GPUs and launch the training run. You can control the per-device batch size via environment variables.

    ```bash
    # Example: Set batch size to 16 (fits on 80GB VRAM)
    export BATCH_SIZE=16
    bash train.sh
    ```

    *Note: `train.sh` handles single-node multi-GPU setups. for multi-node training, please invoke `torchrun` manually with the appropriate rendezvous arguments.*

---

## 📊 Evaluation

Our evaluation framework is adapted from [EncodEval](https://github.com/hgissbkh/EncodEval/tree/MLM_vs_CLM).

### 1. Preparation
If you intend to run long-range Needle-In-A-Haystack (NIAH) benchmarks:
```bash
python gen_niah.py
```

### Running Benchmarks
1.  Navigate to the `EncodEval` directory (`cd EncodEval`).
2.  Open `EncodEval/run.py` and specify:
    *   `model_name`: Local path or HuggingFace ID (e.g., `google-bert/bert-base-uncased`).
    *   `learning_rates`: List of LRs to sweep.
    *   Benchmarks and random seeds.
3.  Ensure YAML configurations for your chosen benchmarks, learning rate, and seeds exist in `EncodEval/configs` (configs for specified values in `run.py` are already provided).
4.  Run the evaluation:
    ```bash
    # cd EncodEval, if not already in the directory
    python run.py
    ```
`run.py` will automatically schedule the benchmarks to run on all GPUs on the machine as they become available.

### NeoBERT Specifics
If evaluating NeoBERT, specific token classification implementations are required:
1.  Download the NeoBERT model.
2.  Move the files from `EncodEval/neobert/` (in this repo) into your downloaded NeoBERT model directory.
3.  Point `model_name` in `EncodEval/run.py` to this local directory.

---

## Efficiency Benchmarks

To reproduce the efficiency plots (throughput and latency) found in the paper:

```bash
# Generate throughput data
python bench_throughput.py

# Generate latency data
python bench_latency.py
```

To run the unoptimized version of Avey-B, the `@torch.compile` decorator can be removed from the implementation. To test the optimized versions of the other models, flash-attention will need to be installed.

**Note on NeoBERT Efficiency Testing:**
To test NeoBERT beyond its training window (solely for efficiency/OOM measurements), you must manually override its config:
1.  Download the [NeoBERT checkpoint](https://huggingface.co/chandar-lab/NeoBERT).
2.  Modify `config.json`: Set `max_length` to a large value (e.g., 100000).
3.  Update the benchmarking scripts to point to this modified local checkpoint.

---

## Citation

If you use Avey-B or this codebase in your research, please cite our paper:

```bibtex
@inproceedings{acharya2026aveyb,
  title={Avey-B},
  author={Acharya, Devang and Hammoud, Mohammad},
  booktitle={Published as a conference paper at ICLR 2026},
  year={2026}
}
```
