# Avey-B

This repository contains the implementation of Avey-B, code for pretraining, running benchmarks, and generating the plots found in the [Avey-B paper](placeholder). Pre-trained checkpoints of Avey-B are available on HuggingFace, and can be found [here](placeholder). All of the available code was tested on instances running Ubuntu 22.04 with NVIDIA A100. The python environment is managed by uv, and all dependencies are pinned in `pyproject.toml` for reproducibility.

## Structure

- `avey_b`: implementation of the Avey-B model
- `EncodEval`: code to run evaluations
- `bench_latency.py`: code for benchmarking model latency
- `bench_throughput.py`: code for benchmarking model throughput
- `neobert`: contains `NeoBERTForTokenClassification` impementation for TC and QA benchmarks

## Setup

`setup.sh` sets up a new VM or container instance running Ubuntu (to be run as root). Feel free to adjust the script as needed. It will update the system, install awscli (for s3 backups if needed), uv, other necessary and useful packages, and initialize a python environment for running the code (with uv sync). To activate the created environment, run `source .venv/bin/activate`.

## Pre-training

- adjust `gpu_config.sh` as needed

```bash
source gpu_config.sh
```

- run training with torch DDP (`train_mlm.py` implements pre-training for the base model by default)

```bash
sh train.sh
```

## Evaluation

`EncodEval` is modified from [here](https://github.com/hgissbkh/EncodEval/tree/MLM_vs_CLM). To run evals, specify models inside `EncodEval/run.py` under `model_name` (supports huggingface compatible models such as `google-bert/bert-base-uncased`), and run `run.py` from inside `EncodEval`.

## Figures

- Set up a clean python environment using `pip install torch transformers matplotlib pandas adjustText xformers -U`
- Optionally install [Flash Attention](https://github.com/Dao-AILab/flash-attention/releases/) for generating plots using it
- Download the Avey-B checkpoint into `avey`
- Download the [NeoBERT](https://huggingface.co/chandar-lab/NeoBERT) checkpoint into `NeoBERT`, and modify the `max_length` parameter inside `NeoBERT/config.json` to a large value, allowing inference with input sequence lenghts longer than the model was trained for (for the sole purpose of measuring efficiency)
- Run `bench_throughput.py` and `bench_latency.py` in environments with and witnout Flash Attention to generate throughput and latency data. Adjust context length inside the scripts in the case of OOM.
