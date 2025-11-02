# Avey-B

This repository contains the implementation of Avey-B, code for pretraining, running benchmarks, and making plots found in the [Avey-B paper](placeholder). The pre-trained checkpoints of Avey-B trained as a part of this paper can be found on [HuggingFace](placeholder).

## Structure

- `avey_b`: implementation of the Avey-B model
- `EncodEval`: All evaluation code
- `bench_latency.py`: code for benchmarking model latency
- `bench_throughput.py`: code for benchmarking model throughput
- `neobert`: contains `NeoBERTForTokenClassification` impementation for TC and QA benchmarks

## Setup

`setup.sh` sets up a new VM or container instance running Ubuntu (to be run as root), adjust as needed. Installs awscli (for s3 backups if needed), miniconda, and two conda envs (one each for training and eval).

## Pre-training

- adjust `gpu_config.sh` as needed

```bash
source gpu_config.sh
```

- run training with torch DDP (`train_mlm.py` implements pre-training for base model by default)

```bash
sh train.sh
```

## Evaluation

`EncodEval` is modified from [here](https://github.com/hgissbkh/EncodEval/tree/MLM_vs_CLM). To run evals, specify models inside `EncodEval/run.py` under `model_name` and `ir_model_name` (supports huggingface compatible models such as `google-bert/bert-base-uncased`), and run `run.py` from inside `EncodEval`.

For Avey-B:

- Download checkpoint to `./avey`
- Generate base model for Sentence Transformers for IR benchmarks:

```bash
cp -r avey avey-model
python save_avey_model.py
```

## Figures

- Set up a clean python environment using `pip install torch transformers matplotlib pandas adjustText xformers -U`
- Optionally install [Flash Attention](https://github.com/Dao-AILab/flash-attention/releases/) for generating plots using it
- Download the Avey-B checkpoint into `avey`
- Download the [NeoBERT](https://huggingface.co/chandar-lab/NeoBERT) checkpoint into `NeoBERT`, and modify the `max_length` parameter inside `NeoBERT/config.json` to a large value, allowing inference with input sequence lenghts longer than the model was trained for (for the sole purpose of measuring efficiency)
- Run `bench_throughput.py` and `bench_latency.py` in environments with and witnout Flash Attention to generate throughput and latency data. Adjust context length inside the scripts in the case of OOM.