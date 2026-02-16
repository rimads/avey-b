# based on https://github.com/rimads/avey-dpa/blob/main/avey/train.py

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
import argparse
import math
import os
import random
import shutil
import traceback

import boto3
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

from avey_b.configuration_avey import AveyConfig
from avey_b.modeling_avey import AveyForMaskedLM
from dataloader import DataLoader_MLM as DataLoader

MODEL_TOKENIZER = "avey-ai/avey1-tokenizer-base"

TRAIN_CONFIG = {
    "total_batch_size": 2**19,  # ~0.5M tokens
    "seq_length": 2048,
    "max_steps": int(2000 * 10),  # 2000 steps = ~1B tokens @ 0.5M/step
    "checkpoint_interval": int(2000 * 10),
    "rng_seed": 11,
    "use_torch_compile": True,
    "adam_beta_1": 0.95,
    "adam_beta_2": 0.95,
    "adam_fused": False,
    "adam_eps": 1e-18,
    "weight_decay": 0.01,
    "grad_norm_clip": 1.0,
}

LR_CONFIG = {
    "decay_type": "cosine",  # "linear" | "cosine" | "exponential" | "none"
    "max_lr": 7e-4,
    "warmup_steps": TRAIN_CONFIG["max_steps"] * 0.1,
    "decay_steps": TRAIN_CONFIG["max_steps"] * 0.9,
}

PATH_CONFIG = {
    "project_name": "Avey-B",
    "model_name": "avey-b-base",
    "backup_to_s3": True,
    "s3_base_dir": "backup",
}
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser(description="training script")
parser.add_argument("--device_bsz", type=int, default=1)
parser.add_argument(
    "--pretrained_path",
    type=str,
    default=None,
)
args, _ = parser.parse_known_args()
d_bsz = int(args.device_bsz)
pretrained_path = args.pretrained_path

TRAIN_CONFIG["micro_batch_size"] = d_bsz
PATH_CONFIG["save_dir"] = PATH_CONFIG["model_name"]

model_name = PATH_CONFIG["model_name"]


# -----------------------------------------------------------------------------
def backup_to_s3(local_directory, bucket_name="backup"):
    s3_client = boto3.client("s3")

    for filename in os.listdir(local_directory):
        local_path = os.path.join(local_directory, filename)
        if os.path.isfile(local_path):
            print(f"Uploading {local_path} to S3 bucket {bucket_name}")
            s3_client.upload_file(
                local_path, bucket_name, f"{PATH_CONFIG['s3_base_dir']}/{local_path}"
            )


def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if master_process:
        print(
            f"Number of decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters"
        )
        print(
            f"Number of non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters"
        )

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(TRAIN_CONFIG["adam_beta_1"], TRAIN_CONFIG["adam_beta_2"]),
        eps=TRAIN_CONFIG["adam_eps"],
        fused=TRAIN_CONFIG["adam_fused"],
    )
    return optimizer


def get_lr(step):
    max_lr = LR_CONFIG["max_lr"]
    max_steps = TRAIN_CONFIG["max_steps"]
    warmup_steps = LR_CONFIG.get("warmup_steps", 0)
    decay_steps = LR_CONFIG.get("decay_steps", max_steps - warmup_steps)
    decay_type = LR_CONFIG.get("decay_type", "linear")  # default = linear

    # Optional params for exponential decay
    exp_decay_rate = LR_CONFIG.get(
        "exp_decay_rate", 0.1
    )  # final_lr = exp_decay_rate * max_lr

    # Warmup
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    # Constant LR before decay starts
    decay_start = max_steps - decay_steps
    if step < decay_start:
        return max_lr

    # Progress of decay (0 → 1)
    decay_progress = (step - decay_start) / decay_steps
    decay_progress = min(max(decay_progress, 0.0), 1.0)

    # Different decay modes
    if decay_type == "linear":
        # same as your original WSD
        return max_lr * (1.0 - decay_progress)

    elif decay_type == "cosine":
        # cosine annealing from max_lr → 0
        return max_lr * 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    elif decay_type == "exponential":
        # exponential from max_lr → exp_decay_rate * max_lr
        return max_lr * (exp_decay_rate**decay_progress)

    elif decay_type == "none":
        # just hold constant after warmup
        return max_lr

    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def main():
    global master_process  # For logging and distributed setup
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from torch._inductor import config

    config.fallback_random = True
    seed = TRAIN_CONFIG["rng_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # DDP Setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA is required for DDP training."
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        print(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    # tf32
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    os.makedirs(PATH_CONFIG["save_dir"], exist_ok=True)
    if master_process:
        wandb.init(project=PATH_CONFIG["project_name"], name=PATH_CONFIG["model_name"])

    # Data Loader Setup & Gradient Accumulation Computation
    B = TRAIN_CONFIG["micro_batch_size"]
    T = TRAIN_CONFIG["seq_length"]
    total_batch_size = TRAIN_CONFIG["total_batch_size"]
    assert total_batch_size % (B * T * ddp_world_size) == 0, (
        "bsz not divisible by B * T * ddp_world_size"
    )
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    TRAIN_CONFIG["grad_accum_steps"] = grad_accum_steps

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)
    tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    train_loader = DataLoader(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        tokenizer=tokenizer,
    )

    if pretrained_path is not None:
        model = AveyForMaskedLM.from_pretrained(pretrained_path)
    else:
        config = AveyConfig(
            vocab_size=50368,
            context_len=4_294_967_296,
            d_embed=768,
            n_layers=30,
            expansion_factor=4,
            chunk_size=256,
            k=3,
            context_proportion=0.5,
        )
        model = AveyForMaskedLM(config)

    model.to(device)
    if TRAIN_CONFIG["use_torch_compile"]:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    if ddp_rank == 0:
        print("PARAMERETS:", sum(p.numel() for p in raw_model.parameters()))

    optimizer = configure_optimizers(
        raw_model,
        weight_decay=TRAIN_CONFIG["weight_decay"],
        learning_rate=LR_CONFIG["max_lr"],
    )

    prog_bar = tqdm(
        range(0, TRAIN_CONFIG["max_steps"]),
        desc="Step",
        initial=0,
        total=TRAIN_CONFIG["max_steps"],
        disable=(not master_process),
    )

    # Training Loop
    for step in prog_bar:
        model.train()
        optimizer.zero_grad()

        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(0.20)
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                output = model(input_ids=x, labels=y)
                loss = output.loss

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            loss_accum_detached = loss_accum.clone().detach()
            dist.all_reduce(loss_accum_detached, op=dist.ReduceOp.AVG)
            log_loss = loss_accum_detached.item()
        else:
            log_loss = loss_accum.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), TRAIN_CONFIG["grad_norm_clip"]
        )
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if master_process:
            wandb.log(
                {
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": lr,
                    "train/loss": log_loss,
                },
                step=step,
            )
        prog_bar.set_postfix({"loss": log_loss})

        if (
            master_process
            and ((step + 1) % TRAIN_CONFIG["checkpoint_interval"] == 0)
            and (step > 0)
        ):
            while True:
                try:
                    ckpt_dir = os.path.join(
                        PATH_CONFIG["save_dir"], f"checkpoint_{step}"
                    )
                    raw_model.save_pretrained(ckpt_dir, safe_serialization=True)
                    shutil.copy("avey_b/modeling_avey.py", ckpt_dir)
                    shutil.copy("avey_b/configuration_avey.py", ckpt_dir)

                    if PATH_CONFIG["backup_to_s3"]:
                        backup_to_s3(ckpt_dir, bucket_name="aveylm-backup")
                        shutil.rmtree(ckpt_dir)
                    break
                except Exception as e:
                    print(f"\nException during checkpointing: {e}")
                    traceback.print_exc()
                    input(
                        "Checkpointing failed. Please fix the issue and press Enter to retry..."
                    )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("-" * 30)
        print(e)
        print("-" * 30)
        traceback.print_exc()
    finally:
        if "dist" in globals() and dist.is_initialized():
            destroy_process_group()
