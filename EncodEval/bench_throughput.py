import importlib
import os
import time
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import torch
from adjustText import adjust_text
from transformers import AutoConfig, AutoModel

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
    print("---->using flash attn")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("---->NOT using flash attn")

ctxt_len = 98304
MODEL_NAMES = [
    ("chandar-lab/NeoBERT", ctxt_len),
    ("answerdotai/ModernBERT-base", ctxt_len),
    ("avey", ctxt_len),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPEATS = 3
BATCH_SIZES = [8]  # Logical batch sizes to benchmark


def is_neobert_model(model):
    # Heuristics: config.model_type == "neobert" or class name contains NeoBERT
    cfg_name = getattr(getattr(model, "config", None), "model_type", "")
    class_name = model.__class__.__name__
    return (isinstance(cfg_name, str) and cfg_name.lower() == "neobert") or (
        "NeoBERT" in class_name
    )


def benchmark_model(model_name, max_len):
    print(f"\nBenchmarking {model_name}...")

    if "avey" in model_name:
        config_import = importlib.import_module(f"{model_name}.configuration_avey")
        model_import = importlib.import_module(f"{model_name}.modeling_avey")
        AveyConfig = config_import.AveyConfig
        AveyModel = model_import.AveyModel

        config = AveyConfig.from_pretrained(model_name)
        archs = getattr(config, "architectures", [])
        is_mlm = any("MaskedLM" in a for a in archs)
        if is_mlm:
            print("saving avey model from masked LM...")
            AveyForMaskedLM = model_import.AveyForMaskedLM
            model = AveyForMaskedLM.from_pretrained(model_name)
            model = model.base_avey_model
            model.save_pretrained(model_name)

        AutoConfig.register("avey", AveyConfig)
        AutoModel.register(AveyConfig, AveyModel)

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
    model.eval()

    seq_lenghts = [128, 256, 512, 1024, 2048] + list(range(4096, max_len + 1, 4096))

    results = []  # store (batch_size, seq_len, throughput)

    with torch.no_grad():
        for batch_size in BATCH_SIZES:
            print(f"  Batch size {batch_size}...")
            for seq_len in seq_lenghts:
                try_packed = (
                    is_neobert_model(model)
                    and DEVICE.type == "cuda"
                    and FLASH_ATTN_AVAILABLE
                )

                if try_packed:
                    # Build packed tensors: concatenate batch_size sequences into 1 row
                    total_len = batch_size * seq_len
                    input_ids_packed = torch.ones((1, total_len), dtype=torch.long).to(
                        DEVICE
                    )
                    # position ids: positions for each original sequence repeated and concatenated
                    pos_list = [
                        torch.arange(seq_len, dtype=torch.long, device=DEVICE)
                        for _ in range(batch_size)
                    ]
                    position_ids_packed = torch.cat(pos_list, dim=0).unsqueeze(
                        0
                    )  # (1, total_len)
                    cu_seqlens = torch.tensor(
                        [i * seq_len for i in range(batch_size + 1)],
                        dtype=torch.int32,
                        device=DEVICE,
                    )
                    max_seqlen = seq_len

                    # Warm-up (packed)
                    # Some NeoBERT implementations assert output_attentions=False for flash path.
                    for _ in range(3):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            model(
                                input_ids=input_ids_packed,
                                position_ids=position_ids_packed,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen,
                                attention_mask=None,
                                output_attentions=False,
                                output_hidden_states=False,
                            )

                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()

                    # Timed runs (packed)
                    start = time.time()
                    for _ in range(REPEATS):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            model(
                                input_ids=input_ids_packed,
                                position_ids=position_ids_packed,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen,
                                attention_mask=None,
                                output_attentions=False,
                                output_hidden_states=False,
                            )
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    end = time.time()

                    avg_time = (end - start) / REPEATS
                    tokens = batch_size * seq_len  # logical tokens processed
                    throughput = tokens / avg_time  # tokens/sec

                    print(
                        f"    Seq len {seq_len} (packed): {throughput:.2f} tokens/sec"
                    )
                    results.append((batch_size, seq_len, throughput))

                else:
                    # Either not NeoBERT or no CUDA, or we will fall back here if packed path fails.
                    # Regular batched input
                    input_ids = torch.ones((batch_size, seq_len), dtype=torch.long).to(
                        DEVICE
                    )
                    attention_mask = torch.ones_like(input_ids).to(DEVICE)

                    # Warm-up
                    for _ in range(3):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            model(input_ids=input_ids, attention_mask=attention_mask)

                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()

                    # Measure time
                    start = time.time()
                    for _ in range(REPEATS):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            model(input_ids=input_ids, attention_mask=attention_mask)
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    end = time.time()

                    avg_time = (end - start) / REPEATS
                    tokens = batch_size * seq_len
                    throughput = tokens / avg_time  # tokens/sec

                    print(f"    Seq len {seq_len}: {throughput:.2f} tokens/sec")
                    results.append((batch_size, seq_len, throughput))

    return results


def main():
    all_results = {}

    for model_name, max_len in MODEL_NAMES:
        try:
            results = benchmark_model(model_name, max_len)
            all_results[model_name] = results
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            traceback.print_exc()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    os.makedirs("benchmark_results", exist_ok=True)

    # Save to CSV for later analysis
    rows = []
    for model_name, results in all_results.items():
        for batch_size, seq_len, throughput in results:
            rows.append(
                {
                    "model": model_name,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "throughput_tokens_per_sec": throughput,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv("benchmark_results/bert_throughput_data.csv", index=False)
    print("\nSaved raw data to 'benchmark_results/bert_throughput_data.csv'")

    # ---- Separate plots: one per batch size ----
    for batch_size in BATCH_SIZES:
        plt.figure(figsize=(12, 7))
        texts = []

        for model_name, results in all_results.items():
            seq_lens = [r[1] for r in results if r[0] == batch_size]
            throughputs = [r[2] for r in results if r[0] == batch_size]

            if not seq_lens:
                continue

            (line,) = plt.plot(seq_lens, throughputs, label=model_name)

            # annotate last point
            x, y = seq_lens[-1], throughputs[-1]
            txt = plt.text(
                x + 50,
                y,
                model_name,
                fontsize=9,
                verticalalignment="center",
                color=line.get_color(),
            )
            texts.append(txt)

        plt.xlabel("Input Size (tokens)")
        plt.ylabel("Throughput (tokens/sec)")
        plt.title(f"Throughput vs. Input Length (batch={batch_size}, {gpu_name})")
        plt.grid(True)
        plt.legend(fontsize=9)

        adjust_text(
            texts,
            only_move={"points": "y", "texts": "y"},
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            expand_text=(1.05, 1.2),
            expand_points=(1.05, 1.2),
        )

        plt.tight_layout()
        save_path = f"benchmark_results/throughput_batch{batch_size}.png"
        plt.savefig(save_path)
        print(f"Saved plot to '{save_path}'")


if __name__ == "__main__":
    main()
