import torch
import time
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoConfig
import os
import importlib
from adjustText import adjust_text
import csv

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

ctxt_len = 98304
MODEL_NAMES = [
    ("NeoBERT", ctxt_len),
    ("answerdotai/ModernBERT-base", ctxt_len),
    ("avey", ctxt_len),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEP_SIZE = 2048
REPEATS = 3
batch_size = 8


def is_neobert_model(model):
    cfg_name = getattr(getattr(model, "config", None), "model_type", "")
    class_name = model.__class__.__name__
    return (isinstance(cfg_name, str) and cfg_name.lower() == "neobert") or ("NeoBERT" in class_name)


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
            model = AveyForMaskedLM.from_pretrained(model_name, dtype=torch.bfloat16)
            model = model.base_avey_model
            model.save_pretrained(model_name)

        AutoConfig.register("avey", AveyConfig)
        AutoModel.register(AveyConfig, AveyModel)

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    input_sizes = []
    times = []

    with torch.no_grad():
        for seq_len in range(STEP_SIZE, max_len + 1, STEP_SIZE):
            try_packed = is_neobert_model(model) and DEVICE.type == "cuda" and FLASH_ATTN_AVAILABLE

            if try_packed:
                # Build packed tensors: concatenate batch_size sequences into 1 row
                total_len = batch_size * seq_len
                input_ids_packed = torch.ones((1, total_len), dtype=torch.long).to(DEVICE)
                # position ids: positions for each original sequence repeated and concatenated
                pos_list = [torch.arange(seq_len, dtype=torch.long, device=DEVICE) for _ in range(batch_size)]
                position_ids_packed = torch.cat(pos_list, dim=0).unsqueeze(0)  # (1, total_len)
                cu_seqlens = torch.tensor(
                    [i * seq_len for i in range(batch_size + 1)], dtype=torch.int32, device=DEVICE
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
                            output_hidden_states=False
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
                            output_hidden_states=False
                        )
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()

                avg_time = (end - start) / REPEATS
                print(f"Seq len {seq_len}: {avg_time:.4f} sec")

                input_sizes.append(seq_len)
                times.append(avg_time)
            else:
                input_ids = torch.ones((batch_size, seq_len), dtype=torch.long).to(DEVICE)
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
                print(f"Seq len {seq_len}: {avg_time:.4f} sec")

                input_sizes.append(seq_len)
                times.append(avg_time)

    return input_sizes, times


def main():
    results = {}

    for model_name, max_len in MODEL_NAMES:
        input_sizes, times = benchmark_model(model_name, max_len)
        results[model_name] = (input_sizes, times)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    os.makedirs("benchmark_results", exist_ok=True)

    # Save results to CSV
    csv_path = "benchmark_results/latency.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Sequence Length", "Average Forward Time (sec)"])
        for model_name, (input_sizes, times) in results.items():
            for seq_len, avg_time in zip(input_sizes, times):
                writer.writerow([model_name, seq_len, avg_time])
    print(f"Saved CSV to '{csv_path}'")

    # Plotting
    plt.figure(figsize=(12, 7))
    texts = []

    for model_name, (input_sizes, times) in results.items():
        line, = plt.plot(input_sizes, times, label=model_name)
        x, y = input_sizes[-1], times[-1]
        txt = plt.text(
            x + 50, y, model_name,
            fontsize=9,
            verticalalignment='center',
            color=line.get_color()
        )
        texts.append(txt)

    plt.xlabel("Input Size (tokens)")
    plt.ylabel("Average Forward Time (seconds)")
    plt.title(f"BERT Model Forward Pass Time vs. Input Length ({gpu_name})")
    plt.grid(True)

    adjust_text(texts,
                only_move={'points': 'y', 'texts': 'y'},
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                expand_text=(1.05, 1.2),
                expand_points=(1.05, 1.2))

    plt.tight_layout()
    plt.savefig("benchmark_results/latency.png")
    print("Saved plot to 'benchmark_results/latency.png'")


if __name__ == "__main__":
    main()
