import subprocess
import time
import os
from queue import Queue

CHECK_INTERVAL = 5  # seconds between GPU usage checks
model_name = "avey"
ir_model_name = "avey-model"


def get_idle_gpus():
    """
    Returns a list of idle GPU indices using nvidia-smi.
    A GPU is considered idle if its memory usage is below a threshold.
    """
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print("Error running nvidia-smi:", result.stderr)
        return []

    idle_gpus = []
    for line in result.stdout.strip().split('\n'):
        index, mem_used = map(int, line.strip().split(','))
        if mem_used < 100:  # < 100 MB usage → consider idle
            idle_gpus.append(index)

    return idle_gpus


def run_command_on_gpu(cmd, gpu_index):
    """
    Runs a shell command with CUDA_VISIBLE_DEVICES set to a specific GPU.
    """
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    print(f"[INFO] Running on GPU {gpu_index}: {cmd}")
    return subprocess.Popen(cmd, shell=True, env=env)


def monitor_and_schedule(commands):
    """
    Monitors GPUs and schedules commands accordingly.
    """
    processes = {}  # gpu_index -> subprocess
    command_queue = Queue()

    for cmd in commands:
        command_queue.put(cmd)

    while not command_queue.empty() or any(p.poll() is None for p in processes.values()):
        # Remove finished processes
        finished_gpus = [gpu for gpu, p in processes.items() if p.poll() is not None]
        for gpu in finished_gpus:
            print(f"[INFO] Process on GPU {gpu} finished.")
            del processes[gpu]

        # Check for available GPUs
        idle_gpus = get_idle_gpus()
        available_gpus = [gpu for gpu in idle_gpus if gpu not in processes]

        for gpu in available_gpus:
            if command_queue.empty():
                break
            cmd = command_queue.get()
            process = run_command_on_gpu(cmd, gpu)
            processes[gpu] = process

        time.sleep(CHECK_INTERVAL)

    print("[INFO] All jobs finished.")


if __name__ == "__main__":
    LRs = [
        "2e-05",
        "6e-05",
        "1e-04",
        "5e-04",
    ]
    tasks = [
        "IR/msmarco_pairs",
        "QA/squad_v2",
        "QA/squad",
        "QA/record",
        "SC/mnli_m",
        "SC/sst2",
        "SC/qqp",
        "TC/conll2003_en",
        "TC/ontonotes",
        "TC/uner_en",
        "IR/mldr_en_msmarco_pairs",
        "IR/msmarco_msmarco_pairs",
        "IR/nq_msmarco_pairs",
    ]
    seeds = list(range(10))
    commands = []

    for task in tasks:
        for lr in LRs:
            for seed in seeds:
                if "IR" in task:
                    commands.append(f"EVAL_MODEL={ir_model_name} python main.py --config_file ./configs/{task}_lr{lr}_sd{seed}.yaml --model_path {ir_model_name}")
                else:
                    commands.append(f"EVAL_MODEL={model_name} python main.py --config_file ./configs/{task}_lr{lr}_sd{seed}.yaml --model_path {model_name}")

    monitor_and_schedule(commands)
