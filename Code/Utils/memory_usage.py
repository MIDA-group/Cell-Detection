import psutil
import os
from contextlib import contextmanager
import torch
 
# Function to measure CPU memory usage (in MB)
def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

# GPU memory usage utility (in MB)
def get_gpu_memory_usage():
    return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB

# Context manager to monitor memory usage
@contextmanager
def measure_memory(device):
    initial_memory = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        initial_memory = get_gpu_memory_usage()
    elif device == "cpu":
        initial_memory = get_cpu_memory_usage()

    yield  # Run the target operation within this block

    final_memory = 0
    peak_memory = 0
    if device == "cuda":
        final_memory = get_gpu_memory_usage()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    elif device == "cpu":
        final_memory = get_cpu_memory_usage()
        peak_memory = final_memory  # Peak memory is the same as final memory for CPU.

    print(f"Initial Memory Usage: {initial_memory:.2f} MB")
    print(f"Final Memory Usage: {final_memory:.2f} MB")
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")