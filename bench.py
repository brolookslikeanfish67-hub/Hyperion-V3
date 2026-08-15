"""
Hyperion-V3: Performance Profiling & Hardware Benchmark Suite.
Measures execution speed, token latency, and VRAM memory-scaling bounds.
"""
import time
import torch
import torch.nn as nn
from model import HyperionV3, HyperionV3Config

# --- BENCHMARK PROFILING MATRICES ---
CONTEXT_TEST_DEPTHS = [1024, 4096, 16384, 65536]  # Scaled up to your 64K target
BATCH_SIZE = 1  # Standard constraint for real-time IDE completion tracking
WARMUP_RUNS = 3
EVAL_RUNS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_sys_header(config: HyperionV3Config):
    print("=" * 60)
    print("       HYPERION-V3 HORIZON-CLASS PROFILE RUNNER")
    print("=" * 60)
    print(f"[Device Workspace] Target Backend: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"[Hardware Details] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[Hardware Details] Total VRAM Capacity: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"[Model Topology] Context Boundary Max: {config.max_seq_len} tokens")
    print(f"[Model Topology] Hidden Dimensions: {config.dim} | Total Layers: {config.n_layers}")
    print(f"[Model Topology] MoE Configuration: {config.n_routed_experts} Experts (Top-{config.top_k_experts})")
    print("=" * 60 + "\n")

@torch.no_grad()
def benchmark_context_layer(model: nn.Module, seq_len: int, dtype: torch.dtype):
    """Profiles a single processing iteration over targeted sequence sizes."""
    # Build a simulated token batch array matching standard data distribution shapes
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH_SIZE, seq_len), device=DEVICE)
    
    # Empty cache allocations to ensure exact tracking metrics
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Warmup runs to spin up CUDA kernels and initialize the engine graph
    for _ in range(WARMUP_RUNS):
        _ = model(input_ids)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        
    # Start tracking time metrics
    start_time = time.perf_counter()
    
    for _ in range(EVAL_RUNS):
        _ = model(input_ids)
        
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Calculate performance averages
    avg_latency = (end_time - start_time) / EVAL_RUNS
    total_tokens_processed = BATCH_SIZE * seq_len
    tokens_per_second = total_tokens_processed / avg_latency
    
    # Calculate precise VRAM footprint markers
    peak_vram_gb = 0.0
    if DEVICE == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(0) / 1e9
        
    return avg_latency, tokens_per_second, peak_vram_gb

def run_performance_suite():
    # Instantiate the base configuration parameters
    config = HyperionV3Config()
    print_sys_header(config)
    
    print("[Pipeline] Instantiating 1.11B MoE architecture model graph blocks...")
    # Initialize the model directly on device using low-precision floating point parameters
    dtype = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    
    with torch.device(DEVICE):
        model = HyperionV3(config=config)
    model.eval()
    
    # Cast parameters to the target floating point format
    model = model.to(dtype=dtype)
    print(f"[Pipeline] Weights successfully cast to floating point representation: {dtype}\n")
    
    print("-" * 75)
    print(f"{'Context Depth':<15} | {'Avg Latency (s)':<18} | {'Tokens / Sec':<15} | {'Peak VRAM (GB)':<15}")
    print("-" * 75)
    
    for depth in CONTEXT_TEST_DEPTHS:
        if depth > config.max_seq_len:
            print(f"{depth:<15} | [Skipped: Exceeds max configuration limits]")
            continue
            
        try:
            latency, tps, vram = benchmark_context_layer(model, depth, dtype)
            print(f"{depth:<15,d} | {latency:<18.4f} | {tps:<15.2f} | {vram:<15.3f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{depth:<15,d} | {'[CRASHED: OOM]':<18} | {'0.00':<15} | {'FAILED':<15}")
                # Clear system state to prevent cascading crashes across remaining test iterations
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            else:
                print(f"{depth:<15,d} | {'[ERROR]':<18} | {'0.00':<15} | {'FAILED':<15}")
                print(f" -> Traceback Details: {e}")
                
    print("-" * 75)
    print("[✓] Profiling sequence completed successfully.")

if __name__ == "__main__":
    run_performance_suite()
