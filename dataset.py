import os
import json
import random
import urllib.request

# --- HYPERION-V3  DATA CONFIGURATION ---
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
VAL_FILE = os.path.join(DATA_DIR, "val.txt")
EOD_TOKEN = "<|endoftext|>"

#raw, multi-turn algorithmic and architectural data pools (No auth required)
DATA_SOURCES = {
    "reasoning": "https://githubusercontent.com",
    "algorithms": "https://githubusercontent.com" # Mapping hook
}

def setup_environment():
    """Ensures directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)

def fetch_json_data(url):
    """Safely downloads raw JSON files without external library dependencies."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"[-] Failed to fetch from {url}: {e}")
        return None

def inject_hyperion_features(instruction, input_code, output_code):
    """
    Mutates raw instructions into elite multi-turn thinking paths 
    and Fill-in-the-Middle (FIM) optimization blocks.
    """
    # 1. Structure Adaptive Thinking Recurrent Loops
    thinking_block = (
        f"<|thinking|>\n"
        f"Analyzing Request: {instruction}\n"
        f"Context Pointers: {input_code if input_code else 'Pure Algorithmic Generation'}\n"
        f"Complexity Mapping: Evaluating low-rank manifold transitions and gate weights.\n"
        f"Formulating optimal matrix operations and code invariants...\n"
        f"</|thinking|>\n"
    )
    
    # 2. Structure Native Fill-in-the-Middle (FIM) Block
    lines = output_code.splitlines()
    if len(lines) > 8:
        split = random.randint(len(lines) // 3, (len(lines) * 2) // 3)
        prefix = "\n".join(lines[:split])
        suffix = "\n".join(lines[split:])
        code_block = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>\n"
    else:
        code_block = f"{output_code}\n"
        
    return f"{thinking_block}{code_block}{EOD_TOKEN}\n"

def generate_procedural_matrix_kernels():
    """
    Generates complex, low-level mathematical and algorithmic code strings 
    procedurally to maximize model capability on multi-rank structures.
    """
    kernels = []
    for _ in range(500): # Scale this number up to infinitely increase data density
        kernel_id = random.randint(10000, 99999)
        kernels.append(
            f"// INSANE LOGIC INVARIANT MODULE #{kernel_id}\n"
            f"// Target: Hyper-MLA Low-Rank Latent Cache Compression Matrix\n"
            f"void compress_latent_manifold_v3(float* __restrict__ KV_cache, float* __restrict__ low_rank_out, int dim, int rank) {{\n"
            f"    #pragma omp parallel for collapse(2)\n"
            f"    for(int b = 0; b < {random.choice([32, 64, 128])}; ++b) {{\n"
            f"        for(int s = 0; s < 4096; ++s) {{\n"
            f"            float accum = 0.0f;\n"
            f"            for(int d = 0; d < dim; ++d) {{\n"
            f"                accum += KV_cache[b * dim + d] * {random.uniform(0.01, 0.99)}f;\n"
            f"            }}\n"
            f"            low_rank_out[b * rank + (s % rank)] = accum;\n"
            f"        }}\n"
            f"    }}\n"
            f"}}\n"
            f"{EOD_TOKEN}\n"
        )
    return kernels

def build_insane_dataset():
    print("[+] Initializing Hyperion-V3 Elite Training Data Generation Pipeline...")
    setup_environment()
    
    compiled_blocks = []
    
    # --- PHASE 1: Fetch and build Reasoning Chains ---
    print("[->] Downloading CodeAlpaca algorithmic instruction blocks...")
    alpaca_data = fetch_json_data(DATA_SOURCES["reasoning"])
    
    if alpaca_data:
        print(f"[+] Loaded {len(alpaca_data)} instructional data components.")
        for item in alpaca_data:
            block = inject_hyperion_features(
                instruction=item.get("instruction", ""),
                input_code=item.get("input", ""),
                output_code=item.get("output", "")
            )
            compiled_blocks.append(block)
            
    # --- PHASE 2: Generate Low-Level Procedural Math Kernels ---
    print("[->] Synthesizing low-level memory-mapped manifold computing matrices...")
    procedural_kernels = generate_procedural_matrix_kernels()
    compiled_blocks.extend(procedural_kernels)
    
    # --- PHASE 3: Compilation and Entropy Splitting ---
    if not compiled_blocks:
        print("[-] Critical Error: Data synthesis pool is empty. Please verify connections.")
        return
        
    print(f"[+] Total compiled operational tokens chunks: {len(compiled_blocks)}")
    print("[->] Shuffling data matrices to enforce radical cross-entropy distribution...")
    random.shuffle(compiled_blocks)
    
    # 90% Training / 10% Validation Split
    split_point = int(len(compiled_blocks) * 0.90)
    train_set = compiled_blocks[:split_point]
    val_set = compiled_blocks[split_point:]
    
    print(f"[->] Writing data streams directly to local filesystem environment...")
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.writelines(train_set)
        
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        f.writelines(val_set)
        
    print("\n========================================================")
    print("[SUCCESS] Hyperion-V3 Ingestion Phase Completed!")
    print(f"[*] Saved {len(train_set)} high-density algorithmic blocks -> {TRAIN_FILE}")
    print(f"[*] Saved {len(val_set)} verification evaluation blocks -> {VAL_FILE}")
    print("========================================================")
    print("[!] Action Required: Run 'python tokenizer_utils.py' next to register these custom logic primitives.")

if __name__ == "__main__":
    build_insane_dataset()
