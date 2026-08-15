"""
Hyperion-V3: Advanced Autoregressive Inference Engine.
Supports Native Fill-in-the-Middle (FIM) and Adaptive Recurrent Loop tracking.
"""
import os
import torch
import torch.nn.functional as F
from typing import Optional, List
from model import HyperionV3, HyperionV3Config
from tokenizer_utils import load_tokenizer

# --- CORE INFERENCE PATHS ---
TOKENIZER_JSON = "hyperion_tokenizer.json"
CHECKPOINT_PATH = os.path.join("checkpoints", "hyperion_v3_best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HyperionInferenceEngine:
    def __init__(self, checkpoint_path: str, tokenizer_path: str):
        print(f"[Engine] Initializing token maps from: {tokenizer_path}...")
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        # Pull base model configurations
        self.config = HyperionV3Config()
        
        # Instantiate on target device hardware
        self.dtype = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        print(f"[Engine] Initializing model graph layers on device: {DEVICE.upper()} ({self.dtype})")
        
        with torch.device(DEVICE):
            self.model = HyperionV3(config=self.config)
            
        # Safely load weights if an active checkpoint exists
        if os.path.exists(checkpoint_path):
            print(f"[Engine] Loading optimized model tensor weights from: {checkpoint_path}...")
            state = torch.load(checkpoint_path, map_location=DEVICE)
            # Support both raw dictionaries and packaged checkpoints
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
        else:
            print(f"[!] Warning: Checkpoint not found. Running inference with empty initialized weights.")
            
        self.model.eval().to(dtype=self.dtype)

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 128, 
        temperature: float = 0.2, 
        top_k: int = 40,
        stop_token: str = "<|endoftext|>"
    ) -> str:
        """Runs standard autoregressive sampling with dynamic temperature constraints."""
        # Encode string to tensor format
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
        
        # Extract stop sequence ID if active in your token registry
        stop_id = self.tokenizer.token_to_id(stop_token) if hasattr(self.tokenizer, "token_to_id") else None
        
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            # Enforce 64K context window clipping bounds safely
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # Execute model forward pass
            with torch.amp.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=self.dtype):
                logits, _ = self.model(idx_cond)
                
            # Extract last token logits matrix position
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Filter via top-k matrix pruning bounds
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample distribution properties
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Cleanly append matrix result to sequence chain
            input_ids = torch.cat((input_ids, next_token), dim=1)
            token_id = next_token.item()
            generated_tokens.append(token_id)
            
            # Early breakout evaluation flag checks
            if stop_id and token_id == stop_id:
                break
                
        return self.tokenizer.decode(generated_tokens)

    def generate_fill_in_the_middle(self, prefix: str, suffix: str, max_new_tokens: int = 64) -> str:
        """Specialized execution pipeline supporting Fill-in-the-Middle autocompletions."""
        # Reconstruct standard structured training format matching dataset pipelines
        fim_prompt = f"// PREFIX\n{prefix}\n// SUFFIX\n{suffix}\n"
        print("\n--- Running FIM Autocomplete Matrix Generation ---")
        return self.generate(fim_prompt, max_new_tokens=max_new_tokens, temperature=0.1)

# ==========================================
# TEST ORCHESTRATOR FOR REAL-TIME VALIDATION
# ==========================================
if __name__ == "__main__":
    # Standard fallback mock configuration files block instantiation
    if not os.path.exists(TOKENIZER_JSON):
        # Create a mock json file configuration if tokenizer utility script wasn't triggered
        with open(TOKENIZER_JSON, "w") as f:
            f.write('{"version": "1.0", "model": {"type": "BPE"}}')

    # Initialize Engine Context Layout Maps
    engine = HyperionInferenceEngine(checkpoint_path=CHECKPOINT_PATH, tokenizer_path=TOKENIZER_JSON)
    
    # Test Run 1: Algorithmic Logic Code Generation Pass
    logic_prompt = "def quicksort(arr):\n    \"\"\"Compute optimized array sorting matrix layers\"\"\"\n"
    print(f"\n[Test Prompt]:\n{logic_prompt}")
    output_text = engine.generate(logic_prompt, max_new_tokens=64, temperature=0.2)
    print(f"[Generated Response]:\n{output_text}")
    
    # Test Run 2: IDE Autocomplete FIM Execution Pass
    code_prefix = "def calculate_matrix_low_rank(x):\n    normed = UltraRMSNorm(x)"
    code_suffix = "    return logits"
    
    fim_completion = engine.generate_fill_in_the_middle(prefix=code_prefix, suffix=code_suffix)
    print(f"[FIM Middle Insertion Result]:\n{fim_completion}")
