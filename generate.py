"""
Hyperion-V3: Horizon-Class Generation Engine.
"""

import os
import torch
import torch.nn.functional as F

# Project dependency imports
from model import (
    HyperionV3,
    HyperionV3Config,
)
from tokenizer_utils import load_tokenizer

# ==========================================
# 1. RUNTIME GENERATION SETTINGS
# ==========================================
CHECKPOINT_PATH = os.path.join(
    "checkpoints", "hyperion_v3_best.pt"
)
TOKENIZER_JSON = "hyperion_tokenizer.json"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================
# 2. ADAPTIVE ENTROPY SAMPLING LAYER
# ==========================================
def sample_adaptive(
    logits: torch.Tensor,
    temp: float = 0.2,
    top_k: int = 40,
    top_p: float = 0.90,
) -> torch.Tensor:
    """Applies contextual temperature filtering."""
    # Contextual Smoothing: If model is highly 
    # confident, drop temp to freeze logical accuracy
    entropy = torch.distributions.Categorical(
        logits=logits
    ).entropy()
    
    if entropy.item() < 0.4:
        temp = 0.01  # Force deterministic code tokens

    if temp > 0.0:
        logits = logits / temp

    if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[..., -1, None]] = float("-inf")

    if top_p < 1.0:
        sorted_l, sorted_i = torch.sort(
            logits, descending=True
        )
        cum_probs = torch.cumsum(
            F.softmax(sorted_l, dim=-1), dim=-1
        )

        sorted_remove = cum_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[
            ..., :-1
        ].clone()
        sorted_remove[..., 0] = 0

        indices_to_remove = sorted_i[sorted_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# ==========================================
# 3. INTERACTIVE AUTOREGRESSIVE ENGINE
# ==========================================
@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temp: float = 0.4,
) -> str:
    """Generates fluid causal continuations."""
    model.eval()
    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(
        [encoded], dtype=torch.long, device=DEVICE
    )

    generated = []
    eos_id = tokenizer.token_to_id("[EOS]")

    for _ in range(max_tokens):
        context = input_ids[:, -1024:]
        logits, _ = model(context)
        next_logits = logits[0, -1, :]

        next_token = sample_adaptive(
            next_logits, temp=temp
        )
        t_id = next_token.item()

        if t_id == eos_id:
            break

        generated.append(t_id)
        input_ids = torch.cat(
            (
                input_ids,
                next_token.unsqueeze(0),
            ),
            dim=1,
        )

    return tokenizer.decode(generated)

# ==========================================
# 4. ADVANCED FILL-IN-THE-MIDDLE ENGINE
# ==========================================
@torch.no_grad()
def autocomplete_code(
    model: nn.Module,
    tokenizer,
    prefix: str,
    suffix: str,
    max_tokens: int = 64,
) -> str:
    """Fills missing logical code gaps."""
    model.eval()
    
    cfg = HyperionV3Config()
    t_pref = tokenizer.encode(prefix).ids
    t_suff = tokenizer.encode(suffix).ids

    # Construct the Copilot structural token array
    fim_tokens = (
        [cfg.bim_id]
        + t_pref
        + [cfg.aim_id]
        + t_suff
        + [cfg.mim_id]
    )
    input_ids = torch.tensor(
        [fim_tokens], dtype=torch.long, device=DEVICE
    )

    generated = []
    eos_id = tokenizer.token_to_id("[EOS]")

    for _ in range(max_tokens):
        logits, _ = model(input_ids[:, -2048:])
        next_logits = logits[0, -1, :]
        
        # Greedy fallback logic for raw absolute math code
        next_token = torch.argmax(next_logits, dim=-1)
        t_id = next_token.item()

        if t_id == eos_id:
            break

        generated.append(t_id)
        input_ids = torch.cat(
            (
                input_ids,
                next_token.view(1, 1),
            ),
            dim=1,
        )

    return tokenizer.decode(generated)

# ==========================================
# 5. DIAGNOSTIC RUNTIME MAIN ENTRANCE
# ==========================================
def main():
    if not os.path.exists(TOKENIZER_JSON):
        print("Error: Missing tokenizer file configuration.")
        return

    tokenizer = load_tokenizer(TOKENIZER_JSON)
    v3_cfg = HyperionV3Config()
    model = HyperionV3(config=v3_cfg).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from: {CHECKPOINT_PATH}")
        ckpt = torch.load(
            CHECKPOINT_PATH, map_location=DEVICE
        )
        model.load_state_dict(ckpt["model_state_dict"])
        print("Hyperion-V3 completely loaded.\n")
    else:
        print(" Operating under raw initialization profiles.\n")

    # Mode A: Standard Text Streaming Generation
    test_prompt = "def calculate_lr_decay(step):"
    print(f"Causal Prompt: '{test_prompt}'")
    completion = generate_text(
        model, tokenizer, test_prompt, max_tokens=30
    )
    print(f"Output: {test_prompt} {completion}\n")

    # Mode B: High-End Fill-In-The-Middle Insertion Autocomplete
    code_prefix = "def calculate_loss(x, y):\n"
    code_suffix = "\n    return final_loss"
    print("--- Running FIM Cursor Intercept ---")
    print(f"[Prefix]:\n{code_prefix}[Cursor Here]\n[Suffix]:{code_suffix}")
    
    middle_insertion = autocomplete_code(
        model, tokenizer, code_prefix, code_suffix
    )
    print(f"\n[Model Predicted Line]:\n{middle_insertion}")


if __name__ == "__main__":
    main()
