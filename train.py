"""
Hyperion-V3: High-Throughput Adaptive Training Pipeline Engine.
Optimized for 1.11B Parameter MoE mixed-precision code pre-training.
"""
import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project dependency imports
from model import HyperionV3, HyperionV3Config
from tokenizer_utils import load_tokenizer
from dataset import HyperionDataset

try:
    from model import UltraRMSNorm
except ImportError:
    UltraRMSNorm = None

# ==========================================
# 1. ARCHITECTURAL & TRAINING CORE CONFIG
# ==========================================
DATA_DIR = "data"
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT = os.path.join(DATA_DIR, "val.txt")
TOKENIZER_JSON = "hyperion_tokenizer.json"
CHECKPOINT_DIR = "checkpoints"
BEST_PATH = os.path.join(CHECKPOINT_DIR, "hyperion_v3_best.pt")

# Memory Optimization Settings
BATCH_SIZE = 8
INITIAL_GRAD_ACCUM = 4
MAX_STEPS = 10000
VAL_INTERVAL = 250
SAVE_INTERVAL = 1000

# Learning Schedule Configuration
LEARNING_RATE = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.1
MAX_GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# 2. OPTIMIZER WEIGHT DECAY CONFIGURATION
# ==========================================
def configure_optimizers(
    model: nn.Module, lr: float, wd: float
) -> torch.optim.Optimizer:
    """
    Decouples regularized weight groups. Ensures Embeddings, Biases, 
    and Normalization layers do not undergo weight decay.
    """
    decay_params = []
    no_decay_params = []
    
    blacklist_classes = (nn.Embedding, nn.LayerNorm, nn.RMSNorm)
    if UltraRMSNorm is not None:
        blacklist_classes += (UltraRMSNorm,)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            # Filter by dimension instead of unstable string parsing
            if pn.endswith("bias") or p.dim() < 2 or isinstance(m, blacklist_classes):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

    opt_groups = [
        {"params": decay_params, "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    # Safe checks for modern fused AdamW optimization routines
    has_fused = hasattr(torch.optim.AdamW, "register_profile")
    var_names = torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = DEVICE == "cuda" and (has_fused or "fused" in var_names)
    
    print(
        f"[Optimizer] Decay groups: {len(decay_params)} tensors | "
        f"No-Decay groups: {len(no_decay_params)} tensors"
    )
    return torch.optim.AdamW(
        opt_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
    )

# ==========================================
# 3. DYNAMIC COSINE TRAINING LR SCHEDULE
# ==========================================
def get_adaptive_lr(
    step: int, max_steps: int, warmup: int, base_lr: float, min_lr: float
) -> float:
    """Calculates scaling learning targets using Cosine Decay with Warmup."""
    if step < warmup:
        return base_lr * float(step) / float(max(1, warmup))
    if step > max_steps:
        return min_lr
    ratio = float(step - warmup) / float(max(1, max_steps - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)

# ==========================================
# 4. EVALUATION SYSTEM MODULE
# ==========================================
@torch.no_grad()
def evaluate_loss(
    model: nn.Module, loader: DataLoader, device: str, steps: int = 50
) -> float:
    """Measures validation metrics across steps cleanly in eval mode."""
    model.eval()
    total_loss = 0.0
    actual_iters = min(len(loader), steps)
    
    has_bf16 = torch.cuda.is_bf16_supported() if device == "cuda" else False
    dtype = torch.bfloat16 if has_bf16 else torch.float32
    device_type = "cuda" if "cuda" in device else "cpu"
    
    for idx, (x, y) in enumerate(loader):
        if idx >= actual_iters:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            _, loss = model(x, targets=y)
            total_loss += loss.item()
            
    model.train()
    return total_loss / max(1, actual_iters)

# ==========================================
# 5. CORE ORCHESTRATION PIPELINE
# ==========================================
def main():
    if not os.path.exists(TOKENIZER_JSON) or not os.path.exists(TRAIN_TXT):
        print("Error: Missing tokenizer or training data files.")
        return

    tokenizer = load_tokenizer(TOKENIZER_JSON)
    v3_config = HyperionV3Config()
    
    print("[Pipeline] Configuring memory-mapped token dataset streams...")
    train_ds = HyperionDataset(
        TRAIN_TXT, tokenizer, max_seq_len=v3_config.max_seq_len
    )
    val_ds = HyperionDataset(
        VAL_TXT, tokenizer, max_seq_len=v3_config.max_seq_len
    ) if os.path.exists(VAL_TXT) else None

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, pin_memory=True
    ) if val_ds else None

    model = HyperionV3(config=v3_config).to(DEVICE)
    optimizer = configure_optimizers(model, LEARNING_RATE, WEIGHT_DECAY)
    
    # In case the machine needs a float16 fallback gradient scaler
    use_scaler = DEVICE == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    
    step = 0
    grad_accum_steps = INITIAL_GRAD_ACCUM
    best_val_loss = float("inf")
    start_time = time.time()
    
    has_bf16 = torch.cuda.is_bf16_supported() if DEVICE == "cuda" else False
    dtype = torch.bfloat16 if has_bf16 else torch.float32
    device_type = "cuda" if "cuda" in DEVICE else "cpu"

    print(f"\n[Pipeline] Running pre-training loop on device: {DEVICE.upper()}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    accumulated_loss = 0.0
    micro_step_count = 0

    while step < MAX_STEPS:
        for x_b, y_b in train_loader:
            if step >= MAX_STEPS:
                break
                
            x_b = x_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                logits, loss = model(x_b, targets=y_b)
                loss = loss / grad_accum_steps
            
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulated_loss += loss.item()
            micro_step_count += 1
            
            if micro_step_count == grad_accum_steps:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_CLIP)
                
                # Dynamic Stability Tuning
                if total_norm > MAX_GRAD_CLIP * 1.5:
                    grad_accum_steps = min(32, grad_accum_steps * 2)
                elif total_norm < MAX_GRAD_CLIP * 0.5:
                    grad_accum_steps = max(INITIAL_GRAD_ACCUM, grad_accum_steps // 2)

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad(set_to_none=True)
                
                current_lr = get_adaptive_lr(
                    step, MAX_STEPS, WARMUP_STEPS, LEARNING_RATE, MIN_LR
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                # Evaluation Log Invariants
                if step % VAL_INTERVAL == 0 and step > 0:
                    train_loss_print = accumulated_loss * (grad_accum_steps / micro_step_count)
                    if val_loader:
                        v_loss = evaluate_loss(model, val_loader, DEVICE)
                        print(
                            f"Step: {step:05d} | Train Loss: {train_loss_print:.4f} | "
                            f"Val Loss: {v_loss:.4f} | GradNorm: {total_norm:.2f} | "
                            f"Accum: {grad_accum_steps} | LR: {current_lr:.2e}"
                        )
                        
                        if v_loss < best_val_loss:
                            best_val_loss = v_loss
                            torch.save({
                                "step": step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": v_loss,
                            }, BEST_PATH)
                    else:
                        print(
                            f"Step: {step:05d} | Train Loss: {train_loss_print:.4f} | "
                            f"GradNorm: {total_norm:.2f} | Accum: {grad_accum_steps} | "
                            f"LR: {current_lr:.2e}"
                        )

                if step % SAVE_INTERVAL == 0 and step > 0:
                    p_path = os.path.join(CHECKPOINT_DIR, f"hyperion_v3_{step}.pt")
                    torch.save({"model_state_dict": model.state_dict()}, p_path)

                # Reset phase flags
                step += 1
                accumulated_loss = 0.0
                micro_step_count = 0

    print(f"\n[Pipeline Finished] Duration: {(time.time() - start_time) / 60:.2f} mins.")

if __name__ == "__main__":
    main()
