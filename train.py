"""
Hyperion-V3: Adaptive Training Pipeline Engine.
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project dependency imports
from model import (
    HyperionV3,
    HyperionV3Config,
)
from tokenizer_utils import load_tokenizer
from dataset import HyperionDataset

# ==========================================
# 1. HORIZON PRE-TRAINING ENGINE VALUES
# ==========================================
DATA_DIR = "data"
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT = os.path.join(DATA_DIR, "val.txt")
TOKENIZER_JSON = "hyperion_tokenizer.json"
CHECKPOINT_DIR = "checkpoints"
BEST_PATH = os.path.join(
    CHECKPOINT_DIR, "hyperion_v3_best.pt"
)

# Core Execution Mechanics
BATCH_SIZE = 8
INITIAL_GRAD_ACCUM = 4
MAX_STEPS = 10000
VAL_INTERVAL = 250
SAVE_INTERVAL = 1000

# Optimization Parameter Manifolds
LEARNING_RATE = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.1
MAX_GRAD_CLIP = 1.0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# 2. OPTIMIZER WEIGHT DECAY MANIFOLD DECOUPLING
# ==========================================
def configure_optimizers(
    model: nn.Module, lr: float, wd: float
) -> torch.optim.Optimizer:
    """Decouples regularized weight groups."""
    decay = set()
    no_decay = set()
    whitelist = (nn.Linear,)
    blacklist = (nn.Embedding, UltraRMSNorm, nn.Parameter)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(
            recurse=False
        ):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(
                m, whitelist
            ):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(
                m, blacklist
            ):
                no_decay.add(fpn)

    param_dict = {
        pn: p for pn, p in model.named_parameters()
    }
    opt_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": wd,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    use_fused = (
        DEVICE == "cuda"
        and "fused"
        in torch.optim.AdamW.__init__.__code__.co_varnames
    )
    return torch.optim.AdamW(
        opt_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )

# ==========================================
# 3. DYNAMIC COSINE TRAINING LR SCHEDULE
# ==========================================
def get_adaptive_lr(
    step: int,
    max_steps: int,
    warmup: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """Calculates scaling learning targets."""
    if step < warmup:
        return base_lr * float(step) / float(max(1, warmup))
    if step > max_steps:
        return min_lr
    ratio = float(step - warmup) / float(
        max(1, max_steps - warmup)
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)

# ==========================================
# 4. EVALUATION SYSTEM MODULE
# ==========================================
@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    steps: int = 50,
) -> float:
    """Measures validation metrics across steps."""
    model.eval()
    total_loss = 0.0
    actual_iters = min(len(loader), steps)
    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float32
    )

    for idx, (x, y) in enumerate(loader):
        if idx >= actual_iters:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=dtype,
        ):
            _, loss = model(x, targets=y)
        total_loss += loss.item()

    model.train()
    return total_loss / max(1, actual_iters)

# ==========================================
# 5. CORE ORCHESTRATION PIPELINE
# ==========================================
def main():
    if not os.path.exists(
        TOKENIZER_JSON
    ) or not os.path.exists(TRAIN_TXT):
        print("Error: Missing text training data files.")
        return

    tokenizer = load_tokenizer(TOKENIZER_JSON)
    v3_config = HyperionV3Config()

    print("Configuring token dataset streams...")
    train_dataset = HyperionDataset(
        TRAIN_TXT,
        tokenizer,
        max_seq_len=v3_config.max_seq_len,
    )
    val_dataset = (
        HyperionDataset(
            VAL_TXT,
            tokenizer,
            max_seq_len=v3_config.max_seq_len,
        )
        if os.path.exists(VAL_TXT)
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        if val_dataset
        else None
    )

    model = HyperionV3(config=v3_config).to(DEVICE)
    optimizer = configure_optimizers(
        model, LEARNING_RATE, WEIGHT_DECAY
    )

    step = 0
    grad_accum_steps = INITIAL_GRAD_ACCUM
    best_val_loss = float("inf")
    start_time = time.time()
    dtype = (
        torch.bfloat16
        if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float32
    )

    print("\nInitiating pre-training execution...")
    model.train()

    while step < MAX_STEPS:
        for x_b, y_b in train_loader:
            if step >= MAX_STEPS:
                break

            x_b = x_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)

            current_lr = get_adaptive_lr(
                step,
                MAX_STEPS,
                WARMUP_STEPS,
                LEARNING_RATE,
                MIN_LR,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            with torch.amp.autocast(
                device_type="cuda" if "cuda" in DEVICE else "cpu",
                dtype=dtype,
            ):
                logits, loss = model(x_b, targets=y_b)
                loss = loss / grad_accum_steps

            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                total_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_CLIP
                )
                
                # Dynamic Scaling Architecture
                # If gradients spikes, expand steps to cushion variance
                if total_norm > MAX_GRAD_CLIP * 1.5:
                    grad_accum_steps = min(
                        32, grad_accum_steps * 2
                    )
                elif total_norm < MAX_GRAD_CLIP * 0.5:
                    grad_accum_steps = max(
                        INITIAL_GRAD_ACCUM, grad_accum_steps // 2
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % VAL_INTERVAL == 0 and step > 0:
                scaled_loss = loss.item() * grad_accum_steps
                if val_loader:
                    v_loss = evaluate_loss(
                        model, val_loader, DEVICE
                    )
                    print(
                        f"Step: {step:05d} | "
                        f"Train Loss: {scaled_loss:.4f} | "
                        f"Val Loss: {v_loss:.4f} | "
                        f"Accum Steps: {grad_accum_steps}"
                    )
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        torch.save(
                            {
                                "step": step,
                                "model_state_dict": model.state_dict(),
                                "loss": v_loss,
                            },
                            BEST_PATH,
                        )
                else:
                    print(
                        f"Step: {step:05d} | "
                        f"Train Loss: {scaled_loss:.4f} | "
                        f"Accum Steps: {grad_accum_steps}"
                    )

            if step % SAVE_INTERVAL == 0 and step > 0:
                p_path = os.path.join(
                    CHECKPOINT_DIR, f"hyperion_v3_{step}.pt"
                )
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    p_path,
                )

            step += 1

    print(
        f"\nPipeline finished in "
        f"{(time.time()-start_time)/60:.2f} mins."
    )

if __name__ == "__main__":
    main()
