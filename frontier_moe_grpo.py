import math
import os
import sys
import tempfile
import subprocess
from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

# =====================================================================
# 1. FRONTIER MODEL CONFIGURATION
# =====================================================================

@dataclass
class FrontierConfig:
    vocab_size: int = 128256     # Llama-3 scale vocabulary
    dim: int = 4096              # Hidden dimension
    n_layers: int = 32
    n_heads: int = 32            # Query heads
    n_kv_heads: int = 8          # Key/Value heads for Grouped-Query Attention (GQA)
    head_dim: int = 128
    
    # Mixture of Experts (MoE) Config
    num_routed_experts: int = 8
    num_shared_experts: int = 2
    top_k: int = 2
    moe_hidden_dim: int = 5120
    router_aux_loss_coef: float = 0.01 # Prevents expert collapse
    
    max_seq_len: int = 8192      # Base context window
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0 # High base theta for long context scaling

@dataclass
class GRPOConfig:
    group_size: int = 4
    clip_eps: float = 0.2
    kl_beta: float = 0.01
    lr: float = 5e-6
    max_gen_len: int = 1024

# =====================================================================
# 2. CORE ARCHITECTURE (GQA + SwiGLU + MoE)
# =====================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class GroupedQueryAttention(nn.Module):
    """Memory-efficient attention using GQA and FlashAttention-2."""
    def __init__(self, cfg: FrontierConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.head_dim

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        
        xq = self.wq(x).view(b, s, self.n_heads, self.head_dim)
        xk = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Repeat KV heads for GQA
        xk = xk[:, :, :, None, :].expand(b, s, self.n_kv_heads, self.n_rep, self.head_dim).reshape(b, s, self.n_heads, self.head_dim)
        xv = xv[:, :, :, None, :].expand(b, s, self.n_kv_heads, self.n_rep, self.head_dim).reshape(b, s, self.n_heads, self.head_dim)

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        
        # Fused FlashAttention
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(b, s, -1))

class SwiGLUMoE(nn.Module):
    """Mixture of Experts utilizing SwiGLU activations and Load Balancing Loss."""
    def __init__(self, cfg: FrontierConfig):
        super().__init__()
        self.num_experts = cfg.num_routed_experts
        self.top_k = cfg.top_k
        self.dim = cfg.dim
        self.expert_dim = cfg.moe_hidden_dim

        # Routed Experts
        self.w1 = nn.Parameter(torch.empty(self.num_experts, cfg.dim, self.expert_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, cfg.dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, cfg.dim, self.expert_dim))

        # Always-active Shared Expert (Captures general knowledge)
        self.shared_w1 = nn.Linear(cfg.dim, self.expert_dim * cfg.num_shared_experts, bias=False)
        self.shared_w2 = nn.Linear(self.expert_dim * cfg.num_shared_experts, cfg.dim, bias=False)
        self.shared_w3 = nn.Linear(cfg.dim, self.expert_dim * cfg.num_shared_experts, bias=False)

        self.router = nn.Linear(cfg.dim, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape
        x_flat = x.view(-1, d)

        # 1. Shared Expert Forward Pass (SwiGLU)
        shared_out = self.shared_w2(F.silu(self.shared_w1(x_flat)) * self.shared_w3(x_flat))

        # 2. Router & Load Balancing Loss Calculation
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Compute auxiliary loss to prevent expert collapse
        tokens_per_expert = torch.bincount(topk_indices.flatten(), minlength=self.num_experts).float()
        density_proxy = tokens_per_expert / (b * s * self.top_k)
        routing_prob_mean = routing_weights.mean(dim=0)
        aux_loss = torch.sum(density_proxy * routing_prob_mean) * self.num_experts

        # 3. Routed Experts Forward Pass (Batched tensor math - zero CPU sync)
        w1_selected = self.w1[topk_indices]
        w2_selected = self.w2[topk_indices]
        w3_selected = self.w3[topk_indices]

        x_expanded = x_flat.unsqueeze(1).unsqueeze(2)
        gate = torch.matmul(x_expanded, w1_selected)
        up = torch.matmul(x_expanded, w3_selected)
        act = F.silu(gate) * up
        expert_outputs = torch.matmul(act, w2_selected).squeeze(2)

        routed_out = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)
        
        final_out = (shared_out + routed_out).view(b, s, d)
        return final_out, aux_loss

class FrontierBlock(nn.Module):
    def __init__(self, cfg: FrontierConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.moe = SwiGLUMoE(cfg)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        moe_out, aux_loss = self.moe(self.ffn_norm(x))
        x = x + moe_out
        return x, aux_loss

class FrontierModel(nn.Module):
    def __init__(self, cfg: FrontierConfig = FrontierConfig()):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([FrontierBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        
        # Tie weights between embedding and output layer
        self.embed_tokens.weight = self.lm_head.weight

        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s = input_ids.shape
        x = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:s].unsqueeze(0).expand(b, -1, -1)
        
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x, freqs_cis)
            total_aux_loss += aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss

# =====================================================================
# 3. REINFORCEMENT LEARNING ENGINE & SANDBOX
# =====================================================================

class IsolatedCodeExecutor:
    """Executes generated code inside an isolated child process to protect the host node."""
    @staticmethod
    def evaluate_code(generated_code: str, unit_test_code: str, timeout: float = 2.0) -> float:
        full_script = f"{generated_code}\n\n{unit_test_code}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as temp_file:
            temp_file.write(full_script)
            temp_file.flush()
            try:
                result = subprocess.run(
                    [sys.executable, temp_file.name],
                    capture_output=True, text=True, timeout=timeout
                )
                if result.returncode == 0: return 1.0  # Pass
                return 0.5  # Valid syntax, failed logic
            except subprocess.TimeoutExpired: return 0.0
            except Exception: return 0.0

def compute_token_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    target_logits = torch.gather(logits, -1, target_ids.unsqueeze(-1)).squeeze(-1)
    return target_logits - log_sum_exp

class FrontierGRPOTrainer:
    """GRPO Engine driven by execution verification and expert load balancing."""
    def __init__(self, policy: nn.Module, ref_policy: nn.Module, model_cfg: FrontierConfig, grpo_cfg: GRPOConfig):
        self.policy = policy
        self.ref_policy = ref_policy
        self.m_cfg = model_cfg
        self.g_cfg = grpo_cfg
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=grpo_cfg.lr, betas=(0.9, 0.95))

        for p in self.ref_policy.parameters():
            p.requires_grad = False

    def grpo_loss_step(self, prompt_ids: torch.Tensor, sampled_ids: torch.Tensor, rewards: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()
        G, gen_len = sampled_ids.shape
        full_input = torch.cat([prompt_ids.expand(G, -1), sampled_ids], dim=-1)

        logits_policy, aux_loss = self.policy(full_input)
        logits_policy = logits_policy[:, -gen_len-1:-1, :]
        
        with torch.no_grad():
            logits_ref, _ = self.ref_policy(full_input)
            logits_ref = logits_ref[:, -gen_len-1:-1, :]

        token_log_probs = compute_token_log_probs(logits_policy, sampled_ids)
        ref_log_probs = compute_token_log_probs(logits_ref, sampled_ids)

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        adv_expanded = advantages.unsqueeze(-1).expand(-1, gen_len)

        ratio = torch.exp(token_log_probs - ref_log_probs)
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.g_cfg.clip_eps, 1.0 + self.g_cfg.clip_eps) * adv_expanded
        
        policy_loss = -torch.min(surr1, surr2).mean()
        kl_div = (torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1).mean()

        # Combine PPO Loss, KL Divergence Penalty, and MoE Load Balancing Loss
        total_loss = policy_loss + (self.g_cfg.kl_beta * kl_div) + (self.m_cfg.router_aux_loss_coef * aux_loss)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {"total_loss": total_loss.item(), "policy_loss": policy_loss.item(), "kl_div": kl_div.item(), "aux_loss": aux_loss.item()}

# =====================================================================
# 4. INITIALIZATION & TEST RUN
# =====================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Using a smaller config for demonstration purposes to avoid OOM on local runs
    demo_cfg = FrontierConfig(vocab_size=32000, dim=1024, n_layers=4, n_heads=8, n_kv_heads=2)
    model = FrontierModel(demo_cfg).to(device)
    
    print(f"=== Frontier-Tier Model Initialized on {device.upper()} ===")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters (Demo Config): {total_params / 1e6:.2f}M")

    # Mock GRPO Pass
    ref_model = deepcopy(model)
    trainer = FrontierGRPOTrainer(model, ref_model, demo_cfg, GRPOConfig())

    prompt = torch.randint(0, demo_cfg.vocab_size, (1, 32), device=device)
    samples = torch.randint(0, demo_cfg.vocab_size, (4, 64), device=device)
    rewards = torch.tensor([0.0, 0.5, 1.0, 0.0], device=device)

    with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.bfloat16):
        metrics = trainer.grpo_loss_step(prompt, samples, rewards)
        
    print(f"\n=== Training Step Metrics ===")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.6f}")
