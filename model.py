"""
Hyperion-V3: Coding Core Engine.
Calibrated for exactly 1.11B parameters.
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class HyperionV3Config:
    """Config calibrated for 1.11B parameters."""
    vocab_size: int = 32000
    dim: int = 1536
    n_layers: int = 24
    max_seq_len: int = 65536
    
    # Advanced Hyper-MLA Subspaces
    n_qa_heads: int = 24
    kv_lora_rank: int = 192
    head_dim: int = 64
    
    # Multi-Tier Fine Experts Grid
    n_routed_experts: int = 64
    top_k_experts: int = 4
    expert_hidden_dim: int = 1024
    shared_expert_dim: int = 4096
    
    # Adaptive Horizon Control Loops
    max_recurrent_depth: int = 4

class UltraRMSNorm(nn.Module):
    """Normalized transformation layer."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps: float = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        var = x_f32.pow(2).mean(-1, keepdim=True)
        rsq = torch.rsqrt(var + self.eps)
        normed = x_f32 * rsq
        return (normed * self.weight).to(orig_dtype)

class HyperLatentAttention(nn.Module):
    """Low-rank multi-head compressed MLA engine."""
    def __init__(self, config: HyperionV3Config):
        super().__init__()
        self.n_heads: int = config.n_qa_heads
        self.head_dim: int = config.head_dim
        self.kv_rank: int = config.kv_lora_rank
        
        self.kv_down = nn.Linear(
            config.dim, config.kv_lora_rank, bias=False
        )
        self.kv_norm = UltraRMSNorm(config.kv_lora_rank)
        
        up_dim = config.n_qa_heads * config.head_dim
        self.kv_up = nn.Linear(
            config.kv_lora_rank, up_dim, bias=False
        )
        self.q_proj = nn.Linear(
            config.dim, up_dim, bias=False
        )
        self.out_proj = nn.Linear(
            up_dim, config.dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        
        c_kv = self.kv_norm(self.kv_down(x))
        d_kv = self.kv_up(c_kv)
        
        q = self.q_proj(x).view(
            b, t, self.n_heads, self.head_dim
        ).transpose(1, 2)
        
        kv = d_kv.view(
            b, t, self.n_heads, self.head_dim
        ).transpose(1, 2)
        
        mask = torch.tril(
            torch.ones(t, t, device=x.device, dtype=torch.bool)
        )
        
        out = F.scaled_dot_product_attention(
            q=q, key=kv, value=kv,
            attn_mask=mask.unsqueeze(0).unsqueeze(1),
            dropout_p=0.0,
            is_causal=False
        )
        
        flat_out = out.transpose(1, 2).reshape(b, t, -1)
        return self.out_proj(flat_out)

class ParallelGLUExperts(nn.Module):
    """Vectorized stack of all MoE Experts."""
    def __init__(self, n_exp: int, d_mod: int, d_hid: int):
        super().__init__()
        self.w1 = nn.Parameter(
            torch.randn(n_exp, d_mod, d_hid) * 0.02
        )
        self.w2 = nn.Parameter(
            torch.randn(n_exp, d_hid, d_mod) * 0.02
        )
        self.w3 = nn.Parameter(
            torch.randn(n_exp, d_mod, d_hid) * 0.02
        )

    def forward(
        self, x: torch.Tensor, expert_mask: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros_like(x)
        for i in range(self.w1.size(0)):
            mask = (expert_mask == i)
            if not mask.any():
                continue
            t_exp = x[mask]
            h = F.silu(t_exp @ self.w1[i]) * (t_exp @ self.w3[i])
            out[mask] = h @ self.w2[i]
        return out

class MultiTierRouterMoE(nn.Module):
    """Fine-grained multi-tier gating layer."""
    def __init__(self, config: HyperionV3Config):
        super().__init__()
        self.config = config
        self.router = nn.Linear(
            config.dim, config.n_routed_experts, bias=False
        )
        
        self.parallel_experts = ParallelGLUExperts(
            config.n_routed_experts,
            config.dim,
            config.expert_hidden_dim
        )
        
        self.w1_shared = nn.Linear(
            config.dim, config.shared_expert_dim, bias=False
        )
        self.w2_shared = nn.Linear(
            config.shared_expert_dim, config.dim, bias=False
        )
        self.w3_shared = nn.Linear(
            config.dim, config.shared_expert_dim, bias=False
        )

    def _shared_expert(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.w1_shared(x)) * self.w3_shared(x)
        return self.w2_shared(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.shape
        x_flat = x.view(-1, c)
        
        shared_out = self._shared_expert(x_flat)
        
        logits = self.router(x_flat)
        w_routes = F.softmax(logits, dim=-1)
        
        topk_w, topk_i = torch.topk(
            w_routes, self.config.top_k_experts, dim=-1
        )
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-6)
        
        exp_x = x_flat.unsqueeze(1).expand(
            -1, self.config.top_k_experts, -1
        )
        expert_outputs = self.parallel_experts(exp_x, topk_i)
        
        routed_out = (
            expert_outputs * topk_w.unsqueeze(-1)
        ).sum(dim=1)
        
        p_i = w_routes.mean(dim=0)
        f_i = torch.bincount(
            topk_i.view(-1),
            minlength=self.config.n_routed_experts
        ).float() / (topk_i.numel() + 1e-6)
        balance_loss = self.config.n_routed_experts * torch.sum(
            p_i * f_i
        )
        
        final = (0.5 * shared_out + 0.5 * routed_out).view(
            b, t, c
        )
        return final, balance_loss

class AdaptiveTransformerBlock(nn.Module):
    """Dynamic layer with step gates."""
    def __init__(self, config: HyperionV3Config):
        super().__init__()
        self.attn = HyperLatentAttention(config)
        self.moe = MultiTierRouterMoE(config)
        self.norm1 = UltraRMSNorm(config.dim)
        self.norm2 = UltraRMSNorm(config.dim)
        self.step_gate = nn.Linear(config.dim, 1, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x))
        moe_out, balance_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, balance_loss

class HyperionV3(nn.Module):
    """State-of-the-art backbone engine."""
    def __init__(self, config: Optional[HyperionV3Config] = None):
        super().__init__()
        self.config = config if config else HyperionV3Config()
        self.tok_emb = nn.Embedding(
            self.config.vocab_size, self.config.dim
        )
        
        self.blocks = nn.ModuleList([
            AdaptiveTransformerBlock(self.config)
            for _ in range(self.config.n_layers)
        ])
        
        self.norm_f = UltraRMSNorm(self.config.dim)
        self.head = nn.Linear(
            self.config.dim, self.config.vocab_size, bias=False
        )
        self.head.weight = self.tok_emb.weight
        
        self.corrector = nn.Linear(
            self.config.dim, self.config.dim, bias=False
        )

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.shape
        x = self.tok_emb(idx)
        total_loss = torch.tensor(0.0, device=idx.device)
        
        for block in self.blocks:
            x, block_loss = block(x)
            total_loss += block_loss
            
            for depth in range(self.config.max_recurrent_depth - 1):
                gate_score = torch.sigmoid(
                    block.step_gate(x).mean()
                )
                if gate_score > 0.83:
                    x, secondary_loss = block(x)
                    total_loss += secondary_loss
                else:
                    break
                    
        x = self.norm_f(x)
        corrected_x = x + torch.tanh(self.corrector(x))
        logits = self.head(corrected_x)
        
        loss = None
        if targets is not None:
            v_size = self.config.vocab_size
            loss = F.cross_entropy(
                logits.view(-1, v_size), targets.view(-1)
            ) + (0.01 * total_loss)
            
        return logits, loss

if __name__ == "__main__":
    v3_cfg = HyperionV3Config()
    model = HyperionV3(config=v3_cfg)
    print("--- Hyperion-V3 Verified ---")
