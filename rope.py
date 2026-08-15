"""
Hyperion-V3: Rotary Positional Embeddings.
"""
import torch
import torch.nn as nn

class HyperionRoPE(nn.Module):
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 65536, 
        base: float = 10000.0
    ):
        super().__init__()
        # dim must be head_dim
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Calculate angular frequencies matrix
        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0, dim, 2
                ).float() / dim
            )
        )
        self.register_buffer(
            "inv_freq", inv_freq, persistent=False
        )
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(
            seq_len, 
            dtype=self.inv_freq.dtype, 
            device=self.inv_freq.device
        )
        # Vectorized outer product space
        freqss = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqss, freqss), dim=-1)
        
        self.register_buffer(
            "cos_cached", 
            emb.cos(), 
            persistent=False
        )
        self.register_buffer(
            "sin_cached", 
            emb.sin(), 
            persistent=False
        )

    def _rotate_half(self, x: torch.Tensor):
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ):
        # x shape: [B, H, T, HeadDim]
        cos = self.cos_cached[:seq_len, :].to(
            x.dtype
        )
        sin = self.sin_cached[:seq_len, :].to(
            x.dtype
        )
        
        # Expand dims for correct broadcast operations
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        
        r_x = self._rotate_half(x)
        return (x * cos) + (r_x * sin)
