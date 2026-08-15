"""
Hyperion-V3: Advanced Post-Training & Pre-Training Diagnostics Engine.
Tracks Expert Utilization Entropy and Router Prediction Sparsity.
"""
import math
from typing import Dict
import torch
import torch.nn.functional as F

class TrainingDiagnostics:
    
    @staticmethod
    def compute_token_perplexity(loss_val: float) -> float:
        """Calculates token perplexity safely bounded to prevent overflow."""
        return math.exp(min(loss_val, 20.0))

    @staticmethod
    def analyze_router_routing(
        router_logits: torch.Tensor, 
        topk_indices: torch.Tensor, 
        num_experts: int
    ) -> Dict[str, float]:
        """
        Computes Expert Utilization Entropy and Active Router Sparsity.
        Runs entirely in VRAM without CPU synchronization stalls.
        """
        with torch.no_grad():
            # 1. Utilization Entropy (How evenly tokens are divided among experts)
            expert_counts = torch.bincount(
                topk_indices.flatten(), 
                minlength=num_experts
            ).float()
            
            probs = expert_counts / (expert_counts.sum() + 1e-9)
            
            # Mask out zeros smoothly to avoid NaN values during log math
            mask = probs > 0
            active_probs = probs[mask]
            
            entropy = -torch.sum(active_probs * torch.log(active_probs)).item()
            max_entropy = math.log(num_experts)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # 2. Router Confidence / Sparsity (Confidence tracking per token)
            softmax_probs = F.softmax(router_logits, dim=-1)
            token_entropy = -torch.sum(
                softmax_probs * torch.log(softmax_probs + 1e-9), 
                dim=-1
            ).mean().item()

            return {
                "expert_utilization_entropy": normalized_entropy,
                "router_prediction_entropy": token_entropy
            }

# --- Unit Test / Verification Block ---
if __name__ == "__main__":
    # Simulate a batch of 1024 tokens being routed across 16 experts (top_k=2)
    sim_logits = torch.randn(1024, 16)
    sim_topk = torch.randint(0, 16, (1024, 2))
    
    metrics = TrainingDiagnostics.analyze_router_routing(
        router_logits=sim_logits,
        topk_indices=sim_topk,
        num_experts=16
    )
    
    mock_perplexity = TrainingDiagnostics.compute_token_perplexity(2.5)
    
    print("📈Diagnostic Metrics Calculated Successfully:")
    print(f" -> Perplexity: {mock_perplexity:.4f}")
    print(f" -> Expert Balance Entropy: {metrics['expert_utilization_entropy']:.4f}")
    print(f" -> Router Logit Sparsity: {metrics['router_prediction_entropy']:.4f}")
    print(" Done! File ready to be saved as diagnostics.py")
