import math

class TrainingDiagnostics:
    @staticmethod
    compute_token_perplexity = staticmethod(lambda loss_val: math.exp(min(loss_val, 20.0)))

    @staticmethod
    def analyze_router_routing(router_logits: torch.Tensor, topk_indices: torch.Tensor, num_experts: int) -> Dict[str, float]:
        """
        Computes Expert Utilization Entropy and Active Router Sparsity.
        """
        with torch.no_grad():
            # 1. Utilization Entropy
            expert_counts = torch.bincount(topk_indices.flatten(), minlength=num_experts).float()
            probs = expert_counts / expert_counts.sum()
            probs = probs[probs > 0] # Filter zero counts for log
            entropy = -torch.sum(probs * torch.log(probs)).item()
            max_entropy = math.log(num_experts)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # 2. Router Confidence / Sparsity (Softmax entropy across expert dimension)
            softmax_probs = F.softmax(router_logits, dim=-1)
            token_entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-9), dim=-1).mean().item()

            return {
                "expert_utilization_entropy": normalized_entropy,
                "router_prediction_entropy": token_entropy
            }
