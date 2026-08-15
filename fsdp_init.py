"""
Hyperion-V3: Multi-Node Fully Sharded Data Parallel (FSDP) Scaling Engine.
Features hybrid sharding, backward pre-fetching, and unified bfloat16 precision policies.
"""
import functools
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def initialize_fsdp_backbone(
    model: nn.Module, 
    block_class: nn.Module, 
    device_id: int
) -> FSDP:
    """
    Wraps standard model in FSDP with hybrid sharding, backward pre-fetching, 
    and hardware-native bfloat16 precision constraints.
    """
    # 1. Move base model parameters to target local GPU device BEFORE wrapping
    if torch.cuda.is_available():
        model.to(device_id)

    # 2. Define mixed precision policy targeting bfloat16 across all boundaries
    bf16_mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # 3. Establish auto-wrap execution policies based on layer boundaries
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={block_class},
    )

    # 4. Instantiate Fully Sharded Data Parallel graph wrapper
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # Intra-node shard, inter-node replicate
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device_id,
        sync_module_states=True,
    )
    
    return fsdp_model


# --- Distributed Test Validation Block ---
if __name__ == "__main__":
    # Simulate a distributed cluster rank context environment loop locally
    if torch.cuda.is_available():
        # Setup mock entities to verify wrapping logic
        class MockLayerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(512, 512)
            def forward(self, x):
                return self.linear(x)

        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([MockLayerBlock() for _ in range(2)])
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x

        print("🚀 Initializing mock model on GPU Lane 0...")
        raw_model = MockTransformer()
        
        # Test function call using local zero rank ID
        wrapped_fsdp_model = initialize_fsdp_backbone(
            model=raw_model,
            block_class=MockLayerBlock,
            device_id=0
        )
        print("✅ FSDP Wrapping Check Passed! Structural layers partitioned successfully.")
    else:
        print("🖥️ System running on CPU: Skipping distributed CUDA tensor validation loops.")
