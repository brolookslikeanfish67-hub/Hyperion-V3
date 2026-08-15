# Hyperion-V3 Core
1.11B MoE Engine

Hyperion-V3 runs locally using
Hyper-MLA, a 64K Context Window,
and under 6GB VRAM.

## Step-by-Step Setup

1. Clone the Codebase:
```bash
git clone https://github.com/\
brolookslikeanfish67-hub/\
Hyperion-V3.git
cd Hyperion-V3
```

2. Install Math Modules:
```bash
pip install torch numpy tokenizers
```

3. Generate Code Files:
```bash
python dataset.py
```

4. Build Vocabulary Map:
```bash
python tokenizer_utils.py
```

5. Profile Local Memory:
```bash
python bench.py
```

6. Launch Training:
```bash
python train.py
```

7. Run Auto Sandbox:
```bash
python agent_executor.py
```

## Core Features

1. Hyper-MLA: Compresses KV caches
by 90% via 192-rank projection.
2. Vectorized MoE: Runs 4 of 64
experts in parallel.
3. Adaptive Loops: Dynamic token
re-processing up to 4x.
4. Compiler Swarm: Sandbox code
execution and debugging.

## Core Files
`model.py` (backbone)
`rope.py` (positional)
`train.py` (training)
`swarm_engine.py` (debugger)
`agent_executor.py` (tools)

## Specs
- Total Size: 1.11B Params
- Active Size: 284M Tokens
- Context Cap: 64K Tokens
- Precision: Native BF16
- License: GNU GPL-3.0
