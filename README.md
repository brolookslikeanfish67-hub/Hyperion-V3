# Hyperion-V3: Horizon-Class Coding Core Engine

Hyperion-V3 is an autonomous, high-performance **Mixture-of-Experts (MoE) Language Model** custom-engineered for elite source code understanding, multi-turn logical reasoning, and real-time code autocomplete tasks. It replaces standard Transformer topologies with advanced inference-acceleration layers.

##  Advanced Architectural Upgrades

*   **Adaptive Thinking Recurrent Loops:** A dynamic, entropy-gated controller block adjusts computation based on complexity, allowing deeper, iterative processing.
*   **Low-Rank Key-Value Latent Attention (Hyper-MLA):** Compresses KV cache states into a low-rank manifold, reducing VRAM usage by over 90% for a **64K maximum context window**.
*   **Self-Correction Feedback Invariants:** Uses a non-linear transformation gate to act as a logical filter, suppressing hallucinations.
*   **Speculative Background OS Streaming:** Asynchronous threads leverage `np.memmap` for zero-stall data ingestion.
*   **Native Fill-in-the-Middle (FIM) Augmentation:** Implements `Prefix-Suffix-Middle` data pipelines to train on code completion tasks.

---

##  Repository Blueprint

*   `model.py`: Core architecture with Hyper-MLA, adaptive routing, and self-correction filters.
*   `tokenizer_utils.py`: Byte-fallback parser for secure, high-precision tokenization.
*   `dataset.py`: Memory-mapped streaming with asynchronous FIM lookahead.
*   `train.py`: High-throughput training loop with dynamic micro-step gradient accumulation.
*   `generate.py`: Dual-mode interface for text generation and FIM code insertion.

---

##  Getting Started & Installation

### 1. Install System Dependencies
Ensure Python 3.10+ and PyTorch are installed:

```bash
pip install torch numpy tokenizers
```

### 2. Dataset Environment
Prepare a `data/` directory with `train.txt` and `val.txt` files.

---

##  How To Run

1.  **Tokenizer:** `python tokenizer_utils.py` to create the vocabulary map.
2.  **Training:** `python train.py` to launch high-throughput, mixed-precision training.
3.  **Generation:** `python generate.py` to trigger inference, text generation, or FIM code insertion tests.

---

##  License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.
