# Hyperion-V3 Engine
 **Autonomous 1.11B MoE Core**

---

Hyperion-V3 is a local MoE model.
It features **Hyper-MLA** attention.
It drives a **64K Context Window**.
It uses under 6GB VRAM.

---

##  Architecture

### 1. Hyper-MLA
Key-Value caches are compressed.
States project into a 192 rank.
This cuts VRAM usage by over 90%.

### 2. Vectorized MoE
Routing runs via matrix stacks.
It runs 64 experts at once.
It uses 4 active experts per token.
It avoids slow serial loops.

### 3. Adaptive Loops
Each block uses an inner step gate.
If complexity scales past 0.83,
the token cycles again.
It loops dynamically up to 4x.

---

##  Repository Blueprint

* `data/train.txt` : Train data.
* `data/val.txt` : Validation loop.
* `model.py` : 1.11B MoE code.
* `tokenizer_utils.py` : 32K BPE.
* `dataset.py` : FIM injector.
* `train.py` : Training loop.
* `bench.py` : VRAM profiler.
* `generate.py` : Inference code.

---

##  Quick Start

Install dependencies:
```bash
pip install torch numpy tokenizers
```

### Run Sequence:

1. Ingest Data:
```bash
python dataset.py
```

2. Compile Vocab:
```bash
python tokenizer_utils.py
```

3. Profile VRAM:
```bash
python bench.py
```

4. Run Training:
```bash
python train.py
```

---

##  Specifications

* **Total Size:** ~1.11B Parameters.
* **Active Size:** ~284M per token.
* **Precision:** Native bfloat16.
* **License:** GNU GPL-3.0.
