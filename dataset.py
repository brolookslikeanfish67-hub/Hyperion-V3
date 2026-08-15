"""
Hyperion-V3: Memory-mapped dataset pipeline.
"""
import os
import random
import torch
from torch.utils.data import Dataset

class HyperionDataset(Dataset):
    def __init__(
        self, 
        txt_path: str, 
        tokenizer, 
        max_seq_len: int = 2048
    ):
        self.max_seq_len = max_seq_len
        
        # Auto-create fallback data files if missing
        if not os.path.exists(txt_path):
            self._generate_fallback_data(txt_path)
            
        print(f"[Dataset] Loading tokens: {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        # Tokenize text string directly to index arrays
        self.tokens = tokenizer.encode(raw_text)
        print(f"[Dataset] Total tokens: {len(self.tokens)}")

    def _generate_fallback_data(self, path: str):
        """Synthesizes sample algorithmic matrices."""
        print(f"[!] File missing. Building template: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Deep logic string blocks to train thinking tracks
        samples = [
            "def sort_array(arr):\n    return sorted(arr)\n",
            "def dot_product(a, b):\n    return sum(x*y for x,y in zip(a,b))\n",
            "// PREFIX\ndef clear_cache():\n// SUFFIX\n    pass\n",
            "<|thinking|>\nOptimizing matrix lora nodes\n</|thinking|>\n"
        ]
        
        # Multiply blocks to build a solid entry binary file size
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(500):
                f.write(random.choice(samples) + "\n<|endoftext|>\n")

    def __len__(self) -> int:
        # Calculate maximum block boundaries safely
        return max(0, len(self.tokens) - self.max_seq_len - 1)

    def __getitem__(self, idx: int):
        # Extract sequence slices and target labels safely
        x = self.tokens[idx : idx + self.max_seq_len]
        y = self.tokens[idx + 1 : idx + self.max_seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
