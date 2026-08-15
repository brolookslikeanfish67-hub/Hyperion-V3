"""
Hyperion-V3: Native Procedural Algorithmic Dataset Factory.
Generates rigorous math, code tracing, and FIM arrays completely locally.
"""
import os
import random
from pathlib import Path

OUTPUT_DIR = Path("data")
TRAIN_FILE = OUTPUT_DIR / "train.txt"
VAL_FILE = OUTPUT_DIR / "val.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class NativeDatasetFactory:
    """Procedurally builds high-density logical inputs without external APIs."""

    @staticmethod
    def generate_matrix_puzzle() -> str:
        """Creates unique matrix multiplication algorithmic test targets."""
        r = random.randint(2, 5)
        c = random.randint(2, 5)
        matrix_a = [[random.randint(-10, 10) for _ in range(c)] for _ in range(r)]
        matrix_b = [[random.randint(-10, 10) for _ in range(r)] for _ in range(c)]
        
        prompt = (
            f"Task: Write a Python function `multiply()` to process these arrays:\n"
            f"A = {matrix_a}\nB = {matrix_b}\n"
            f"Verify your row-column alignments inside reasoning blocks."
        )
        # We output a blank reasoning shell. Your GRPO loop will force the 
        # model to fill this space with its own self-generated thinking traces!
        return f"<|reasoning_start|>\nPrompt: {prompt}\n<|thinking|>\n"

    @staticmethod
    def generate_fim_template() -> str:
        """Procedurally builds fill-in-the-middle structural syntax arrays."""
        structures = [
            ("def binary_search(arr, target):\n    low, high = 0, len(arr) - 1",
             "    return -1\n# End of search logic"),
            ("class Node:\n    def __init__(self, val):\n        self.val = val",
             "        self.right = None\n# End of structural graph initialization")
        ]
        prefix, suffix = random.choice(structures)
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

def main(total_samples: int = 5000):
    print(f" Procedurally generating {total_samples} local logical targets...")
    
    train_count, val_count = 0, 0
    
    with open(TRAIN_FILE, "w", encoding="utf-8") as f_train, \
         open(VAL_FILE, "w", encoding="utf-8") as f_val:
             
        for i in range(total_samples):
            # Alternate between math puzzles and fill-in-the-middle syntax templates
            if i % 2 == 0:
                sample = NativeDatasetFactory.generate_matrix_puzzle()
            else:
                sample = NativeDatasetFactory.generate_fim_template()
                
            # Append samples into 90/10 train/validation splits
            if i % 10 == 0:
                f_val.write(sample + "\n")
                val_count += 1
            else:
                f_train.write(sample + "\n")
                train_count += 1
                
    print(f" Setup complete! Created {train_count} train and {val_count} val samples natively.")

if __name__ == "__main__":
    main()
