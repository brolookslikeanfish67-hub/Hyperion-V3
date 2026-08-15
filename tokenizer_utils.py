"""
Hyperion-V3: Tokenizer Utilities Module.
Calibrated for a 32,000 vocabulary size limit.
"""
import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as BLPost

# --- CORE TOKENIZER CONFIG DATA ---
TOKENIZER_JSON = "hyperion_tokenizer.json"
VOCAB_SIZE = 32000

# Critical structural logic tokens for MoE and FIM
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|thinking|>",
    "</|thinking|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>"
]

def build_and_train_tokenizer(
    source_txt: str = "data/train.txt"
):
    """
    Trains a custom Byte-Level BPE tokenizer
    calibrated to match Hyperion's 1.11B matrix space.
    """
    print(f"[Tokenizer] Training on: {source_txt}")
    
    if not os.path.exists(source_txt):
        print("[-] Error: Source training data missing.")
        return None

    # Initialize a clean Byte-Fallback BPE model
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    
    # Configure Byte-Level character splits
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )
    
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet()
    )
    
    # Train directly from your local text matrices
    tokenizer.train([source_txt], trainer)
    
    # Enable post-processing for clean serialization
    tokenizer.post_processor = BLPost(
        trim_offsets=False
    )
    
    tokenizer.save(TOKENIZER_JSON)
    print(f"[✓] Tokenizer saved to: {TOKENIZER_JSON}")
    return tokenizer

def load_tokenizer(path: str = TOKENIZER_JSON):
    """Loads tokenizer from local JSON workspace."""
    if not os.path.exists(path):
        # Auto-train backup if configuration file is missing
        print("[!] Tokenizer config not found. Auto-building...")
        return build_and_train_tokenizer()
        
    return Tokenizer.from_file(path)

if __name__ == "__main__":
    # Create mock text if data directory doesn't exist
    os.makedirs("data", exist_ok=True)
    train_path = "data/train.txt"
    
    if not os.path.exists(train_path):
        with open(train_path, "w", encoding="utf-8") as out:
            out.write("def hello_world():\n    print('Hyperion')\n")
            
    build_and_train_tokenizer(train_path)
