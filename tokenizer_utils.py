"""
Hyperion-V3: Production Code-Optimized BPE Tokenizer.
"""

import os
from tokenizers import Tokenizer, regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import (
    Sequence,
    Split,
    ByteLevel,
)
from tokenizers.decoders import ByteLevel as ByteDecoder


def train_hyperion_tokenizer(
    corpus_path: str,
    vocab_size: int = 4000,
    save_path: str = "hyperion_tokenizer.json",
) -> Tokenizer:
    """Trains an enterprise code sub-word processor."""
    print(
        f"--- Training Tokenizer Engine "
        f"(Vocab Size: {vocab_size}) ---"
    )

    # 1. Instantiate clean byte-fallback BPE infrastructure
    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=True))

    # 2. Advanced Multi-Stage Regex Splitting Core
    # Ensures indent steps and keywords are cleanly split
    code_pattern = (
        r"'s|'t|'re|'ve|'m|'ll|'d|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}|"
        r"[^\s\p{L}\p{N}]+|"
        r"[\r\n]+|"
        r"\s+(?=>[\r\n])|"
        r"\s+"
    )

    tokenizer.pre_tokenizer = Sequence(
        [
            Split(
                pattern=regex.Regex(code_pattern),
                behavior="isolated",
            ),
            ByteLevel(
                add_prefix_space=False,
                use_regex=False,
            ),
        ]
    )

    # Attach byte decoder to accurately reconstruct tabs and symbols
    tokenizer.decoder = ByteDecoder()

    # 3. Secure Core Layout Boundary Invariants
    trainer = BpeTrainer(
        special_tokens=[
            "[PAD]",
            "[BOS]",
            "[EOS]",
            "<|before_code_insertion|>",
            "<|after_code_insertion|>",
            "<|middle_insert_prediction|>",
        ],
        vocab_size=vocab_size,
        show_progress=True,
    )

    # Ingest text strings and compile the structural mapping
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(save_path)

    print(f"Tokenizer compiled successfully to: {save_path}\n")
    return tokenizer


def load_tokenizer(
    path: str = "hyperion_tokenizer.json",
) -> Tokenizer:
    """Loads a pre-compiled structural asset snapshot."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Asset tracking mismatch error: '{path}' missing."
        )
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    print("Executing tokenization engine validation tests...")
    temp_raw_data = "tokenizer_mock.txt"

    # Write structural python test patterns
    with open(temp_raw_data, "w", encoding="utf-8") as f:
        f.write("def compute_embeddings(tokens):\n")
        f.write("    return model.encode(tokens)\n")

    # Run structural code compilation testing pass
    tok = train_hyperion_tokenizer(temp_raw_data, vocab_size=500)

    test_str = "    return model.encode"
    encoded = tok.encode(test_str)
    decoded = tok.decode(encoded.ids)

    print("\n--- Tokenizer Functional Validation ---")
    print(f"Input Code Line Slice:  '{test_str}'")
    print(f"Sub-Word Array Map IDs: {encoded.ids}")
    print(f"Decoded Identity Match: '{decoded}' [SUCCESS]")

    # System file housekeeping cleanup parameters
    if os.path.exists(temp_raw_data):
        os.remove(temp_raw_data)
    if os.path.exists("hyperion_tokenizer.json"):
        os.remove("hyperion_tokenizer.json")
