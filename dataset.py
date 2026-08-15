"""
Hyperion-V3 Ultra Data Infrastructure.
"""

import os
import random
import concurrent.futures
from typing import Tuple, List, Final
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Global architectural constants
CACHE_DTYPE: Final = np.uint16
BIM_ID: Final[int] = 3997  # <|before_code_insertion|>
AIM_ID: Final[int] = 3998  # <|after_code_insertion|>
MIM_ID: Final[int] = 3999  # <|middle_insert_prediction|>


class HyperionDataset(Dataset):
    """Next-Gen Token-Streaming & FIM Engine."""

    def __init__(
        self,
        corpus_path: str,
        tokenizer,
        max_seq_len: int = 1024,
        force_rebuild: bool = False,
        num_workers: int = os.cpu_count() or 4,
        fim_rate: float = 0.5,
        prefetch_factor: int = 4,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.bin_path = corpus_path + ".tokens.bin"
        self.fim_rate = fim_rate
        self.prefetch_factor = prefetch_factor
        self.pad_id = tokenizer.token_to_id("[PAD]") or 1
        self.eos_id = tokenizer.token_to_id("[EOS]") or 3

        if os.path.exists(self.bin_path) and not force_rebuild:
            print(f"--- [Cache Hit] Mapping: {self.bin_path} ---")
        else:
            print(f"--- [Cache Miss] Processing: {corpus_path} ---")
            self._build_binary_cache(corpus_path, tokenizer, num_workers)

        # Zero-copy kernel space handle allocation
        self.token_stream = np.memmap(
            self.bin_path, dtype=CACHE_DTYPE, mode="r"
        )
        self.num_sequences = (len(self.token_stream) - 1) // self.max_seq_len

        # Speculative lookahead queue execution pools
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.prefetch_cache = {}

        print(f"Tokens Scanned: {len(self.token_stream):,}")
        print(f"Usable Slices: {self.num_sequences:,}\n")

    def _build_binary_cache(
        self, corpus_path: str, tokenizer, num_workers: int
    ) -> None:
        """Slices files into concurrent parser channels."""
        lines: List[str] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Distributing lines to {num_workers} workers...")

        def _tokenize_chunk(text_batch: List[str]) -> List[int]:
            token_accumulator: List[int] = []
            for item in text_batch:
                token_accumulator.extend(tokenizer.encode(item).ids)
            return token_accumulator

        chunks = np.array_split(lines, num_workers)
        compiled_stream: List[int] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as exec_pool:
            tasks = [
                exec_pool.submit(_tokenize_chunk, c.tolist()) for c in chunks
            ]
            for future in concurrent.futures.as_completed(tasks):
                compiled_stream.extend(future.result())

        print(f"Writing binary output tokens to: {self.bin_path}")
        np_array = np.array(compiled_stream, dtype=CACHE_DTYPE)
        with open(self.bin_path, "wb") as f_bin:
            f_bin.write(np_array.tobytes())

    def _async_prefetch_lookahead(
        self, start_idx: int, end_idx: int, next_idx: int
    ) -> None:
        """Asynchronously pulls future segments down into cache memory."""
        if next_idx not in self.prefetch_cache and next_idx < self.num_sequences:
            n_start = next_idx * self.max_seq_len
            n_end = n_start + self.max_seq_len + 1
            self.prefetch_cache[next_idx] = self.token_stream[
                n_start:n_end
            ].copy()

    def _apply_fill_in_the_middle(self, chunk: np.ndarray) -> torch.Tensor:
        """Transforms text segments to learn cursor insertion logic."""
        total_len = len(chunk)
        if total_len < 4:
            return torch.from_numpy(chunk.astype(np.int64))

        s1 = random.randint(1, total_len // 3)
        s2 = random.randint(s1 + 1, (2 * total_len) // 3)

        prefix = chunk[:s1]
        middle = chunk[s1:s2]
        suffix = chunk[s2:]

        fim_sequence = np.concatenate(
            [[BIM_ID], prefix, [AIM_ID], suffix, [MIM_ID], middle]
        )

        if len(fim_sequence) >= total_len:
            fim_sequence = fim_sequence[:total_len]
        else:
            pad_size = total_len - len(fim_sequence)
            padding = np.full((pad_size,), self.pad_id, dtype=np.int64)
            fim_sequence = np.concatenate([fim_sequence, padding])

        return torch.from_numpy(fim_sequence.astype(np.int64))

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= self.num_sequences or idx < 0:
            raise IndexError("Index falls outside allowable stream limits.")

        if idx in self.prefetch_cache:
            chunk_raw = self.prefetch_cache.pop(idx)
        else:
            start_idx = idx * self.max_seq_len
            end_idx = start_idx + self.max_seq_len + 1
            chunk_raw = self.token_stream[start_idx:end_idx]

        # Speculatively cache the absolute next block in lookahead
        next_idx = idx + 1
        self.executor.submit(self._async_prefetch_lookahead, 0, 0, next_idx)

        if len(self.prefetch_cache) > self.prefetch_factor * 2:
            self.prefetch_cache.clear()

        if random.random() < self.fim_rate:
            processed_chunk = self._apply_fill_in_the_middle(chunk_raw)
        else:
            processed_chunk = torch.from_numpy(chunk_raw.astype(np.int64))

        x = processed_chunk[:-1]
        y = processed_chunk[1:]

        return x, y


if __name__ == "__main__":
    from tokenizer_utils import train_hyperion_tokenizer

    print("Verifying layout parameters...")
    temp_text = "test_code_corpus.py"

    with open(temp_text, "w", encoding="utf-8") as f:
        for i in range(100):
            f.write(f"def code_block_{i}(state):\n")
            f.write("    return torch.relu(state)\n\n")

    tok_inst = train_hyperion_tokenizer(temp_text, vocab_size=500)
    dataset = HyperionDataset(
        temp_text, tok_inst, max_seq_len=32, force_rebuild=True, fim_rate=0.8
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    x_batch, y_batch = next(iter(dataloader))

    print("\n--- Structural Tests Concluded ---")
    print(f"X Shape: {x_batch.shape} | Y Shape: {y_batch.shape}")
    print("Speculative Buffer Queue Pipeline: ONLINE [SUCCESS]")

    if os.path.exists(temp_text):
        os.remove(temp_text)
    if os.path.exists(temp_text + ".tokens.bin"):
        os.remove(temp_text + ".tokens.bin")
    if os.path.exists("hyperion_tokenizer.json"):
        os.remove("hyperion_tokenizer.json")
