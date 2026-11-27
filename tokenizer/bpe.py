"""A small, educational byte-level Byte Pair Encoding tokenizer.

The tokenizer starts with the 256 possible byte values and learns frequent
adjacent-byte/token merges from a training corpus. It is lossless for UTF-8
text and therefore does not need an unknown token.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


class BPETokenizer:
    """Learn frequent adjacent byte-token merges from training text."""

    FORMAT_VERSION = 1
    BASE_VOCAB_SIZE = 256

    def __init__(
        self,
        vocab_size: int,
        base_encoding: str = "bytes",
        unknown_token: Optional[str] = None,
    ):
        """Initialize a byte vocabulary and empty merge table.

        Args:
            vocab_size: Desired total vocabulary size, including the 256
                single-byte tokens.
            base_encoding: Currently only UTF-8 byte encoding is supported.
            unknown_token: Not used for byte-level BPE; all byte values are
                already representable.
        """
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer.")
        if vocab_size < self.BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be at least {self.BASE_VOCAB_SIZE} for byte-level BPE."
            )
        if base_encoding != "bytes":
            raise ValueError("base_encoding must be 'bytes'.")
        if unknown_token is not None:
            raise ValueError(
                "Byte-level BPE does not need an unknown token; all 256 byte values are covered."
            )

        self.target_vocab_size = vocab_size
        self.base_encoding = base_encoding
        self.unknown_token = None
        self.vocab: list[bytes] = [bytes([value]) for value in range(256)]
        self.merges: list[tuple[int, int]] = []
        self.merge_ranks: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        """Return the number of available token IDs."""
        return len(self.vocab)

    def get_initial_symbols(self, text: str) -> list[int]:
        """Convert text to its UTF-8 byte values."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        return list(text.encode("utf-8"))

    def count_adjacent_pairs(
        self, symbol_sequences: Sequence[Sequence[Any]]
    ) -> dict[tuple[Any, Any], int]:
        """Count adjacent pairs without crossing sequence boundaries."""
        pair_counts: dict[tuple[Any, Any], int] = {}
        for sequence in symbol_sequences:
            for index in range(len(sequence) - 1):
                pair = (sequence[index], sequence[index + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def merge_pair(
        self,
        symbol_sequences: Sequence[Sequence[Any]],
        pair: tuple[Any, Any],
        new_symbol: Any,
    ) -> list[list[Any]]:
        """Replace non-overlapping occurrences of pair in each sequence."""
        merged_sequences: list[list[Any]] = []
        for sequence in symbol_sequences:
            merged: list[Any] = []
            index = 0
            while index < len(sequence):
                if (
                    index + 1 < len(sequence)
                    and sequence[index] == pair[0]
                    and sequence[index + 1] == pair[1]
                ):
                    merged.append(new_symbol)
                    index += 2
                else:
                    merged.append(sequence[index])
                    index += 1
            merged_sequences.append(merged)
        return merged_sequences

    def train(self, training_text: str) -> None:
        """Learn merges until the target vocabulary size or no pairs remain.

        Ties are resolved lexicographically by the pair of token IDs, making
        training deterministic for the same input text.
        """
        if not isinstance(training_text, str):
            raise TypeError("training_text must be a string.")

        # Reset to the byte-level base vocabulary so train() can be repeated.
        self.vocab = [bytes([value]) for value in range(self.BASE_VOCAB_SIZE)]
        self.merges = []
        self.merge_ranks = {}

        sequences = [self.get_initial_symbols(training_text)]

        while self.vocab_size < self.target_vocab_size:
            pair_counts = self.count_adjacent_pairs(sequences)
            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key=lambda pair: (pair_counts[pair], pair),
            )
            new_token_id = self.vocab_size
            left_id, right_id = best_pair
            self.vocab.append(self.vocab[left_id] + self.vocab[right_id])
            self.merge_ranks[best_pair] = len(self.merges)
            self.merges.append(best_pair)
            sequences = self.merge_pair(sequences, best_pair, new_token_id)

    def encode(self, text: str) -> list[int]:
        """Encode text by applying the learned merges in their training order."""
        symbols = self.get_initial_symbols(text)
        for rank, pair in enumerate(self.merges):
            new_token_id = self.BASE_VOCAB_SIZE + rank
            symbols = self.merge_pair([symbols], pair, new_token_id)[0]
        return symbols

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode token IDs back to the original UTF-8 text."""
        byte_parts: list[bytes] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError(f"Invalid token ID: {token_id!r}")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Invalid token ID: {token_id!r}")
            byte_parts.append(self.vocab[token_id])

        try:
            return b"".join(byte_parts).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Token IDs do not decode to valid UTF-8 text.") from exc

    def save(self, path: str | Path) -> None:
        """Save the vocabulary and ordered merge rules as JSON."""
        tokenizer_path = Path(path)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format_version": self.FORMAT_VERSION,
            "target_vocab_size": self.target_vocab_size,
            "base_encoding": self.base_encoding,
            "unknown_token": self.unknown_token,
            "vocab": [token.hex() for token in self.vocab],
            "merges": [list(pair) for pair in self.merges],
        }
        tokenizer_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Restore a trained tokenizer without retraining it."""
        tokenizer_path = Path(path)
        data = json.loads(tokenizer_path.read_text(encoding="utf-8"))

        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported BPE tokenizer file format.")
        if data.get("base_encoding") != "bytes" or data.get("unknown_token") is not None:
            raise ValueError("The saved tokenizer is not a supported byte-level BPE tokenizer.")

        tokenizer = cls(
            vocab_size=data["target_vocab_size"],
            base_encoding=data["base_encoding"],
            unknown_token=data.get("unknown_token"),
        )

        saved_vocab = data.get("vocab")
        saved_merges = data.get("merges")
        if not isinstance(saved_vocab, list) or not isinstance(saved_merges, list):
            raise ValueError("The saved BPE vocabulary or merge list is invalid.")

        tokenizer.vocab = [
            bytes([value]) for value in range(cls.BASE_VOCAB_SIZE)
        ]
        tokenizer.merges = []
        tokenizer.merge_ranks = {}

        for rank, pair in enumerate(saved_merges):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(
                    isinstance(token_id, bool) or not isinstance(token_id, int)
                    for token_id in pair
                )
            ):
                raise ValueError(f"Invalid merge rule at position {rank}.")

            left_id, right_id = pair
            if not (
                0 <= left_id < len(tokenizer.vocab)
                and 0 <= right_id < len(tokenizer.vocab)
            ):
                raise ValueError(f"Merge rule {rank} refers to an unknown token.")

            tokenizer.merges.append((left_id, right_id))
            tokenizer.merge_ranks[(left_id, right_id)] = rank
            tokenizer.vocab.append(
                tokenizer.vocab[left_id] + tokenizer.vocab[right_id]
            )

        try:
            decoded_saved_vocab = [bytes.fromhex(token) for token in saved_vocab]
        except (TypeError, ValueError) as exc:
            raise ValueError("The saved vocabulary contains invalid byte data.") from exc

        if decoded_saved_vocab != tokenizer.vocab:
            raise ValueError("The saved vocabulary does not match its merge rules.")
        if tokenizer.vocab_size > tokenizer.target_vocab_size:
            raise ValueError("The saved vocabulary exceeds its target size.")

        return tokenizer
