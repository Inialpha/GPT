"""Sentence-level text segmentation (not a subword tokenizer)."""
from __future__ import annotations
from typing import Sequence

class SentenceTokenizer:
    """Split text into sentence strings and optionally reconstruct it."""
    def __init__(self, keep_delimiter: bool = True, split_on_newline: bool = True):
        """Store punctuation-retention and newline-boundary options."""
        raise NotImplementedError

    def tokenize(self, text: str) -> list[str]:
        """Return sentences in order; consider abbreviations, decimals, quotes and empty segments."""
        raise NotImplementedError

    def detokenize(self, sentences: Sequence[str]) -> str:
        """Recombine sentence strings into readable text."""
        raise NotImplementedError
