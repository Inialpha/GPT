"""Small utilities for inspecting and evaluating tokenizer behavior."""
from __future__ import annotations
from typing import Any

def inspect_tokenizer(tokenizer: Any, sample_text: str) -> dict[str, Any]:
    """Return encoded IDs, token count, vocabulary size, decoded text, and round-trip status."""
    raise NotImplementedError
