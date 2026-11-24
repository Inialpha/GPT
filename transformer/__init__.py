"""Decoder-only Transformer components for the GPT project."""
from .config import TransformerConfig
from .model import GPTLanguageModel

__all__ = ["TransformerConfig", "GPTLanguageModel"]
