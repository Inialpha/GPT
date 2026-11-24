"""Language-model head and complete decoder-only GPT model."""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import TransformerConfig
from .embeddings import TokenAndPositionEmbedding
from .block import TransformerBody

class LanguageModelHead(nn.Module):
    """Project hidden states (B,T,C) to raw vocabulary logits (B,T,V)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class GPTLanguageModel(nn.Module):
    """Embeddings -> Transformer blocks -> final LayerNorm -> vocabulary logits."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return logits (B,T,V) and optional scalar cross-entropy loss; targets are pre-shifted."""
        raise NotImplementedError
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Crop context, sample from final-position logits, and append one token per step."""
        raise NotImplementedError
