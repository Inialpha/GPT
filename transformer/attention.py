"""Causal self-attention layers."""
import torch
import torch.nn as nn
from .config import TransformerConfig

class AttentionHead(nn.Module):
    """One scaled dot-product causal attention head; input (B,T,C), output (B,T,D)."""
    def __init__(self, config: TransformerConfig, head_size: int):
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.n_embed, config.head_size)
        self.k = nn.Linear(config.n_embed, config.head_size)
        self.v = nn.Linear(config.n_embed, config.head_size)
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = q @ k.T(-2, -1)
        scores = scores
        torch.ones(
        """Project Q/K/V, scale scores, mask future positions, softmax and mix values."""
        raise NotImplementedError

class MultiHeadSelfAttention(nn.Module):
    """Run parallel heads, concatenate their outputs, project and apply dropout."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
