"""Causal self-attention layers."""
import torch
import torch.nn as nn
from .config import TransformerConfig

class AttentionHead(nn.Module):
    """One scaled dot-product causal attention head; input (B,T,C), output (B,T,D)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.n_embed, config.head_size)
        self.k = nn.Linear(config.n_embed, config.head_size)
        self.v = nn.Linear(config.n_embed, config.head_size)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.config.head_size)
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        return out
        """Project Q/K/V, scale scores, mask future positions, softmax and mix values."""

class MultiHeadSelfAttention(nn.Module):
    """Run parallel heads, concatenate their outputs, project and apply dropout."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
