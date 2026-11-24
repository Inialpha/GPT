"""Position-wise feedforward network used inside each Transformer block."""
import torch
import torch.nn as nn
from .config import TransformerConfig

class FeedForward(nn.Module):
    """MLP: C -> 4C -> GELU -> C, with dropout; input/output (B,T,C)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
