"""Pre-LayerNorm Transformer block and stack."""
import torch
import torch.nn as nn
from .config import TransformerConfig
from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward

class TransformerBlock(nn.Module):
    """Apply attention and feedforward sublayers with residual connections."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply x += attention(LN(x)), then x += feedforward(LN(x))."""
        raise NotImplementedError

class TransformerBody(nn.Module):
    """Sequential stack of n_layer TransformerBlocks."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
