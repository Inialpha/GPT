"""Token and positional embedding layers."""
import torch
import torch.nn as nn
from .config import TransformerConfig

class TokenEmbedding(nn.Module):
    """Map token IDs (B,T) to learned vectors (B,T,C)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class PositionalEmbedding(nn.Module):
    """Map positions to learned vectors broadcastable over the batch."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

class TokenAndPositionEmbedding(nn.Module):
    """Add token and position vectors, then apply dropout."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
