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
        self.layer_norm_1 = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.layer_norm_2 = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.attention = MultiHeadSelfAttention(config)
        self.feedforward = FeedForward(config)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply x += attention(LN(x)), then x += feedforward(LN(x))."""
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feedforward(self.layer_norm_2(x))
        return x

class TransformerBody(nn.Module):
    """Sequential stack of n_layer TransformerBlocks."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.network = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
