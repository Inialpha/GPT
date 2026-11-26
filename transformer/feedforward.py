"""Position-wise feedforward network used inside each Transformer block."""
import torch
import torch.nn as nn
from .config import TransformerConfig

class FeedForward(nn.Module):
    """MLP: C -> 4C -> GELU -> C, with dropout; input/output (B,T,C)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * 4),
            nn.GELU(),
            nn.Linear(config.n_embed * 4, config.n_embed),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
