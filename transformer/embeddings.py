"""Token and positional embedding layers."""
import torch
import torch.nn as nn
from .config import TransformerConfig

class TokenEmbedding(nn.Module):
    """Map token IDs (B,T) to learned vectors (B,T,C)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.n_embed
        )
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(idx)

class PositionalEmbedding(nn.Module):
    """Map positions to learned vectors broadcastable over the batch."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            self.config.block_size,
            self.config.n_embed
        )

    def forward(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        position_ids = torch.arange(sequence_length, device=device)
        return self.embedding(position_ids)



class TokenAndPositionEmbedding(nn.Module):
    """Add token and position vectors, then apply dropout."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(self.config)
        self.dropout = nn.Dropout(config.dropout)
        self.positional_embedding = PositionalEmbedding(self.config)
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        tp_embedding = self.token_embedding(idx) + self.positional_embedding(idx.shape[1], idx.device)
        return self.dropout(tp_embedding)
