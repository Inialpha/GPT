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
        self.layer = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class GPTLanguageModel(nn.Module):
    """Embeddings -> Transformer blocks -> final LayerNorm -> vocabulary logits."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.layer_norm = nn.LayerNorm(config.n_embed)
        self.token_and_positional_embedding = TokenAndPositionEmbedding(config)
        self.transformer_body = TransformerBody(config)
        self.language_model_head = LanguageModelHead(config)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return logits (B,T,V) and optional scalar cross-entropy loss."""

        if targets is not None:
            assert idx.shape == targets.shape, (
                "Input and target tensors must have the same shape."
            )

        embedding = self.token_and_positional_embedding(idx)
        x = self.transformer_body(embedding)
        x_norm = self.layer_norm(x)
        logits = self.language_model_head(x_norm)

        if targets is None:
            return logits, None

        batch_size, sequence_length, vocab_size = logits.shape
        flat_logits = logits.reshape(batch_size * sequence_length, vocab_size)
        flat_targets = targets.reshape(batch_size * sequence_length)
        loss = F.cross_entropy(flat_logits, flat_targets)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Crop context, sample from final-position logits, and append tokens."""

        for _ in range(max_new_tokens):
            idx_cont = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cont)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
