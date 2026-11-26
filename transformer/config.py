"""Configuration shared by Transformer components."""
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    """Model hyperparameters: vocabulary, context, width, heads, layers, dropout and bias."""
    vocab_size: int
    block_size: int
    batch_size: int
    n_embed: int
    n_head: int
    n_layer: int
    dropout: float
    bias: bool

    @property
    def head_size(self):
        return self.n_embed // self.n_head
