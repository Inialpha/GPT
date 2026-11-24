"""Configuration shared by Transformer components."""
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    """Model hyperparameters: vocabulary, context, width, heads, layers, dropout and bias."""
    vocab_size: int
    block_size: int
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    bias: bool = True
