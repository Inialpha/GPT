"""Implementation scaffold for a decoder-only GPT-style Transformer.

All model methods are intentionally unimplemented. Fill in the TODOs and test
shapes after each component. Tensor dimensions:
B=batch, T=sequence length, C=model width, V=vocabulary size,
H=attention heads, D=head width=C//H.

Pipeline: token IDs -> token+position embeddings -> N TransformerBlocks ->
final LayerNorm -> vocabulary projection -> logits for next-token prediction.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    """Shared hyperparameters.

    vocab_size: number of tokenizer IDs; block_size: maximum context length;
    n_embd: representation width C; n_head: parallel heads H (C divisible by H);
    n_layer: number of blocks; dropout: dropout probability; bias: Linear bias flag.
    """
    vocab_size: int
    block_size: int
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    bias: bool = True

class TokenEmbedding(nn.Module):
    """Map token IDs (B,T) to learned vectors (B,T,C)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: nn.Embedding(vocab_size, n_embd).
        raise NotImplementedError
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Look up one learned vector for every token ID."""
        raise NotImplementedError

class PositionalEmbedding(nn.Module):
    """Map positions 0..T-1 to learned vectors (1,T,C), broadcastable over B."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: Learnable nn.Embedding(block_size, n_embd).
        raise NotImplementedError
    def forward(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Create position IDs on device, look them up, and handle T>block_size."""
        raise NotImplementedError

class TokenAndPositionEmbedding(nn.Module):
    """Add token and position vectors; input (B,T), output (B,T,C)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: Instantiate both embedding modules and dropout.
        raise NotImplementedError
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Look up both embeddings, add elementwise, apply dropout."""
        raise NotImplementedError

class AttentionHead(nn.Module):
    """One causal scaled dot-product attention head.

    Input (B,T,C); output (B,T,D), D=C//H. Project x into Q, K, V; compute
    QK^T/sqrt(D); mask future positions with -inf before softmax; softmax over
    key positions; apply dropout; multiply weights by V. Query is what a token
    seeks, Key is what it offers for matching, Value is information contributed.
    """
    def __init__(self, config: TransformerConfig, head_size: int):
        super().__init__()
        # TODO: Q/K/V Linear(C, head_size), causal lower-triangular mask buffer,
        # and attention dropout. A registered buffer follows model device moves.
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return weighted value mixtures with shape (B,T,head_size)."""
        raise NotImplementedError

class MultiHeadSelfAttention(nn.Module):
    """Run H heads, concatenate to width C, project C->C, apply dropout.

    Input/output: (B,T,C). Validate that n_embd is divisible by n_head.
    Different heads can learn different token relationships in parallel.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: Validate dimensions; create n_head AttentionHead modules;
        # output projection Linear(C,C) and dropout.
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run heads, concatenate on final dimension, project, apply dropout."""
        raise NotImplementedError

class FeedForward(nn.Module):
    """Position-wise MLP: Linear(C,4C)->GELU->Linear(4C,C)->Dropout.

    Input/output (B,T,C). The same MLP is applied at every position. It
    transforms each token representation but does not mix information between
    positions; self-attention performs the token-to-token mixing.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: Define expansion Linear, GELU, projection Linear and dropout.
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply expansion, activation, projection and dropout."""
        raise NotImplementedError

class TransformerBlock(nn.Module):
    """One repeatable pre-LayerNorm Transformer block.

    Operations: x = x + attention(ln1(x)); x = x + feedforward(ln2(x)).
    Input/output remain (B,T,C). Residual additions preserve a direct path for
    information and gradients. Components: two LayerNorms, multi-head
    self-attention and position-wise feedforward network.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: Instantiate ln1, ln2, MultiHeadSelfAttention and FeedForward.
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implement pre-norm attention + residual, then MLP + residual."""
        raise NotImplementedError

class TransformerBody(nn.Module):
    """Stack n_layer TransformerBlocks sequentially; shape stays (B,T,C)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: Store blocks in nn.ModuleList so PyTorch registers parameters.
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass output of each block as input to the next block."""
        raise NotImplementedError

class LanguageModelHead(nn.Module):
    """Project (B,T,C) to raw vocabulary logits (B,T,V).

    Do not softmax here: torch.nn.functional.cross_entropy expects raw logits.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: Linear(n_embd, vocab_size).
        raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return vocabulary scores for every position."""
        raise NotImplementedError

class GPTLanguageModel(nn.Module):
    """Complete decoder-only Transformer for autoregressive next-token learning.

    Inputs: idx LongTensor (B,T); optional targets LongTensor (B,T), already
    shifted so each position's target is the next token. Outputs: logits
    (B,T,V), and loss=None without targets or scalar cross-entropy with targets.

    Pipeline: embeddings -> TransformerBody -> final LayerNorm -> vocab head.
    For loss, flatten logits to (B*T,V) and targets to (B*T,). Do not shift
    targets again here if the data loader already returns shifted x/y batches.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: Instantiate embeddings, TransformerBody, final LayerNorm,
        # and LanguageModelHead. Optional: deliberate parameter initialization.
        raise NotImplementedError
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run complete pipeline; optionally calculate flattened cross-entropy."""
        raise NotImplementedError
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressively append IDs and return prompt plus generated IDs.

        Each step: crop context to block_size; forward; select final-position
        logits (B,V); divide by positive temperature; optionally suppress all
        but top_k logits; softmax; sample with torch.multinomial; append token.
        Validate temperature and top_k. The model must not see future tokens.
        """
        raise NotImplementedError

@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, get_batch, eval_iters: int = 100) -> dict[str, float]:
    """Estimate average train/validation loss.

    get_batch accepts 'train' or 'val' and returns (x,y). Temporarily switch to
    eval mode, average eval_iters losses per split without gradients, then
    restore the model's previous train/eval mode. Return {'train': ..., 'val': ...}.
    """
    raise NotImplementedError

# Suggested implementation order:
# TokenEmbedding -> PositionalEmbedding -> TokenAndPositionEmbedding
# -> AttentionHead -> MultiHeadSelfAttention -> FeedForward -> TransformerBlock
# -> TransformerBody -> LanguageModelHead -> GPTLanguageModel.forward/generate
# -> estimate_loss.
#
# Shape checks for B=2,T=16,C=64,H=4,D=16,V=100:
# token embedding (2,16,64); one head (2,16,16); multi-head (2,16,64);
# feedforward/block/body (2,16,64); logits (2,16,100); loss is scalar.
