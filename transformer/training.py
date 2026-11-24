"""Training and evaluation helpers."""
import torch
from .model import GPTLanguageModel

@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, get_batch, eval_iters: int = 100) -> dict[str, float]:
    """Average train/validation losses without gradients and restore prior model mode."""
    raise NotImplementedError
