"""Character-to-ID tokenizer scaffold."""
from __future__ import annotations
from typing import Iterable, Optional

class CharacterTokenizer:
    """Map Unicode characters to integer IDs; fit vocabulary on training text only."""
    def __init__(self, training_text: str, unknown_token: Optional[str] = "<unk>"):
        self.training_text = training_text
        self.unknown_token = unknown_token
        # Your existing vocabulary-building attempt is retained here for you to refine.
        self.vocab = list(set(training_text))
        self.vocab + [self.unknown_token]
        self.vocab_size = len(self.vocab)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Return the number of token IDs in the vocabulary."""
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Map characters to IDs, applying the chosen unknown-character policy."""
        raise NotImplementedError

    def decode(self, token_ids: Iterable[int]) -> str:
        """Map IDs back to characters and handle invalid IDs deliberately."""
        raise NotImplementedError

    def save_vocabulary(self, path: str) -> None:
        """Persist the exact vocabulary ordering and special-token state."""
        raise NotImplementedError

    @classmethod
    def load_vocabulary(cls, path: str) -> "CharacterTokenizer":
        """Restore mappings exactly without rebuilding or reordering the vocabulary."""
        raise NotImplementedError
