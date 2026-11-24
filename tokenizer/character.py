"""Character-to-ID tokenizer scaffold."""
from __future__ import annotations
from typing import Iterable, Optional
import json
class CharacterTokenizer:
    """Map Unicode characters to integer IDs; fit vocabulary on training text only."""
    def __init__(self, training_text: str, unknown_token: Optional[str] = "<unk>"):
        self.training_text = training_text
        self.unknown_token = unknown_token
        # Your existing vocabulary-building attempt is retained here for you to refine.
        self.vocab = sorted(list(set(training_text)))
        self.vocab += [self.unknown_token]
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        """Return the number of token IDs in the vocabulary."""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Map characters to IDs, applying the chosen unknown-character policy."""
        ids = []
        for c in text:
            ids.append(self.stoi.get(c, self.unknown_token))
        return ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """Map IDs back to characters and handle invalid IDs deliberately."""
        chars = []
        for i in token_ids:
            chars.append(self.itos.get(i, self.unknown_token))
        return chars

    def save_vocabulary(self, path: str) -> None:
        """Persist the exact vocabulary ordering and special-token state."""
        vocab = {
            "vocab": self.vocab,
            "stoi": self.stoi,
            "itos": self.itos
        }

        json.dump(vocab, path)

    @classmethod
    def load_vocabulary(cls, path: str) -> "CharacterTokenizer":
        """Restore mappings exactly without rebuilding or reordering the vocabulary."""
        vocab = json.load(path)
        self.vocab = vocab.get("vocab")
        self.stoi = vocab.get("stoi")
        self.itos = vocab.get("itos")

