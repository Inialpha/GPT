"""Character-to-ID tokenizer."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import json


class CharacterTokenizer:
    """Map Unicode characters to integer IDs; fit vocabulary on training text only."""

    def __init__(self, training_text: str, unknown_token: Optional[str] = "<unk>"):
        if not isinstance(training_text, str):
            raise TypeError("training_text must be a string.")
        if unknown_token is not None and not isinstance(unknown_token, str):
            raise TypeError("unknown_token must be a string or None.")
        if unknown_token == "":
            raise ValueError("unknown_token cannot be an empty string.")

        self.training_text = training_text
        self.unknown_token = unknown_token
        self.vocab = sorted(set(training_text))

        if self.unknown_token is not None and self.unknown_token not in self.vocab:
            self.vocab.append(self.unknown_token)

        self.stoi = {token: index for index, token in enumerate(self.vocab)}
        self.itos = {index: token for index, token in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        """Return the number of token IDs in the vocabulary."""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Map characters to IDs, applying the configured unknown-character policy."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        ids = []
        for character in text:
            token_id = self.stoi.get(character)
            if token_id is None:
                if self.unknown_token is None:
                    raise ValueError(
                        f"Character {character!r} is not present in the vocabulary."
                    )
                token_id = self.stoi[self.unknown_token]
            ids.append(token_id)
        return ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """Map IDs back to text; invalid IDs raise a ValueError."""
        characters = []
        for token_id in token_ids:
            try:
                characters.append(self.itos[token_id])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Invalid token ID: {token_id!r}") from exc
        return "".join(characters)

    def save_vocabulary(self, path: str | Path) -> None:
        """Save the vocabulary order and unknown-token setting as JSON."""
        vocabulary_path = Path(path)
        vocabulary_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab": self.vocab,
            "unknown_token": self.unknown_token,
        }
        vocabulary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_vocabulary(cls, path: str | Path) -> "CharacterTokenizer":
        """Restore the exact saved vocabulary without refitting or reordering it."""
        vocabulary_path = Path(path)
        data = json.loads(vocabulary_path.read_text(encoding="utf-8"))

        vocab = data.get("vocab")
        unknown_token = data.get("unknown_token")

        if not isinstance(vocab, list) or not all(
            isinstance(token, str) for token in vocab
        ):
            raise ValueError("The saved vocabulary must be a list of strings.")
        if len(vocab) != len(set(vocab)):
            raise ValueError("The saved vocabulary contains duplicate tokens.")
        if unknown_token is not None and unknown_token not in vocab:
            raise ValueError("The saved unknown token is missing from the vocabulary.")

        # Bypass __init__: loading must preserve the saved order, not refit it.
        tokenizer = cls.__new__(cls)
        tokenizer.training_text = ""
        tokenizer.unknown_token = unknown_token
        tokenizer.vocab = vocab
        tokenizer.stoi = {token: index for index, token in enumerate(vocab)}
        tokenizer.itos = {index: token for index, token in enumerate(vocab)}
        return tokenizer
