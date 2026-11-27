"""Train the decoder-only GPT model and generate a sample."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import torch
import torch.optim as optim

from tokenizer import BPETokenizer, CorpusBuilder
from tokenizer.character import CharacterTokenizer
from transformer.config import TransformerConfig
from transformer.model import GPTLanguageModel


# Change this to "character" to run the character-tokenizer experiment.
TOKENIZER_TYPE = "bpe"
BPE_VOCAB_SIZE = 512

# Each execution performs this many additional optimizer updates.
# The checkpoint records the cumulative number of completed updates.
TRAINING_STEPS_PER_RUN = 200
CHECKPOINT_EVERY = 100
CHECKPOINT_PATH = Path("checkpoints/gpt_checkpoint.pt")


def sampler(tokens, block_size, batch_size):
    """Sample random next-token input/target windows from a 1D token tensor."""
    starts = torch.randint(0, len(tokens) - block_size, (batch_size,))
    x = []
    y = []
    for start in starts:
        block = tokens[start:start + block_size + 1]
        x.append(block[:block_size])
        y.append(block[1:])

    return torch.stack(x), torch.stack(y)


def load_tokenizer(corpus: str):
    """Load the selected tokenizer and return it with its saved artifact path."""
    if TOKENIZER_TYPE == "character":
        tokenizer_path = Path("data/character_vocab.json")
        tokenizer = CharacterTokenizer.load_vocabulary(tokenizer_path)
        return tokenizer, tokenizer_path

    if TOKENIZER_TYPE == "bpe":
        tokenizer_path = Path("data/bpe_tokenizer.json")
        if tokenizer_path.exists():
            tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            tokenizer = BPETokenizer(vocab_size=BPE_VOCAB_SIZE)
            tokenizer.train(corpus)
            tokenizer.save(tokenizer_path)
        return tokenizer, tokenizer_path

    raise ValueError(f"Unsupported tokenizer type: {TOKENIZER_TYPE!r}")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 fingerprint of a tokenizer artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(
    model: GPTLanguageModel,
    optimizer: optim.Optimizer,
    completed_steps: int,
    config: TransformerConfig,
    tokenizer_identity: dict,
    corpus_sha256: str,
) -> None:
    """Atomically save everything required to resume compatible training."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "completed_steps": completed_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(config),
        "tokenizer": tokenizer_identity,
        "corpus_sha256": corpus_sha256,
        "torch_rng_state": torch.get_rng_state(),
    }

    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(CHECKPOINT_PATH)


def load_checkpoint(
    model: GPTLanguageModel,
    optimizer: optim.Optimizer,
    config: TransformerConfig,
    tokenizer_identity: dict,
    corpus_sha256: str,
) -> int:
    """Restore a checkpoint only when its model, tokenizer, and corpus match."""
    if not CHECKPOINT_PATH.exists():
        print("No checkpoint found. Starting training from scratch.")
        return 0

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with PyTorch versions predating weights_only.
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")

    if checkpoint.get("model_config") != asdict(config):
        raise ValueError(
            "The saved checkpoint uses a different model configuration. "
            "Restore the original configuration or move the checkpoint aside "
            "before starting a new experiment."
        )

    if checkpoint.get("tokenizer") != tokenizer_identity:
        raise ValueError(
            "The saved checkpoint was trained with a different tokenizer "
            "artifact. Refusing to load weights with mismatched token IDs. "
            "Restore the matching tokenizer file or move the checkpoint aside "
            "to start a new experiment."
        )

    if checkpoint.get("corpus_sha256") != corpus_sha256:
        raise ValueError(
            "The training corpus differs from the corpus used by this checkpoint. "
            "Restore the original corpus or move the checkpoint aside to start fresh."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

    completed_steps = int(checkpoint["completed_steps"])
    print(f"Loaded checkpoint: {completed_steps} optimizer updates completed.")
    return completed_steps


def main() -> None:
    """Prepare data, resume or train GPT, save checkpoints, and generate text."""
    builder = CorpusBuilder.load("data/corpus.txt")
    corpus = builder.build()

    tokenizer, tokenizer_path = load_tokenizer(corpus)
    encoded_corpus = tokenizer.encode(corpus)

    tokenizer_identity = {
        "type": TOKENIZER_TYPE,
        "artifact_path": str(tokenizer_path),
        "artifact_sha256": sha256_file(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
    }
    corpus_sha256 = hashlib.sha256(corpus.encode("utf-8")).hexdigest()

    print(f"Tokenizer: {TOKENIZER_TYPE}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"UTF-8 corpus bytes: {len(corpus.encode('utf-8'))}")
    print(f"Encoded token count: {len(encoded_corpus)}")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        batch_size=32,
        n_embed=128,
        n_head=4,
        n_layer=4,
        dropout=0.1,
        bias=True,
    )
    model = GPTLanguageModel(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    completed_steps = load_checkpoint(
        model,
        optimizer,
        config,
        tokenizer_identity,
        corpus_sha256,
    )
    target_step = completed_steps + TRAINING_STEPS_PER_RUN
    tokens = torch.tensor(encoded_corpus, dtype=torch.long)

    model.train()
    for step in range(completed_steps + 1, target_step + 1):
        x, y = sampler(tokens, config.block_size, config.batch_size)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step: {step}, Loss: {loss.item():.4f}")

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                config,
                tokenizer_identity,
                corpus_sha256,
            )
            print(f"Checkpoint saved at step {step}.")

    # Save the final state too, including when a run ends between 100-step marks.
    save_checkpoint(
        model,
        optimizer,
        target_step,
        config,
        tokenizer_identity,
        corpus_sha256,
    )
    print(f"Training state saved: {target_step} total optimizer updates.")

    model.eval()
    prompt = "Psychology"
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
    ).unsqueeze(0)

    generated_ids = model.generate(input_ids, max_new_tokens=100)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
