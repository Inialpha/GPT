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


def sampler(tokens, block_size, batch_size):
    starts = torch.randint(0, len(tokens) - block_size, (batch_size,))
    x = []
    y = []
    for start in starts:
        block = tokens[start:start + block_size + 1]
        x.append(block[:block_size])
        y.append(block[1:])

    return torch.stack(x), torch.stack(y)


def load_tokenizer(corpus: str):
    """Load the selected tokenizer, training and saving BPE on first use."""
    if TOKENIZER_TYPE == "character":
        return CharacterTokenizer.load_vocabulary(
            "data/character_vocab.json"
        )

    if TOKENIZER_TYPE == "bpe":
        tokenizer_path = Path("data/bpe_tokenizer.json")
        if tokenizer_path.exists():
            return BPETokenizer.load(tokenizer_path)

        tokenizer = BPETokenizer(vocab_size=BPE_VOCAB_SIZE)
        tokenizer.train(corpus)
        tokenizer.save(tokenizer_path)
        return tokenizer

    raise ValueError(f"Unsupported tokenizer type: {TOKENIZER_TYPE!r}")


def main() -> None:
    """Prepare data, tokenize it, train the GPT model, and generate text."""
    builder = CorpusBuilder.load("data/corpus.txt")
    corpus = builder.build()

    tokenizer = load_tokenizer(corpus)
    encoded_corpus = tokenizer.encode(corpus)

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

    tokens = torch.tensor(encoded_corpus, dtype=torch.long)

    model.train()
    for step in range(200):
        x, y = sampler(tokens, config.block_size, config.batch_size)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step: {step}, Loss: {loss.item():.4f}")

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
