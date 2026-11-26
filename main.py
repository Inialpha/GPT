from tokenizer import CorpusBuilder
from tokenizer.character import CharacterTokenizer
from transformer.embeddings import TokenAndPositionEmbedding
from transformer.attention import MultiHeadSelfAttention
from transformer.config import TransformerConfig
import torch
from transformer.model import GPTLanguageModel
import torch.optim as optim

"""Project entry point.

Import and run the individual data, tokenizer, and model components here once
those components have been implemented and tested. Keep experiments explicit
so each stage can be run independently during development.
"""
def sampler(tokens, block_size, batch_size):
    starts = torch.randint(0, len(tokens) - block_size, (batch_size,))
    x = []
    y = []
    for start in starts:
        block = tokens[start:start + block_size + 1]
        x.append(block[:block_size])
        y.append(block[1:])

    return torch.stack(x), torch.stack(y)

def main() -> None:
    """Coordinate the end-to-end data preparation, tokenization, and model workflow."""

    builder = CorpusBuilder.load("data/corpus.txt")
    corpus = builder.build()
    tokenizer = CharacterTokenizer.load_vocabulary(
    "data/character_vocab.json"
    )
    encoded_corpus = tokenizer.encode(corpus)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        batch_size=32,
        n_embed=128,
        n_head=4,
        n_layer=4,
        dropout=0.1,
        bias=True
    )
    model = GPTLanguageModel(config)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

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
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    generated_ids = model.generate(input_ids, max_new_tokens=100)

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)



if __name__ == "__main__":
    main()
