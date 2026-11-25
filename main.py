from tokenizer import CorpusBuilder
from tokenizer.character import CharacterTokenizer
from transformer.embeddings import TokenAndPositionEmbedding
from transformer.config import TransformerConfig
import torch

"""Project entry point.

Import and run the individual data, tokenizer, and model components here once
those components have been implemented and tested. Keep experiments explicit
so each stage can be run independently during development.
"""

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
        n_embed=128,
        n_head=4,
        n_layer=4,
        dropout=0.1,
        bias=True
    )
    
    input_ids = encoded_corpus[:config.block_size]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    embedding = TokenAndPositionEmbedding(config)
    output = embedding(input_ids)
    print(output)
    print(tokenizer.decode(input_ids[0].tolist()))
    print("Input shape:", input_ids.shape)
    print("Output shape:", output.shape)



if __name__ == "__main__":
    main()
