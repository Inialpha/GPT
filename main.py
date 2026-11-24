from tokenizer import CorpusBuilder
from tokenizer.character import CharacterTokenizer


"""Project entry point.

Import and run the individual data, tokenizer, and model components here once
those components have been implemented and tested. Keep experiments explicit
so each stage can be run independently during development.
"""

def main() -> None:
    """Coordinate the end-to-end data preparation, tokenization, and model workflow."""
    builder = CorpusBuilder.load("data/corpus.txt")
    corpus = builder.build()
    print(len(corpus))
    tokenizer = CharacterTokenizer(corpus)
    print(tokenizer.stoi)
    print(tokenizer.itos)
    print(tokenizer.encode("hello world"))
    print("".join(tokenizer.decode(tokenizer.encode("hello world"))) == "hello world")

if __name__ == "__main__":
    main()
