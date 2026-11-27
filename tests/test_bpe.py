import tempfile
import unittest
from pathlib import Path

from tokenizer.bpe import BPETokenizer


class BPETokenizerTests(unittest.TestCase):
    def test_base_vocabulary_contains_all_byte_values(self):
        tokenizer = BPETokenizer(vocab_size=256)
        self.assertEqual(tokenizer.vocab_size, 256)
        self.assertEqual(tokenizer.get_initial_symbols("A"), [65])

    def test_training_learns_merges_and_round_trips_utf8(self):
        text = "banana banana 🍌\nHello, 世界!"
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(text)

        self.assertGreater(tokenizer.vocab_size, 256)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_unseen_unicode_text_is_still_encodable(self):
        tokenizer = BPETokenizer(vocab_size=270)
        tokenizer.train("some training text")
        text = "🧪 Καλημέρα — नमस्ते"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_repeated_patterns_are_compressed(self):
        text = "abababababababab"
        tokenizer = BPETokenizer(vocab_size=270)
        tokenizer.train(text)
        self.assertLess(len(tokenizer.encode(text)), len(text.encode("utf-8")))

    def test_pair_count_does_not_cross_sequence_boundaries(self):
        tokenizer = BPETokenizer(vocab_size=256)
        self.assertEqual(
            tokenizer.count_adjacent_pairs([[1], [2]]),
            {},
        )
        self.assertEqual(
            tokenizer.count_adjacent_pairs([[1, 2]]),
            {(1, 2): 1},
        )

    def test_merge_replaces_non_overlapping_occurrences(self):
        tokenizer = BPETokenizer(vocab_size=256)
        result = tokenizer.merge_pair([[1, 1, 1]], (1, 1), 256)
        self.assertEqual(result, [[256, 1]])

    def test_save_and_load_preserve_tokenizer(self):
        text = "banana 🍌 banana"
        tokenizer = BPETokenizer(vocab_size=280)
        tokenizer.train(text)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = BPETokenizer.load(path)

        self.assertEqual(restored.vocab_size, tokenizer.vocab_size)
        self.assertEqual(restored.merges, tokenizer.merges)
        self.assertEqual(restored.decode(restored.encode(text)), text)

    def test_invalid_token_id_raises(self):
        tokenizer = BPETokenizer(vocab_size=256)
        with self.assertRaises(ValueError):
            tokenizer.decode([256])

    def test_vocab_size_cannot_be_smaller_than_byte_base(self):
        with self.assertRaises(ValueError):
            BPETokenizer(vocab_size=255)


if __name__ == "__main__":
    unittest.main()
