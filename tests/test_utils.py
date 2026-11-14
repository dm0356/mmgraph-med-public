import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import make_config  # noqa: E402

import numpy as np  # noqa: E402

from mmgraph_agent.utils import (  # noqa: E402
    digest,
    hash_embed,
    proper_noun_phrases,
    split_sentences,
    tokenize,
    truncate_words,
)


class TestUtils(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("The Eiffel Tower is in Paris."), ["the", "eiffel", "tower", "is", "in", "paris"])

    def test_split_sentences(self):
        self.assertEqual(len(split_sentences("A. B? C!")), 3)

    def test_decimals_are_not_sentence_boundaries(self):
        self.assertEqual(len(split_sentences("窗口 650.0-655.0 秒。")), 1)
        self.assertEqual(len(split_sentences("血糖 8.74 mmol/L，复查 7.2。")), 1)
        self.assertEqual(len(split_sentences("A. B? C!")), 3)

    def test_hash_embed_dim_and_norm(self):
        vec = hash_embed("hello world", 64)
        self.assertEqual(vec.shape, (64,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_truncate_words(self):
        self.assertEqual(truncate_words("a b c d e", 3), "a b c")

    def test_digest_bounds(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        self.assertLessEqual(len(digest(text, 4).split()), 4)

    def test_proper_noun_phrases(self):
        ents = proper_noun_phrases("The Eiffel Tower is located in Paris, France.")
        self.assertIn("Eiffel Tower", ents)
        self.assertIn("Paris", ents)
        self.assertNotIn("The", ents)

    def test_chinese_tokenize_and_sentences(self):
        from mmgraph_agent.utils import split_sentences, tokenize, truncate_words

        self.assertEqual(tokenize("糖尿病患者"), ["糖", "尿", "病", "患", "者"])
        self.assertEqual(len(split_sentences("患者就诊。心电图正常。")), 2)
        self.assertEqual(truncate_words("糖尿病患者需要控制血糖", 4), "糖尿病患")


if __name__ == "__main__":
    unittest.main()
