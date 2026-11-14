import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import make_config  # noqa: E402,F401

from mmgraph_agent.eval.metrics import answer_recall, token_f1  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(token_f1("Paris", "Paris"), 1.0)

    def test_no_overlap(self):
        self.assertEqual(token_f1("London", "Paris"), 0.0)

    def test_partial(self):
        f1 = token_f1("Paris France", "Paris")
        self.assertAlmostEqual(f1, 2 / 3, places=4)

    def test_answer_recall(self):
        self.assertEqual(answer_recall("Paris", "the tower is in paris"), 1.0)
        self.assertEqual(answer_recall("Paris France", "the tower is in paris"), 0.5)


if __name__ == "__main__":
    unittest.main()
