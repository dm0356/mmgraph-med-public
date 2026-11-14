import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ROOT, make_config  # noqa: E402

from mmgraph_agent.eval.datasets import load_smoke  # noqa: E402
from mmgraph_agent.eval.runner import evaluate  # noqa: E402


class TestEvaluate(unittest.TestCase):
    def test_smoke_end_to_end(self):
        config = make_config()
        docs, qa = load_smoke(ROOT / "data" / "smoke")
        result = evaluate(docs, qa, config)
        self.assertEqual(len(result["rows"]), len(qa))
        self.assertGreater(result["store"]["entities"], 0)
        self.assertGreater(result["token_f1"], 0.5)
        self.assertGreater(result["answer_recall"], 0.7)


if __name__ == "__main__":
    unittest.main()
