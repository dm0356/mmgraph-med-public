import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ROOT, make_config  # noqa: E402

from mmgraph_agent.eval.datasets import load_smoke  # noqa: E402
from mmgraph_agent.hypergraph import build_hypergraph  # noqa: E402
from mmgraph_agent.hypergraph.encoder import HashingEmbedder  # noqa: E402
from mmgraph_agent.memory import TwoTierMemory  # noqa: E402


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        docs, _ = load_smoke(ROOT / "data" / "smoke")
        self.store = build_hypergraph(docs, self.config)
        self.embedder = HashingEmbedder(256)

    def test_retrieve_returns_top_c_and_bounded(self):
        memory = TwoTierMemory(self.store, self.config, self.embedder)
        evidence = memory.retrieve("how tall is the eiffel tower")
        self.assertGreaterEqual(len(evidence), 1)
        self.assertLessEqual(len(evidence), memory.C)
        self.assertLessEqual(memory.prompt_tokens(), self.config["memory"]["L_cap"])

    def test_admission_truncates_and_digests(self):
        config = copy.deepcopy(self.config)
        config["memory"] = {"C": 5, "L_max": 4, "L_dig": 3, "L_cap": 40}
        memory = TwoTierMemory(self.store, config, self.embedder)
        evidence = memory.retrieve("how tall is the eiffel tower")
        for item in evidence:
            self.assertLessEqual(len(item.text.split()), 3)
        self.assertLessEqual(memory.prompt_tokens(), 40)

    def test_reset_trajectory(self):
        memory = TwoTierMemory(self.store, self.config, self.embedder)
        memory.retrieve("eiffel")
        self.assertGreater(len(memory.l1), 0)
        memory.reset_trajectory()
        self.assertEqual(len(memory.l1), 0)


if __name__ == "__main__":
    unittest.main()
