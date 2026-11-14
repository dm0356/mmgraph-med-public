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
from mmgraph_agent.retrieval import CannedWebSearch, UnifiedRetriever  # noqa: E402


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        docs, _ = load_smoke(ROOT / "data" / "smoke")
        self.store = build_hypergraph(docs, self.config)
        self.embedder = HashingEmbedder(256)

    def test_offline_retrieval(self):
        memory = TwoTierMemory(self.store, self.config, self.embedder)
        retriever = UnifiedRetriever(self.store, memory, self.config, self.embedder)
        evidence = retriever.retrieve("how tall is the eiffel tower", "text")
        self.assertTrue(evidence)
        self.assertTrue(all(item.source == "offline" for item in evidence))

    def test_canned_web_channel(self):
        hits = CannedWebSearch(["a web snippet"]).search("q")
        self.assertEqual(hits[0].source, "web")

    def test_web_integration(self):
        config = copy.deepcopy(self.config)
        config["retrieval"]["enable_web"] = True
        docs = [{"doc_id": "d1", "text": "The Eiffel Tower is in Paris.", "images": []}]
        store = build_hypergraph(docs, config)
        memory = TwoTierMemory(store, config, self.embedder)
        retriever = UnifiedRetriever(
            store, memory, config, self.embedder, web=CannedWebSearch(["Paris is a city."])
        )
        evidence = retriever.retrieve("paris", "text")
        self.assertTrue(any(item.source == "web" for item in evidence))


if __name__ == "__main__":
    unittest.main()
