import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import make_config  # noqa: E402

from mmgraph_agent.hypergraph.encoder import HashingEmbedder  # noqa: E402
from mmgraph_agent.hypergraph.store import Store  # noqa: E402
from mmgraph_agent.memory import TwoTierMemory  # noqa: E402


def _store():
    entities = [
        {"id": "e1", "name": "糖尿病", "type": "concept", "description": "", "embedding": None},
        {"id": "e2", "name": "血糖", "type": "concept", "description": "", "embedding": None},
        {"id": "e3", "name": "肺炎", "type": "concept", "description": "", "embedding": None},
    ]
    embedder = HashingEmbedder(256)
    vecs = embedder.encode_texts(["糖尿病与血糖相关", "血糖异常与肺炎风险", "肺炎患者发热"])
    hyperedges = [
        {"id": "h0", "doc_id": "d1", "unit_id": "u0", "modality": "text",
         "text": "糖尿病与血糖相关", "entities": ["e1", "e2"], "relations": [], "images": [],
         "embedding": vecs[0].tolist()},
        {"id": "h1", "doc_id": "d1", "unit_id": "u1", "modality": "text",
         "text": "血糖异常与肺炎风险", "entities": ["e2", "e3"], "relations": [], "images": [],
         "embedding": vecs[1].tolist()},
        {"id": "h2", "doc_id": "d1", "unit_id": "u2", "modality": "text",
         "text": "肺炎患者发热", "entities": ["e3"], "relations": [], "images": [],
         "embedding": vecs[2].tolist()},
    ]
    return Store(meta={"embed_dim": 256}, entities=entities, images=[], units=[], hyperedges=hyperedges)


class TestMultiHop(unittest.TestCase):
    def test_neighbor_one_hop(self):
        store = _store()
        ids = [edge["id"] for edge in store.neighbor_hyperedges(["h0"], hops=1)]
        self.assertEqual(ids, ["h0", "h1"])

    def test_neighbor_two_hops(self):
        store = _store()
        ids = [edge["id"] for edge in store.neighbor_hyperedges(["h0"], hops=2)]
        self.assertEqual(ids, ["h0", "h1", "h2"])

    def test_memory_retrieve_with_hops(self):
        config = make_config()
        config["retrieval"]["hops"] = 2
        config["memory"]["C"] = 10
        store = _store()
        memory = TwoTierMemory(store, config, HashingEmbedder(256))
        evidence = memory.retrieve("糖尿病")
        ids = {item.id for item in evidence}
        self.assertIn("h0", ids)
        self.assertIn("h1", ids)


if __name__ == "__main__":
    unittest.main()
