import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ROOT, make_config  # noqa: E402

from mmgraph_agent.eval.datasets import load_smoke  # noqa: E402
from mmgraph_agent.hypergraph import build_hypergraph  # noqa: E402
from mmgraph_agent.hypergraph.builder import _merge_duplicate_entities  # noqa: E402


class TestEntityMerge(unittest.TestCase):
    def test_merge_duplicate_names(self):
        entities = [
            {"id": "e0", "name": "冠心病", "description": "短"},
            {"id": "e1", "name": "冠心病", "description": "更长的描述"},
        ]
        hyperedges = [{"entities": ["e0", "e1"]}, {"entities": ["e1"]}]
        merged = _merge_duplicate_entities(entities, hyperedges)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["id"], "e0")
        self.assertEqual(merged[0]["description"], "更长的描述")
        self.assertEqual(hyperedges[0]["entities"], ["e0"])
        self.assertEqual(hyperedges[1]["entities"], ["e0"])


class TestBuilder(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        self.docs, _ = load_smoke(ROOT / "data" / "smoke")
        self.store = build_hypergraph(self.docs, self.config)

    def test_non_empty(self):
        self.assertGreater(len(self.store.entities), 0)
        self.assertGreater(len(self.store.hyperedges), 0)
        self.assertEqual(self.store.embed_dim, 256)

    def test_coverage_property(self):
        docs_with_images = {img["doc_id"] for img in self.store.images}
        for edge in self.store.hyperedges:
            if edge["doc_id"] in docs_with_images:
                self.assertGreaterEqual(len(edge["images"]), 1)

    def test_entity_ids_resolve(self):
        for edge in self.store.hyperedges:
            names = self.store.entity_names(edge["entities"])
            self.assertEqual(len(names), len(edge["entities"]))


if __name__ == "__main__":
    unittest.main()
