import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import make_config  # noqa: E402

from mmgraph_agent.hypergraph import Store  # noqa: E402
from mmgraph_agent.memory import TwoTierMemory  # noqa: E402


class FakeEmbedder:
    name = "fake-text"

    def encode_texts(self, texts):
        import numpy as np

        return [np.asarray([1.0, 0.0], dtype=np.float32) for _ in texts]


class FakeCLIPTextEncoder:
    name = "fake-clip"

    def encode_texts(self, texts):
        import numpy as np

        return [np.asarray([0.0, 1.0], dtype=np.float32) for _ in texts]


def _store():
    return Store(
        meta={"embedder": "fake-text", "embed_dim": 2},
        entities=[],
        images=[
            {"id": "i1", "caption": "心电图波形", "embedding": [1.0, 0.0],
             "image_embedding": [0.0, 1.0], "image_embedder": "fake-clip", "image_embed_dim": 2},
        ],
        units=[],
        hyperedges=[
            {"id": "h1", "doc_id": "d1", "unit_id": "d1:u0", "modality": "text",
             "text": "患者心电图提示窦性心律。", "entities": [], "relations": [],
             "images": ["i1"], "embedding": [1.0, 0.0]},
        ],
    )


class TestCrossModalRetrieval(unittest.TestCase):
    def test_clip_image_search(self):
        hits = _store().search_images_clip([0.0, 1.0], top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["image"]["id"], "i1")
        self.assertEqual(hits[0]["space"], "clip")

    def test_memory_fuses_clip_image_hits(self):
        config = make_config()
        config["retrieval"]["cross_modal"] = True
        config["image"] = {"embedder": "clip"}
        memory = TwoTierMemory(_store(), config, FakeEmbedder(), image_text_encoder=FakeCLIPTextEncoder())
        evidence = memory.retrieve("心电图")
        ids = {item.id for item in evidence}
        self.assertIn("i1", ids)
        self.assertTrue(any(item.modality == "image" for item in evidence))

    def test_cross_modal_disabled_by_default(self):
        config = make_config()
        config["retrieval"]["cross_modal"] = False
        memory = TwoTierMemory(_store(), config, FakeEmbedder(), image_text_encoder=FakeCLIPTextEncoder())
        self.assertIsNone(memory.image_text_encoder)


if __name__ == "__main__":
    unittest.main()
