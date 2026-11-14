import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ROOT, make_config  # noqa: E402

from mmgraph_agent.hypergraph import build_hypergraph  # noqa: E402
from mmgraph_agent.hypergraph.encoder import get_image_embedder  # noqa: E402

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z1ZkAAAAASUVORK5CYII="
)


class FakeImageEncoder:
    name = "fake-clip"
    dim = 4

    def encode_images(self, paths):
        return [np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32) for _ in paths]


class TestImageModality(unittest.TestCase):
    def _docs(self, path):
        return [{
            "doc_id": "d1",
            "text": "患者胸痛，心电图提示窦性心律。",
            "images": [{"image_id": "i1", "caption": "心电图波形", "path": path}],
            "tables": [],
            "diagrams": [],
        }]

    def test_clip_image_vector_is_stored(self):
        config = make_config()
        directory = ROOT / "out" / "test_image_modality"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "img.png").write_bytes(_PNG)
        with patch("mmgraph_agent.hypergraph.builder.get_image_embedder",
                   return_value=FakeImageEncoder()):
            store = build_hypergraph(self._docs("img.png"), config, base_dir=directory)
        self.assertEqual(len(store.images), 1)
        image = store.images[0]
        self.assertIn("image_embedding", image)
        self.assertEqual(image["image_embedder"], "fake-clip")
        self.assertEqual(image["image_embed_dim"], 4)
        self.assertEqual(store.meta["image_embedder"], "fake-clip")
        self.assertEqual(store.meta["image_dim"], 4)
        # caption embedding remains in the text space for normal retrieval
        self.assertEqual(len(image["embedding"]), store.embed_dim)

    def test_missing_image_path_falls_back_to_caption_only(self):
        config = make_config()
        with patch("mmgraph_agent.hypergraph.builder.get_image_embedder",
                   return_value=FakeImageEncoder()):
            store = build_hypergraph(self._docs("missing.png"), config)
        self.assertEqual(len(store.images), 1)
        self.assertNotIn("image_embedding", store.images[0])

    def test_image_encoder_disabled_returns_none(self):
        config = make_config()
        config["image"] = {"embedder": "none"}
        self.assertIsNone(get_image_embedder(config))


if __name__ == "__main__":
    unittest.main()
