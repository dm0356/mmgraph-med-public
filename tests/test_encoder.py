import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import make_config  # noqa: E402

from mmgraph_agent.hypergraph.encoder import APIEmbedder, HashingEmbedder, get_embedder  # noqa: E402


class TestEncoder(unittest.TestCase):
    def test_hashing(self):
        config = make_config()
        config["hypergraph"]["embedder"] = "hashing"
        embedder = get_embedder(config)
        self.assertIsInstance(embedder, HashingEmbedder)
        vector = embedder.encode_texts(["糖尿病"])[0]
        self.assertEqual(vector.shape, (256,))

    def test_api_embedder_type(self):
        config = make_config()
        config["hypergraph"]["embedder"] = "api"
        config["embedding"] = {"base_url": "http://127.0.0.1:8002/v1/embeddings", "dim": 1024}
        embedder = get_embedder(config)
        self.assertIsInstance(embedder, APIEmbedder)
        self.assertEqual(embedder.dim, 1024)

    def test_api_embedder_requires_base_url(self):
        config = make_config()
        config["hypergraph"]["embedder"] = "api"
        config["embedding"] = {"base_url": ""}
        embedder = get_embedder(config)
        with self.assertRaises(RuntimeError):
            embedder.encode_texts(["x"])


if __name__ == "__main__":
    unittest.main()
