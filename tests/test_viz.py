import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ROOT, make_config  # noqa: E402

from mmgraph_agent.eval.datasets import load_docs  # noqa: E402
from mmgraph_agent.hypergraph import build_hypergraph  # noqa: E402
from mmgraph_agent.viz import build_view_data, render_hypergraph_html  # noqa: E402


def _store_for(data_root: str, config):
    docs = load_docs(ROOT / "data" / data_root / "corpus.jsonl")
    return build_hypergraph(docs, config)


def _tmpdir() -> Path:
    """A fixed writable dir inside the workspace (git-ignored out/)."""
    base = ROOT / "out" / "test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    return base


class TestViz(unittest.TestCase):
    def test_build_view_data(self):
        store = _store_for("smoke", make_config())
        data = build_view_data(store)
        self.assertEqual(
            len(data["nodes"]), len(store.entities) + len(store.hyperedges) + len(store.images)
        )
        kinds = {node["type"] for node in data["nodes"]}
        self.assertIn("entity", kinds)
        self.assertIn("unit", kinds)
        self.assertIn("image", kinds)
        self.assertGreater(len(data["links"]), 0)

    def test_render_html_self_contained(self):
        store = _store_for("smoke", make_config())
        out = _tmpdir() / "mmkg.html"
        stats = render_hypergraph_html(store, out, "测试超图")
        self.assertTrue(out.exists())
        html = out.read_text(encoding="utf-8")
        self.assertIn("测试超图", html)
        self.assertIn('"nodes"', html)
        self.assertIn('"links"', html)
        self.assertNotIn("<script src=", html)  # no external assets
        self.assertGreater(stats["entities"], 0)

    def test_chinese_labels(self):
        store = _store_for("smoke_zh", make_config())
        labels = " ".join(node["label"] for node in build_view_data(store)["nodes"])
        self.assertIn("糖尿病", labels)
        self.assertIn("冠心病", labels)

    def test_render_from_chinese_corpus(self):
        store = _store_for("smoke_zh", make_config())
        out = _tmpdir() / "mmkg_zh.html"
        render_hypergraph_html(store, out, "中文超图")
        html = out.read_text(encoding="utf-8")
        self.assertIn("糖尿病", html)


if __name__ == "__main__":
    unittest.main()
