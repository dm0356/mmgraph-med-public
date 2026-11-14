import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import make_config  # noqa: E402

from mmgraph_agent.hypergraph import build_hypergraph  # noqa: E402
from mmgraph_agent.hypergraph.llm_extractor import (  # noqa: E402
    Extractor,
    LLMExtractor,
    RegexExtractor,
    _parse_json_payload,
    get_extractor,
)


class FakeExtractor(Extractor):
    name = "fake"

    def extract(self, text):
        return ["假实体A", "假实体B"]

    def extract_rich(self, text):
        return {
            "entities": ["假实体A", "假实体B"],
            "relations": [{"head": "假实体A", "relation": "关联", "tail": "假实体B"}],
        }


class TestLLMExtractor(unittest.TestCase):
    def test_parse_json_variants(self):
        self.assertEqual(_parse_json_payload('{"entities": ["a"]}')["entities"], ["a"])
        self.assertEqual(_parse_json_payload('```json\n{"entities": ["b"]}\n```')["entities"], ["b"])
        self.assertEqual(_parse_json_payload('前缀 {"entities": ["c"]} 后缀')["entities"], ["c"])
        self.assertIsNone(_parse_json_payload("not json at all"))
        self.assertIsNone(_parse_json_payload(None))

    def test_regex_extractor_chinese(self):
        config = make_config()
        entities = RegexExtractor(config).extract("患者有糖尿病，空腹血糖升高。")
        self.assertIn("糖尿病", entities)
        self.assertIn("空腹血糖", entities)

    def test_get_extractor_modes(self):
        config = make_config()
        config["extractor"] = {"mode": "regex"}
        self.assertIsInstance(get_extractor(config), RegexExtractor)
        config["extractor"] = {"mode": "llm", "base_url": "http://x"}
        self.assertIsInstance(get_extractor(config), LLMExtractor)
        config["extractor"] = {"mode": "auto", "base_url": ""}
        self.assertIsInstance(get_extractor(config), RegexExtractor)

    def test_llm_falls_back_on_error(self):
        config = make_config()
        config["extractor"] = {"mode": "llm", "base_url": "http://localhost:1", "max_retries": 0}
        extractor = LLMExtractor(config)

        def boom(prompt):
            raise RuntimeError("no network")

        extractor._chat = boom  # noqa: SLF001 - test seam
        result = extractor.extract_rich("患者有糖尿病")
        self.assertIn("糖尿病", result["entities"])

    def test_builder_uses_custom_extractor(self):
        config = make_config()
        docs = [{"doc_id": "d1", "text": "甲乙丙发生了关联。", "images": []}]
        store = build_hypergraph(docs, config, extractor=FakeExtractor())
        names = [entity["name"] for entity in store.entities]
        self.assertIn("假实体A", names)
        self.assertIn("假实体B", names)
        self.assertTrue(any(edge["relations"] for edge in store.hyperedges))

    def test_default_api_keys_are_separate(self):
        from mmgraph_agent.config import DEFAULTS

        self.assertNotEqual(
            DEFAULTS["extractor"]["api_key_env"], DEFAULTS["captioner"]["api_key_env"]
        )

    def test_captioner_has_ping(self):
        from mmgraph_agent.hypergraph.captioner import OpenAICompatibleCaptioner

        self.assertTrue(hasattr(OpenAICompatibleCaptioner, "ping"))

    def test_env_overrides_endpoints(self):
        import os
        from unittest.mock import patch

        from mmgraph_agent.config import load_config

        env = {
            "MMGRAPH_EXTRACTOR_BASE_URL": "https://api.deepseek.com/v1/chat/completions",
            "MMGRAPH_CAPTIONER_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "MMGRAPH_CAPTIONER_ENABLED": "true",
        }
        with patch.dict(os.environ, env):
            config = load_config()
        self.assertEqual(
            config["extractor"]["base_url"], "https://api.deepseek.com/v1/chat/completions"
        )
        self.assertTrue(config["captioner"]["enabled"])


if __name__ == "__main__":
    unittest.main()
