import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import make_config  # noqa: E402

from mmgraph_agent.hypergraph import build_hypergraph  # noqa: E402
from mmgraph_agent.hypergraph.construction import ConstructionPipeline  # noqa: E402
from mmgraph_agent.hypergraph.llm_extractor import Extractor  # noqa: E402


class FakeChat:
    def __call__(self, prompt: str) -> str:
        if "改写为" in prompt:
            return "患者李女士有2型糖尿病病史，空腹血糖升高。"
        if "规范化" in prompt:
            return '{"mapping": {"李女士": "李女士", "2型糖尿病": "2型糖尿病", "空腹血糖": "空腹血糖"}}'
        if "百科式描述" in prompt:
            return '{"descriptions": {"2型糖尿病": "一种慢性代谢性疾病", "空腹血糖": "空腹状态下的血糖水平"}}'
        if "候选描述" in prompt:
            return (
                '{"descriptions": {'
                '"2型糖尿病": "一种以高血糖为特征的慢性代谢性疾病", '
                '"空腹血糖": "空腹状态下测得的血糖浓度"}}'
            )
        if "打分" in prompt:
            return '{"scores": {"2型糖尿病": 8, "空腹血糖": 7}, "rewrite": []}'
        return "{}"


class FakeExtractor(Extractor):
    name = "fake"

    def extract(self, text, context=None):
        return ["2型糖尿病", "空腹血糖"]

    def extract_rich(self, text, context=None):
        return {
            "entities": ["2型糖尿病", "空腹血糖"],
            "relations": [{"head": "李女士", "relation": "患有", "tail": "2型糖尿病"}],
        }


def _config():
    config = make_config()
    config["construction"] = {
        "enabled": True,
        "rewriter": True,
        "grounding": True,
        "refinement": True,
        "inspector": False,
    }
    return config


class TestConstruction(unittest.TestCase):
    def test_pipeline_stages(self):
        pipeline = ConstructionPipeline(_config(), chat=FakeChat())
        rewritten = pipeline.rewrite({"doc_id": "d1", "text": "李女士有糖尿病。"})
        self.assertIn("糖尿病", rewritten)
        canonical, descriptions = pipeline.enrich_entities(
            {"doc_id": "d1"}, ["李女士", "2型糖尿病", "空腹血糖"], rewritten
        )
        self.assertEqual(canonical["2型糖尿病"], "2型糖尿病")
        self.assertIn("慢性代谢", descriptions["2型糖尿病"])

    def test_builder_with_construction(self):
        config = _config()
        docs = [{"doc_id": "d1", "text": "李女士有2型糖尿病，空腹血糖升高。", "images": []}]
        pipeline = ConstructionPipeline(config, chat=FakeChat())
        store = build_hypergraph(docs, config, extractor=FakeExtractor(), construction=pipeline)
        names = [entity["name"] for entity in store.entities]
        self.assertIn("2型糖尿病", names)
        entity = next(entity for entity in store.entities if entity["name"] == "2型糖尿病")
        self.assertIn("慢性代谢", entity["description"])
        self.assertEqual(store.meta["construction"], "on")


if __name__ == "__main__":
    unittest.main()
