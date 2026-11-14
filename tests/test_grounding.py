import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ROOT, make_config  # noqa: E402

from mmgraph_agent.hypergraph.construction import ConstructionPipeline  # noqa: E402
from mmgraph_agent.hypergraph.knowledge_searcher import (  # noqa: E402
    ICD10Searcher,
    KnowledgeSearcher,
    LocalGlossarySearcher,
    WikipediaSearcher,
)


class FakeSearcher(KnowledgeSearcher):
    name = "fake"

    def search(self, term):
        return ["外部知识库描述"] if term == "2型糖尿病" else []


class FakeChat:
    def __call__(self, prompt):
        if "规范化" in prompt:
            return '{"mapping": {"2型糖尿病": "2型糖尿病"}}'
        if "百科式描述" in prompt:
            return '{"descriptions": {"2型糖尿病": "模型生成的描述"}}'
        if "候选描述列表" in prompt:
            return '{"descriptions": {"2型糖尿病": "外部知识库描述"}}'
        return "{}"


class TestKnowledgeGrounding(unittest.TestCase):
    def test_local_glossary_tsv(self):
        searcher = LocalGlossarySearcher(ROOT / "data" / "glossary" / "medical_terms.sample.tsv")
        hits = searcher.search("2型糖尿病")
        self.assertTrue(hits)
        self.assertIn("代谢", hits[0])

    def test_local_glossary_jsonl_with_aliases(self):
        path = ROOT / "out" / "test_grounding" / "terms.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"term": "2型糖尿病", "description": "慢性代谢病", "aliases": ["T2DM"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        searcher = LocalGlossarySearcher(path)
        self.assertEqual(searcher.search("t2dm"), ["慢性代谢病"])

    def test_wikipedia_searcher_parses_extract(self):
        searcher = WikipediaSearcher(make_config())
        searcher._get_json = lambda params: {"query": {"pages": {"1": {"extract": "一种慢性代谢性疾病。"}}}}
        self.assertEqual(searcher.search("2型糖尿病"), ["一种慢性代谢性疾病。"])

    def test_pipeline_uses_external_candidate(self):
        config = make_config()
        config["construction"] = {
            "enabled": True,
            "rewriter": True,
            "grounding": True,
            "refinement": False,
            "inspector": False,
        }
        pipeline = ConstructionPipeline(config, chat=FakeChat(), searcher=FakeSearcher())
        canonical, descriptions = pipeline.enrich_entities(
            {"doc_id": "d1"}, ["2型糖尿病"], "患者有2型糖尿病。"
        )
        self.assertEqual(descriptions["2型糖尿病"], "外部知识库描述")


class ICD10Fixture:
    def __init__(self):
        self.dir = ROOT / "out" / "test_icd10"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "diseases.json"
        self.path.write_text(
            json.dumps(
                [
                    {
                        "value": "I25",
                        "label": "慢性缺血性心脏病",
                        "children": [{"value": "I25.1", "label": "冠状动脉粥样硬化性心脏病"}],
                    },
                    {
                        "value": "E10-E14",
                        "label": "糖尿病",
                        "children": [{"value": "E11", "label": "2型糖尿病"}],
                    },
                    {
                        "value": "I10-I15",
                        "label": "高血压病",
                        "children": [{"value": "I10", "label": "原发性高血压"}],
                    },
                    {"value": "I50", "label": "心力衰竭"},
                    {"value": "E06", "label": "甲状腺炎"},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def close(self):
        pass


class TestICD10Searcher(unittest.TestCase):
    def setUp(self):
        self.fixture = ICD10Fixture()
        self.searcher = ICD10Searcher(self.fixture.path)

    def tearDown(self):
        self.fixture.close()

    def test_alias_resolves_to_icd_name(self):
        self.assertEqual(self.searcher.search("冠心病"), ["ICD-10 I25.1 冠状动脉粥样硬化性心脏病"])

    def test_canonical_and_preferred(self):
        self.assertEqual(self.searcher.canonical("冠心病"), "冠状动脉粥样硬化性心脏病")
        self.assertEqual(self.searcher.preferred("2型糖尿病"), "ICD-10 E11 2型糖尿病")

    def test_exact_name_beats_contains(self):
        self.assertEqual(self.searcher.canonical("心力衰竭"), "心力衰竭")

    def test_history_and_grade_suffixes(self):
        self.assertEqual(self.searcher.search("糖尿病史"), ["ICD-10 E10-E14 糖尿病"])
        self.assertEqual(self.searcher.search("高血压一级"), ["ICD-10 I10-I15 高血压病"])

    def test_anatomy_is_not_coded(self):
        self.assertIsNone(self.searcher.preferred("甲状腺"))
        self.assertIsNone(self.searcher.preferred("下肢静脉"))
        self.assertIsNone(self.searcher.preferred("肌张力"))

    def test_missing_dictionary_is_safe(self):
        searcher = ICD10Searcher(self.fixture.dir / "missing.json")
        self.assertEqual(searcher.search("冠心病"), [])
        self.assertIsNone(searcher.canonical("冠心病"))


class TestConstructionWithICD(unittest.TestCase):
    def test_icd_name_and_description_are_authoritative(self):
        fixture = ICD10Fixture()
        try:
            config = make_config()
            config["construction"] = {
                "enabled": True,
                "rewriter": False,
                "grounding": True,
                "refinement": False,
                "inspector": False,
            }
            pipeline = ConstructionPipeline(
                config, chat=lambda prompt: "{}", searcher=ICD10Searcher(fixture.path)
            )
            canonical, descriptions = pipeline.enrich_entities(
                {"doc_id": "d1"}, ["冠心病"], "患者诊断为冠心病。"
            )
        finally:
            fixture.close()
        self.assertEqual(canonical["冠心病"], "冠状动脉粥样硬化性心脏病")
        self.assertEqual(
            descriptions["冠状动脉粥样硬化性心脏病"],
            "ICD-10 I25.1 冠状动脉粥样硬化性心脏病",
        )


if __name__ == "__main__":
    unittest.main()
