import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ROOT  # noqa: E402

from mmgraph_agent.hypergraph.knowledge_searcher import ICD10Searcher  # noqa: E402
from scripts.import_icd10 import convert, write_tree  # noqa: E402


class TestICD10Importer(unittest.TestCase):
    def setUp(self):
        self.dir = ROOT / "out" / "test_icd10_import"
        self.dir.mkdir(parents=True, exist_ok=True)

    def test_convert_chinese_header(self):
        rows = [
            ["编码", "名称"],
            ["I25.101", "冠状动脉粥样硬化性心脏病"],
            ["I21.401", "急性非ST段抬高型心肌梗死"],
            ["I50.900", "心力衰竭"],
            ["E11.900", "2型糖尿病"],
        ]
        entries = convert(rows)
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[0], {"value": "I25.101", "label": "冠状动脉粥样硬化性心脏病"})

    def test_roundtrip_to_searcher(self):
        rows = [
            ["code", "name"],
            ["I25.101", "冠状动脉粥样硬化性心脏病"],
            ["I21.401", "急性非ST段抬高型心肌梗死"],
        ]
        path = self.dir / "diseases.json"
        write_tree(convert(rows), path)
        searcher = ICD10Searcher(path)
        self.assertEqual(
            searcher.search("冠心病"), ["ICD-10 I25.101 冠状动脉粥样硬化性心脏病"]
        )
        self.assertEqual(
            searcher.search("急性非ST段抬高型心肌梗死"),
            ["ICD-10 I21.401 急性非ST段抬高型心肌梗死"],
        )

    def test_dedup_codes(self):
        rows = [["编码", "名称"], ["I50.900", "心力衰竭"], ["I50.900", "心力衰竭（重复）"]]
        entries = convert(rows)
        self.assertEqual(len(entries), 1)

    def test_official_header_names(self):
        rows = [
            ["疾病诊断编码", "疾病诊断名称"],
            ["I25.103", "冠状动脉粥样硬化性心脏病"],
            ["I50.900", "心力衰竭"],
        ]
        entries = convert(rows)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["value"], "I25.103")


if __name__ == "__main__":
    unittest.main()
