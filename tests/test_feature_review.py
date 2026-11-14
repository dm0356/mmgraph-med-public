import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmgraph_agent.eval.clinical_features_llm import (  # noqa: E402
    _supported,
    extract_report_hybrid,
)
from mmgraph_agent.eval.clinical_features import VARIABLES  # noqa: E402
from mmgraph_agent.eval.clinical_review import make_review_rows, score_review  # noqa: E402


class FakeLLM:
    """Minimal LLM stand-in: returns fixed records for requested variables."""

    def __init__(self, records):
        self.records = records
        self.requested = []

    def extract(self, text, variables, patient_id="", report_time="", source=""):
        self.requested.extend(variables)
        return [
            {**record, "patient_id": patient_id, "time": report_time, "source": source,
             "extractor": "llm"}
            for record in self.records
            if record["variable"] in variables
        ]


class TestHybridExtraction(unittest.TestCase):
    def test_llm_fills_missing_and_keeps_rule_values(self):
        fake = FakeLLM([
            {"variable": "potassium", "value": 3.4, "unit": "mmol/L", "status": "present",
             "confidence": 0.75, "snippet": "血钾3.4", "kind": "numeric"},
            {"variable": "SAM", "value": True, "unit": "", "status": "present",
             "confidence": 0.75, "snippet": "SAM阳性", "kind": "boolean"},
        ])
        records, conflicts = extract_report_hybrid(
            "LVEF 35%，可疑SAM，血钾未在报告中给出。", patient_id="P1", llm=fake
        )
        by = {record["variable"]: record for record in records}
        self.assertEqual(by["LVEF"]["value"], 35.0)          # rule present kept
        self.assertEqual(by["SAM"]["value"], True)            # uncertain upgraded
        self.assertEqual(by["potassium"]["value"], 3.4)       # missing filled
        self.assertEqual(by["potassium"]["extractor"], "llm")
        self.assertEqual(conflicts, [])

    def test_conflict_when_rule_uncertain_llm_absent(self):
        fake = FakeLLM([
            {"variable": "LGE_present", "value": False, "unit": "", "status": "absent",
             "confidence": 0.5, "snippet": "LGE未见", "kind": "boolean"},
        ])
        records, conflicts = extract_report_hybrid("可疑LGE。", patient_id="P1", llm=fake)
        by = {record["variable"]: record for record in records}
        self.assertFalse(by["LGE_present"]["value"])
        self.assertEqual(len(conflicts), 1)

    def test_llm_evidence_support(self):
        spec = VARIABLES["LVEF"]
        self.assertTrue(_supported("左心室射血分数 35%，", "LVEF", spec, 35,
                                   evidence="左心室射血分数 35%"))
        self.assertFalse(_supported("左心室射血分数 35%，", "LVEF", spec, 50,
                                    evidence="左心室射血分数 35%"))


class TestReviewWorkflow(unittest.TestCase):
    def test_review_rows_prioritize_uncertain(self):
        records = [
            {"patient_id": "P1", "variable": "SAM", "value": None, "status": "uncertain",
             "confidence": 0.6, "snippet": "可疑SAM", "extractor": "rule"},
            {"patient_id": "P1", "variable": "LVEF", "value": 35, "status": "present",
             "confidence": 0.95, "snippet": "LVEF 35%", "extractor": "rule"},
        ]
        rows = make_review_rows(records, max_rows=10)
        self.assertEqual(rows[0]["variable"], "SAM")
        self.assertEqual(rows[0]["needs_review"], "uncertain")
        self.assertIn("review_value", rows[0])

    def test_score_review(self):
        records = [
            {"patient_id": "P1", "variable": "LVEF", "value": 35, "status": "present",
             "confidence": 0.95, "kind": "numeric"},
            {"patient_id": "P1", "variable": "SAM", "value": None, "status": "missing",
             "confidence": 0.0, "kind": "boolean"},
        ]
        reviewed = [
            {"patient_id": "P1", "variable": "LVEF", "review_value": "35", "review_status": "present"},
            {"patient_id": "P1", "variable": "SAM", "review_value": "false", "review_status": "absent"},
        ]
        summary = score_review(records, reviewed)
        self.assertEqual(summary["reviewed"], 2)
        self.assertEqual(summary["status_accuracy"], 0.5)
        self.assertEqual(summary["present_precision"], 1.0)
        self.assertEqual(summary["present_recall"], 1.0)
        self.assertAlmostEqual(summary["numeric_mae"], 0.0)
        self.assertIn("cohen_kappa", summary)


if __name__ == "__main__":
    unittest.main()
