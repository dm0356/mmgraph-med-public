import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ROOT  # noqa: E402

from mmgraph_agent.eval.clinical_features import VARIABLES  # noqa: E402
from mmgraph_agent.eval.crf import (  # noqa: E402
    case_to_crf_row,
    load_crf,
    validate_crf,
    write_crf_exports,
)

CRF = load_crf(ROOT / "configs" / "crf_variables.json")
CASES = [
    json.loads(line) for line in (ROOT / "data" / "clinical_demo" / "cases.jsonl")
    .read_text(encoding="utf-8").splitlines() if line.strip()
]


class TestCRF(unittest.TestCase):
    def test_crf_is_valid(self):
        self.assertEqual(validate_crf(CRF), [])
        self.assertGreaterEqual(len(CRF["variables"]), 50)

    def test_extraction_variables_exist(self):
        for variable in CRF["variables"]:
            extraction = variable.get("extraction_variable")
            if extraction:
                self.assertIn(extraction, VARIABLES, variable["name"])

    def test_case_mapping(self):
        row = case_to_crf_row(CASES[0])
        self.assertEqual(row["age"], 78)
        self.assertEqual(row["sex"], "M")
        self.assertEqual(row["LVEF"], 35)
        self.assertEqual(row["potassium"], 3.4)
        self.assertEqual(row["NT_proBNP"], 2100.0)
        self.assertEqual(row["HR"], 85)
        self.assertEqual(row["QRS"], 110)
        self.assertTrue(row["SCDA_event"])
        self.assertEqual(row["event_type"], "VT")
        self.assertTrue(row["amiodarone"])
        self.assertFalse(row["beta_blocker"])
        # I25.103 is chronic CAD, not prior MI
        self.assertFalse(row["prior_MI"])

    def test_features_merge(self):
        features = [{
            "patient_id": "P0001", "variable": "NSVT", "value": 1,
            "status": "present", "confidence": 0.9, "kind": "boolean",
        }]
        row = case_to_crf_row(CASES[0], features=features)
        self.assertEqual(row["NSVT"], 1)

    def test_export_files(self):
        out = ROOT / "out" / "test_crf"
        summary = write_crf_exports(CRF, out, cases=CASES)
        self.assertEqual(summary["demo_rows"], len(CASES))
        self.assertTrue((out / "crf_variable_dictionary.csv").exists())
        self.assertTrue((out / "crf_form_template.csv").exists())
        self.assertTrue((out / "crf_mapping.json").exists())
        self.assertTrue((out / "crf_demo_rows.csv").exists())
        header = (out / "crf_variable_dictionary.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        self.assertIn("label_cn", header)
        demo = (out / "crf_demo_rows.csv").read_text(encoding="utf-8-sig")
        self.assertIn("P0001", demo)


if __name__ == "__main__":
    unittest.main()
