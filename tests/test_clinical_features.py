import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmgraph_agent.eval.clinical_features import extract_report, to_wide_csv  # noqa: E402


def _by_variable(records):
    return {record["variable"]: record for record in records}


class TestClinicalFeatures(unittest.TestCase):
    def test_numeric_and_units(self):
        records = _by_variable(extract_report(
            "左室射血分数约35%，室间隔基底段厚度约1.2cm，血钾3.4mmol/L。", patient_id="P1"
        ))
        self.assertEqual(records["LVEF"]["value"], 35.0)
        self.assertEqual(records["LVEF"]["status"], "present")
        self.assertEqual(records["IVS_thickness"]["value"], 12.0)  # cm -> mm
        self.assertEqual(records["potassium"]["value"], 3.4)

    def test_negation_uncertainty_and_missing(self):
        records = _by_variable(extract_report(
            "LGE未见明显强化，可疑SAM，未见心尖部血栓。", patient_id="P1"
        ))
        self.assertFalse(records["LGE_present"]["value"])
        self.assertEqual(records["LGE_present"]["status"], "absent")
        self.assertIsNone(records["SAM"]["value"])
        self.assertEqual(records["SAM"]["status"], "uncertain")
        self.assertFalse(records["thrombus"]["value"])
        # not mentioned -> explicit missing
        self.assertEqual(records["QTc"]["status"], "missing")
        self.assertIsNone(records["QTc"]["value"])

    def test_categorical(self):
        records = _by_variable(extract_report("二尖瓣反流中度，未见心包积液。", patient_id="P1"))
        self.assertEqual(records["mitral_regurgitation"]["value"], "中度")
        self.assertEqual(records["pericardial_effusion"]["status"], "absent")

    def test_out_of_range_flagged(self):
        records = _by_variable(extract_report("LVEF 120%。", patient_id="P1"))
        self.assertEqual(records["LVEF"]["status"], "out_of_range")

    def test_provenance_and_wide_csv(self):
        records = extract_report("LVEF 42%，QRS 110ms。", patient_id="P1",
                                 report_time="2026-01-01", source="echo")
        lvef = _by_variable(records)["LVEF"]
        self.assertIn("LVEF 42%", lvef["snippet"])
        self.assertEqual(lvef["source"], "echo")
        self.assertEqual(lvef["time"], "2026-01-01")
        csv = to_wide_csv(records)
        self.assertIn("patient_id,", csv)
        self.assertIn("P1,", csv)


if __name__ == "__main__":
    unittest.main()
