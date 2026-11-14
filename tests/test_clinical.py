import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmgraph_agent.eval.clinical_metrics import (  # noqa: E402
    auroc,
    auprc,
    brier_score,
    expected_calibration_error,
    evaluate_predictions,
    lead_time_hours,
    sensitivity_specificity,
)
from mmgraph_agent.eval.clinical_schema import (  # noqa: E402
    case_to_qa,
    cases_to_corpus,
    validate_case,
    window_label,
)


def _case(event="VT", event_time="2026-01-27T02:30:00", horizon=48):
    return {
        "case_id": "P1_W1",
        "patient_id": "P1",
        "encounter_id": "E1",
        "window": {"start": "2026-01-26T00:00:00", "end": "2026-01-27T00:00:00", "hours": 24},
        "demographics": {"age": 78, "sex": "M"},
        "diagnoses": [{"code": "I50.900", "name": "心力衰竭"}],
        "labs": [],
        "medications": [],
        "ecg": [],
        "echo": [],
        "cmr": [],
        "notes": [],
        "labels": {
            "event": event,
            "event_time": event_time,
            "event_source": "adjudicated",
            "outcome_horizon_hours": horizon,
        },
        "split": "train",
        "deidentified": True,
    }


class TestClinicalSchema(unittest.TestCase):
    def test_valid_case(self):
        self.assertEqual(validate_case(_case()), [])

    def test_positive_pre_event_window(self):
        self.assertEqual(window_label(_case()), 1)

    def test_negative_window(self):
        self.assertEqual(window_label(_case(event="none", event_time=None)), 0)

    def test_event_inside_window_is_excluded(self):
        self.assertIsNone(window_label(_case(event_time="2026-01-26T12:00:00")))

    def test_event_beyond_horizon_is_negative(self):
        self.assertEqual(window_label(_case(event_time="2026-02-10T00:00:00")), 0)

    def test_invalid_sex_and_deidentification(self):
        case = _case()
        case["demographics"]["sex"] = "X"
        case["deidentified"] = False
        errors = validate_case(case)
        self.assertTrue(any("sex" in error for error in errors))
        self.assertTrue(any("deidentified" in error for error in errors))

    def test_case_to_corpus(self):
        docs, qa = cases_to_corpus([_case()])
        self.assertGreaterEqual(len(docs), 1)
        self.assertEqual(qa[0]["answer"], "是")
        self.assertEqual(case_to_qa(_case())["answer"], "是")


class TestClinicalMetrics(unittest.TestCase):
    def test_auroc_perfect_and_reversed(self):
        self.assertEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)

    def test_auprc_perfect(self):
        self.assertAlmostEqual(auprc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0, places=6)

    def test_sensitivity_specificity(self):
        sensitivity, specificity = sensitivity_specificity([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 0.5)
        self.assertEqual(sensitivity, 1.0)
        self.assertEqual(specificity, 1.0)

    def test_calibration_perfect(self):
        self.assertEqual(brier_score([0, 1], [0.0, 1.0]), 0.0)
        self.assertEqual(expected_calibration_error([0, 1], [0.0, 1.0]), 0.0)

    def test_lead_time(self):
        stats = lead_time_hours(
            ["2026-01-01T00:00:00", "2026-01-01T06:00:00"],
            ["2026-01-01T10:00:00", "2026-01-01T10:00:00"],
        )
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["median_hours"], 7.0)

    def test_evaluate_predictions(self):
        rows = [
            {"label": 0, "score": 0.1},
            {"label": 0, "score": 0.2},
            {"label": 1, "score": 0.8},
            {"label": 1, "score": 0.9},
        ]
        summary = evaluate_predictions(rows, specificity=0.5, n_boot=20)
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
