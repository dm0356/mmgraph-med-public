import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import ROOT  # noqa: E402

from mmgraph_agent.eval.validation import validate_corpus, validate_qa  # noqa: E402


def _write(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.dir = ROOT / "out" / "test_validate"
        self.dir.mkdir(parents=True, exist_ok=True)

    def test_valid_corpus(self):
        path = self.dir / "ok.jsonl"
        _write(path, [{"doc_id": "d1", "text": "患者发热。", "images": [{"image_id": "i1", "caption": "图"}]}])
        report = validate_corpus(path)
        self.assertTrue(report["ok"])
        self.assertEqual(report["stats"]["docs"], 1)
        self.assertEqual(report["stats"]["images"], 1)

    def test_missing_doc_id(self):
        path = self.dir / "missing.jsonl"
        _write(path, [{"text": "x"}])
        self.assertFalse(validate_corpus(path)["ok"])

    def test_duplicate_doc_id(self):
        path = self.dir / "dup.jsonl"
        _write(path, [{"doc_id": "d1", "text": "a"}, {"doc_id": "d1", "text": "b"}])
        self.assertFalse(validate_corpus(path)["ok"])

    def test_missing_image_path(self):
        path = self.dir / "badimg.jsonl"
        _write(path, [{"doc_id": "d1", "text": "a", "images": [{"image_id": "i1", "path": "nope.png"}]}])
        self.assertFalse(validate_corpus(path)["ok"])

    def test_empty_content_warns(self):
        path = self.dir / "empty.jsonl"
        _write(path, [{"doc_id": "d1"}])
        report = validate_corpus(path)
        self.assertTrue(report["ok"])
        self.assertTrue(report["warnings"])

    def test_valid_qa(self):
        path = self.dir / "qa.jsonl"
        _write(path, [{"qid": "q1", "question": "x?", "answer": "y", "has_image": True}])
        report = validate_qa(path)
        self.assertTrue(report["ok"])
        self.assertEqual(report["stats"]["has_image"], 1)

    def test_qa_missing_answer(self):
        path = self.dir / "qa_bad.jsonl"
        _write(path, [{"qid": "q1", "question": "x?"}])
        self.assertFalse(validate_qa(path)["ok"])


if __name__ == "__main__":
    unittest.main()
