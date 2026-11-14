import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import make_config  # noqa: E402

from mmgraph_agent.hypergraph.parser import extract_entities, is_text_unit, parse_documents  # noqa: E402


class TestParser(unittest.TestCase):
    def setUp(self):
        self.config = make_config()

    def test_units_and_modalities(self):
        docs = [
            {
                "doc_id": "d1",
                "text": "Alpha one. Beta two.",
                "images": [{"image_id": "i1", "caption": "a picture"}],
                "tables": [{"table_id": "t1", "text": "row one"}],
            }
        ]
        units = parse_documents(docs, self.config)
        modalities = [unit["modality"] for unit in units]
        self.assertIn("text", modalities)
        self.assertIn("image", modalities)
        self.assertIn("table", modalities)
        self.assertEqual(len([u for u in units if is_text_unit(u)]), 3)

    def test_extract_entities_regex(self):
        ents = extract_entities("The Eiffel Tower is located in Paris, France.", self.config)
        self.assertIn("Eiffel Tower", ents)
        self.assertIn("Paris", ents)


if __name__ == "__main__":
    unittest.main()
