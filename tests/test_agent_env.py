import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmgraph_agent.policy.agent_env import RetrievalEnv  # noqa: E402


class FakeStore:
    def search_text(self, vec, top_k=3, threshold=0.0):
        return [{"hyperedge": {"text": "心电图显示窦性心律，标签 baseline。"}}]


class FakeEmbedder:
    def encode_texts(self, texts):
        return [[0.0, 0.0]]


class TestRetrievalEnv(unittest.TestCase):
    def setUp(self):
        config = {"policy": {"regulator_base": 0.5, "structural_penalty": 0.5}}
        self.env = RetrievalEnv(FakeStore(), FakeEmbedder(), config, max_rounds=2, top_k=1)

    def test_reset_contains_question(self):
        prompt = self.env.reset("该窗口的标签是什么？")
        self.assertIn("该窗口的标签是什么", prompt)
        self.assertIn("<search>", prompt)
        self.assertIn("<answer>", prompt)

    def test_search_action(self):
        self.env.reset("问题")
        step = self.env.parse("<search>VFDB 记录 424</search>")
        self.assertFalse(step.done)
        self.assertEqual(step.info["action"], "search")
        self.assertEqual(self.env.search_count, 1)
        self.assertIn("窦性心律", step.observation)

    def test_answer_action(self):
        self.env.reset("问题")
        step = self.env.parse("<answer>baseline</answer>")
        self.assertTrue(step.done)
        self.assertEqual(self.env.answer, "baseline")

    def test_invalid_action(self):
        self.env.reset("问题")
        step = self.env.parse("我觉得是 baseline")
        self.assertTrue(step.done)
        self.assertEqual(step.info["action"], "invalid")
        self.assertEqual(self.env.answer, "")

    def test_max_rounds(self):
        self.env.reset("问题")
        self.env.parse("<search>a</search>")
        self.env.parse("<search>b</search>")
        step = self.env.parse("<search>c</search>")
        self.assertTrue(step.done)
        self.assertEqual(step.info["action"], "invalid")


if __name__ == "__main__":
    unittest.main()
