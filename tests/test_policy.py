import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import make_config  # noqa: E402

from mmgraph_agent.memory import Evidence  # noqa: E402
from mmgraph_agent.policy import (  # noqa: E402
    ANSWER,
    TEXT_SEARCH,
    HeuristicPolicy,
    compute_reward,
    parse_action,
    regulator,
)


class TestPolicy(unittest.TestCase):
    def test_parse_action(self):
        self.assertEqual(
            parse_action("<reason>r</reason><text_search>eiffel tower</text_search>"),
            (TEXT_SEARCH, "eiffel tower"),
        )
        self.assertEqual(parse_action("<reason>r</reason><answer>Paris</answer>"), (ANSWER, "Paris"))
        self.assertEqual(parse_action("plain answer"), (ANSWER, "plain answer"))

    def test_regulator(self):
        self.assertEqual(regulator(0), 0.0)
        self.assertEqual(regulator(1), 1.0)
        self.assertEqual(regulator(2), 0.5)
        self.assertEqual(regulator(3), 0.25)

    def test_compute_reward(self):
        config = make_config()
        self.assertEqual(compute_reward(1.0, 1, True, config), 1.0)
        self.assertEqual(compute_reward(1.0, 2, True, config), 0.5)
        self.assertEqual(compute_reward(1.0, 0, False, config), -0.5)

    def test_heuristic_short_answer(self):
        config = make_config()
        evidence = [Evidence(id="h", source="offline", modality="text", text="The tower is 330 metres tall.", score=0.9)]
        self.assertEqual(HeuristicPolicy(config).answer("How tall is the tower?", evidence), "330 metres")


if __name__ == "__main__":
    unittest.main()
