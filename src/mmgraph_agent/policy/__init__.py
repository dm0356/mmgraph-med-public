"""Unified action space, heuristic policy and GRPO reward (paper §3.3)."""

from .actions import (
    ANSWER,
    IMAGE_SEARCH,
    PROMPT_TEMPLATE,
    TEXT_SEARCH,
    format_prompt,
    parse_action,
)
from .agent_env import RetrievalEnv
from .grpo import GRPOTrainer, compute_reward, regulator
from .infer import HeuristicPolicy, best_sentence, short_answer

# The functional trainer is imported from ``mmgraph_agent.policy.grpo_trainer``
# directly, so that this package stays importable on CPU-only / no-torch setups.

__all__ = [
    "TEXT_SEARCH",
    "IMAGE_SEARCH",
    "ANSWER",
    "PROMPT_TEMPLATE",
    "format_prompt",
    "parse_action",
    "HeuristicPolicy",
    "best_sentence",
    "short_answer",
    "regulator",
    "compute_reward",
    "GRPOTrainer",
    "RetrievalEnv",
]
