"""GRPO reward + training scaffold (Module 3, paper §3.3).

The paper trains a Qwen2.5-VL-3B policy with GRPO and a *hit-aware* reward:
a token-level F1 term multiplied by a concave call-count regulator that pays
for the first retrieval and decays the marginal bonus of extra calls, plus a
structural penalty on malformed or retrieval-free trajectories.

``compute_reward`` is pure-Python and unit-testable. The functional single-GPU
trainer lives in :mod:`mmgraph_agent.policy.grpo_trainer`; the legacy
``GRPOTrainer`` below is kept only as a lightweight compatibility stub.
"""

from __future__ import annotations

from typing import Any


def regulator(num_retrievals: int, base: float = 0.5) -> float:
    """Concave call-count regulator: gamma(1)=1, gamma(2)=base, gamma(3)=base^2 ..."""
    if num_retrievals <= 0:
        return 0.0
    if num_retrievals == 1:
        return 1.0
    return float(base ** (num_retrievals - 1))


def compute_reward(
    token_f1: float,
    num_retrievals: int,
    well_formed: bool,
    config: dict[str, Any],
) -> float:
    """Reward used during GRPO rollouts (reproduction of paper Eq. (6)).

    Parameters
    ----------
    token_f1 : token-level F1 of the final answer against the gold answer.
    num_retrievals : number of retrieval calls in the trajectory.
    well_formed : whether the trajectory ended in a valid ``<answer>`` and
        respected the output schema.
    config : full config dict (reads ``[policy]``).
    """
    policy = config["policy"]
    base = float(policy.get("regulator_base", 0.5))
    penalty = float(policy.get("structural_penalty", 0.5))

    reward = token_f1 * regulator(num_retrievals, base)
    if not well_formed:
        reward -= penalty
    if num_retrievals == 0:
        reward -= penalty
    return max(-penalty, float(reward))


class GRPOTrainer:
    """Optional GRPO training scaffold.

    This class only exists so a GPU reproduction can be wired in without
    changing the rest of the codebase. Constructing it requires the ``gpu``
    extras; everything below is intentionally a documented stub.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._torch = None
        try:  # pragma: no cover - requires optional deps
            import torch  # type: ignore

            self._torch = torch
        except Exception:
            self._torch = None

    @property
    def available(self) -> bool:
        return self._torch is not None

    def train(self, dataset: Any, output_dir: str) -> None:  # pragma: no cover
        """TODO(reproduction): implement GRPO updates around a Qwen2.5-VL-3B."""
        raise NotImplementedError(
            "GRPO training needs the GPU stack (torch + transformers + a "
            "Qwen2.5-VL-3B checkpoint). Install `pip install -e '.[gpu]'`, "
            "point at a dataset with rollouts, and implement this loop; the "
            "reward is provided by `compute_reward`."
        )


__all__ = ["regulator", "compute_reward", "GRPOTrainer"]
