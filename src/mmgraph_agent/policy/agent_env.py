"""Retrieval-and-answer environment for the GRPO policy.

The policy sees a clinical question, may issue ``<search>query</search>`` up to
``max_rounds`` times, and must finish with ``<answer>...</answer>``. The
environment executes retrieval against the hypergraph store and appends the
evidence to the observation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ACTION_RE = re.compile(r"<search>(.*?)</search>|<answer>(.*?)</answer>", re.S | re.I)

SYSTEM_PROMPT = (
    "你是临床预警助手。你只能输出以下两种动作之一，且每次只输出一个动作：\n"
    "1) <search>检索词</search>\n"
    "2) <answer>最终答案</answer>\n"
    "规则：最多检索 3 次；必须先检索至少一次；最终必须用 <answer> 给出答案。\n"
    "不要输出动作之外的任何文字。"
)


@dataclass
class StepResult:
    observation: str
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class RetrievalEnv:
    def __init__(self, store, embedder, config: dict, max_rounds: int = 3, top_k: int = 3) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.question = ""
        self.search_count = 0
        self.evidence: list[str] = []
        self.answer = ""

    def reset(self, question: str) -> str:
        self.question = question
        self.search_count = 0
        self.evidence = []
        self.answer = ""
        return self.system_prompt()

    def system_prompt(self) -> str:
        return (
            SYSTEM_PROMPT
            + "\n示例：\n"
            "<search>VFDB 记录 424 窗口 650.0</search>\n"
            "（系统会返回检索证据，然后你继续输出动作）\n"
            "<answer>baseline</answer>\n"
            f"问题：{self.question}\n"
        )

    def parse(self, text: str) -> StepResult:
        match = ACTION_RE.search(text or "")
        if match:
            search, answer = match.group(1), match.group(2)
            if search is not None and search.strip() and self.search_count < self.max_rounds:
                self.search_count += 1
                query = search.strip()[:200]
                vec = self.embedder.encode_texts([query])[0]
                hits = self.store.search_text(vec, top_k=self.top_k, threshold=0.0)
                lines = [f"[检索结果 {index + 1}] {hit['hyperedge']['text'][:400]}"
                         for index, hit in enumerate(hits)]
                self.evidence.extend(hit["hyperedge"]["text"] for hit in hits)
                return StepResult("\n".join(lines), False, {"action": "search", "query": query})
            if answer is not None and answer.strip():
                self.answer = answer.strip()
                return StepResult("", True, {"action": "answer", "answer": self.answer})
        return StepResult("", True, {"action": "invalid"})
