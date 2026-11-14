"""Unified action space and prompt template (Module 3, paper §3.3).

The policy emits one of three actions in a single autoregressive stream:
text retrieval, image retrieval, or answer. This mirrors Table 2 of the paper.
"""

from __future__ import annotations

import re

TEXT_SEARCH = "text_search"
IMAGE_SEARCH = "image_search"
ANSWER = "answer"

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the given question based on the provided "
    "images; you can query the knowledge base as many times as you want. You must "
    "first conduct reasoning inside <reason>...</reason>. For image search, append "
    "<search><img></search> behind the reasoning, which runs Google Lens on the "
    "original image; for text search, output a concise query inside "
    "<text_search>...</text_search>, answered by KG/Memory retrieval (and Google "
    "Search when enabled). Retrieved evidence returns inside "
    "<information>...</information>; when you have the final answer, output it "
    "inside <answer>...</answer>. Here is the image and the question: <image> "
    "{question} Assistant:"
)

_TEXT_RE = re.compile(r"<text_search>(.*?)</text_search>", re.S)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def format_prompt(question: str, evidence_text: str = "") -> str:
    prompt = PROMPT_TEMPLATE.format(question=question)
    if evidence_text:
        prompt = prompt.replace(
            "<image> {question}".format(question=question),
            evidence_text + " <image> {question}".format(question=question),
        )
    return prompt


def parse_action(model_output: str) -> tuple[str, str | None]:
    """Parse a raw model string into ``(action, payload)``."""
    match = _TEXT_RE.search(model_output)
    if match:
        return TEXT_SEARCH, match.group(1).strip()
    if "<search>" in model_output and "<img>" in model_output:
        return IMAGE_SEARCH, None
    match = _ANSWER_RE.search(model_output)
    if match:
        return ANSWER, match.group(1).strip()
    stripped = model_output.strip()
    if "<reason>" in stripped:
        stripped = stripped.split("</reason>")[-1].strip()
    return ANSWER, stripped


__all__ = [
    "TEXT_SEARCH",
    "IMAGE_SEARCH",
    "ANSWER",
    "PROMPT_TEMPLATE",
    "format_prompt",
    "parse_action",
]
