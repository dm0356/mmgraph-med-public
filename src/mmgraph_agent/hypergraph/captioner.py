"""Optional API-based multimodal decoding.

The zero-API reproduction path embeds an image unit's *caption*. When the user
wants to decode raw image pixels with a multimodal model (Qwen-VL / DeepSeek-VL
behind an OpenAI-compatible endpoint), enable ``[captioner]`` in the config and
the builder replaces the image unit text with an API-generated description before
it is embedded and linked.

The captioner uses its **own** ``[captioner].api_key_env`` (by default
``DASHSCOPE_API_KEY``) so the text extractor and the vision model can use two
different keys. Set ``[captioner].enabled = true`` to activate it.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from ..http_utils import post_json


class Captioner:
    def describe(self, image_unit: dict) -> str | None:  # pragma: no cover - interface
        return None


class NullCaptioner(Captioner):
    def describe(self, image_unit: dict) -> str | None:
        return None


class OpenAICompatibleCaptioner(Captioner):
    """Call a Qwen-VL / DeepSeek-VL OpenAI-compatible chat endpoint."""

    def __init__(self, config: dict) -> None:
        cap = config.get("captioner", {})
        self.model = cap.get("model") or "qwen3-vl-plus"
        self.base_url = (cap.get("base_url") or "").strip()
        self.api_key_env = cap.get("api_key_env") or "DASHSCOPE_API_KEY"
        self.timeout = float(cap.get("timeout", 120))
        self.prompt = cap.get("prompt") or "请用一句中文描述这张医学图像的关键内容。"
        self.temperature = float(cap.get("temperature", 0.0))

    @property
    def ready(self) -> bool:
        return bool(self.base_url)

    def ping(self) -> str:
        """Text-only probe: verifies endpoint + key without needing an image."""
        return self._post(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "请只回复：OK"}],
                "temperature": self.temperature,
            }
        )

    def describe(self, image_unit: dict) -> str | None:  # pragma: no cover - network
        path = image_unit.get("path")
        if not path or not Path(path).exists() or not self.ready:
            return None
        try:
            payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content = self._post(
                {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
                            ],
                        }
                    ],
                    "temperature": self.temperature,
                }
            )
            return content.strip() if content else None
        except Exception:
            return None

    def _post(self, body: dict) -> str:  # pragma: no cover - network
        if not self.ready:
            raise RuntimeError("captioner is not configured: set [captioner].base_url")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = post_json(self.base_url, body, headers, self.timeout)
        return data["choices"][0]["message"]["content"]


def get_captioner(config: dict) -> Captioner | None:
    cap = config.get("captioner", {})
    if not cap.get("enabled"):
        return None
    return OpenAICompatibleCaptioner(config)


__all__ = ["Captioner", "NullCaptioner", "OpenAICompatibleCaptioner", "get_captioner"]
