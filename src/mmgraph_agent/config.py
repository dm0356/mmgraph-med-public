"""Configuration loading.

Configs are plain TOML (``tomllib`` is in the standard library) so the project
has no YAML dependency. Any key omitted by the user falls back to ``DEFAULTS``.
"""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "project": {"name": "mmgraph-agent"},
    "hypergraph": {
        "parser": "auto",
        "embedder": "auto",
        "embed_dim": 256,
        "clip_model": "openai/clip-vit-base-patch32",
        "max_entities_per_unit": 12,
        "modalities": ["text", "image", "table", "diagram"],
    },
    "embedding": {
        "base_url": "",
        "model": "BAAI/bge-m3",
        "api_key_env": "MMGRAPH_EMBED_API_KEY",
        "dim": 1024,
        "timeout": 60,
    },
    "captioner": {
        "enabled": False,
        "provider": "qwen",
        "model": "qwen3-vl-plus",
        "base_url": "",
        "api_key_env": "DASHSCOPE_API_KEY",
        "timeout": 120,
        "temperature": 0.0,
        "prompt": "请用一句中文描述这张医学图像的关键内容。",
    },
    "extractor": {
        "mode": "auto",
        "language": "zh",
        "base_url": "",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.0,
        "timeout": 60,
        "max_retries": 2,
        "cache_dir": "out/.extract_cache",
    },
    "construction": {
        "enabled": False,
        "rewriter": True,
        "grounding": True,
        "refinement": True,
        "inspector": False,
    },
    "grounding": {
        "searcher": "none",
        "local_path": "",
        "icd10_path": "data/icd10/diseases.json",
        "lang": "zh",
        "timeout": 10,
        "max_candidates": 3,
        "max_chars": 200,
    },
    "memory": {"C": 5, "L_max": 1200, "L_dig": 300, "L_cap": 5000},
    "retrieval": {"top_k": 5, "distance_threshold": 0.05, "enable_web": False, "hops": 0},
    "policy": {
        "max_rounds": 5,
        "regulator_base": 0.5,
        "structural_penalty": 0.5,
        "well_formed_required": True,
    },
    "eval": {"datasets": ["smoke"], "metric": "token_f1"},
}


def _deep_merge(base: dict, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    """Let environment variables override a few endpoint settings.

    This keeps keys and endpoints out of the TOML file and makes the
    ``run_check_api.bat`` double-click flow work without editing config.
    """
    mapping = {
        ("extractor", "base_url"): "MMGRAPH_EXTRACTOR_BASE_URL",
        ("extractor", "model"): "MMGRAPH_EXTRACTOR_MODEL",
        ("captioner", "base_url"): "MMGRAPH_CAPTIONER_BASE_URL",
        ("captioner", "model"): "MMGRAPH_CAPTIONER_MODEL",
    }
    for (section, key), env_name in mapping.items():
        value = os.environ.get(env_name)
        if value:
            cfg[section][key] = value
    enabled = os.environ.get("MMGRAPH_CAPTIONER_ENABLED")
    if enabled is not None and enabled != "":
        cfg["captioner"]["enabled"] = enabled.lower() in ("1", "true", "yes", "on")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return a merged config dict; ``None`` yields the built-in defaults."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(Path(path), "rb") as handle:
            _deep_merge(cfg, tomllib.load(handle))
    _apply_env_overrides(cfg)
    return cfg
