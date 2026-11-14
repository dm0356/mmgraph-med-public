"""Frozen multimodal encoders (Module 1, paper §3.1).

The paper embeds entities and image units into a shared space with a frozen
CLIP encoder in one streaming pass. For the CPU smoke path we ship a
deterministic hashing embedder; ``CLIPEmbedder`` is the swap-in adapter used
once torch + transformers are available.
"""

from __future__ import annotations

import os

import numpy as np

from ..utils import hash_embed


class BaseEmbedder:
    name = "base"
    dim = 256

    def encode_texts(self, texts: list[str]) -> list[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_captions(self, captions: list[str]) -> list[np.ndarray]:
        """Image units are embedded from their captions in the CPU fallback."""
        return self.encode_texts(captions)

    def encode_images(self, paths: list[str]) -> list[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError("this embedder does not support pixel-level images")


class HashingEmbedder(BaseEmbedder):
    """Deterministic bag-of-words hashing; no model, no network, no GPU."""

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [hash_embed(text, self.dim) for text in texts]


class APIEmbedder(BaseEmbedder):
    """OpenAI-compatible ``/v1/embeddings`` client (e.g. vLLM serving BGE-M3).

    This lets the cluster serve local embedding models without any Python ML
    dependency in this project: configure ``[embedding].base_url`` to a local
    vLLM/TEI endpoint and set ``[hypergraph].embedder = "api"``.
    """

    name = "api"

    def __init__(self, config: dict) -> None:
        emb = config.get("embedding", {})
        self.base_url = (emb.get("base_url") or "").strip()
        self.model = emb.get("model") or "BAAI/bge-m3"
        self.api_key_env = emb.get("api_key_env") or "MMGRAPH_EMBED_API_KEY"
        self.dim = int(emb.get("dim", 1024))
        self.timeout = float(emb.get("timeout", 60))

    def encode_texts(self, texts: list[str]) -> list[np.ndarray]:  # pragma: no cover - network
        if not self.base_url:
            raise RuntimeError(
                "APIEmbedder requires [embedding].base_url, e.g. http://127.0.0.1:8002/v1/embeddings"
            )
        from ..http_utils import post_json

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = post_json(self.base_url, {"model": self.model, "input": list(texts)}, headers, self.timeout)
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [np.asarray(item["embedding"], dtype=np.float32) for item in items]
        out: list[np.ndarray] = []
        for vector in vectors:
            norm = float(np.linalg.norm(vector))
            out.append(vector / norm if norm > 0 else vector)
        return out


class CLIPEmbedder(BaseEmbedder):
    """Frozen CLIP text encoder adapter.

    Constructing this class requires the ``gpu`` extras (torch + transformers).
    If they are missing a :class:`RuntimeError` explains how to install them so
    :func:`get_embedder` can fall back to :class:`HashingEmbedder`.
    """

    name = "clip"

    def __init__(self, dim: int = 512, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu") -> None:
        try:  # pragma: no cover - requires optional deps
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "CLIPEmbedder requires torch + transformers; install with "
                "`pip install -e '.[gpu]'` (or set [hypergraph].embedder = 'hashing')."
            ) from exc
        self.dim = dim
        self.device = device
        self._torch = torch
        self._model = CLIPModel.from_pretrained(model_name).to(device)
        self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()

    def encode_texts(self, texts: list[str]) -> list[np.ndarray]:  # pragma: no cover
        torch = self._torch
        with torch.no_grad():
            tokens = self._tokenizer(
                texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            feats = self._model.get_text_features(**tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return [f.cpu().numpy().astype(np.float32) for f in feats]

    def encode_images(self, paths: list[str]) -> list[np.ndarray]:  # pragma: no cover
        from PIL import Image

        torch = self._torch
        images = [Image.open(path).convert("RGB") for path in paths]
        with torch.no_grad():
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return [f.cpu().numpy().astype(np.float32) for f in feats]


class ChineseCLIPEmbedder:
    """Chinese-CLIP text/image encoder (OFA-Sys/chinese-clip-vit-base-patch16).

    Chinese-CLIP is trained on Chinese image-text pairs, so it is the preferred
    adapter for Chinese clinical images. It uses a different processor/model
    family from OpenAI CLIP, hence a separate class.
    """

    name = "chinese-clip"

    def __init__(self, dim: int = 512,
                 model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
                 device: str = "cpu") -> None:
        try:  # pragma: no cover - requires optional deps
            import torch  # type: ignore
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "ChineseCLIPEmbedder requires torch + transformers with ChineseCLIP support"
            ) from exc
        self.dim = dim
        self.device = device
        self._torch = torch
        self._model = ChineseCLIPModel.from_pretrained(model_name).to(device)
        self._processor = ChineseCLIPProcessor.from_pretrained(model_name)
        self._model.eval()

    def encode_texts(self, texts: list[str]) -> list[np.ndarray]:  # pragma: no cover
        torch = self._torch
        with torch.no_grad():
            inputs = self._processor(
                text=list(texts), return_tensors="pt", padding=True,
                truncation=True, max_length=52,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return [f.cpu().numpy().astype(np.float32) for f in feats]

    def encode_images(self, paths: list[str]) -> list[np.ndarray]:  # pragma: no cover
        from PIL import Image

        torch = self._torch
        images = [Image.open(path).convert("RGB") for path in paths]
        with torch.no_grad():
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return [f.cpu().numpy().astype(np.float32) for f in feats]


def get_embedder(config: dict) -> BaseEmbedder:
    """Build the configured embedder, silently degrading to hashing on CPU."""
    name = config["hypergraph"].get("embedder", "auto")
    dim = int(config["hypergraph"].get("embed_dim", 256))
    clip_model = config["hypergraph"].get("clip_model", "openai/clip-vit-base-patch32")

    if name == "hashing":
        return HashingEmbedder(dim)
    if name == "api":
        return APIEmbedder(config)
    if name in ("clip", "chinese-clip", "biomedclip"):
        return CLIPEmbedder(dim, clip_model)
    # auto: prefer CLIP, fall back quietly for the zero-dependency smoke path.
    try:  # pragma: no cover - depends on optional deps
        return CLIPEmbedder(dim, clip_model)
    except Exception:
        return HashingEmbedder(dim)


def get_image_text_encoder(config: dict):
    """Return the CLIP text encoder used to query the CLIP image space."""
    return get_image_embedder(config)


def get_image_embedder(config: dict):
    """Return a pixel-level image encoder (CLIP) or ``None``.

    Configured through ``[image]``::

        [image]
        embedder = "clip"          # none | clip
        model = "openai/clip-vit-base-patch32"
        dim = 512
        device = "cpu"

    ``MMGRAPH_CLIP_MODEL`` overrides ``model`` (useful for a local snapshot).
    """
    section = config.get("image") or {}
    kind = str(section.get("embedder", "none") or "none").lower()
    if kind not in ("clip", "chinese-clip", "biomedclip"):
        return None
    model = (
        os.environ.get("MMGRAPH_CLIP_MODEL")
        or section.get("model")
        or config["hypergraph"].get("clip_model")
        or "openai/clip-vit-base-patch32"
    )
    dim = int(section.get("dim", 512))
    device = section.get("device", "cpu")
    try:  # pragma: no cover - depends on optional deps
        if kind == "chinese-clip":
            return ChineseCLIPEmbedder(dim, model, device)
        if kind in ("clip", "biomedclip"):
            return CLIPEmbedder(dim, model, device)
    except Exception:
        return None
    return None
