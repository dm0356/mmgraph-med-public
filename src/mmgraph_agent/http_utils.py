"""Minimal stdlib HTTP JSON client.

The reproduction intentionally has **no third-party HTTP dependency**: the two
API adapters (text extraction and image captioning) POST JSON through
``urllib.request`` so the project runs on a stock Python environment (and on the
bundled runtime) with only NumPy.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class HTTPJSONError(RuntimeError):
    """Raised with the HTTP status and response body so failures are diagnosable."""


def post_json(url: str, body: dict, headers: dict[str, str] | None = None, timeout: float = 60) -> dict:
    """POST ``body`` as JSON and return the decoded JSON response.

    On failure raises :class:`HTTPJSONError` carrying the server's status and
    error body (which is where DashScope/DeepSeek put the real reason, e.g.
    ``model not found`` or ``invalid api key``).
    """
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:
            pass
        raise HTTPJSONError(f"HTTP {exc.code} {exc.reason}: {detail[:800]}") from exc
    except urllib.error.URLError as exc:
        raise HTTPJSONError(f"connection error: {exc.reason}") from exc

    return json.loads(raw)


__all__ = ["post_json", "HTTPJSONError"]
