"""Real Ollama HTTP client (W7 follow-up to P11-05 stub).

Implements ``LlmClient`` Protocol from ``reflection_brief`` against the local
Ollama daemon at ``http://localhost:11434`` (default). Zero new dependencies —
uses ``urllib`` from the stdlib.

Rule 8.3.1 governance: "Ollama / model unreachable → 503 + reason; never
fabricate a brief." This module raises ``OllamaUnreachableError`` (or its
subclasses) on any failure so the upstream LLM consumer
(``generate_reflection_brief`` / ``generate_per_ticker_brief``) can surface a
clean error rather than retry or fabricate.

24-hour file cache: keyed by ``sha256(model|prompt)``, stored at
``{cache_dir}/{key[:16]}.json``. Same prompt + model → cache hit; expired
entries are silently re-fetched. Cache disabled when ``cache_dir=None``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


__all__ = [
    "DEFAULT_OLLAMA_HOST",
    "OllamaClient",
    "OllamaUnreachableError",
    "compute_cache_key",
]


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_CACHE_TTL_HOURS = 24.0


class OllamaUnreachableError(RuntimeError):
    """Raised when Ollama is not reachable, times out, or returns an empty body.

    Rule 8.3.1: upstream consumers MUST surface this as a 503-style refusal
    rather than fabricate a brief.
    """


def compute_cache_key(*, model: str, prompt: str) -> str:
    """Deterministic 64-hex sha256 key over (model, prompt). Public for tests."""
    return sha256(f"{model}|{prompt}".encode("utf-8")).hexdigest()


HttpPostFn = Callable[[str, bytes, float, Mapping[str, str]], Mapping[str, Any]]
NowFn = Callable[[], datetime]


@dataclass
class OllamaClient:
    """Real Ollama HTTP client implementing the ``LlmClient`` Protocol.

    Parameters
    ----------
    host : str
        Ollama base URL. Defaults to local daemon.
    timeout_s : float
        HTTP timeout for each generate call.
    cache_dir : Path | None
        File-cache directory. Set to ``None`` to disable caching (e.g., when
        the prompt carries volatile context that would poison the cache).
    cache_ttl_hours : float
        Cache entry lifetime. After expiry the entry is silently re-fetched.
    http_post_fn / now_fn : optional injected callables
        For tests. Default implementations use ``urllib`` and ``datetime.now``.
    """

    host: str = DEFAULT_OLLAMA_HOST
    timeout_s: float = _DEFAULT_TIMEOUT_S
    cache_dir: Optional[Path] = None
    cache_ttl_hours: float = _DEFAULT_CACHE_TTL_HOURS
    http_post_fn: Optional[HttpPostFn] = None
    now_fn: Optional[NowFn] = None

    def generate(self, *, prompt: str, model: str) -> str:
        """Generate a completion. Cache-first, then HTTP. Fail-closed."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise OllamaUnreachableError("prompt must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise OllamaUnreachableError("model must be a non-empty string")

        key = compute_cache_key(model=model, prompt=prompt)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        url = f"{self.host.rstrip('/')}/api/generate"
        headers = {"Content-Type": "application/json"}

        try:
            payload = (self.http_post_fn or _default_http_post)(
                url, body, self.timeout_s, headers,
            )
        except OllamaUnreachableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OllamaUnreachableError(
                f"Ollama at {self.host} unreachable: {type(exc).__name__}: {exc}"
            ) from exc

        text = payload.get("response", "") if isinstance(payload, Mapping) else ""
        if not isinstance(text, str) or not text.strip():
            raise OllamaUnreachableError(
                f"Ollama returned empty/invalid response for model {model!r}"
            )

        self._cache_put(key, text=text, model=model)
        return text

    # ─── cache layer ───────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return Path(self.cache_dir) / f"{key[:16]}.json"

    def _now(self) -> datetime:
        return (self.now_fn or _default_now)()

    def _cache_get(self, key: str) -> Optional[str]:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        try:
            cached_at = datetime.fromisoformat(str(payload["cached_at"]))
        except (KeyError, ValueError):
            return None
        if cached_at.tzinfo is None:
            return None
        now = self._now()
        if (now - cached_at) > timedelta(hours=self.cache_ttl_hours):
            return None
        text = payload.get("response")
        if isinstance(text, str) and text.strip():
            return text
        return None

    def _cache_put(self, key: str, *, text: str, model: str) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "response": text,
            "cached_at": self._now().isoformat(),
            "model": model,
            "key_prefix": key[:16],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )


# ─── default implementations ───────────────────────────────────────────────


def _default_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _default_http_post(
    url: str,
    body: bytes,
    timeout_s: float,
    headers: Mapping[str, str],
) -> Mapping[str, Any]:
    """POST JSON via stdlib urllib; raise OllamaUnreachableError on any error."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise OllamaUnreachableError(
            f"Ollama HTTP error {exc.code}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OllamaUnreachableError(
            f"Ollama URL error (connection refused or DNS failure): {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise OllamaUnreachableError(
            f"Ollama timeout after {timeout_s}s"
        ) from exc

    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OllamaUnreachableError(
            f"Ollama returned non-JSON response: {exc}"
        ) from exc
