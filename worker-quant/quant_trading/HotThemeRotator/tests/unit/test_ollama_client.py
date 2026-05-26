"""Real Ollama HTTP client tests (W7, P11-05 follow-up)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.llm.ollama_client import (  # noqa: E402
    DEFAULT_OLLAMA_HOST,
    OllamaClient,
    OllamaUnreachableError,
    compute_cache_key,
)


# ─── helpers: fake http_post and clock ─────────────────────────────────────


def _ok_post_factory(response_text="generated narrative"):
    calls = []

    def post(url, body, timeout_s, headers):
        calls.append({"url": url, "body": body, "timeout_s": timeout_s, "headers": dict(headers)})
        return {"response": response_text}

    post.calls = calls  # type: ignore[attr-defined]
    return post


def _raising_post_factory(exc):
    def post(url, body, timeout_s, headers):
        raise exc
    return post


def _frozen_clock(dt):
    def now():
        return dt
    return now


_T0 = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)


# ─── compute_cache_key ─────────────────────────────────────────────────────


def test_cache_key_deterministic():
    a = compute_cache_key(model="gemma4:e4b", prompt="hello")
    b = compute_cache_key(model="gemma4:e4b", prompt="hello")
    assert a == b
    assert len(a) == 64


def test_cache_key_differs_across_models():
    a = compute_cache_key(model="gemma4:e4b", prompt="hello")
    b = compute_cache_key(model="gemma4:26b", prompt="hello")
    assert a != b


def test_cache_key_differs_across_prompts():
    a = compute_cache_key(model="m", prompt="A")
    b = compute_cache_key(model="m", prompt="B")
    assert a != b


# ─── successful HTTP path ──────────────────────────────────────────────────


def test_generate_returns_response_text():
    post = _ok_post_factory("hello world")
    client = OllamaClient(http_post_fn=post)
    out = client.generate(prompt="say hi", model="gemma4:e4b")
    assert out == "hello world"
    assert len(post.calls) == 1


def test_generate_posts_to_default_host_path():
    post = _ok_post_factory()
    client = OllamaClient(http_post_fn=post)
    client.generate(prompt="x", model="gemma4:e4b")
    assert post.calls[0]["url"] == f"{DEFAULT_OLLAMA_HOST}/api/generate"


def test_generate_posts_json_body_with_stream_false():
    post = _ok_post_factory()
    client = OllamaClient(http_post_fn=post)
    client.generate(prompt="test prompt", model="gemma4:e4b")
    body = json.loads(post.calls[0]["body"].decode("utf-8"))
    assert body == {"model": "gemma4:e4b", "prompt": "test prompt", "stream": False}


# ─── failure paths ─────────────────────────────────────────────────────────


def test_generate_empty_response_raises():
    post = _ok_post_factory("   ")
    client = OllamaClient(http_post_fn=post)
    with pytest.raises(OllamaUnreachableError, match="empty/invalid"):
        client.generate(prompt="x", model="m")


def test_generate_missing_response_field_raises():
    def post(url, body, timeout_s, headers):
        return {"unexpected": "shape"}
    client = OllamaClient(http_post_fn=post)
    with pytest.raises(OllamaUnreachableError, match="empty/invalid"):
        client.generate(prompt="x", model="m")


def test_generate_post_raises_wrapped_into_unreachable():
    client = OllamaClient(http_post_fn=_raising_post_factory(ConnectionError("refused")))
    with pytest.raises(OllamaUnreachableError, match="unreachable"):
        client.generate(prompt="x", model="m")


def test_generate_propagates_existing_unreachable_error():
    err = OllamaUnreachableError("explicit reason")
    client = OllamaClient(http_post_fn=_raising_post_factory(err))
    with pytest.raises(OllamaUnreachableError, match="explicit reason"):
        client.generate(prompt="x", model="m")


def test_generate_rejects_empty_prompt():
    client = OllamaClient(http_post_fn=_ok_post_factory())
    with pytest.raises(OllamaUnreachableError, match="prompt"):
        client.generate(prompt="   ", model="m")


def test_generate_rejects_empty_model():
    client = OllamaClient(http_post_fn=_ok_post_factory())
    with pytest.raises(OllamaUnreachableError, match="model"):
        client.generate(prompt="x", model="")


# ─── cache hit / miss / expiry ─────────────────────────────────────────────


def test_cache_hit_skips_http(tmp_path):
    post = _ok_post_factory("from http")
    client = OllamaClient(
        http_post_fn=post, cache_dir=tmp_path, now_fn=_frozen_clock(_T0),
    )
    first = client.generate(prompt="p", model="m")
    second = client.generate(prompt="p", model="m")
    assert first == second == "from http"
    assert len(post.calls) == 1  # second call cache-hit, no HTTP


def test_cache_miss_after_ttl_expires_re_fetches(tmp_path):
    post = _ok_post_factory("first")
    client = OllamaClient(
        http_post_fn=post, cache_dir=tmp_path,
        cache_ttl_hours=1.0, now_fn=_frozen_clock(_T0),
    )
    client.generate(prompt="p", model="m")
    # Advance clock past TTL.
    later = _T0 + timedelta(hours=2)
    client.now_fn = _frozen_clock(later)
    # Swap the post to return new content so we can prove a re-fetch happened.
    post2 = _ok_post_factory("second")
    client.http_post_fn = post2
    out = client.generate(prompt="p", model="m")
    assert out == "second"
    assert len(post2.calls) == 1


def test_cache_disabled_when_cache_dir_none():
    post = _ok_post_factory("ans")
    client = OllamaClient(http_post_fn=post, cache_dir=None)
    client.generate(prompt="p", model="m")
    client.generate(prompt="p", model="m")
    assert len(post.calls) == 2  # no cache, both calls hit HTTP


def test_cache_malformed_json_silently_re_fetches(tmp_path):
    cache_file = tmp_path / (
        compute_cache_key(model="m", prompt="p")[:16] + ".json"
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("not json", encoding="utf-8")
    post = _ok_post_factory("fresh")
    client = OllamaClient(
        http_post_fn=post, cache_dir=tmp_path, now_fn=_frozen_clock(_T0),
    )
    out = client.generate(prompt="p", model="m")
    assert out == "fresh"


def test_cache_file_written_with_iso_ts_and_response(tmp_path):
    post = _ok_post_factory("payload text")
    client = OllamaClient(
        http_post_fn=post, cache_dir=tmp_path, now_fn=_frozen_clock(_T0),
    )
    client.generate(prompt="p", model="m")
    key = compute_cache_key(model="m", prompt="p")
    cache_file = tmp_path / f"{key[:16]}.json"
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["response"] == "payload text"
    assert payload["model"] == "m"
    assert payload["cached_at"].startswith("2026-05-26")


def test_different_models_get_different_cache_entries(tmp_path):
    post = _ok_post_factory()
    client = OllamaClient(
        http_post_fn=post, cache_dir=tmp_path, now_fn=_frozen_clock(_T0),
    )
    client.generate(prompt="same", model="gemma4:e4b")
    client.generate(prompt="same", model="gemma4:26b")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2


# ─── host customization ────────────────────────────────────────────────────


def test_custom_host_used_in_url():
    post = _ok_post_factory()
    client = OllamaClient(host="http://remote:11434/", http_post_fn=post)
    client.generate(prompt="x", model="m")
    assert post.calls[0]["url"] == "http://remote:11434/api/generate"
