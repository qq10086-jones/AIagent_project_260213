"""
test_supervisor_routing.py

WS-33-01-03: Classifier pipeline test coverage.

Covers:
  1. supervisor_node — coding mode, no prior facts → routes to "coder"
  2. supervisor_node — coding mode, coder fact present → routes to "writer"
  3. supervisor_node — discovery mode, no facts → routes to "discovery"
  4. supervisor_node — discovery mode, intel fact present → routes to "screening"
  5. supervisor_node — analysis mode (default), no facts → routes to "quant"
  6. supervisor_node — analysis mode, quant fact present, browser disabled → routes to "writer"
  7. poll_for_fact — gateway exception → returns None without crashing (unavailability path)
  8. get_llm — no API key set in environment → returns None (low-confidence: no LLM available)
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add brain/ root to path so supervisor imports work from the tests/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import supervisor
from supervisor import supervisor_node, poll_for_fact, get_llm


# ── supervisor_node tests (pure routing — no DB or LLM needed) ────────────────

class TestSupervisorNodeCodingMode:
    def test_coding_no_facts_routes_to_coder(self):
        """Happy path: coding mode with no prior agent facts → dispatch coder."""
        state = {"mode": "coding", "facts": []}
        result = supervisor_node(state)
        assert result == {"next_step": "coder"}

    def test_coding_with_coder_fact_routes_to_writer(self):
        """Happy path: coding mode, coder already ran → dispatch writer."""
        state = {"mode": "coding", "facts": [{"agent": "coder", "data": {}}]}
        result = supervisor_node(state)
        assert result == {"next_step": "writer"}


class TestSupervisorNodeDiscoveryMode:
    def test_discovery_no_facts_routes_to_discovery(self):
        """Happy path: discovery mode with no facts → dispatch discovery (intel) stage."""
        state = {"mode": "discovery", "facts": []}
        result = supervisor_node(state)
        assert result == {"next_step": "discovery"}

    def test_discovery_with_intel_routes_to_screening(self):
        """Happy path: discovery mode, intel gathered → dispatch screening."""
        state = {"mode": "discovery", "facts": [{"agent": "intel", "data": {}}]}
        result = supervisor_node(state)
        assert result == {"next_step": "screening"}

    def test_discovery_with_intel_and_screener_routes_to_writer(self):
        """Happy path: all stages complete → dispatch writer."""
        state = {
            "mode": "discovery",
            "facts": [
                {"agent": "intel", "data": {}},
                {"agent": "screener", "data": {}},
            ],
        }
        result = supervisor_node(state)
        assert result == {"next_step": "writer"}


class TestSupervisorNodeAnalysisMode:
    def test_analysis_no_facts_routes_to_quant(self):
        """Happy path: default analysis mode, no facts → dispatch quant."""
        state = {"mode": "analysis", "facts": [], "symbol": "AAPL"}
        result = supervisor_node(state)
        assert result == {"next_step": "quant"}

    def test_analysis_quant_done_browser_disabled_routes_to_writer(self):
        """Happy path: quant complete, browser disabled → skip browser, dispatch writer."""
        # symbol without .T suffix → browser disabled in "auto" mode
        state = {
            "mode": "analysis",
            "facts": [{"agent": "quant", "data": {}}],
            "symbol": "AAPL",
        }
        with patch.dict(os.environ, {"ENABLE_BROWSER_FACTS": "0"}):
            result = supervisor_node(state)
        assert result == {"next_step": "writer"}

    def test_analysis_missing_mode_defaults_to_analysis(self):
        """Edge case: mode key absent → treated as analysis mode."""
        state = {"facts": [], "symbol": "TEST"}
        result = supervisor_node(state)
        assert result == {"next_step": "quant"}


# ── poll_for_fact — unavailability path ───────────────────────────────────────

class TestPollForFactUnavailability:
    def test_gateway_exception_returns_none(self):
        """Unavailability path: fact gateway failure → returns None, no crash."""
        with patch("supervisor.requests.get", side_effect=Exception("connection refused")):
            result = poll_for_fact("run-001", "quant", timeout=0.1)
        assert result is None

    def test_gateway_no_rows_returns_none(self, patch_requests_get):
        """Low-confidence path: gateway available but no rows for agent → returns None."""
        result = poll_for_fact("run-002", "quant", timeout=0.1)
        assert result is None

    def test_gateway_payload_returns_dict(self, patch_requests_get):
        """Happy path: gateway returns a payload object."""
        patched, mock_resp = patch_requests_get
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "fact": {
                "payload": {"agent": "quant", "data": {"symbol": "AAPL"}}
            },
        }
        result = poll_for_fact("run-003", "quant", timeout=0.1)
        assert result == {"agent": "quant", "data": {"symbol": "AAPL"}}
        assert patched.called


# ── get_llm — no API key (low-confidence / unavailability) ────────────────────

class TestGetLlmNoApiKey:
    def test_returns_none_when_no_api_key(self):
        """Unavailability path: neither QWEN_API_KEY nor OPENAI_API_KEY set → returns None."""
        env_without_keys = {k: v for k, v in os.environ.items()
                            if k not in ("QWEN_API_KEY", "OPENAI_API_KEY")}
        with patch.dict(os.environ, env_without_keys, clear=True):
            with patch("supervisor.LLM_PROVIDER", "dashscope"):
                result = get_llm()
        assert result is None
