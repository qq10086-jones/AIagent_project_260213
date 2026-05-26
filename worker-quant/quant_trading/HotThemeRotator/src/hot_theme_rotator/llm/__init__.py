"""LLM narrative-synthesis package (Rule 8.3.1 + 13.4 enforced).

Defaults to local Ollama with ``gemma4:e4b`` per ``feedback_v7_llm_model`` +
``feedback_quant_htr_llm_role``. Output schemas NEVER carry probability, win
rate, or any ``\\d+%`` numeric quantity — regex post-check rejects violations.

Surface so far:

- ``reflection_brief`` (P11-05): consumes L0-L4 outputs and produces a
  Chinese research note with conditional language matching the upstream
  ``counterfactual_validity``.

Future modules: ``per_ticker_brief`` (P10-06) and ``ollama_client`` for real
endpoint integration are deferred to a later cycle.
"""
from __future__ import annotations

from hot_theme_rotator.llm.reflection_brief import (
    LlmClient,
    ReflectionBrief,
    ReflectionBriefError,
    ReflectionBriefInput,
    forbidden_pattern,
    generate_reflection_brief,
    regex_check_narrative,
    scan_brief_for_forbidden_tokens,
)

__all__ = [
    "LlmClient",
    "ReflectionBrief",
    "ReflectionBriefError",
    "ReflectionBriefInput",
    "forbidden_pattern",
    "generate_reflection_brief",
    "regex_check_narrative",
    "scan_brief_for_forbidden_tokens",
]
