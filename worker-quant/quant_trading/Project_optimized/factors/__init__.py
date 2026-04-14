"""v3 Phase A — Alpha factor expansion (Alpha158-style subset).

Goal: migrate project from 25 → 80+ academically-grounded factors.
Each factor lives in its family module, is a pure function, has a
literature citation, and is registered in factors.registry.

Design rules (from v3 plan):
- **Do not invent factors.** Every addition must have published evidence.
- Each factor is pure: same inputs → same output.
- Outputs are pandas Series indexed by date.
- NaN tolerated at series start (min-window).
- Every factor preregistered in experiment_log.jsonl at wire-in time.
"""
from .registry import FACTOR_REGISTRY, factor_names, compute  # noqa: F401
