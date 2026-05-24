"""HotThemeRotator FastAPI layer (P8-09 / ADR-0004).

Read-only JSON over the Python data layer. No POST/PUT/DELETE endpoints,
no execution paths, no order placement. Rule 3 (advice-only) is enforced
by design at this layer.
"""
from api.main import app, create_app

__all__ = ["app", "create_app"]
