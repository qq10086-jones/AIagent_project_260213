"""Remote Personal Access token gate (P22-02, Rule 15.9).

Active ONLY when ``HTR_ACCESS_TOKEN`` is set in the environment (the guarded
runner ``tools/serve_remote.py`` enforces this for any non-loopback bind —
fail-closed). When active, EVERY request — API and pages — must present the
token via the ``X-HTR-Token`` header, an ``Authorization: Bearer`` header, or
the session cookie set by ``/login?token=…``. Anything else receives 401.

This is defense-in-depth behind the Rule 15.9 exposure boundary (a private
overlay network such as Tailscale/WireGuard or an SSH tunnel); it is NOT an
account system and creates no new write path (Rule 11.5 whitelist unchanged).
The token is never logged or rendered (Rule 15.9.3).
"""
from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

COOKIE_NAME = "htr_token"
LOGIN_PATH = "/login"

# Hosts treated as loopback by the fail-closed guard. "testserver" is the
# Starlette TestClient default and carries no network exposure.
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver", None}


def _safe_compare(supplied: str, secret: str) -> bool:
    """Constant-time compare that treats undecodable/odd input as a plain
    mismatch (security review 2026-07-06: a non-ASCII header raised TypeError →
    500; deny with 401 instead)."""
    try:
        return hmac.compare_digest(supplied, secret)
    except (TypeError, ValueError):
        return False


class LoopbackOnlyGuard(BaseHTTPMiddleware):
    """Rule 15.0/15.9 fail-closed AT THE APP LEVEL (security review C2).

    Installed whenever HTR_ACCESS_TOKEN is NOT configured: any request that
    arrives on a non-loopback listening address is refused, so a fat-fingered
    ``uvicorn --host 0.0.0.0`` (bypassing tools/serve_remote.py) serves
    nothing instead of serving the whole cockpit ungated.
    """

    async def dispatch(self, request: Request, call_next):
        server = request.scope.get("server")
        host = server[0] if server else None
        if host in LOOPBACK_HOSTS:
            return await call_next(request)
        return JSONResponse(
            {"detail": "Rule 15.9 fail-closed: non-loopback serving requires "
                       "HTR_ACCESS_TOKEN (use tools/serve_remote.py)"},
            status_code=403,
        )


class AccessTokenMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str):
        super().__init__(app)
        if not token:
            raise ValueError("AccessTokenMiddleware requires a non-empty token")
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path == LOGIN_PATH:
            supplied = request.query_params.get("token", "")
            if supplied and _safe_compare(supplied, self._token):
                resp = RedirectResponse(url="/", status_code=303)
                resp.set_cookie(COOKIE_NAME, self._token, httponly=True, samesite="lax")
                return resp
            return JSONResponse({"detail": "invalid token (Rule 15.9)"}, status_code=401)
        if self._authorized(request):
            return await call_next(request)
        return JSONResponse(
            {"detail": "unauthorized — Rule 15.9 remote personal access token required"},
            status_code=401,
        )

    def _authorized(self, request: Request) -> bool:
        header = request.headers.get("x-htr-token", "")
        if header and _safe_compare(header, self._token):
            return True
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and _safe_compare(auth[7:], self._token):
            return True
        cookie = request.cookies.get(COOKIE_NAME, "")
        return bool(cookie) and _safe_compare(cookie, self._token)
