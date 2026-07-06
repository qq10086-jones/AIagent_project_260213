"""Rule 15.9 guarded runner — the ONLY approved way to serve beyond loopback.

Remote Personal Access mode (single operator, private overlay network such as
Tailscale/WireGuard or an SSH tunnel — see docs/02_GOVERNANCE.md Rule 15.9).
Fail-closed: refuses to start on a non-loopback bind unless HTR_ACCESS_TOKEN
is set (and long enough to not be guessable). Loopback serving without a
token remains exactly Local Beta v0.

Usage (PowerShell):
    $env:HTR_BIND_HOST = "100.x.y.z"   # your Tailscale IP (NOT a public IP)
    $env:HTR_ACCESS_TOKEN = "<random >=16 chars>"
    python tools\\serve_remote.py
Then on your device: open http://100.x.y.z:8000/login?token=<token>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
MIN_TOKEN_LEN = 16
MIN_TOKEN_DISTINCT = 8  # degenerate-token floor (security review 2026-07-06 C3)


def bind_guard_error(host: str, token: str) -> str | None:
    """Return the Rule 15.9 refusal reason, or None when the bind is allowed."""
    if host in LOOPBACK_HOSTS:
        return None
    if not token:
        return (
            f"Rule 15.9 fail-closed: non-loopback bind ({host}) requires "
            "HTR_ACCESS_TOKEN to be set."
        )
    if len(token) < MIN_TOKEN_LEN:
        return (
            f"Rule 15.9: HTR_ACCESS_TOKEN too short (>= {MIN_TOKEN_LEN} chars "
            "required for a non-loopback bind)."
        )
    if len(set(token)) < MIN_TOKEN_DISTINCT:
        # The 2026-07-03 all-zero-token incident (PS5.1 RNG pitfall) passed the
        # length check; a degenerate token is as good as none. Prefer
        # --gen-token, which cannot produce this failure class.
        return (
            f"Rule 15.9: HTR_ACCESS_TOKEN looks degenerate (<{MIN_TOKEN_DISTINCT} "
            "distinct chars). Generate one with: python tools\\serve_remote.py --gen-token"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-token", action="store_true",
                    help="print a cryptographically random token and exit")
    args = ap.parse_args(argv)
    if args.gen_token:
        import secrets

        print(secrets.token_urlsafe(24))
        return 0

    host = os.environ.get("HTR_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("HTR_BIND_PORT", "8000"))
    token = os.environ.get("HTR_ACCESS_TOKEN", "")
    err = bind_guard_error(host, token)
    if err:
        print(err, file=sys.stderr)
        return 2
    import uvicorn

    # access_log=False — the login URL carries the token as a query param; the
    # default access log would persist it verbatim (security review 2026-07-06
    # C1 / Rule 15.9.3 "never logged").
    uvicorn.run("api.main:app", host=host, port=port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
