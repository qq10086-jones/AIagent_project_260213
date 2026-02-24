from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            return self._send(200, {"ok": True, "service": "openclaw-mock"})
        return self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, {"ok": False, "error": "not_found"})

        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(n).decode("utf-8") if n else "{}"
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {"raw": "<unparseable>"}

        # Minimal deterministic response to validate E2E pipeline.
        out = {
            "ok": True,
            "engine": "openclaw-mock",
            "result": {
                "message": "mock executed",
                "received": payload,
            },
        }
        return self._send(200, out)

if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 18080), Handler).serve_forever()
