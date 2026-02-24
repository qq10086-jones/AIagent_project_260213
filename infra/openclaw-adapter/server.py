from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import subprocess
import urllib.request
import urllib.error
from typing import Tuple, Dict, Any
import re
import shlex
from typing import Dict, Any
import os, time
import boto3
import hashlib
from datetime import datetime

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "nexus")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "nexuspassword")

LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "18081"))

OPENCLAW_CONTAINER_NAME = os.getenv("OPENCLAW_CONTAINER_NAME", "infra-openclaw-1")
OPENCLAW_BROWSER_PORT = int(os.getenv("OPENCLAW_BROWSER_PORT", "18790"))
OPENCLAW_GATEWAY_HTTP = os.getenv("OPENCLAW_GATEWAY_HTTP", "http://openclaw:18789/")
DOCKER_BIN = os.getenv("DOCKER_BIN", "docker")

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/tmp/openclaw-artifacts")

def _json(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def docker_version() -> Tuple[bool, str]:
    try:
        cp = subprocess.run([DOCKER_BIN, "--version"], capture_output=True, text=True, timeout=8)
        if cp.returncode == 0:
            return True, (cp.stdout or cp.stderr).strip()
        return False, (cp.stderr or cp.stdout).strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def docker_inspect_container(name: str) -> Tuple[bool, str]:
    try:
        cp = subprocess.run([DOCKER_BIN, "inspect", name], capture_output=True, text=True, timeout=8)
        if cp.returncode == 0:
            return True, "ok"
        return False, (cp.stderr or cp.stdout).strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def http_status(url: str) -> Tuple[bool, int, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return True, int(getattr(resp, "status", 200)), "ok"
    except urllib.error.HTTPError as e:
        # reachable but non-200
        return True, int(e.code), "http_error"
    except Exception as e:
        return False, -1, f"{type(e).__name__}: {e}"


def docker_exec_sh(container: str, sh_cmd: str, timeout_s: int = 30) -> Dict[str, Any]:
    """
    Run `sh -lc <sh_cmd>` inside the given container via docker CLI.
    Returns dict: {ok, rc, stdout, stderr}.
    """
    try:
        cp = subprocess.run(
            [DOCKER_BIN, "exec", container, "sh", "-lc", sh_cmd],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "ok": cp.returncode == 0,
            "rc": cp.returncode,
            "stdout": (cp.stdout or "").strip(),
            "stderr": (cp.stderr or "").strip(),
        }
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "rc": 124, "stdout": (e.stdout or "").strip() if e.stdout else "", "stderr": "timeout"}
    except Exception as e:
        return {"ok": False, "rc": 127, "stdout": "", "stderr": f"{type(e).__name__}: {e}"}

def _parse_last_json(stdout: str) -> Dict[str, Any]:
    """
    openclaw CLI 的 --json 可能会输出多行日志 + 最后一行 JSON。
    这里尽量稳：优先取最后一个 {...} 片段并 json.loads。
    """
    s = (stdout or "").strip()
    if not s:
        return {}
    # 先尝试整段就是 JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 否则找最后一个 { ... }（粗略但够用）
    m = None
    for m in re.finditer(r"\{.*\}", s, re.DOTALL):
        pass
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def docker_cp_from(container: str, container_path: str, out_dir: str = ARTIFACTS_DIR, timeout_s: int = 60) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    # 给文件名加时间戳，避免覆盖
    base = os.path.basename(container_path.rstrip("/"))
    if not base:
        base = f"artifact-{int(time.time())}"
    host_path = os.path.join(out_dir, base)

    # docker cp <container>:<path> <host_path>
    sh_cmd = f"docker cp {shlex.quote(container)}:{shlex.quote(container_path)} {shlex.quote(host_path)}"
    result = sh(sh_cmd, timeout_s=timeout_s)  # 你 server.py 里已有 sh(...) / docker_exec_sh(...) 的那套，按你实际函数名替换

    ok = (result.get("rc") == 0) and os.path.exists(host_path)
    return {
        "ok": bool(ok),
        "rc": result.get("rc"),
        "stdout": result.get("stdout"),
        "stderr": result.get("stderr"),
        "host_path": host_path if ok else None,
        "target": {"cli_command": sh_cmd},
    }

def openclaw_browser(cmd: str, args: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
    """
    在 openclaw 容器里执行：node openclaw.mjs browser <cmd> ... --json
    cmd: start | stop | open | navigate | screenshot
    args: 参数字典（url/...）
    """
    if cmd in ("start", "stop"):
        sh_cmd = (
            "PLAYWRIGHT_BROWSERS_PATH=/app/.playwright-browsers "
            f"node openclaw.mjs browser {cmd} "
            "--browser-profile openclaw "
            "--token dev-openclaw-token "
            "--json"
        )
        
    elif cmd == "open":
        url = (args or {}).get("url")
        if not url:
            return {"ok": False, "error": "missing_url"}
        # 【关键修复】使用 shlex.quote 防止命令注入
        safe_url = shlex.quote(url)
        sh_cmd = (
            "PLAYWRIGHT_BROWSERS_PATH=/app/.playwright-browsers "
            f"node openclaw.mjs browser open {safe_url} "
            "--browser-profile openclaw "
            "--json"
        )
        
    elif cmd == "navigate":
        url = (args or {}).get("url")
        if not url:
            return {"ok": False, "error": "missing_url"}
        safe_url = shlex.quote(url)
        # 【关键修复】移除未经验证的 targetId 传参，保持和手动测试一致
        sh_cmd = (
            "PLAYWRIGHT_BROWSERS_PATH=/app/.playwright-browsers "
            f"node openclaw.mjs browser navigate {safe_url} "
            "--browser-profile openclaw "
            "--json"
        )
        
    elif cmd == "screenshot":
        # 【关键修复】不要强校验 targetId，也不要拼接不支持的 --targetId 参数
        sh_cmd = (
            "PLAYWRIGHT_BROWSERS_PATH=/app/.playwright-browsers "
            f"node openclaw.mjs browser screenshot --browser-profile openclaw --json"
        )
        
    else:
        return {"ok": False, "error": "unsupported_browser_cmd", "cmd": cmd}

    # 调用 server.py 中的执行函数
    result = docker_exec_sh(OPENCLAW_CONTAINER_NAME, sh_cmd, timeout_s=timeout_s)
    
    # 假设你已经有了 _parse_last_json 这个函数
    parsed = _parse_last_json(result.get("stdout", ""))
    
    ok = (result.get("rc") == 0) and (parsed.get("ok", True) is True)

    return {
        "ok": bool(ok),
        "rc": result.get("rc"),
        "stdout": result.get("stdout"),
        "stderr": result.get("stderr"),
        "parsed": parsed,
        "target": {"container": OPENCLAW_CONTAINER_NAME, "cli_command": sh_cmd},
    }

def archive_to_minio(container_name: str, container_path: str) -> dict:
    filename = container_path.split("/")[-1]
    ext = filename.split(".")[-1]
    local_tmp_path = f"/tmp/{filename}"
    
    # 1. 用 docker cp 把图片拷贝到 adapter 的临时目录
    cp_cmd = ["docker", "cp", f"{container_name}:{container_path}", local_tmp_path]
    subprocess.run(cp_cmd, check=True)
    
    # 2. 计算 SHA256 指纹 (严格遵守蓝图的去重规范)
    sha256_hash = hashlib.sha256()
    with open(local_tmp_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()
    
    # 3. 构造完美的蓝图 Object Key: {namespace}/{yyyy}/{mm}/{dd}/{sha256[:2]}/{sha256}.{ext}
    now = datetime.now()
    object_key = f"browser/{now.strftime('%Y/%m/%d')}/{file_hash[:2]}/{file_hash}.{ext}"
    
    # 4. 上传到 MinIO
    s3 = boto3.client('s3',
                      endpoint_url=MINIO_ENDPOINT,
                      aws_access_key_id=MINIO_ACCESS_KEY,
                      aws_secret_access_key=MINIO_SECRET_KEY,
                      region_name='us-east-1')
    
    s3.upload_file(local_tmp_path, 'nexus-evidence', object_key)
    
    # 5. 阅后即焚，清理临时文件
    os.remove(local_tmp_path)
    
    return {
        "bucket": "nexus-evidence",
        "object_key": object_key,
        "sha256": file_hash
    }

def handle_op(payload: Dict[str, Any]) -> Dict[str, Any]:
    op = (payload or {}).get("op")
    args = (payload or {}).get("args") or {}
    if not op:
        return {"ok": False, "error": "missing_op"}

    if op == "browser.start":
        out = openclaw_browser("start", args, timeout_s=60)
        return {"engine": "openclaw-adapter", "op": op, **out}

    if op == "browser.stop":
        out = openclaw_browser("stop", args, timeout_s=60)
        return {"engine": "openclaw-adapter", "op": op, **out}

    if op == "browser.open":
        out = openclaw_browser("open", args, timeout_s=60)
        return {"engine": "openclaw-adapter", "op": op, **out}

    if op == "browser.navigate":
        out = openclaw_browser("navigate", args, timeout_s=60)
        return {"engine": "openclaw-adapter", "op": op, **out}

    if op == "browser.screenshot":
        out = openclaw_browser("screenshot", args, timeout_s=90)
        return {"engine": "openclaw-adapter", "op": op, **out}
    
    if op == "artifact.pull":
        container = (args or {}).get("container", OPENCLAW_CONTAINER_NAME)
        path = (args or {}).get("path")
        
        if not path:
            return {"ok": False, "error": "missing_path"}

        import shlex
        safe_path = shlex.quote(path)
        
        # 魔法在这里：在容器内用 base64 命令把图片编码，-w 0 表示不换行
        sh_cmd = f"base64 -w 0 {safe_path}"
        result = docker_exec_sh(container, sh_cmd, timeout_s=30)

        if result["rc"] != 0:
            return {"ok": False, "error": "read_failed", "stderr": result["stderr"]}

        return {
            "engine": "openclaw-adapter",
            "op": op,
            "ok": True,
            "file_name": path.split("/")[-1],
            "base64_data": result["stdout"]  # 这里就是图片的完整 base64 字符串
        }
    
    if op == "artifact.archive":
        container = (args or {}).get("container", OPENCLAW_CONTAINER_NAME)
        path = (args or {}).get("path")
        
        if not path:
            return {"ok": False, "error": "missing_path"}
            
        try:
            minio_info = archive_to_minio(container, path)
            return {
                "engine": "openclaw-adapter",
                "op": op,
                "ok": True,
                **minio_info
            }
        except Exception as e:
            return {"ok": False, "error": "archive_failed", "details": str(e)}

    return {"ok": False, "error": "unsupported_op", "op": op}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # quieter logs; keep if you want:
        return

    def _send(self, code: int, obj: Dict[str, Any]):
        body = _json(obj)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            dv_ok, dv = docker_version()
            di_ok, di_msg = docker_inspect_container(OPENCLAW_CONTAINER_NAME)
            gw_ok, gw_status, gw_msg = http_status(OPENCLAW_GATEWAY_HTTP)
            return self._send(
                200,
                {
                    "ok": True,
                    "service": "openclaw-adapter",
                    "docker_cli": {"ok": dv_ok, "version": dv},
                    "openclaw_container": {"name": OPENCLAW_CONTAINER_NAME, "ok": di_ok, "msg": di_msg},
                    "gateway_http": {"url": OPENCLAW_GATEWAY_HTTP, "ok": gw_ok, "status": gw_status, "msg": gw_msg},
                    "browser_control": {"loopback_url": f"http://127.0.0.1:{OPENCLAW_BROWSER_PORT}/start"},
                },
            )
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

        out = handle_op(payload)
        return self._send(200 if out.get("ok") else 400, out)


if __name__ == "__main__":
    HTTPServer((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
