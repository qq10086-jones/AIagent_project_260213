import os, json, time, subprocess, hashlib, base64, io, re, traceback, math, statistics, socket, shutil
import html
import xml.etree.ElementTree as ET
from pathlib import Path
import redis
import yfinance as yf
import pandas as pd
import boto3
from datetime import datetime, timedelta
import urllib.request
import urllib.error
import urllib.parse
import tomllib
from PIL import Image, ImageDraw, ImageStat
import requests
from jinja2 import Template
import uuid
import psycopg2
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from email.utils import parsedate_to_datetime
from worker_result_helpers import (
    build_worker_result_envelope,
    is_single_agent_payload,
    validate_single_agent_payload_guardrails,
    validate_single_agent_worker_result_guardrails,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _load_runtime_config() -> tuple[dict, str | None]:
    candidates = [
        str(os.getenv("RUNTIME_CONFIG_PATH", "")).strip(),
        str(Path.cwd() / "configs" / "runtime" / "runtime_defaults.json"),
        str(Path.cwd().parent / "configs" / "runtime" / "runtime_defaults.json"),
        "/workspace/configs/runtime/runtime_defaults.json",
    ]
    for p in candidates:
        if not p:
            continue
        try:
            fp = Path(p)
            if fp.exists():
                return json.loads(fp.read_text(encoding="utf-8")), str(fp)
        except Exception:
            continue
    return {}, None


RUNTIME_CONFIG, RUNTIME_CONFIG_PATH_LOADED = _load_runtime_config()
RUNTIME_QUANT = (RUNTIME_CONFIG or {}).get("worker_quant", {})
RUNTIME_STREAMS = (RUNTIME_CONFIG or {}).get("streams", {})

# --- Env Config ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_TASK = os.getenv("STREAM_TASK", "stream:task")
STREAM_RESULT = os.getenv("STREAM_RESULT", "stream:result")
STREAM_DLQ = os.getenv("STREAM_DLQ", RUNTIME_STREAMS.get("task_dlq", "stream:dlq"))
GROUP = os.getenv("GROUP", "cg:workers")
CONSUMER = os.getenv("CONSUMER", "worker-quant-1")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
VISIBILITY_TIMEOUT_S = int(os.getenv("VISIBILITY_TIMEOUT_S", "300"))
RECLAIM_INTERVAL_S = int(os.getenv("RECLAIM_INTERVAL_S", "30"))
EXECUTION_AUDIT_SCHEMA_READY = False

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "nexus")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "nexuspassword")
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "/workspace"))
WORKSPACE_RUNTIME_ROOT = Path(os.getenv("WORKSPACE_RUNTIME_ROOT", str(WORKSPACE_ROOT / "runtime")))
RUNTIME_ARTIFACTS_ROOT = Path(os.getenv("RUNTIME_ARTIFACTS_ROOT", str(WORKSPACE_RUNTIME_ROOT / "artifacts")))
QUANT_PROJECT_ROOT = Path(os.getenv("QUANT_PROJECT_ROOT", "/app/quant_trading/Project_optimized"))
QUANT_RUNTIME_ROOT = Path(os.getenv("QUANT_RUNTIME_ROOT", str(WORKSPACE_RUNTIME_ROOT / "worker-quant" / "quant_trading" / "Project_optimized")))
QUANT_DB_PATH = Path(os.getenv("QUANT_DB_PATH", str(QUANT_RUNTIME_ROOT / "japan_market.db")))
RENDER_API_KEY = os.getenv("RENDER_API_KEY", "")
RENDER_API_BASE_URL = os.getenv("RENDER_API_BASE_URL", "https://api.render.com/v1")
PREVIEW_PORT_START = int(os.getenv("PREVIEW_PORT_START", "46000"))
PREVIEW_PORT_END = int(os.getenv("PREVIEW_PORT_END", "46020"))
PREVIEW_PUBLIC_HOST = os.getenv("PREVIEW_PUBLIC_HOST", "localhost").strip() or "localhost"
PREVIEW_HEALTH_TIMEOUT_S = int(os.getenv("PREVIEW_HEALTH_TIMEOUT_S", "15"))
PREVIEW_REGISTRY_PATH = RUNTIME_ARTIFACTS_ROOT / "preview_runtime_registry.json"

# --- LLM Config ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", RUNTIME_QUANT.get("llm_provider", "ollama")).lower() # ollama, openai, dashscope, gemini
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", RUNTIME_QUANT.get("ollama_base_url", "http://host.docker.internal:11434"))
OLLAMA_CHAT_API = f"{OLLAMA_BASE}/api/chat"
OLLAMA_GENERATE_API = f"{OLLAMA_BASE}/api/generate"

# Cloud Provider Configs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", RUNTIME_QUANT.get("openai_base_url", "https://api.openai.com/v1"))
DASH_SCOPE_API_KEY = os.getenv("DASH_SCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
DASH_SCOPE_BASE_URL = (
    os.getenv("DASH_SCOPE_BASE_URL")
    or os.getenv("QWEN_BASE_URL")
    or RUNTIME_QUANT.get("dash_scope_base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
)
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = (
    os.getenv("MINIMAX_BASE_URL")
    or RUNTIME_QUANT.get("minimax_base_url", "https://api.minimaxi.com/v1")
)

QUANT_LLM_MODEL = os.getenv("QUANT_LLM_MODEL", RUNTIME_QUANT.get("quant_llm_model", "deepseek-r1:1.5b"))
CODE_LLM_MODEL = os.getenv("CODE_LLM_MODEL", RUNTIME_QUANT.get("code_llm_model", "glm-4.7-flash:latest"))
DISCOVERY_LEARNING_PATH = Path(os.getenv("DISCOVERY_LEARNING_PATH", str(QUANT_RUNTIME_ROOT / "nexus_discovery_learning.json")))
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", str(RUNTIME_QUANT.get("llm_timeout_s", 25))))
DISCOVERY_TIME_BUDGET_S = int(os.getenv("DISCOVERY_TIME_BUDGET_S", "75"))
DISCOVERY_QUICK_UNIVERSE_CAP = int(os.getenv("DISCOVERY_QUICK_UNIVERSE_CAP", "28"))
DISCOVERY_FULL_UNIVERSE_CAP = int(os.getenv("DISCOVERY_FULL_UNIVERSE_CAP", "120"))
print(
    f"[runtime-config] path={RUNTIME_CONFIG_PATH_LOADED} llm_provider={LLM_PROVIDER} "
    f"quant_model={QUANT_LLM_MODEL} code_model={CODE_LLM_MODEL} timeout_s={LLM_TIMEOUT_S}"
)

r = redis.from_url(REDIS_URL, decode_responses=True)

def get_llm_response(system: str, user: str, model: str | None = None, provider: str | None = None, timeout_s: int | None = None) -> str:
    """Unified LLM client supporting multiple providers."""
    prov = (provider or LLM_PROVIDER).lower()
    model = model or QUANT_LLM_MODEL
    timeout_s = int(timeout_s or LLM_TIMEOUT_S)
    timeout_s = max(5, min(timeout_s, 180))
    
    # Strip provider prefix if present (e.g. "ollama/deepseek-r1:1.5b")
    if isinstance(model, str) and "/" in model:
        p_prefix, m_name = model.split("/", 1)
        if p_prefix in ["ollama", "openai", "dashscope", "gemini", "minimax"]:
            prov = p_prefix
            model = m_name

    if prov == "ollama":
        chat_data = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False
        }).encode("utf-8")
        req = urllib.request.Request(OLLAMA_CHAT_API, data=chat_data, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw_res = json.loads(resp.read().decode("utf-8"))
                return raw_res.get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Compatibility fallback for Ollama versions without /api/chat.
                gen_data = json.dumps({
                    "model": model,
                    "prompt": f"System:\n{system}\n\nUser:\n{user}",
                    "stream": False
                }).encode("utf-8")
                gen_req = urllib.request.Request(OLLAMA_GENERATE_API, data=gen_data, method="POST")
                gen_req.add_header("Content-Type", "application/json")
                try:
                    with urllib.request.urlopen(gen_req, timeout=timeout_s) as gen_resp:
                        gen_res = json.loads(gen_resp.read().decode("utf-8"))
                        return gen_res.get("response", "")
                except Exception as gen_err:
                    print(f"[LLM] Ollama generate fallback error: {gen_err}")
                    return ""
            print(f"[LLM] Ollama HTTP error: {e}")
            return ""
        except Exception as e:
            print(f"[LLM] Ollama error: {e}")
            return ""
            
    elif prov in ["openai", "dashscope", "gemini", "minimax"]:
        # Standard OpenAI compatible API call
        if prov == "openai":
            api_key = OPENAI_API_KEY
            base_url = OPENAI_BASE_URL
        elif prov == "minimax":
            api_key = MINIMAX_API_KEY
            base_url = MINIMAX_BASE_URL
        elif prov == "gemini":
            base_url = os.getenv("GEMINI_BASE_URL", DASH_SCOPE_BASE_URL)
            api_key = os.getenv("GEMINI_API_KEY", DASH_SCOPE_API_KEY)
        else:
            api_key = DASH_SCOPE_API_KEY
            base_url = DASH_SCOPE_BASE_URL

        if not api_key:
            print(f"[LLM] Provider {prov} requested but API key is missing.")
            print("[LLM] Falling back to Ollama due to missing cloud API key...")
            return get_llm_response(system, user, model=model, provider="ollama", timeout_s=min(timeout_s, 25))

        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(f"[LLM] Cloud provider {prov} error: {e}")
            # Fallback to Ollama if it's not the primary
            if prov != "ollama":
                print("[LLM] Falling back to Ollama...")
                return get_llm_response(system, user, model=model, provider="ollama", timeout_s=min(timeout_s, 25))
            return ""
    
    return ""

def get_s3_client():
    return boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)

def archive_file(file_path: Path):
    s3 = get_s3_client()
    with open(file_path, "rb") as f:
        data = f.read()
        file_hash = hashlib.sha256(data).hexdigest()
    ext = file_path.suffix.lower().replace(".", "")
    now = datetime.now()
    object_key = f"quant/{now.strftime('%Y/%m/%d')}/{file_hash[:2]}/{file_hash}.{ext}"
    s3.put_object(Bucket='nexus-artifacts', Key=object_key, Body=data)
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "html": "text/html",
        "htm": "text/html",
        "md": "text/markdown",
        "csv": "text/csv",
        "pdf": "application/pdf",
        "json": "application/json",
    }
    return {"name": file_path.name, "object_key": object_key, "sha256": file_hash, "size": len(data), "mime": mime_map.get(ext, "text/plain")}

def _now_date_str():
    return datetime.now().strftime("%Y-%m-%d")

def _detect_lang(text: str) -> str:
    if not text:
        return "unknown"
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"

def _load_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None

def _load_watchlists():
    base_dir = Path("/app/quant_trading/Project_optimized")
    selected_path = base_dir / "selected_tickers.json"
    jp_symbols = []
    jp_names = {}

    selected = _load_json(selected_path) or {}
    if isinstance(selected, dict) and isinstance(selected.get("symbols"), list):
        jp_symbols = [s for s in selected.get("symbols") if isinstance(s, str)]

    # Pull names from db_update TARGET_UNIVERSE when available.
    try:
        import sys
        sys.path.insert(0, str(base_dir))
        import db_update  # type: ignore
        for sym, name, _sector in getattr(db_update, "TARGET_UNIVERSE", []):
            if isinstance(sym, str):
                jp_names[sym] = name
        sys.path.pop(0)
    except Exception:
        pass

    us_path = Path(os.getenv("NEWS_US_UNIVERSE_PATH", "/app/configs/universe_us.json"))
    us_list = _load_json(us_path) or []
    us_symbols = []
    us_names = {}
    if isinstance(us_list, list):
        for it in us_list:
            if isinstance(it, dict) and it.get("symbol"):
                sym = str(it.get("symbol"))
                us_symbols.append(sym)
                if it.get("name"):
                    us_names[sym] = str(it.get("name"))

    return {
        "jp_symbols": jp_symbols,
        "jp_names": jp_names,
        "us_symbols": us_symbols,
        "us_names": us_names,
    }

def _gdelt_doc_search(query: str, start_dt: datetime, end_dt: datetime, max_records: int = 250):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("articles", []) if isinstance(data, dict) else []
    except Exception:
        return []

def _safe_snippet_from_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return ""
        html = resp.text
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except Exception:
            return ""
        soup = BeautifulSoup(html, "lxml")
        for key in ["description", "og:description", "twitter:description"]:
            tag = soup.find("meta", attrs={"name": key}) or soup.find("meta", attrs={"property": key})
            if tag and tag.get("content"):
                return str(tag.get("content")).strip()
        return ""
    except Exception:
        return ""

def _make_bar_chart(data: list[tuple[str, int]], title: str, path: Path):
    if not plt:
        return False
    if not data:
        return False
    labels = [k for k, _ in data]
    values = [v for _, v in data]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values, color="#1f77b4")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True

def _img_to_data_uri(path: Path) -> str:
    try:
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""

def _calculate_atr(symbol: str, period: int = 14) -> float | None:
    """Calculate Average True Range (ATR) for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo") # Get enough data for 14-day ATR
        if len(hist) < period + 1:
            return None
        
        # Calculate True Range (TR)
        high = hist['High']
        low = hist['Low']
        close_prev = hist['Close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean().iloc[-1]
        return float(atr)
    except Exception as e:
        print(f"[worker] ATR calculation error for {symbol}: {e}")
        return None

def _get_tick_size(price: float, symbol: str = "") -> float:
    """Get minimum tick size. Defaulting to TOPIX tick size logic for .T stocks."""
    if symbol.upper().endswith(".T"):
        # Simplified TOPIX tick size rules
        if price <= 1000: return 0.1
        if price <= 3000: return 0.5
        if price <= 5000: return 1.0
        if price <= 10000: return 1.0
        if price <= 30000: return 5.0
        if price <= 50000: return 10.0
        return 10.0
    else:
        # Default US/Generic (cent)
        return 0.01

def _calculate_limit_prices(symbol: str, side: str, close: float, atr: float, ma5: float | None = None) -> dict:
    """Calculate three levels of limit prices based on r1 strategy."""
    # 1. ATR clamp
    atr_min = close * 0.003
    atr_max = close * 0.08
    atr_eff = max(atr_min, min(atr_max, atr))
    
    # 2. Anchor
    anchor = close
    if ma5 is not None:
        if side.upper() == "BUY":
            anchor = min(close, ma5)
        else:
            anchor = max(close, ma5)
            
    tick = _get_tick_size(close, symbol)
    
    def _finalize(raw_px, side):
        # 5. Price safety guard (10%)
        p_guard = 0.10
        lower_guard = close * (1 - p_guard)
        upper_guard = close * (1 + p_guard)
        clamped = max(lower_guard, min(upper_guard, raw_px))
        
        # 6. Tick size alignment
        if side.upper() == "BUY":
            return math.floor(clamped / tick) * tick
        else:
            return math.ceil(clamped / tick) * tick

    res = {}
    if side.upper() == "BUY":
        res["aggressive"] = _finalize(anchor + 0.15 * atr_eff, "BUY")
        res["balanced"]   = _finalize(anchor - 0.10 * atr_eff, "BUY")
        res["patient"]    = _finalize(anchor - 0.35 * atr_eff, "BUY")
    else:
        res["aggressive"] = _finalize(anchor - 0.15 * atr_eff, "SELL")
        res["balanced"]   = _finalize(anchor + 0.10 * atr_eff, "SELL")
        res["patient"]    = _finalize(anchor + 0.35 * atr_eff, "SELL")
        
    return res

def calc_limit_price_tool(payload: dict):
    """Tool: Calculate limit prices for a symbol."""
    symbol = payload.get("symbol")
    if not symbol:
        return {"ok": False, "error": "Symbol is required."}
    
    symbol = str(symbol).upper()
    side = str(payload.get("side", "BUY")).upper()
    
    quote = _fetch_quote_facts(symbol)
    close = quote.get("price")
    if close is None:
        return {"ok": False, "error": "Could not fetch current price."}
    
    atr = _calculate_atr(symbol) or (close * 0.02) # Fallback to 2% vol
    
    # Try to get MA5 from technical metrics
    metrics = _compute_quant_metrics(symbol)
    ma5 = None # Existing _compute_quant_metrics doesn't have MA5 specifically yet, maybe add later
    
    prices = _calculate_limit_prices(symbol, side, close, atr, ma5)
    
    return {
        "ok": True,
        "symbol": symbol,
        "side": side,
        "current_price": close,
        "atr_used": atr,
        "limit_prices": prices,
        "tick_size": _get_tick_size(close, symbol)
    }

# --- AI Analysis with Fallback Request ---
def ai_analyze_report(payload: dict):
    model = payload.get("model") or QUANT_LLM_MODEL
    direct_prompt = str(payload.get("prompt") or "").strip()
    report_path = Path("/app/quant_trading/Project_optimized/reports/strategy_report.html")
    
    content = "Summary of backtest results unavailable."
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()[:2500]

    if direct_prompt:
        system = "You are a concise financial AI assistant."
        user = direct_prompt
    else:
        system = "You are a senior quantitative analyst. Provide 3 sharp insights based on the HTML report provided."
        user = f"Here is the report content:\n{content}"
    
    analysis = get_llm_response(system, user, model=model)
    if analysis:
        return {"ok": True, "model_used": model, "analysis": analysis}
    else:
        return {"ok": False, "error": "LLM_FAILURE", "suggestion": "Check API keys or Ollama connectivity."}


def fetch_stock_price(payload: dict):
    symbol = _normalize_symbol_for_lookup(payload.get("symbol", "NVDA"))
    quote = _fetch_quote_facts(symbol)
    return {"symbol": symbol, "price": quote.get("price"), "ts": time.time(), "company_name": quote.get("company_name")}

def _quote_url(symbol: str) -> str:
    sym = str(symbol or "").upper()
    if sym.endswith(".T"):
        return f"https://finance.yahoo.co.jp/quote/{sym}"
    return f"https://finance.yahoo.com/quote/{sym}"

def _normalize_symbol_for_lookup(symbol: str) -> str:
    sym = str(symbol or "").upper().strip()
    # Most JP cash equities are 4-digit codes; normalize to Yahoo JP suffix.
    if re.fullmatch(r"\d{4}", sym):
        return f"{sym}.T"
    return sym

def _infer_market_from_symbol(symbol: str) -> str:
    sym = _normalize_symbol_for_lookup(symbol)
    return "JP" if sym.endswith(".T") else "US"

def _safe_float(v):
    try:
        if v is None:
            return None
        fv = float(v)
        if fv != fv:
            return None
        return fv
    except Exception:
        return None

def _fetch_quote_facts(symbol: str) -> dict:
    sym = _normalize_symbol_for_lookup(symbol)
    ticker = yf.Ticker(sym)
    company_name = None
    currency = None
    price = None
    market_cap = None
    source = "yfinance"
    recent_news = []

    try:
        info = ticker.info or {}
        company_name = info.get("longName") or info.get("shortName")
        currency = info.get("currency")
        market_cap = _safe_float(info.get("marketCap"))
    except Exception:
        pass

    try:
        fi = ticker.fast_info or {}
        price = _safe_float(fi.get("last_price"))
        currency = currency or fi.get("currency")
        market_cap = market_cap or _safe_float(fi.get("market_cap"))
    except Exception:
        pass

    hist = None
    try:
        hist = ticker.history(period="10d", interval="1d", auto_adjust=False)
        if price is None and hist is not None and not hist.empty:
            price = _safe_float(hist["Close"].dropna().iloc[-1])
    except Exception:
        hist = None

    if hist is not None and not hist.empty and "Close" in hist:
        closes = hist["Close"].dropna()
        if len(closes) >= 2:
            prev_close = _safe_float(closes.iloc[-2])
            if price is not None and prev_close not in (None, 0):
                change_pct = (price - prev_close) / prev_close * 100.0
            else:
                change_pct = None
        else:
            prev_close = None
            change_pct = None
    else:
        prev_close = None
        change_pct = None

    try:
        news_raw = ticker.news or []
        for n in news_raw[:5]:
            title = str(n.get("title") or "").strip()
            link = str(n.get("link") or "").strip()
            pub = str(n.get("publisher") or "").strip()
            ts = n.get("providerPublishTime")
            if title:
                recent_news.append({
                    "title": title,
                    "url": link,
                    "publisher": pub,
                    "published_ts": ts,
                })
    except Exception:
        recent_news = []

    return {
        "symbol": sym,
        "company_name": company_name,
        "price": price,
        "currency": currency,
        "market_cap": market_cap,
        "prev_close": prev_close,
        "change_pct": change_pct,
        "quote_url": _quote_url(sym),
        "recent_news": recent_news,
        "source": source,
    }

def _fetch_ss6_signal(symbol: str) -> dict:
    """Fetch professional regression signal from ss6_sqlite database if available."""
    sym = str(symbol or "").upper()
    db_path = QUANT_DB_PATH
    if not db_path.exists():
        return {"found": False}
    
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            import sys
            sys.path.insert(0, str(db_path.parent))
            import trade_schema
            trade_schema.ensure_trade_tables(conn)
            sys.path.pop(0)
        except Exception:
            pass
        cur = conn.cursor()
        # Try to find the latest order reason for this symbol which often contains the alpha score
        cur.execute(
            "SELECT reason, asof, side FROM orders WHERE symbol = ? ORDER BY asof DESC LIMIT 1",
            (sym,)
        )
        row = cur.fetchone()
        conn.close()
        
        if row:
            reason, asof, side = row
            # Extract score from reason string like "score=0.45"
            score = None
            match = re.search(r"score=([0-9.-]+)", str(reason))
            if match:
                score = float(match.group(1))
            
            return {
                "found": True,
                "asof": asof,
                "side": side,
                "reason": reason,
                "model_score": score,
                "source": "ss6_sqlite_regression"
            }
    except Exception as e:
        print(f"[worker] db fetch error: {e}")
    return {"found": False}

def _compute_quant_metrics(symbol: str) -> dict:
    sym = _normalize_symbol_for_lookup(symbol)
    ticker = yf.Ticker(sym)
    
    # NEW: Fetch professional signal first
    ss6 = _fetch_ss6_signal(sym)
    
    try:
        # E-03: Use auto_adjust=True (split/dividend-adjusted) to avoid spurious signals on ex-div dates
        hist = ticker.history(period="1y", interval="1d", auto_adjust=True)
    except Exception:
        hist = None

    if hist is None or hist.empty or "Close" not in hist:
        return {
            "ok": False,
            "signal": "N/A",
            "ret_5d_pct": None,
            "ret_20d_pct": None,
            "ret_60d_pct": None,
            "vol_20d_pct": None,
            "rsi14": None,
            "sma20": None,
            "sma60": None,
            "z20": None,
            "alpha_score": None,
            "risk_state": "unknown",
            "benchmark_ret_20d_pct": None,
            "relative_20d_pct": None,
            "ss6_signal": ss6
        }

    close = hist["Close"].dropna()
    if len(close) < 60:
        return { "ok": False, "signal": "Insufficient Data", "risk_state": "unknown", "ss6_signal": ss6 }

    def _pct_return(series, n):
        if len(series) <= n:
            return None
        a = _safe_float(series.iloc[-1])
        b = _safe_float(series.iloc[-n-1])
        if a is None or b in (None, 0):
            return None
        return (a / b - 1.0) * 100.0

    ret_5d = _pct_return(close, 5)
    ret_20d = _pct_return(close, 20)
    ret_60d = _pct_return(close, 60)

    returns = close.pct_change().dropna()
    vol_20d = None
    if len(returns) >= 20:
        try:
            vol_20d = float(returns.tail(20).std() * (252 ** 0.5) * 100.0)
        except Exception:
            vol_20d = None

    # Use tail(120) for indicators to ensure enough history for rolling
    sma20 = _safe_float(close.rolling(20).mean().iloc[-1])
    sma60 = _safe_float(close.rolling(60).mean().iloc[-1])
    std20 = _safe_float(close.rolling(20).std().iloc[-1])
    
    z20 = None
    latest = _safe_float(close.iloc[-1])
    if sma20 is not None and std20 not in (None, 0) and latest is not None:
        z20 = (latest - sma20) / std20

    rsi14 = None
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean().iloc[-1]
        avg_loss = loss.rolling(14).mean().iloc[-1]
        if avg_loss == 0 and avg_gain > 0:
            rsi14 = 100.0
        elif avg_loss and avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi14 = 100.0 - (100.0 / (1.0 + rs))

    # Benchmark comparison
    benchmark_ret_20d = None
    relative_20d = None
    benchmark_symbol = "^N225" if sym.endswith(".T") else "^GSPC"
    try:
        bench = yf.Ticker(benchmark_symbol).history(period="6mo", interval="1d", auto_adjust=False)
        if bench is not None and not bench.empty and "Close" in bench:
            benchmark_ret_20d = _pct_return(bench["Close"].dropna(), 20)
    except Exception:
        pass
        
    if ret_20d is not None and benchmark_ret_20d is not None:
        relative_20d = ret_20d - benchmark_ret_20d

    # Scoring logic optimization
    score = 0.0
    if ret_20d is not None:
        score += 1.0 if ret_20d > 0 else -1.0
    if sma20 is not None and sma60 is not None:
        score += 1.5 if sma20 > sma60 else -1.5 # Weight SMA crossover more
    if rsi14 is not None:
        if 45 <= rsi14 <= 65:
            score += 0.5
        elif rsi14 >= 75:
            score -= 1.0 # Overbought
        elif rsi14 <= 30:
            score += 1.0 # Oversold / Reversal potential
    if relative_20d is not None:
        score += 1.0 if relative_20d > 0 else -1.0

    # INTEGRATION: Boost score if professional model agrees
    if ss6.get("found"):
        m_score = ss6.get("model_score")
        if m_score is not None:
            if m_score > 0: score += 2.0  # Professional model confirmation
            else: score -= 2.0

    if score >= 2.5:
        signal = "Strong Buy"
        risk_state = "Risk-ON"
    elif score >= 1.0:
        signal = "Overweight"
        risk_state = "Risk-ON"
    elif score <= -2.5:
        signal = "Strong Sell"
        risk_state = "Risk-OFF"
    elif score <= -1.0:
        signal = "Underweight"
        risk_state = "Risk-OFF"
    else:
        signal = "Neutral"
        risk_state = "Neutral"

    return {
        "ok": True,
        "signal": signal,
        "ret_5d_pct": ret_5d,
        "ret_20d_pct": ret_20d,
        "ret_60d_pct": ret_60d,
        "vol_20d_pct": vol_20d,
        "rsi14": rsi14,
        "sma20": sma20,
        "sma60": sma60,
        "z20": z20,
        "alpha_score": score,
        "risk_state": risk_state,
        "benchmark_ret_20d_pct": benchmark_ret_20d,
        "relative_20d_pct": relative_20d,
        "ss6_signal": ss6
    }

def _extract_direct_url_from_google_link(link: str) -> str:
    """
    Try fast extraction from known query params used by Google News redirect links.
    """
    try:
        parsed = urllib.parse.urlparse(str(link or ""))
        if not parsed.scheme or not parsed.netloc:
            return ""
        q = urllib.parse.parse_qs(parsed.query or "")
        for key in ("url", "u", "q"):
            vals = q.get(key) or []
            if vals:
                cand = urllib.parse.unquote(str(vals[0]).strip())
                if cand.startswith(("http://", "https://")):
                    return cand
    except Exception:
        return ""
    return ""

def _resolve_google_news_link(link: str, timeout_s: int = 8) -> str:
    """
    Resolve a Google News RSS URL to a direct publisher URL when possible.
    Fail-safe: return input link.
    """
    src = str(link or "").strip()
    if not src:
        return src
    if "news.google." not in src:
        return src

    extracted = _extract_direct_url_from_google_link(src)
    if extracted:
        return extracted

    try:
        # Google RSS links usually 30x to publisher URL. requests will follow redirects.
        resp = requests.get(
            src,
            timeout=timeout_s,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NexusBot/1.0)"},
        )
        final_url = str(getattr(resp, "url", "") or "").strip()
        if final_url and "news.google." not in final_url:
            return final_url
    except Exception:
        pass
    return src

def _search_direct_url_by_title(title: str, publisher: str = "", timeout_s: int = 10) -> str:
    """
    Fallback resolver: search by headline and publisher, return first likely direct source URL.
    """
    t = str(title or "").strip()
    p = str(publisher or "").strip()
    if not t:
        return ""
    query = f"\"{t[:140]}\" {p}".strip()
    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=timeout_s,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NexusBot/1.0)"},
        )
        if resp.status_code != 200:
            return ""
        text = resp.text or ""
        # DuckDuckGo redirect style: /l/?uddg=<url-encoded>
        candidates = re.findall(r'href=\"(?:https?:)?//duckduckgo\\.com/l/\\?[^\"#]*?uddg=([^\"&]+)', text)
        candidates += re.findall(r'href=\"/l/\\?[^\"#]*?uddg=([^\"&]+)', text)
        for c in candidates:
            u = urllib.parse.unquote(str(c))
            if not u.startswith(("http://", "https://")):
                continue
            low = u.lower()
            if "news.google." in low or "duckduckgo.com" in low:
                continue
            if any(x in low for x in ["google.com", "gstatic.com", "doubleclick.net"]):
                continue
            return u
    except Exception:
        return ""
    return ""

def _build_article_search_url(title: str, publisher: str = "") -> str:
    q = f"{str(title or '').strip()} {str(publisher or '').strip()}".strip()
    if not q:
        return ""
    return "https://www.google.com/search?q=" + urllib.parse.quote_plus(q)

def _fetch_news_from_google_rss(
    symbol: str,
    company_name: str | None,
    max_items: int = 5,
    lang: str = "en",
    resolve_original_links: bool = False,
) -> list[dict]:

    query_terms = [symbol]
    if company_name:
        query_terms.append(company_name)
    if lang != "ja":
        query_terms.append("stock")
    q = " ".join([x for x in query_terms if x])
    url = "https://news.google.com/rss/search"
    
    params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    if lang == "ja":
        params = {"q": q, "hl": "ja", "gl": "JP", "ceid": "JP:ja"}
    elif lang == "zh":
        params = {"q": q, "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"}

    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        out = []
        resolve_cache = {}
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("source") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            published_ts = None
            try:
                if pub_date:
                    published_ts = parsedate_to_datetime(pub_date).timestamp()
            except Exception:
                published_ts = None
            if title and link:
                final_link = link
                if resolve_original_links:
                    if link in resolve_cache:
                        final_link = resolve_cache[link]
                    else:
                        final_link = _resolve_google_news_link(link)
                        resolve_cache[link] = final_link
                out.append({
                    "title": title,
                    "url": final_link,
                    "url_google": link,
                    "publisher": pub,
                    "published_at": pub_date,
                    "published_ts": published_ts,
                    "source": "google_news_rss",
                    "via_google_redirect": bool(final_link and final_link != link),
                })
        return out
    except Exception:
        return []

def _parse_article_dt(article: dict) -> Optional[datetime]:
    if not isinstance(article, dict):
        return None
    candidates = [
        article.get("published_ts"),
        article.get("providerPublishTime"),
        article.get("published_at"),
        article.get("pubDate"),
        article.get("seendate"),
        article.get("pubdate"),
        article.get("date"),
    ]
    for v in candidates:
        if v in (None, ""):
            continue
        try:
            if isinstance(v, (int, float)):
                return datetime.utcfromtimestamp(float(v))
            text = str(v).strip()
            if not text:
                continue
            # GDELT format: 20260227T061500Z or 20260227T061500
            if re.match(r"^\d{8}T\d{6}Z?$", text):
                fmt = "%Y%m%dT%H%M%SZ" if text.endswith("Z") else "%Y%m%dT%H%M%S"
                return datetime.strptime(text, fmt)
            try:
                return parsedate_to_datetime(text).astimezone().replace(tzinfo=None)
            except Exception:
                pass
            # ISO fallback
            if "T" in text:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
    return None

def _apply_freshness_policy(articles: List[dict], max_age_hours: float) -> tuple[List[dict], dict]:
    now_utc = datetime.utcnow()
    fresh = []
    dropped_stale = 0
    dropped_undated = 0
    ages_h = []
    for a in articles or []:
        dt = _parse_article_dt(a)
        if dt is None:
            dropped_undated += 1
            continue
        age_hours = (now_utc - dt).total_seconds() / 3600.0
        if age_hours < 0:
            age_hours = 0.0
        if age_hours <= max_age_hours:
            b = dict(a)
            b["_age_hours"] = age_hours
            fresh.append(b)
            ages_h.append(age_hours)
        else:
            dropped_stale += 1
    fresh.sort(key=lambda x: x.get("_age_hours", 1e9))
    for it in fresh:
        it.pop("_age_hours", None)
    stats = {
        "max_age_hours": max_age_hours,
        "input_count": len(articles or []),
        "fresh_count": len(fresh),
        "dropped_stale": dropped_stale,
        "dropped_undated": dropped_undated,
        "freshness_p50_hours": round(statistics.median(ages_h), 2) if ages_h else None,
        "freshness_max_hours": round(max(ages_h), 2) if ages_h else None,
    }
    return fresh, stats

def _fetch_news_from_quote_page(symbol: str, max_items: int = 5) -> list[dict]:
    quote_url = _quote_url(symbol)
    urls = [quote_url]
    if symbol.endswith(".T"):
        urls.insert(0, f"{quote_url}/news")
    else:
        urls.insert(0, f"https://finance.yahoo.com/quote/{symbol}/news")
    banned_exact = {"ニュース", "もっと見る", "news", "more", "latest news"}
    banned_fragments = ["/nisa/", "/finance/", "/calendar/"]
    seen = set()
    out = []
    try:
        from bs4 import BeautifulSoup  # type: ignore
        for url in urls:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = str(a.get("href") or "").strip()
                text = str(a.get_text(" ", strip=True) or "").strip()
                if not text:
                    continue
                if len(text) < 10:
                    continue
                if text.lower() in banned_exact:
                    continue
                href_low = href.lower()
                if any(x in href_low for x in banned_fragments):
                    continue
                if "news" not in href_low and "article" not in href_low:
                    continue
                if href.startswith("/"):
                    if symbol.endswith(".T"):
                        href = "https://finance.yahoo.co.jp" + href
                    else:
                        href = "https://finance.yahoo.com" + href
                if not href.startswith("http"):
                    continue
                key = (text[:160], href)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "title": text,
                    "url": href,
                    "publisher": "",
                    "published_at": "",
                    "source": "quote_page",
                })
                if len(out) >= max_items:
                    return out
        return out
    except Exception:
        return []

def _fetch_news_from_yfinance(symbol: str, max_items: int = 5) -> list[dict]:
    sym = _normalize_symbol_for_lookup(symbol)
    try:
        ticker = yf.Ticker(sym)
        news_raw = ticker.news or []
        out = []
        for n in news_raw[:max_items]:
            title = str(n.get("title") or "").strip()
            link = str(n.get("link") or "").strip()
            pub = str(n.get("publisher") or "").strip()
            ts = n.get("providerPublishTime")
            if title:
                out.append({"title": title, "url": link, "publisher": pub, "published_ts": ts, "source": "yfinance"})
        return out
    except Exception as e:
        print(f"[yfinance] Failed to fetch news for {symbol}: {e}")
        return []


def _fetch_news_from_openbb(symbol: str, max_items: int = 5) -> list[dict]:
    """Fetch news using OpenBB Platform SDK v4 with timeout and environment variable credentials."""
    import concurrent.futures
    import os
    try:
        from openbb import obb
        
        # Inject API keys if present in the environment to avoid free tier rate limits
        fmp_key = os.getenv("OPENBB_FMP_KEY")
        if fmp_key:
            # We use try/except because openbb structure might slightly differ across versions
            try: obb.user.credentials.fmp_api_key = fmp_key
            except Exception: pass

        q_sym = symbol.replace('.T', '') if symbol.endswith('.T') else symbol
        
        # Run the fetch in a separate thread with a strict 10-second timeout
        # to prevent blocking the worker event loop
        def do_fetch():
            return obb.news.company(symbol=q_sym, limit=max_items)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(do_fetch)
            res = future.result(timeout=10)

        if not res or not res.results:
            return []
        
        out = []
        for item in res.results:
            if isinstance(item, dict):
                title = item.get('title', '')
                url = item.get('url', '')
                pub_date = item.get('date', '')
                publisher = item.get('source', 'openbb')
            else:
                title = getattr(item, 'title', '')
                url = getattr(item, 'url', '')
                pub_date = getattr(item, 'date', '')
                publisher = getattr(item, 'source', 'openbb')
                
            out.append({
                "title": str(title),
                "url": str(url),
                "publisher": str(publisher),
                "published_at": str(pub_date),
                "source": "openbb"
            })
        return out
    except concurrent.futures.TimeoutError:
        print(f"[openbb] Timeout (10s) fetching news for {symbol}")
        return []
    except Exception as e:
        print(f"[openbb] Failed to fetch news for {symbol}: {e}")
        return []

def _merge_recent_news(quote: dict, max_items: int = 5) -> list[dict]:
    news = list((quote or {}).get("recent_news") or [])
    symbol = str((quote or {}).get("symbol") or "")
    company_name = (quote or {}).get("company_name")
    
    try:
        yfinance_news = _fetch_news_from_yfinance(symbol, max_items=max_items * 2)
    except Exception as e:
        print(f"[merge_news] yfinance fetch failed for {symbol}: {e}")
        yfinance_news = []
    news.extend(yfinance_news)

    if len(yfinance_news) < max_items:
        news.extend(_fetch_news_from_quote_page(symbol, max_items=max_items * 2))
        news.extend(_fetch_news_from_google_rss(symbol, company_name, max_items=max_items * 3))

    trusted_publishers = [
        "reuters", "nikkei", "bloomberg", "wsj", "financial times", "marketwatch",
        "yahoo", "traders-web", "minkabu", "cabotan",
    ]
    blocked_publishers = ["meyka"]

    out = []
    seen = set()
    for item in news:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        publisher = str(item.get("publisher") or "").strip()
        if not title:
            continue
        if "login.yahoo.co.jp" in url.lower():
            continue
        publisher_low = publisher.lower()
        if any(b in publisher_low for b in blocked_publishers):
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        title_low = title.lower()
        url_low = url.lower()
        score = 0
        if symbol and (symbol.lower() in title_low or symbol.lower() in url_low):
            score += 3
        if company_name and str(company_name).lower() in title_low:
            score += 2
        if "finance.yahoo.co.jp/news/detail/" in url_low or "finance.yahoo.com/news/" in url_low:
            score += 1
        if publisher_low:
            if any(p in publisher_low for p in trusted_publishers):
                score += 3
        out.append({
            "title": title,
            "url": url,
            "publisher": publisher,
            "published_ts": item.get("published_ts"),
            "published_at": str(item.get("published_at") or ""),
            "source": str(item.get("source") or quote.get("source") or "unknown"),
            "_score": score,
        })
    out.sort(key=lambda x: (x.get("_score", 0), str(x.get("published_ts") or ""), str(x.get("published_at") or "")), reverse=True)
    trimmed = out[:max_items]
    for item in trimmed:
        item.pop("_score", None)
    return trimmed

def _fmt_num(v, digits=2, suffix=""):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{digits}f}{suffix}"
    except Exception:
        return "N/A"

def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n| N/A | " + " | ".join([""] * (len(headers)-1)) + " |"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [str(x).replace("\n", " ").replace("|", "/") for x in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def _render_quant_template_report(run_id: str, quote: dict, metrics: dict, news: list[dict]) -> tuple[str, str]:
    template_path = Path("/app/quant_trading/execution_report_template.md")
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        template = "# Execution Report\n\n{{target_weights_table}}\n\n{{trade_list_table}}"

    symbol = quote.get("symbol") or "N/A"
    company_name = quote.get("company_name") or "N/A"
    ccy = quote.get("currency") or "N/A"
    price = quote.get("price")
    quote_url = quote.get("quote_url") or "N/A"
    today = datetime.now().strftime("%Y-%m-%d")

    feature_rows = [[
        symbol,
        _fmt_num(1.0 if metrics.get("signal") == "Overweight" else 0.0, 2),
        _fmt_num(metrics.get("ret_60d_pct"), 2, "%"),
        _fmt_num(metrics.get("rsi14"), 2),
        _fmt_num(metrics.get("vol_20d_pct"), 2, "%"),
        _fmt_num(metrics.get("z20"), 2),
        _fmt_num(
            (metrics.get("sma20") - metrics.get("sma60")) / metrics.get("sma60") * 100.0
            if metrics.get("sma20") is not None and metrics.get("sma60") not in (None, 0)
            else None,
            2,
            "%",
        ),
        "derived from 6m daily close",
    ]]
    feature_table = _markdown_table(
        ["Ticker", "Weight", "slope60", "rsi14", "vol20", "z20", "ma_gap", "Notes"],
        feature_rows,
    )

    alpha_table = _markdown_table(
        ["Ticker", "Alpha Score", "Rank", "Selected?", "Comment"],
        [[
            symbol,
            _fmt_num(metrics.get("alpha_score"), 2),
            "1",
            "yes" if metrics.get("signal") == "Overweight" else "no",
            f"signal={metrics.get('signal', 'N/A')}",
        ]],
    )

    trade_side = "BUY" if metrics.get("signal") == "Overweight" else ("SELL" if metrics.get("signal") == "Underweight" else "HOLD")
    trade_qty = "100" if trade_side in ("BUY", "SELL") else "0"
    trade_table = _markdown_table(
        ["Ticker", "Side", "Qty (shares)", "Qty (lots)", "Est. Price", "Notional", "Reason"],
        [[
            symbol,
            trade_side,
            trade_qty,
            "1" if trade_qty != "0" else "0",
            _fmt_num(price, 3),
            _fmt_num((price or 0.0) * (100 if trade_qty != "0" else 0.0), 0),
            f"score={_fmt_num(metrics.get('alpha_score'), 2)}",
        ]],
    )

    target_table = _markdown_table(
        ["Ticker", "Name", "Current Weight", "Target Weight", "Delta"],
        [[
            symbol,
            company_name,
            "N/A",
            "100%" if metrics.get("signal") == "Overweight" else ("0%" if metrics.get("signal") == "Underweight" else "50%"),
            "N/A",
        ]],
    )

    top_news = []
    for n in news[:5]:
        t = n.get("title") or "N/A"
        u = n.get("url") or ""
        top_news.append(f"- {t} ({u})")
    top_news_block = "\n".join(top_news) if top_news else "- N/A"

    replacements = {
        "run_id": run_id or "manual",
        "asof_date": today,
        "exec_date": today,
        "universe_desc": f"single-name focus: {symbol}",
        "benchmark_ticker": "^N225" if str(symbol).endswith(".T") else "^GSPC",
        "horizon_days": "20",
        "rebalance_every": "20",
        "train_window": "252",
        "ccy": ccy,
        "risk_state_badge": metrics.get("risk_state", "unknown"),
        "risk_trigger_desc": f"alpha_score={_fmt_num(metrics.get('alpha_score'), 2)}, ret_20d={_fmt_num(metrics.get('ret_20d_pct'), 2, '%')}",
        "pv_pre": "N/A",
        "cash_pre": "N/A",
        "turnover_notional": "N/A",
        "cost_total_est": "N/A",
        "cash_post_est": "N/A",
        "lot_size_rule": "100 shares for JP stocks",
        "rounding_policy": "floor to tradable lot",
        "pretrade_notes": f"quote_url={quote_url}",
        "target_weights_table": target_table,
        "n_pos_target": "1",
        "top3_concentration": "100%",
        "cash_target_implied": "0-100% (regime dependent)",
        "trade_list_table": trade_table,
        "feature_readings_table": feature_table,
        "rsi_rule_text": "70 above overbought, 30 below oversold",
        "z20_rule_text": "positive implies above 20d mean",
        "vol_rule_text": "higher implies higher short-term risk",
        "alpha_score_table": alpha_table,
        "model_name": "quant_trading_signal_v1",
        "fit_method": "deterministic factor scoring",
        "reg_desc": "rule-based score aggregation",
        "feature_set_desc": "ret/vol/rsi/ma/zscore",
        "top_contrib_table": "| 1 | N/A | N/A | N/A | N/A |",
        "bottom_contrib_table": "| 1 | N/A | N/A | N/A | N/A |",
        "adv_check_table": "| N/A | N/A | N/A | N/A | N/A | N/A | N/A |",
        "liquidity_flag_summary": "N/A",
        "liquidity_mitigation_policy": "slice orders if utilization exceeds threshold",
        "impact_power": "0.5",
        "fee_bps": "N/A",
        "fee_amt": "N/A",
        "fee_notes": "broker dependent",
        "slip_bps": "N/A",
        "slip_amt": "N/A",
        "slip_notes": "depends on volatility and liquidity",
        "impact_bps": "N/A",
        "impact_amt": "N/A",
        "impact_k": "N/A",
        "total_cost_bps": "N/A",
        "total_cost_amt": "N/A",
        "gross_exposure": "N/A",
        "net_exposure": "N/A",
        "n_holdings": "1",
        "largest_pos_desc": f"{symbol} (single-name report)",
        "port_vol_est": _fmt_num(metrics.get("vol_20d_pct"), 2, "%"),
        "model_dd_est": "N/A",
        "risk_notes": f"signal={metrics.get('signal', 'N/A')}, relative_20d={_fmt_num(metrics.get('relative_20d_pct'), 2, '%')}",
        "initial_capital": "N/A",
        "final_equity": "N/A",
        "bt_start": "N/A",
        "bt_end": "N/A",
        "n_rebalances": "N/A",
        "ret_total": _fmt_num(metrics.get("ret_60d_pct"), 2, "%"),
        "ret_bench": _fmt_num(metrics.get("benchmark_ret_20d_pct"), 2, "%"),
        "cagr": "N/A",
        "cagr_bench": "N/A",
        "vol": _fmt_num(metrics.get("vol_20d_pct"), 2, "%"),
        "vol_bench": "N/A",
        "sharpe": "N/A",
        "sharpe_bench": "N/A",
        "max_dd": "N/A",
        "max_dd_bench": "N/A",
        "turnover_avg": "N/A",
        "turnover_median": "N/A",
        "cost_paid_total": "N/A",
        "cost_bps_avg": "N/A",
        "pct_risk_on": "N/A",
        "data_sources": "yfinance, Yahoo quote page, Google News RSS",
        "corp_action_policy": "provider-adjusted historical series where available",
        "leakage_control_desc": "uses only up-to-now observations",
        "missing_data_policy": "explicitly marked as N/A",
        "known_limits": "single-name lightweight analysis, not full portfolio backtest",
        "target_weights_raw_block": f"- symbol: {symbol}\n- company: {company_name}\n- latest_price: {_fmt_num(price, 4)} {ccy}",
        "trade_blotter_raw_block": trade_table,
        "config_json": json.dumps({
            "symbol": symbol,
            "quote_url": quote_url,
            "risk_state": metrics.get("risk_state"),
            "signal": metrics.get("signal"),
            "alpha_score": metrics.get("alpha_score"),
        }, ensure_ascii=False, indent=2),
        "changelog_item_1": "Added template-driven single-name report rendering",
        "changelog_item_2": "Added fallback news sources and factor metrics",
    }

    md = template
    for k, v in replacements.items():
        md = md.replace("{{" + k + "}}", str(v))
    md = re.sub(r"\{\{[^{}]+\}\}", "N/A", md)
    md += "\n\n## Latest News Snapshot\n" + top_news_block + "\n"

    # HTML Styling
    signal_color = "#10b981" if metrics.get("signal") in ["Strong Buy", "Overweight"] else ("#ef4444" if metrics.get("signal") in ["Strong Sell", "Underweight"] else "#6b7280")
    
    ss6 = metrics.get("ss6_signal", {})
    ss6_html = ""
    if ss6.get("found"):
        ss6_html = f"""
      <div class="card" style="border-left: 4px solid #3b82f6;">
        <h3>Professional Model (ss6_sqlite)</h3>
        <div class="stat"><span class="stat-label">Model Alpha Score</span><span class="stat-value" style="color:#3b82f6;">{_fmt_num(ss6.get('model_score'), 3)}</span></div>
        <div class="stat"><span class="stat-label">Recommended Side</span><span class="stat-value">{ss6.get('side')}</span></div>
        <div class="stat"><span class="stat-label">As-of Date</span><span class="stat-value">{ss6.get('asof')}</span></div>
        <p style="font-size: 12px; color: #64748b; margin-top: 10px;"><strong>Reason:</strong> {html.escape(str(ss6.get('reason')))}</p>
      </div>
        """

    html_report = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Quant Report: {html.escape(symbol)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{ --primary: #0f172a; --accent: {signal_color}; --bg: #f8fafc; --card-bg: #ffffff; --text: #334155; }}
    body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; line-height: 1.6; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    .header {{ background: var(--primary); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
    .header h1 {{ margin: 0; font-size: 28px; display: flex; align-items: center; gap: 12px; }}
    .badge {{ background: var(--accent); color: white; padding: 4px 12px; border-radius: 9999px; font-size: 14px; font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 24px; }}
    .card {{ background: var(--card-bg); padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1); }}
    .card h3 {{ margin-top: 0; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px; font-size: 16px; color: var(--primary); }}
    .stat {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
    .stat-label {{ color: #64748b; font-size: 14px; }}
    .stat-value {{ font-weight: 600; }}
    .news-item {{ margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #f1f5f9; }}
    .news-title {{ font-size: 14px; font-weight: 500; display: block; color: #2563eb; text-decoration: none; }}
    .news-title:hover {{ text-decoration: underline; }}
    .news-meta {{ font-size: 12px; color: #94a3b8; }}
    pre {{ background: #1e293b; color: #f8fafc; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 13px; margin: 0; }}
    .markdown-block {{ background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ text-align: left; background: #f1f5f9; padding: 10px; }}
    td {{ padding: 10px; border-bottom: 1px solid #f1f5f9; }}
    .positive {{ color: #10b981; }}
    .negative {{ color: #ef4444; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>{html.escape(symbol)} <span class="badge">{metrics.get("signal", "N/A")}</span></h1>
      <p style="margin: 8px 0 0 0; opacity: 0.8;">{html.escape(company_name)} | {today}</p>
    </div>

    <div class="grid">
      <div class="card">
        <h3>Market Snapshot</h3>
        <div class="stat"><span class="stat-label">Latest Price</span><span class="stat-value">{_fmt_num(price, 2)} {ccy}</span></div>
        <div class="stat"><span class="stat-label">1D Change</span><span class="stat-value { 'positive' if (quote.get('change_pct') or 0) > 0 else 'negative' }">{_fmt_num(quote.get('change_pct'), 2, '%')}</span></div>
        <div class="stat"><span class="stat-label">Alpha Score</span><span class="stat-value">{_fmt_num(metrics.get('alpha_score'), 2)}</span></div>
        <div class="stat"><span class="stat-label">Risk Regime</span><span class="stat-value">{metrics.get('risk_state', 'N/A')}</span></div>
      </div>
      <div class="card">
        <h3>Technical Factors</h3>
        <div class="stat"><span class="stat-label">RSI (14)</span><span class="stat-value">{_fmt_num(metrics.get('rsi14'), 1)}</span></div>
        <div class="stat"><span class="stat-label">Z-Score (20)</span><span class="stat-value">{_fmt_num(metrics.get('z20'), 2)}</span></div>
        <div class="stat"><span class="stat-label">Vol (20D Ann.)</span><span class="stat-value">{_fmt_num(metrics.get('vol_20d_pct'), 1, '%')}</span></div>
        <div class="stat"><span class="stat-label">Return (60D)</span><span class="stat-value { 'positive' if (metrics.get('ret_60d_pct') or 0) > 0 else 'negative' }">{_fmt_num(metrics.get('ret_60d_pct'), 1, '%')}</span></div>
      </div>
      {ss6_html}
    </div>

    <div class="card" style="margin-bottom: 24px;">
      <h3>Recent Intelligence</h3>
      {"".join([f'<div class="news-item"><a href="{html.escape(n["url"])}" class="news-title" target="_blank">{html.escape(n["title"])}</a><div class="news-meta">{html.escape(n.get("publisher",""))} | {html.escape(n.get("published_at") or n.get("published_ts") or "")}</div></div>' for n in news[:4]])}
      <a href="{html.escape(quote_url)}" style="font-size: 13px; color: #64748b;">View Full Quote &rarr;</a>
    </div>

    <div class="markdown-block">
      <h3 style="margin-top:0">Full Execution Report (Markdown)</h3>
      <pre>{html.escape(md)}</pre>
    </div>
    
    <p style="text-align: center; color: #94a3b8; font-size: 12px; margin-top: 24px;">
      Generated by OpenClaw Nexus Quant Engine v1.2.6
    </p>
  </div>
</body>
</html>
"""
    return md, html_report

def _grounded_quant_summary(quote: dict) -> str:
    symbol = quote.get("symbol", "unknown")
    company_name = quote.get("company_name") or "Data unavailable"
    price = quote.get("price")
    currency = quote.get("currency") or ""
    change_pct = quote.get("change_pct")
    quote_url = quote.get("quote_url") or "Data unavailable"
    news = quote.get("recent_news") or []

    if price is None:
        price_line = "Data unavailable"
    else:
        price_line = f"{price:.4f} {currency}".strip()

    if change_pct is None:
        change_line = "Data unavailable"
    else:
        change_line = f"{change_pct:.2f}%"

    lines = [
        f"{symbol} factual snapshot",
        f"- company: {company_name}",
        f"- latest_price: {price_line}",
        f"- 1d_change_pct: {change_line}",
        f"- quote_url: {quote_url}",
        "- recent_news:",
    ]
    if news:
        for idx, item in enumerate(news[:5], start=1):
            lines.append(f"  {idx}. {item.get('title','Data unavailable')} ({item.get('url','')})")
    else:
        lines.append("  Data unavailable")
    lines.append("- note: facts only, no speculative forecast.")
    return "\n".join(lines)

def run_optimized_pipeline(payload: dict):
    run_id = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()
    base_dir = Path("/app/quant_trading/Project_optimized")
    reports_dir = base_dir / "reports"
    if reports_dir.exists():
        for f in reports_dir.glob("*"):
            try: f.unlink()
            except: pass
            
    # --- 透传 orchestrator payload 中的模型参数，run_pipeline.py 会以 os.environ 为优先级覆盖 config ---
    db_path = base_dir / "japan_market.db"
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = str(base_dir) + ":" + my_env.get("PYTHONPATH", "")
    if db_path.exists():
        my_env["SS6_NEWS_DB"] = str(db_path)
    # 允许 orchestrator 通过 payload 动态指定 signal_mode 和 compare_signal_modes
    if payload.get("signal_mode"):
        my_env["SS6_SIGNAL_MODE"] = str(payload["signal_mode"])
    if payload.get("compare_signal_modes"):
        v = payload["compare_signal_modes"]
        my_env["SS6_COMPARE_SIGNAL_MODES"] = ",".join(v) if isinstance(v, list) else str(v)
    if payload.get("db_path"):
        my_env["SS6_DB_PATH"] = str(payload["db_path"])

    res = subprocess.run(["python", "run_pipeline.py", "--config", "config.yaml"], cwd=str(base_dir), capture_output=True, text=True, timeout=600, env=my_env)
    
    artifacts = []
    if reports_dir.exists():
        for f in reports_dir.glob("*"):
            if f.is_file(): artifacts.append(archive_file(f))

    stdout_tail = (res.stdout or "")[-1200:]
    stderr_tail = (res.stderr or "")[-800:]
    status_text = "SUCCESS" if res.returncode == 0 else "FAILED"
    summary = (
        f"Daily quant pipeline {date_str}: {status_text}. "
        f"Artifacts={len(artifacts)}. "
        f"stdout_tail={stdout_tail[:240].replace(chr(10), ' ')}"
    )
    if res.returncode == 0 and stdout_tail:
        llm_summary = get_llm_response(
            "You are a quantitative research assistant. Summarize execution result in Chinese within 5 bullet points.",
            stdout_tail,
            model=QUANT_LLM_MODEL
        )
        if llm_summary:
            summary = llm_summary.strip()

    if run_id:
        record_fact(
            run_id,
            "quant",
            "daily_pipeline",
            {
                "date": date_str,
                "ok": res.returncode == 0,
                "artifacts": [a.get("name") for a in artifacts],
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "summary": summary,
            },
        )

    return {
        "ok": res.returncode == 0,
        "date": date_str,
        "artifacts": artifacts,
        "stdout": stdout_tail,
        "stderr": stderr_tail,
        "analysis": summary,
    }


def generate_report_card(payload: dict):
    # 此处省略复杂的绘图代码，保持之前的逻辑
    img = Image.new('RGB', (800, 400), color=(10, 10, 20))
    d = ImageDraw.Draw(img)
    d.text((50, 50), f"NEXUS QUANT REPORT - {datetime.now().strftime('%Y-%m-%d')}", fill=(0, 255, 255))
    d.text((50, 150), "Status: STRATEGY_OPTIMIZED", fill=(255, 255, 255))
    out_path = Path("/tmp/report_card.png")
    img.save(out_path)
    meta = archive_file(out_path)
    return {"ok": True, "artifacts": [meta]} # Changed from card_asset to artifacts list

def _openclaw_run(op: str, args: dict):
    base = os.getenv("OPENCLAW_BASE_URL")
    if not base:
        return {"ok": False, "error": "OPENCLAW_BASE_URL not set"}
    url = base.rstrip("/") + "/run"
    try:
        resp = requests.post(url, json={"op": op, "args": args}, timeout=60)
        try:
            return resp.json()
        except Exception:
            return {"ok": False, "error": "invalid_json", "status": resp.status_code, "text": resp.text[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _is_probably_blank_image(bucket: str, object_key: str) -> bool:
    """
    Heuristic blank-image detector for archived screenshots.
    Returns True only when image is very likely a near-solid frame.
    """
    try:
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=object_key)
        data = obj["Body"].read()
        if not data:
            return True
        img = Image.open(io.BytesIO(data)).convert("L")
        # Downsample for fast stats while keeping global luminance structure.
        img = img.resize((256, 256))
        stat = ImageStat.Stat(img)
        stddev = float(stat.stddev[0]) if stat.stddev else 0.0
        hist = img.histogram()
        total = float(sum(hist) or 1.0)
        very_bright = float(sum(hist[245:])) / total
        very_dark = float(sum(hist[:10])) / total
        # Very low variance + dominated by bright/dark pixels => likely blank.
        if stddev < 2.2 and (very_bright > 0.985 or very_dark > 0.985):
            return True
        return False
    except Exception:
        # Fail open: keep artifact when detector cannot verify.
        return False

def _openclaw_capture_urls(
    urls: list[str],
    full_page: bool = True,
    max_shots: int = 3,
    stats_out: Optional[dict] = None,
):
    artifacts = []
    stats = {
        "requested": len(urls or []),
        "attempted": 0,
        "archived_ok": 0,
        "blank_filtered": 0,
        "kept": 0,
        "start_ok": None,
        "skipped_reason": "",
    }
    if isinstance(stats_out, dict):
        stats_out.clear()
        stats_out.update(stats)
    if not urls:
        return artifacts
    urls = urls[:max_shots]
    start = _openclaw_run("browser.start", {})
    if not start.get("ok", False):
        stats["start_ok"] = False
        stats["skipped_reason"] = str(start.get("error") or "browser_start_failed")
        if isinstance(stats_out, dict):
            stats_out.clear()
            stats_out.update(stats)
        return artifacts
    stats["start_ok"] = True
    for u in urls:
        stats["attempted"] += 1
        _openclaw_run("browser.open", {"url": u})
        shot = _openclaw_run("browser.screenshot", {"fullPage": full_page})
        path = None
        if isinstance(shot, dict):
            parsed = shot.get("parsed") or {}
            if isinstance(parsed, dict):
                path = parsed.get("path")
        if path:
            archived = _openclaw_run("artifact.archive", {"path": path})
            if archived.get("ok"):
                stats["archived_ok"] += 1
                bucket = archived.get("bucket", "nexus-evidence")
                object_key = archived.get("object_key")
                if object_key and _is_probably_blank_image(bucket, object_key):
                    stats["blank_filtered"] += 1
                    continue
                artifacts.append({
                    "name": f"browser_screenshot_{len(artifacts)+1}.png",
                    "bucket": bucket,
                    "object_key": object_key,
                    "sha256": archived.get("sha256"),
                    "size": 0,
                    "mime": "image/png",
                })
    stats["kept"] = len(artifacts)
    if isinstance(stats_out, dict):
        stats_out.clear()
        stats_out.update(stats)
    return artifacts

def _make_report_card(title: str, subtitle: str, stats: list[str], out_path: Path):
    img = Image.new("RGB", (1200, 675), color=(14, 20, 30))
    d = ImageDraw.Draw(img)
    d.text((60, 50), title, fill=(0, 255, 255))
    d.text((60, 110), subtitle, fill=(200, 200, 200))
    y = 200
    for line in stats:
        d.text((60, y), f"- {line}", fill=(240, 240, 240))
        y += 40
    img.save(out_path)
    return out_path

def news_daily_report(payload: dict):
    run_id = payload.get("run_id")
    lookback_hours = int(payload.get("lookback_hours", 24))
    max_items = int(payload.get("max_items", 40))
    date_str = payload.get("date") or _now_date_str()

    wl = _load_watchlists()
    jp_syms = wl.get("jp_symbols", [])
    us_syms = wl.get("us_symbols", [])
    name_map = {}
    name_map.update(wl.get("jp_names", {}))
    name_map.update(wl.get("us_names", {}))

    all_symbols = list(dict.fromkeys(jp_syms + us_syms))
    company_terms = [name_map.get(s, "") for s in all_symbols if name_map.get(s)]
    symbol_terms = [s for s in all_symbols]

    finance_terms = ["stock", "shares", "earnings", "guidance", "profit", "forecast", "dividend", "buyback"]
    query_parts = []
    if company_terms or symbol_terms:
        company_query = " OR ".join([f'"{t}"' for t in (company_terms + symbol_terms) if t])
        if company_query:
            query_parts.append(f"({company_query})")
    if finance_terms:
        query_parts.append("(" + " OR ".join(finance_terms) + ")")

    query = " AND ".join(query_parts) if query_parts else "stocks"
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    articles = _gdelt_doc_search(query, start_dt, end_dt, max_records=max_items * 3)
    if not articles:
        articles = _gdelt_doc_search("stocks", start_dt, end_dt, max_records=max_items * 2)
    if not articles:
        # GDELT can lag by 1-2 days; widen window if 24h is empty.
        start_dt = end_dt - timedelta(days=3)
        articles = _gdelt_doc_search("stocks", start_dt, end_dt, max_records=max_items * 2)

    items = []
    for art in articles:
        if not isinstance(art, dict):
            continue
        title = str(art.get("title") or "").strip()
        url = str(art.get("url") or "").strip()
        if not title or not url:
            continue
        source = str(art.get("source") or art.get("domain") or "").strip()
        seendate = str(art.get("seendate") or art.get("pubdate") or "").strip()
        lang = str(art.get("language") or "").strip() or _detect_lang(title)
        items.append({
            "title": title,
            "url": url,
            "source": source,
            "seendate": seendate,
            "lang": lang,
        })
        if len(items) >= max_items:
            break

    # Fallback to Google News if GDELT totally fails
    if not items:
        fallback_news = _fetch_news_from_google_rss("market", "finance", max_items=10)
        for n in fallback_news:
            items.append({
                "title": n.get("title", ""),
                "url": n.get("url", ""),
                "source": n.get("publisher", "Google News"),
                "seendate": n.get("published_at", ""),
                "lang": "en"
            })

    # Add short snippets for the top items.
    for it in items[:10]:
        it["snippet"] = _safe_snippet_from_url(it["url"])

    # Tag impacted tickers by simple string match.
    for it in items:
        text = f"{it.get('title','')} {it.get('snippet','')}".lower()
        matched = []
        for sym in all_symbols:
            if sym.lower() in text:
                matched.append(sym)
                continue
            name = name_map.get(sym, "")
            if name and name.lower() in text:
                matched.append(sym)
        it["tickers"] = list(dict.fromkeys(matched))

    # Aggregate stats.
    ticker_counts = {}
    lang_counts = {}
    source_counts = {}
    for it in items:
        for sym in it.get("tickers", []):
            ticker_counts[sym] = ticker_counts.get(sym, 0) + 1
        lang_counts[it.get("lang", "unknown")] = lang_counts.get(it.get("lang", "unknown"), 0) + 1
        src = it.get("source") or "unknown"
        source_counts[src] = source_counts.get(src, 0) + 1

    top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Calculate Freshness Telemetry
    import statistics
    from datetime import timezone
    
    lag_seconds_list = []
    now_utc = datetime.now(timezone.utc)
    
    for it in items:
        # Try to parse seendate
        try:
            # GDELT typical format: YYYYMMDDTHHMMSSZ
            pub_str = it.get("seendate", "")
            if pub_str:
                # Basic parsing attempt
                if len(pub_str) == 15 and "T" in pub_str: # 20260226T120000
                    pub_dt = datetime.strptime(pub_str[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                else:
                    # Generic fallback
                    from dateutil import parser
                    pub_dt = parser.parse(pub_str)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        
                lag = (now_utc - pub_dt).total_seconds()
                lag_seconds_list.append(max(0, lag))
        except Exception:
            pass

    lag_p50 = statistics.median(lag_seconds_list) if lag_seconds_list else 0
    lag_max = max(lag_seconds_list) if lag_seconds_list else 0
    
    coverage_flag = "degraded" if len(items) < 5 else "fresh"
    
    freshness_stats = f"Freshness: P50={int(lag_p50//60)}m, Max={int(lag_max//3600)}h | Items: {len(items)} | Status: {coverage_flag}"

    summary_lines = [freshness_stats]
    use_llm = str(os.getenv("NEWS_USE_LLM", "1")).lower() in ["1", "true", "yes"]
    if use_llm and items:
        titles = "\n".join([f"- {it['title']}" for it in items[:20]])
        system = "You summarize market news in Chinese. Provide 5 concise bullet points."
        out = get_llm_response(system, titles, model=QUANT_LLM_MODEL)
        if out:
            summary_lines = [l.strip("- ").strip() for l in out.splitlines() if l.strip()]

    charts = {}
    chart_dir = Path("/tmp")
    if top_tickers:
        p = chart_dir / f"news_tickers_{date_str}.png"
        if _make_bar_chart(top_tickers, "Mentions by Ticker", p):
            charts["tickers"] = _img_to_data_uri(p)
    if top_sources:
        p = chart_dir / f"news_sources_{date_str}.png"
        if _make_bar_chart(top_sources, "Mentions by Source", p):
            charts["sources"] = _img_to_data_uri(p)

    card_path = chart_dir / f"news_card_{date_str}.png"
    _make_report_card(
        title=f"Daily Market News Report {date_str}",
        subtitle=f"Items: {len(items)} | Lookback: {lookback_hours}h",
        stats=[
            f"JP watchlist: {len(jp_syms)}",
            f"US watchlist: {len(us_syms)}",
            f"Top tickers: {', '.join([t for t, _ in top_tickers[:5]]) or 'n/a'}",
        ],
        out_path=card_path,
    )

    template = Template("""
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Daily Market News Report {{ date }}</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #e2e8f0;
      --brand: #0b6f5f;
      --chip: #e8f4ef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", Arial, sans-serif;
      background: linear-gradient(160deg, #f2f8f7 0%, var(--bg) 45%, #eef2ff 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1180px;
      margin: 28px auto;
      padding: 0 14px 28px;
    }
    .hero {
      background: radial-gradient(1200px 280px at 10% -20%, #32c3a6 0%, #0b6f5f 55%, #134e4a 100%);
      color: #f8fafc;
      border-radius: 16px;
      padding: 20px 22px;
      box-shadow: 0 14px 34px rgba(11, 111, 95, 0.24);
    }
    .hero h1 {
      margin: 0 0 8px;
      letter-spacing: .2px;
      font-size: 27px;
      line-height: 1.2;
    }
    .hero .meta {
      color: #dff7f0;
      font-size: 14px;
      opacity: .95;
    }
    .chips {
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,.18);
      border: 1px solid rgba(255,255,255,.25);
      font-size: 12px;
      color: #f8fafc;
    }
    .grid {
      margin-top: 16px;
      display: grid;
      grid-template-columns: 1.35fr .9fr;
      gap: 14px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 5px 14px rgba(15, 23, 42, .05);
    }
    .card h3 {
      margin: 0 0 10px;
      font-size: 16px;
      color: #0f172a;
    }
    .summary-list {
      margin: 0;
      padding-left: 18px;
    }
    .summary-list li {
      margin: 6px 0;
      line-height: 1.45;
    }
    .highlight {
      border-top: 1px dashed var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }
    .item {
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid #f1f5f9;
    }
    .item:last-child {
      border-bottom: 0;
      margin-bottom: 0;
      padding-bottom: 0;
    }
    .item-title {
      font-size: 15px;
      font-weight: 600;
      line-height: 1.4;
      margin-bottom: 6px;
    }
    .item-title a {
      color: #0f172a;
      text-decoration: none;
    }
    .item-title a:hover {
      color: #0b6f5f;
      text-decoration: underline;
    }
    .small {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .tag {
      display: inline-block;
      padding: 2px 8px;
      margin-right: 6px;
      margin-top: 6px;
      border-radius: 999px;
      background: var(--chip);
      color: #116149;
      border: 1px solid #cbe9de;
      font-size: 11px;
      font-weight: 600;
    }
    .charts img {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      margin-bottom: 10px;
      background: #fff;
    }
    .all-items {
      margin-top: 14px;
    }
    .all-items ol {
      margin: 0;
      padding-left: 20px;
    }
    .all-items li {
      margin-bottom: 10px;
      line-height: 1.45;
    }
    @media (max-width: 960px) {
      .grid { grid-template-columns: 1fr; }
      .hero h1 { font-size: 23px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>NEXUS Daily Market News</h1>
      <div class="meta">Date: {{ date }} | Items: {{ items|length }} | Lookback: {{ lookback_hours }}h</div>
      <div class="chips">
        <span class="chip">Auto-Collected</span>
        <span class="chip">Freshness-Aware</span>
        <span class="chip">Ticker-Tagged</span>
      </div>
    </section>

    <section class="grid">
      <article class="card">
        <h3>Summary</h3>
        {% if summary_lines %}
          <ul class="summary-list">
            {% for s in summary_lines %}
              <li>{{ s }}</li>
            {% endfor %}
          </ul>
        {% else %}
          <div class="small">No summary generated.</div>
        {% endif %}

        <div class="highlight">
          <h3>Top Highlights</h3>
          {% for it in items[:10] %}
          <div class="item">
            <div class="item-title"><a href="{{ it.url }}" target="_blank" rel="noreferrer">{{ it.title }}</a></div>
            <div class="small">{{ it.source }} | {{ it.seendate }} | {{ it.lang }}</div>
            {% if it.tickers %}
              <div>
                {% for t in it.tickers %}
                  <span class="tag">{{ t }}</span>
                {% endfor %}
              </div>
            {% endif %}
            {% if it.snippet %}
              <div class="small" style="margin-top:6px;">{{ it.snippet }}</div>
            {% endif %}
          </div>
          {% endfor %}
        </div>
      </article>

      <aside class="card charts">
        <h3>Visual Snapshot</h3>
        {% if charts.tickers %}<img src="{{ charts.tickers }}" alt="Ticker Mentions" />{% endif %}
        {% if charts.sources %}<img src="{{ charts.sources }}" alt="Source Mentions" />{% endif %}
        {% if not charts.tickers and not charts.sources %}
          <div class="small">No chart data available.</div>
        {% endif %}
      </aside>
    </section>

    <section class="card all-items">
      <h3>All Collected Items</h3>
      <ol>
        {% for it in items %}
          <li>
            <div class="item-title"><a href="{{ it.url }}" target="_blank" rel="noreferrer">{{ it.title }}</a></div>
            <div class="small">{{ it.source }} | {{ it.seendate }} | {{ it.lang }}</div>
          </li>
        {% endfor %}
      </ol>
    </section>
  </div>
</body>
</html>
""")

    html = template.render(date=date_str, items=items, lookback_hours=lookback_hours, charts=charts, summary_lines=summary_lines)
    html_path = Path(f"/tmp/news_report_{date_str}.html")
    html_path.write_text(html, encoding="utf-8")

    artifacts = [archive_file(html_path), archive_file(card_path)]
    browser_artifacts = _openclaw_capture_urls([it["url"] for it in items[:3]])
    artifacts.extend([a for a in browser_artifacts if a.get("object_key")])

    top_ticker_text = ", ".join([f"{t}:{c}" for t, c in top_tickers[:5]]) if top_tickers else "n/a"
    summary_text = "\n".join(summary_lines[:5]).strip()
    if not summary_text:
        summary_text = f"News items: {len(items)}. Top tickers: {top_ticker_text}."

    if run_id:
        record_fact(
            run_id,
            "news",
            "daily_report",
            {
                "date": date_str,
                "count": len(items),
                "top_tickers": top_tickers,
                "top_sources": top_sources[:5],
                "summary": summary_text,
            },
        )

    return {
        "ok": True,
        "date": date_str,
        "count": len(items),
        "artifacts": artifacts,
        "analysis": summary_text,
    }

def _get_current_positions_from_fills(max_symbols: int = 20) -> list[dict]:
    db_path = QUANT_DB_PATH
    if not db_path.exists():
        return []
    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                UPPER(symbol) AS symbol,
                SUM(CASE WHEN UPPER(side)='BUY' THEN qty ELSE -qty END) AS net_qty
            FROM fills
            GROUP BY UPPER(symbol)
            HAVING net_qty > 0
            ORDER BY net_qty DESC
            LIMIT ?
            """,
            (max_symbols,),
        )
        out = []
        for row in cur.fetchall() or []:
            sym = _normalize_symbol_for_lookup(row[0])
            qty = _safe_float(row[1]) or 0.0
            if sym and qty > 0:
                out.append({
                    "symbol": sym,
                    "net_qty": qty,
                    "market": _infer_market_from_symbol(sym),
                })
        return out
    except Exception:
        return []
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def _get_account_state_snapshot() -> dict:
    db_path = QUANT_DB_PATH
    if not db_path.exists():
        return {}
    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT starting_capital, base_ccy
            FROM account_state
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            return {}
        return {
            "starting_capital": _safe_float(row[0]),
            "base_ccy": str(row[1] or "JPY").upper().strip(),
        }
    except Exception:
        return {}
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def _score_hot_article(item: dict, lookback_hours: int, trusted_publishers: list[str], hot_keywords: list[str]) -> float:
    title = str(item.get("title") or "").lower()
    source = str(item.get("source") or item.get("publisher") or "").lower()
    dt = _parse_article_dt(item)
    age_h = lookback_hours
    if dt is not None:
        age_h = max(0.0, (datetime.utcnow() - dt).total_seconds() / 3600.0)
    recency = max(0.0, 1.0 - (age_h / max(1.0, float(lookback_hours))))
    src_score = 1.0 if any(p in source for p in trusted_publishers) else 0.4
    kw_hits = sum(1 for k in hot_keywords if k in title)
    kw_score = min(1.0, kw_hits / 4.0)
    return round(100.0 * (0.55 * recency + 0.25 * src_score + 0.20 * kw_score), 2)

def news_active_hot_search(payload: dict):
    run_id = payload.get("run_id")
    lookback_hours = int(payload.get("lookback_hours", 24))
    top_n = int(payload.get("top_n", 8))
    top_n = max(5, min(top_n, 10))
    include_positions = _to_bool(payload.get("include_positions"), default=True)
    date_str = payload.get("date") or _now_date_str()

    trusted_publishers = [
        "reuters", "bloomberg", "nikkei", "ft", "financial times", "wsj",
        "marketwatch", "yahoo", "cnbc", "investing.com"
    ]
    hot_keywords = [
        "earnings", "guidance", "fed", "fomc", "inflation", "cpi", "rate",
        "yield", "ai", "semiconductor", "chip", "layoff", "buyback",
        "并购", "财报", "加息", "降息", "日银", "boj", "业绩"
    ]

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    seed_queries = [
        "global stock market OR macro OR inflation OR central bank OR rate decision OR guidance",
        "Middle East OR Ukraine OR conflict OR oil OR crude OR sanctions AND stock market",
        "AI OR semiconductor OR chip OR Nvidia OR TSMC OR Microsoft OR OpenAI",
        "Nikkei OR TOPIX OR Japan stocks OR BOJ OR yen",
    ]

    raw_items = []
    for q in seed_queries:
        arts = _gdelt_doc_search(q, start_dt, end_dt, max_records=80)
        for art in arts:
            if not isinstance(art, dict):
                continue
            title = str(art.get("title") or "").strip()
            url = str(art.get("url") or "").strip()
            if not title or not url:
                continue
            raw_items.append({
                "title": title,
                "url": url,
                "source": str(art.get("source") or art.get("domain") or "gdelt").strip(),
                "seendate": str(art.get("seendate") or art.get("pubdate") or "").strip(),
                "publisher": str(art.get("source") or art.get("domain") or "gdelt").strip(),
            })

    # Broaden sources to avoid channel lock-in
    for n in _fetch_news_from_google_rss("global market", "finance economy geopolitics", max_items=40, lang="en"):
        raw_items.append({
            "title": str(n.get("title") or "").strip(),
            "url": str(n.get("url") or "").strip(),
            "source": str(n.get("publisher") or "google_news").strip(),
            "published_at": str(n.get("published_at") or "").strip(),
            "published_ts": n.get("published_ts"),
            "publisher": str(n.get("publisher") or "").strip(),
        })
    for n in _fetch_news_from_google_rss("world economy", "energy oil rates geopolitics", max_items=25, lang="zh"):
        raw_items.append({
            "title": str(n.get("title") or "").strip(),
            "url": str(n.get("url") or "").strip(),
            "source": str(n.get("publisher") or "google_news_zh").strip(),
            "published_at": str(n.get("published_at") or "").strip(),
            "published_ts": n.get("published_ts"),
            "publisher": str(n.get("publisher") or "").strip(),
        })
    for n in _fetch_news_from_google_rss("日本株", "ニュース", max_items=20, lang="ja"):
        raw_items.append({
            "title": str(n.get("title") or "").strip(),
            "url": str(n.get("url") or "").strip(),
            "source": str(n.get("publisher") or "google_news_ja").strip(),
            "published_at": str(n.get("published_at") or "").strip(),
            "published_ts": n.get("published_ts"),
            "publisher": str(n.get("publisher") or "").strip(),
        })

    dedup = []
    seen = set()
    for it in raw_items:
        t = str(it.get("title") or "").strip()
        u = str(it.get("url") or "").strip()
        if not t or not u:
            continue
        key = (t.lower(), u.lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)

    fresh_items, fresh_stats = _apply_freshness_policy(dedup, max_age_hours=float(lookback_hours))
    scored = []
    for it in fresh_items:
        hotness = _score_hot_article(it, lookback_hours, trusted_publishers, hot_keywords)
        row = dict(it)
        row["hotness"] = hotness
        scored.append(row)
    scored.sort(key=lambda x: (x.get("hotness", 0), str(x.get("seendate") or x.get("published_at") or "")), reverse=True)
    hot_news = scored[:top_n]

    # Position-linked news
    position_news = []
    positions = _get_current_positions_from_fills(max_symbols=12) if include_positions else []
    for p in positions:
        sym = p.get("symbol")
        if not sym:
            continue
        quote = _fetch_quote_facts(sym)
        company_name = str(quote.get("company_name") or "").strip()
        merged = _merge_recent_news(quote, max_items=6)
        # Extra global fetch for holdings: improves recall beyond fixed channels.
        pos_query = f"{sym} OR {company_name or sym} stock OR earnings OR guidance OR outlook"
        for art in _gdelt_doc_search(pos_query, start_dt, end_dt, max_records=30):
            if not isinstance(art, dict):
                continue
            t = str(art.get("title") or "").strip()
            u = str(art.get("url") or "").strip()
            if not t or not u:
                continue
            merged.append({
                "title": t,
                "url": u,
                "publisher": str(art.get("source") or art.get("domain") or "gdelt").strip(),
                "published_at": str(art.get("seendate") or art.get("pubdate") or "").strip(),
                "source": "gdelt_holding",
            })
        merged.extend(_fetch_news_from_google_rss(sym, company_name, max_items=6, lang="en"))
        merged.extend(_fetch_news_from_google_rss(sym, company_name, max_items=4, lang="ja"))
        merged_fresh, _ = _apply_freshness_policy(merged, max_age_hours=float(lookback_hours))
        if not merged_fresh:
            # Holdings-first fallback: widen freshness window when 24h has no usable articles.
            merged_fresh, _ = _apply_freshness_policy(merged, max_age_hours=float(max(72, lookback_hours)))
        for n in merged_fresh[:4]:
            title = str(n.get("title") or "").strip()
            url = str(n.get("url") or "").strip()
            if not title or not url:
                continue
            row = {
                "symbol": sym,
                "net_qty": p.get("net_qty"),
                "title": title,
                "url": url,
                "publisher": str(n.get("publisher") or n.get("source") or "").strip(),
                "published_at": str(n.get("published_at") or ""),
                "published_ts": n.get("published_ts"),
            }
            row["hotness"] = _score_hot_article(row, lookback_hours, trusted_publishers, hot_keywords)
            position_news.append(row)

    # Dedup position news
    pos_seen = set()
    pos_out = []
    for n in sorted(position_news, key=lambda x: x.get("hotness", 0), reverse=True):
        key = (str(n.get("symbol") or ""), str(n.get("title") or "").lower(), str(n.get("url") or "").lower())
        if key in pos_seen:
            continue
        pos_seen.add(key)
        pos_out.append(n)
        if len(pos_out) >= top_n:
            break

    all_titles_blob = " ".join([str(x.get("title") or "").lower() for x in hot_news + pos_out])
    risk_kw = ["conflict", "war", "missile", "attack", "sanction", "strike", "中东", "冲突", "袭击", "制裁", "战争"]
    oil_kw = ["oil", "crude", "brent", "wti", "gas", "lng", "原油", "油价", "天然气"]
    safe_kw = ["defensive", "utilities", "telecom", "dividend", "防御", "公用事业", "通信", "高股息"]
    growth_kw = ["ai", "semiconductor", "chip", "cloud", "gpu", "人工智能", "半导体", "芯片"]

    risk_hits = sum(1 for k in risk_kw if k in all_titles_blob)
    oil_hits = sum(1 for k in oil_kw if k in all_titles_blob)
    safe_hits = sum(1 for k in safe_kw if k in all_titles_blob)
    growth_hits = sum(1 for k in growth_kw if k in all_titles_blob)
    avg_hot = round(sum(float(x.get("hotness") or 0.0) for x in hot_news) / max(1, len(hot_news)), 2)

    if risk_hits >= 3 or (risk_hits >= 2 and oil_hits >= 1):
        risk_level = "high"
        market_view = "偏风险厌恶，指数层面以防守为主。"
    elif risk_hits >= 1 or oil_hits >= 1:
        risk_level = "medium"
        market_view = "情绪中性偏谨慎，盘中可能出现快速切换。"
    else:
        risk_level = "low"
        market_view = "外部冲击有限，可维持结构性轮动。"

    preferred = ["高股息", "公用事业", "通信", "军工/能源链"] if risk_level != "low" else ["科技成长", "景气改善链"]
    avoid = ["高估值高波动小盘题材"] if risk_level != "low" else ["纯防御过度拥挤标的"]
    watch_signals = [
        "WTI/Brent 夜盘方向与波动率",
        "美元/日元（USDJPY）与美债收益率",
        "日经期货开盘缺口与前30分钟成交量"
    ]
    action_plan = [
        "开盘前先看风险指标，再决定是否降仓位。",
        "若油价和避险资产同步走强，优先防御仓位。",
        "避免追高，采用分批与回撤确认后再加仓。"
    ]

    next_day_advice = {
        "risk_level": risk_level,
        "market_view": market_view,
        "preferred_themes": preferred,
        "avoid_themes": avoid,
        "action_plan": action_plan,
        "watch_signals": watch_signals,
        "confidence_hint": {
            "avg_hotness": avg_hot,
            "risk_hits": risk_hits,
            "oil_hits": oil_hits,
            "safe_hits": safe_hits,
            "growth_hits": growth_hits,
        },
    }

    analysis_lines = [
        f"主动热点扫描完成：窗口={lookback_hours}h，热点候选={len(fresh_items)}，输出Top{len(hot_news)}。",
        f"持仓关联新闻：持仓标的={len(positions)}，输出Top{len(pos_out)}。",
        f"次日风险级别：{risk_level.upper()} | 平均热度：{avg_hot}",
        f"市场判断：{market_view}",
        "建议动作：",
        f"1) {action_plan[0]}",
        f"2) {action_plan[1]}",
        f"3) {action_plan[2]}",
        f"优先关注：{', '.join(preferred)}",
        f"回避方向：{', '.join(avoid)}",
    ]
    if pos_out:
        analysis_lines.append("持仓相关（优先关注）:")
        for idx, it in enumerate(pos_out[:5], start=1):
            analysis_lines.append(
                f"P{idx}. [{it.get('symbol')}|热度{it.get('hotness')}] {it.get('title')} ({it.get('publisher')})"
            )
    for idx, it in enumerate(hot_news[:5], start=1):
        analysis_lines.append(f"{idx}. [热度{it.get('hotness')}] {it.get('title')} ({it.get('source')})")

    artifacts = []
    capture_stats = {}
    try:
        capture_urls = [it.get("url") for it in hot_news[:2] if str(it.get("url") or "").startswith("http")]
        if capture_urls:
            artifacts.extend(
                _openclaw_capture_urls(
                    capture_urls,
                    full_page=False,
                    max_shots=2,
                    stats_out=capture_stats,
                )
            )
    except Exception:
        pass

    result = {
        "ok": True,
        "type": "active_hot_news",
        "date": date_str,
        "lookback_hours": lookback_hours,
        "freshness": fresh_stats,
        "hot_news": hot_news,
        "position_news": pos_out,
        "positions": positions,
        "artifacts": artifacts,
        "artifact_capture_stats": capture_stats,
        "next_day_advice": next_day_advice,
        "analysis": "\n".join(analysis_lines),
        "summary": f"24h热点Top{len(hot_news)} + 持仓相关Top{len(pos_out)}",
    }

    if run_id:
        record_fact(
            run_id,
            "news",
            "active_hot_search",
            {
                "lookback_hours": lookback_hours,
                "hot_count": len(hot_news),
                "position_count": len(pos_out),
                "positions": positions,
                "artifact_capture_stats": capture_stats,
            },
        )
    return result

def github_skills_daily_report(payload: dict):
    date_str = payload.get("date") or _now_date_str()
    max_items = int(payload.get("max_items", 10))

    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    queries = [
        "ai agent skills",
        "agent skill",
        "autonomous agent",
        "ai agent toolkit",
    ]

    repos = {}
    for q in queries:
        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                headers=headers,
                params={"q": q, "sort": "updated", "order": "desc", "per_page": 20},
                timeout=20,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for item in data.get("items", []):
                full = item.get("full_name")
                if not full:
                    continue
                repos[full] = item
        except Exception:
            continue

    ranked = sorted(
        repos.values(),
        key=lambda r: (r.get("stargazers_count", 0), r.get("updated_at", "")),
        reverse=True,
    )
    ranked = ranked[:max_items]

    items = []
    for r in ranked:
        items.append({
            "name": r.get("name"),
            "full_name": r.get("full_name"),
            "url": r.get("html_url"),
            "desc": r.get("description") or "",
            "stars": r.get("stargazers_count", 0),
            "updated": r.get("updated_at", ""),
        })

    # Optional LLM summary for functions/uses.
    use_llm = str(os.getenv("GITHUB_USE_LLM", "1")).lower() in ["1", "true", "yes"]
    if use_llm and items:
        prompt = "\n".join([f"- {it['full_name']}: {it['desc']}" for it in items])
        system = "You summarize GitHub agent skill projects in Chinese. Provide a short line per repo with functions and possible uses."
        out = get_llm_response(system, prompt, model=CODE_LLM_MODEL)
        if out:
            lines = [l.strip("- ").strip() for l in out.splitlines() if l.strip()]
            for it, line in zip(items, lines):
                it["summary"] = line

    star_counts = [(it["name"], it["stars"]) for it in items if it.get("name")]
    chart_dir = Path("/tmp")
    charts = {}
    if star_counts:
        p = chart_dir / f"github_stars_{date_str}.png"
        if _make_bar_chart(star_counts, "Stars by Repo", p):
            charts["stars"] = _img_to_data_uri(p)

    card_path = chart_dir / f"github_card_{date_str}.png"
    _make_report_card(
        title=f"Daily GitHub Agent Skills {date_str}",
        subtitle=f"Repos: {len(items)}",
        stats=[
            f"Top repo: {(items[0]['full_name'] if items else 'n/a')}",
            f"Token: {'yes' if token else 'no'}",
        ],
        out_path=card_path,
    )

    template = Template("""
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Daily GitHub Agent Skills {{ date }}</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #111; }
    h1 { margin-bottom: 6px; }
    .meta { color: #555; margin-bottom: 18px; }
    .card { border: 1px solid #ddd; padding: 12px; border-radius: 8px; margin-bottom: 16px; }
    .item { margin-bottom: 12px; }
    .item-title { font-weight: bold; }
    .small { color: #666; font-size: 12px; }
    img { max-width: 100%; }
  </style>
</head>
<body>
  <h1>Daily GitHub Agent Skills</h1>
  <div class="meta">Date: {{ date }} | Repos: {{ items|length }}</div>
  <div class="card">
    <h3>Charts</h3>
    {% if charts.stars %}<img src="{{ charts.stars }}" />{% endif %}
  </div>
  <div class="card">
    <h3>Projects</h3>
    <ol>
      {% for it in items %}
        <li class="item">
          <div class="item-title"><a href="{{ it.url }}">{{ it.full_name }}</a></div>
          <div class="small">Stars: {{ it.stars }} | Updated: {{ it.updated }}</div>
          <div>{{ it.desc }}</div>
          {% if it.summary %}<div class="small">Summary: {{ it.summary }}</div>{% endif %}
        </li>
      {% endfor %}
    </ol>
  </div>
</body>
</html>
""")

    html = template.render(date=date_str, items=items, charts=charts)
    html_path = Path(f"/tmp/github_skills_{date_str}.html")
    html_path.write_text(html, encoding="utf-8")

    artifacts = [archive_file(html_path), archive_file(card_path)]
    browser_artifacts = _openclaw_capture_urls([it["url"] for it in items[:3]])
    artifacts.extend([a for a in browser_artifacts if a.get("object_key")])

    return {"ok": True, "date": date_str, "count": len(items), "artifacts": artifacts}

def _db_connect():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "db"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "nexus"),
        password=os.getenv("PGPASSWORD", "nexus"),
        database=os.getenv("PGDATABASE", "nexus"),
    )

def ensure_run_exists(run_id: str):
    if not run_id:
        return
    conn = None
    cur = None
    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (run_id, client_msg_id, user_id, status, input_text)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO NOTHING
            """,
            (run_id, f"worker-{run_id}", "worker-quant", "running", "worker-fallback-init"),
        )
        conn.commit()
    except Exception as e:
        print(f"[worker] ensure_run_exists failed: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def record_fact(run_id, agent, kind, data):
    if not run_id:
        return None
    ensure_run_exists(run_id)
    conn = None
    cur = None
    fact_id = uuid.uuid4().hex
    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO fact_items (fact_id, run_id, agent_name, kind, payload_json) VALUES (%s, %s, %s, %s, %s)",
            (fact_id, run_id, agent, kind, json.dumps(data)),
        )
        conn.commit()
        return fact_id
    except Exception as e:
        print(f"[worker] DB Record Error: {e}")
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def record_event(task_id: str, event_type: str, payload: dict | None = None):
    if not task_id:
        return
    conn = None
    cur = None
    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO event_log (task_id, event_type, payload_json) VALUES (%s, %s, %s)",
            (task_id, event_type, json.dumps(payload or {})),
        )
        conn.commit()
    except Exception as e:
        print(f"[worker] event_log error: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def record_execution_audit(run_id: str, task_id: str, worker_name: str, event_type: str, payload: dict | None = None):
    if not run_id:
        return
    global EXECUTION_AUDIT_SCHEMA_READY
    conn = None
    cur = None
    try:
        ensure_run_exists(run_id)
        conn = _db_connect()
        cur = conn.cursor()
        if not EXECUTION_AUDIT_SCHEMA_READY:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_audit_log (
                  id BIGSERIAL PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  task_id TEXT,
                  worker_name TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_audit_log_run_id ON execution_audit_log(run_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_audit_log_task_id ON execution_audit_log(task_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_execution_audit_log_event_type ON execution_audit_log(event_type)")
            EXECUTION_AUDIT_SCHEMA_READY = True
        cur.execute(
            """
            INSERT INTO execution_audit_log (run_id, task_id, worker_name, event_type, payload_json)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, task_id or None, worker_name, event_type, json.dumps(payload or {})),
        )
        conn.commit()
    except Exception as e:
        print(f"[worker] execution_audit_log error: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def record_evidence_link(fact_id: str | None, url: str, screenshot_ref: dict | None, extracted_text: str = ""):
    if not fact_id:
        return
    conn = None
    cur = None
    try:
        conn = _db_connect()
        cur = conn.cursor()
        evidence_id = uuid.uuid4().hex
        content_hash = hashlib.sha256((url + extracted_text + json.dumps(screenshot_ref or {}, sort_keys=True)).encode("utf-8")).hexdigest()
        cur.execute(
            """
            INSERT INTO evidence (evidence_id, url, screenshot_ref_json, extracted_text, content_hash)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (evidence_id, url, json.dumps(screenshot_ref or {}), extracted_text, content_hash),
        )
        cur.execute("INSERT INTO links (fact_id, evidence_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (fact_id, evidence_id))
        conn.commit()
    except Exception as e:
        print(f"[worker] evidence link error: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def deep_analysis(payload: dict):
    symbol = _normalize_symbol_for_lookup(payload.get("symbol", "NVDA"))
    run_id = payload.get("run_id")
    
    # Capital constraints from payload or defaults
    capital_base = float(payload.get("capital_base_jpy") or payload.get("capital_base") or 400000)
    headroom = float(payload.get("capital_headroom_pct") or 0.25)
    max_pos_pct = float(payload.get("max_position_pct") or 0.25)
    
    quote = _fetch_quote_facts(symbol)
    metrics = _compute_quant_metrics(symbol)
    news = _merge_recent_news(quote, max_items=5)
    quote["recent_news"] = news
    analysis = _grounded_quant_summary(quote)

    # E-02: Integrate news sentiment into signal score (small weight to avoid overfitting)
    news_sentiment = 0.0
    try:
        nr = compute_news_risk_factor({"symbol": symbol, "run_id": run_id})
        news_sentiment = float(nr.get("sentiment", 0) or 0)
        if "alpha_score" in metrics and metrics["alpha_score"] is not None:
            metrics["alpha_score"] = round(metrics["alpha_score"] + 0.5 * news_sentiment, 3)
    except Exception:
        pass
    metrics["news_sentiment"] = round(news_sentiment, 3)

    # B-05: Check for TDnet announcements for this symbol
    today_announcements = []
    try:
        today_announcements = _fetch_tdnet_announcements(symbols=[symbol], lookback_hours=24)
        if today_announcements:
            # Catalyst adjustment: up/down one tier based on announcement category
            cats_up   = {"earnings_revision_up", "dividend_increase", "special_dividend", "buyback"}
            cats_down = {"earnings_revision_down", "dividend_cut", "scandal", "regulatory"}
            for ann in today_announcements:
                cat = ann.get("category", "")
                if cat in cats_up and metrics.get("alpha_score") is not None:
                    metrics["alpha_score"] = round(metrics["alpha_score"] + 1.0, 3)
                    metrics["catalyst_boost"] = f"+1.0 ({cat})"
                elif cat in cats_down and metrics.get("alpha_score") is not None:
                    metrics["alpha_score"] = round(metrics["alpha_score"] - 1.5, 3)
                    metrics["catalyst_boost"] = f"-1.5 ({cat})"
    except Exception:
        pass
    metrics["today_announcements"] = today_announcements

    # D-01: Log signal for IC tracking
    _log_signal(symbol, metrics.get("signal", "N/A"), metrics.get("alpha_score"), quote.get("price"))

    # --- Execution Logic (v1.2.7) ---
    raw_signal = (metrics.get("signal") or "").upper()
    side = metrics.get("signal_side") or payload.get("mode")
    if not side:
        if "BUY" in raw_signal or "OVERWEIGHT" in raw_signal:
            side = "BUY"
        elif "SELL" in raw_signal or "UNDERWEIGHT" in raw_signal:
            side = "SELL"
        else:
            side = "HOLD"
    
    side = side.upper()
    price = quote.get("price")
    atr = _calculate_atr(symbol) or (price * 0.02 if price else 0)
    
    exec_sug = {}
    if price and side != "HOLD":
        limit_prices = _calculate_limit_prices(symbol, side, price, atr)
        
        # Sizing logic
        lot_size = 100 if symbol.endswith(".T") else 1
        max_pos_jpy = capital_base * max_pos_pct
        budget_jpy = capital_base * (1 + headroom)
        
        shares = 0
        reason = ""
        if side == "BUY":
            if price * lot_size > max_pos_jpy:
                reason = f"Price {price} * lot {lot_size} exceeds max position {max_pos_jpy}"
            else:
                lots = math.floor(max_pos_jpy / (price * lot_size))
                shares = lots * lot_size
        
        exec_sug = {
            "side": side,
            "urgency": "balanced",
            "limit_prices": limit_prices,
            "sizing": {
                "lot_size": lot_size,
                "shares": shares,
                "notional_jpy": shares * price,
                "reason_if_zero": reason
            },
            "constraints": {
                "capital_base_jpy": capital_base,
                "max_position_pct": max_pos_pct,
                "budget_jpy": budget_jpy
            }
        }

    report_markdown, report_html = _render_quant_template_report(run_id or "manual", quote, metrics, news)
    ts = int(time.time())
    safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)
    md_path = Path(f"/tmp/quant_execution_report_{safe_symbol}_{ts}.md")
    html_path = Path(f"/tmp/quant_execution_report_{safe_symbol}_{ts}.html")
    md_path.write_text(report_markdown, encoding="utf-8")
    html_path.write_text(report_html, encoding="utf-8")

    md_artifact = archive_file(md_path)
    html_artifact = archive_file(html_path)

    record_fact(
        run_id,
        "quant",
        "analysis_report",
        {
            "symbol": symbol,
            "company_name": quote.get("company_name"),
            "price": quote.get("price"),
            "currency": quote.get("currency"),
            "quote_url": quote.get("quote_url"),
            "recent_news": news,
            "metrics": metrics,
            "execution_suggestions": exec_sug,
            "summary": analysis,
            "report_markdown": report_markdown,
            "report_md_object_key": md_artifact.get("object_key"),
            "report_html_object_key": html_artifact.get("object_key"),
            "source": quote.get("source"),
        },
    )

    return {
        "ok": True,
        "symbol": symbol,
        "company_name": quote.get("company_name"),
        "price": price,
        "currency": quote.get("currency"),
        "quote_url": quote.get("quote_url"),
        "recent_news": news,
        "metrics": metrics,
        "execution_suggestions": exec_sug,
        "analysis": analysis,
        "report_markdown": report_markdown,
        "report_md_object_key": md_artifact.get("object_key"),
        "report_html_object_key": html_artifact.get("object_key"),
        "artifacts": [md_artifact, html_artifact],
    }

def browser_screenshot(payload: dict):
    symbol = str(payload.get("symbol", "9432.T")).upper()
    run_id = payload.get("run_id")
    url = f"https://finance.yahoo.co.jp/quote/{symbol}" if ".T" in symbol else f"https://finance.yahoo.com/quote/{symbol}"
    artifacts = _openclaw_capture_urls([url], full_page=False)
    
    if artifacts:
        first = artifacts[0]
        fact_id = record_fact(
            run_id,
            "browser",
            "visual_evidence",
            {"url": url, "object_key": first.get("object_key"), "sha256": first.get("sha256")},
        )
        record_evidence_link(
            fact_id,
            url=url,
            screenshot_ref={"object_key": first.get("object_key"), "sha256": first.get("sha256")},
            extracted_text="",
        )
    else:
        record_fact(
            run_id,
            "browser",
            "visual_evidence",
            {"url": url, "error": "capture_failed", "object_key": None},
        )
    
    return {
        "ok": len(artifacts) > 0,
        "url": url,
        "analysis": f"Browser evidence captured from {url}" if artifacts else f"Browser evidence unavailable for {url}",
        "artifacts": artifacts,
    }

def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default

def _normalize_risk_profile(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"low", "conservative", "defensive", "稳健", "低风险"}:
        return "low"
    if text in {"high", "aggressive", "进取", "高风险"}:
        return "high"
    return "medium"

def _infer_goal_profile(goal: str, risk_profile: str) -> dict:
    g = (goal or "").lower()
    if any(k in g for k in ["稳", "保本", "低波", "低回撤", "drawdown", "defensive"]):
        style = "defensive"
    elif any(k in g for k in ["分红", "股息", "现金流", "income", "dividend"]):
        style = "income"
    elif any(k in g for k in ["成长", "增长", "进取", "翻倍", "growth", "alpha"]):
        style = "growth"
    else:
        style = "balanced"

    if style == "defensive":
        risk = "low"
    elif style == "growth":
        risk = "high" if risk_profile == "medium" else risk_profile
    else:
        risk = risk_profile

    style_map = {
        "growth": {"entry_style": "momentum_2_step", "horizon_days": 90},
        "balanced": {"entry_style": "balanced_3_step", "horizon_days": 120},
        "income": {"entry_style": "pullback_3_step", "horizon_days": 150},
        "defensive": {"entry_style": "defensive_3_step", "horizon_days": 180},
    }
    out = style_map.get(style, style_map["balanced"]).copy()
    out["style"] = style
    out["risk_profile"] = risk
    return out

def _build_position_plan(
    candidates: List[dict],
    capital_base_jpy: float,
    max_pos_pct: float,
    goal: str,
    goal_profile: dict,
    target_return_pct: Optional[float],
    horizon_days: int,
) -> dict:
    risk = goal_profile.get("risk_profile", "medium")
    defaults = {
        "low": {"n": 3, "cash_reserve_pct": 0.20, "entry_steps": [0.35, 0.35, 0.30]},
        "medium": {"n": 4, "cash_reserve_pct": 0.12, "entry_steps": [0.50, 0.30, 0.20]},
        "high": {"n": 5, "cash_reserve_pct": 0.08, "entry_steps": [0.60, 0.25, 0.15]},
    }.get(risk, {"n": 4, "cash_reserve_pct": 0.12, "entry_steps": [0.50, 0.30, 0.20]})

    selected = candidates[: defaults["n"]]
    if not selected:
        return {
            "goal": goal or "balanced growth",
            "risk_profile": risk,
            "entry_style": goal_profile.get("entry_style"),
            "capital_base_jpy": capital_base_jpy,
            "target_return_pct": target_return_pct,
            "horizon_days": horizon_days,
            "positions": [],
            "cash_reserve_jpy": round(capital_base_jpy, 2),
            "planned_investment_jpy": 0.0,
            "estimated_cash_left_jpy": round(capital_base_jpy, 2),
        }

    investable_pct = max(0.50, 1.0 - defaults["cash_reserve_pct"])
    # Keep per-name cap aligned with discovery candidate generation range.
    max_alloc_per_name = max(0.10, min(float(max_pos_pct), 0.65))

    raw_scores = [max(0.05, float(c.get("selection_score") or c.get("score") or 0.05)) for c in selected]
    score_sum = sum(raw_scores) or 1.0
    weights = [min(max_alloc_per_name, investable_pct * s / score_sum) for s in raw_scores]

    assigned = sum(weights)
    if assigned < investable_pct:
        room = [max(0.0, max_alloc_per_name - w) for w in weights]
        room_sum = sum(room)
        if room_sum > 0:
            leftover = investable_pct - assigned
            weights = [w + leftover * (r / room_sum) for w, r in zip(weights, room)]

    steps = defaults["entry_steps"]
    positions = []
    total_planned = 0.0
    for c, w in zip(selected, weights):
        lot_size = int(c.get("lot_size") or 1)
        cost_per_lot_jpy = float(c.get("cost_per_lot_jpy") or 0.0)
        if cost_per_lot_jpy <= 0:
            continue
        planned_budget = capital_base_jpy * w
        lots = int(math.floor(planned_budget / cost_per_lot_jpy))
        if lots <= 0 and cost_per_lot_jpy <= capital_base_jpy * max_alloc_per_name:
            lots = 1
        shares = lots * lot_size
        planned_cost = lots * cost_per_lot_jpy
        if shares <= 0 or planned_cost <= 0:
            continue
        total_planned += planned_cost
        positions.append({
            "symbol": c.get("symbol"),
            "market": c.get("market"),
            "weight_pct": round((planned_cost / capital_base_jpy) * 100.0, 2),
            "planned_budget_jpy": round(planned_budget, 2),
            "planned_cost_jpy": round(planned_cost, 2),
            "planned_lots": lots,
            "planned_shares": shares,
            "entry_schedule": [
                {
                    "stage": idx + 1,
                    "allocation_pct_of_position": round(pct * 100.0, 2),
                    "planned_notional_jpy": round(planned_cost * pct, 2),
                }
                for idx, pct in enumerate(steps)
            ],
            "thesis": f"alpha={round(float(c.get('score') or 0.0), 3)}, affordability={round(float(c.get('affordability_score') or 0.0), 3)}",
        })

    total_planned = round(total_planned, 2)
    if total_planned <= 0 and selected:
        # Fallback: if candidate ranking produced at least one tradable lot, avoid empty plan.
        for c in selected:
            lot_size = int(c.get("lot_size") or 1)
            cost_per_lot_jpy = float(c.get("cost_per_lot_jpy") or 0.0)
            if cost_per_lot_jpy <= 0:
                continue
            if cost_per_lot_jpy > capital_base_jpy * max_alloc_per_name:
                continue
            planned_cost = cost_per_lot_jpy
            total_planned = round(planned_cost, 2)
            positions.append({
                "symbol": c.get("symbol"),
                "market": c.get("market"),
                "weight_pct": round((planned_cost / capital_base_jpy) * 100.0, 2),
                "planned_budget_jpy": round(capital_base_jpy * max_alloc_per_name, 2),
                "planned_cost_jpy": round(planned_cost, 2),
                "planned_lots": 1,
                "planned_shares": lot_size,
                "entry_schedule": [
                    {
                        "stage": idx + 1,
                        "allocation_pct_of_position": round(pct * 100.0, 2),
                        "planned_notional_jpy": round(planned_cost * pct, 2),
                    }
                    for idx, pct in enumerate(steps)
                ],
                "thesis": f"fallback_one_lot alpha={round(float(c.get('score') or 0.0), 3)}",
            })
            break

    cash_left = round(max(0.0, capital_base_jpy - total_planned), 2)
    return {
        "goal": goal or "balanced growth",
        "risk_profile": risk,
        "entry_style": goal_profile.get("entry_style"),
        "capital_base_jpy": round(capital_base_jpy, 2),
        "target_return_pct": target_return_pct,
        "horizon_days": int(horizon_days),
        "positions": positions,
        "planned_investment_jpy": total_planned,
        "cash_reserve_jpy": cash_left,
        "estimated_cash_left_jpy": cash_left,
    }

def _load_discovery_learning_store() -> dict:
    data = _load_json(DISCOVERY_LEARNING_PATH)
    return data if isinstance(data, dict) else {}

def _save_discovery_learning_store(store: dict):
    try:
        DISCOVERY_LEARNING_PATH.parent.mkdir(parents=True, exist_ok=True)
        DISCOVERY_LEARNING_PATH.write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[discovery] unable to save learning store: {e}")

def discovery_workflow(payload: dict):
    """
    Stage 3: Screening high-potential stocks based on intelligence.
    v1.3.4: Capital+goal driven discovery with adaptive multi-attempt search.
    """
    run_id = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()

    requested_market = str(payload.get("market") or "").upper().strip()
    market_explicit = requested_market in {"JP", "US", "ALL"}
    market_focus = requested_market if market_explicit else str(os.getenv("NEWS_FOCUS_MARKET", "ALL")).upper().strip()
    market_focus = market_focus if market_focus in {"JP", "US", "ALL"} else "ALL"

    positions_snapshot = _get_current_positions_from_fills(max_symbols=12)
    if not market_explicit and positions_snapshot:
        mk_counter = Counter([str(p.get("market") or "").upper() for p in positions_snapshot if p.get("market")])
        if mk_counter.get("JP", 0) > 0 and mk_counter.get("US", 0) == 0:
            market_focus = "JP"
        elif mk_counter.get("US", 0) > 0 and mk_counter.get("JP", 0) == 0:
            market_focus = "US"

    account_snapshot = _get_account_state_snapshot()
    account_capital = _safe_float(account_snapshot.get("starting_capital"))
    account_ccy = str(account_snapshot.get("base_ccy") or "JPY").upper()

    capital_raw = payload.get("capital_base_jpy")
    capital_input_ccy = "JPY"
    if capital_raw is None:
        capital_raw = payload.get("capital_base")
    if capital_raw is None and account_capital:
        capital_raw = account_capital
        capital_input_ccy = account_ccy if account_ccy in {"JPY", "USD"} else "JPY"

    base_max_pos_pct = max(0.10, min(float(payload.get("max_position_pct") or 0.25), 0.65))

    goal_text = str(payload.get("goal") or "").strip()
    risk_profile = _normalize_risk_profile(payload.get("risk_profile"))
    goal_profile = _infer_goal_profile(goal_text, risk_profile)
    target_return_pct = _safe_float(payload.get("target_return_pct"))
    if target_return_pct is None and goal_text:
        m_target = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", goal_text)
        if m_target:
            target_return_pct = _safe_float(m_target.group(1))
    horizon_days = int(payload.get("horizon_days") or goal_profile.get("horizon_days") or 120)

    usdjpy = 150.0
    try:
        usdjpy_ticker = yf.Ticker("JPY=X")
        usdjpy = _safe_float(usdjpy_ticker.fast_info.get("last_price")) or 150.0
    except Exception:
        pass

    capital_base_jpy = max(1.0, float(capital_raw or 400000))
    if capital_input_ccy == "USD":
        capital_base_jpy = capital_base_jpy * usdjpy

    prefer_small_mid_cap = _to_bool(payload.get("prefer_small_mid_cap"), default=(capital_base_jpy <= 600000))
    avoid_mega_cap = _to_bool(payload.get("avoid_mega_cap"), default=prefer_small_mid_cap)
    auto_evolve = _to_bool(payload.get("auto_evolve"), default=True)
    auto_expand_default = not market_explicit and (market_focus == "ALL")
    if not market_explicit and positions_snapshot and market_focus in {"JP", "US"}:
        auto_expand_default = False
    auto_expand_market = _to_bool(payload.get("auto_expand_market"), default=auto_expand_default)
    enable_learning = _to_bool(payload.get("enable_learning"), default=True)
    quick_mode = _to_bool(payload.get("quick_mode"), default=False)
    default_budget = DISCOVERY_TIME_BUDGET_S if quick_mode else max(DISCOVERY_TIME_BUDGET_S * 2, 120)
    time_budget_s = int(payload.get("time_budget_s") or default_budget)
    time_budget_s = max(20, min(time_budget_s, 600))
    started_at = time.time()

    max_attempts = int(payload.get("max_attempts") or 4)
    max_attempts = max(1, min(max_attempts, 6))
    risk_target_map = {"low": 2, "medium": 3, "high": 4}
    min_candidates_target = int(payload.get("min_candidates") or risk_target_map.get(goal_profile.get("risk_profile"), 3))
    min_candidates_target = max(1, min(min_candidates_target, 8))
    if quick_mode:
        max_attempts = min(max_attempts, 2)
        min_candidates_target = min(min_candidates_target, 2)

    def _deadline_reached() -> bool:
        return (time.time() - started_at) >= time_budget_s

    print(
        "[discovery] "
        f"Market={market_focus} ExplicitMarket={market_explicit} AutoExpand={auto_expand_market} "
        f"CapitalJPY={capital_base_jpy:.2f} InputCCY={capital_input_ccy} BaseMaxPosPct={base_max_pos_pct} "
        f"Goal={goal_text or 'N/A'} Risk={goal_profile.get('risk_profile')} "
        f"AutoEvolve={auto_evolve} Attempts={max_attempts} Quick={quick_mode} BudgetS={time_budget_s}"
    )

    wl = _load_watchlists()
    jp_list = wl.get("jp_symbols", [])
    us_list = wl.get("us_symbols", [])
    ideal_lot_cost_jpy = max(capital_base_jpy * 0.10, 30000.0)
    base_mega_cap_cutoff_jpy = 2.5e13 if capital_base_jpy <= 600000 else 8.0e13
    risk_alpha_floor_map = {"low": -0.7, "medium": -1.0, "high": -1.3}
    base_alpha_floor = risk_alpha_floor_map.get(goal_profile.get("risk_profile"), -1.0)
    scanned_symbols = set()

    def _capital_bucket(amount_jpy: float) -> str:
        if amount_jpy < 300000:
            return "lt_300k"
        if amount_jpy < 800000:
            return "300k_800k"
        if amount_jpy < 2000000:
            return "800k_2m"
        return "ge_2m"

    def _build_universe(market_key: str) -> List[str]:
        if market_key == "JP":
            return list(jp_list)
        if market_key == "US":
            return list(us_list)
        
        # Only mix markets if no explicit market was requested or if explicitly set to ALL
        if market_focus == "ALL" or not market_explicit:
            if capital_base_jpy <= 600000:
                return list(jp_list) + list(us_list)
            
            merged = []
            for i in range(max(len(jp_list), len(us_list))):
                if i < len(jp_list):
                    merged.append(jp_list[i])
                if i < len(us_list):
                    merged.append(us_list[i])
            return merged
        
        # Fallback to focused market if defined, else empty
        return list(jp_list) if market_focus == "JP" else list(us_list) if market_focus == "US" else []

    # --- Integration: Macro News Factor & Historical Memory ---
    print("[discovery] Retrieving historical news facts for context...")
    historical_bias = 0.0
    try:
        # Search for recent macro facts from dedicated news tools
        recent_facts = query_facts(fact_type="macro_sentiment_impact", limit=3)
        for f in recent_facts:
            data = f.get("payload_json", {}).get("data", {})
            historical_bias += float(data.get("sentiment", 0.0)) * 0.8
        print(f"[discovery] Historical Sentiment Bias: {historical_bias:+.2f}")
    except: pass

    print("[discovery] Computing live macro news risk factor...")
    macro_risk = compute_news_risk_factor({"symbol": "^N225" if market_focus == "JP" else "^GSPC", "run_id": run_id, "date": date_str})
    news_sentiment_bias = macro_risk.get("scores", {}).get("sentiment", 0.0) + historical_bias
    news_risk_z = macro_risk.get("effective_news_risk_z", 0.0)
    print(f"[discovery] Final Combined News Factor: Sentiment={news_sentiment_bias}, Risk_Z={news_risk_z}")

    # --- D-03: Cross-Sectional Rank Pre-computation ---
    # Compute 20d excess-return percentile rank for the relevant universe.
    # This captures cross-sectional momentum: top-quartile stocks get a boost,
    # bottom-quartile stocks are penalized. Computed once, reused across all attempts.
    cs_rank_map: dict = {}
    if not quick_mode:
        try:
            cs_universe = list(jp_list)[:40] if market_focus in ("JP", "ALL") else list(us_list)[:30]
            if market_focus == "US":
                cs_universe = list(us_list)[:30]
            print(f"[discovery] Computing cross-sectional ranks for {len(cs_universe)} symbols...")
            cs_rank_map = _compute_cross_sectional_rank(cs_universe)
            print(f"[discovery] CS rank computed: {len(cs_rank_map)} symbols")
        except Exception as _e:
            print(f"[discovery] CS rank skipped: {_e}")

    def _run_attempt(profile: dict) -> tuple:
        market_key = str(profile.get("market") or market_focus).upper()
        market_key = market_key if market_key in {"JP", "US", "ALL"} else market_focus
        max_pos_pct = max(0.10, min(float(profile.get("max_position_pct") or base_max_pos_pct), 0.65))
        max_pos_jpy = capital_base_jpy * max_pos_pct
        alpha_floor = max(-3.0, min(float(profile.get("alpha_floor") or base_alpha_floor), 0.5))
        avoid_mega_cap_local = bool(profile.get("avoid_mega_cap"))
        prefer_small_mid_cap_local = bool(profile.get("prefer_small_mid_cap"))
        mega_cap_cutoff_jpy = float(profile.get("mega_cap_cutoff_jpy") or base_mega_cap_cutoff_jpy)

        universe = _build_universe(market_key)
        scanned_count = 0
        results_local = []

        universe_cap = DISCOVERY_QUICK_UNIVERSE_CAP if quick_mode else DISCOVERY_FULL_UNIVERSE_CAP
        for sym in universe[:universe_cap]:
            if _deadline_reached():
                break
            is_jp = str(sym).upper().endswith(".T")
            
            # --- PHYSICAL BLOCKADE ---
            if market_key == "JP" and not is_jp:
                continue # Skip US stocks if we explicitly want JP
            if market_key == "US" and is_jp:
                continue # Skip JP stocks if we explicitly want US

            scanned_count += 1
            scanned_symbols.add(sym)
            quote = _fetch_quote_facts(sym)
            price = _safe_float(quote.get("price"))
            if price is None or price <= 0:
                continue

            is_jp = sym.endswith(".T")
            lot_size = 100 if is_jp else 1
            price_jpy = price if is_jp else price * usdjpy
            cost_per_lot_jpy = price_jpy * lot_size
            if cost_per_lot_jpy > max_pos_jpy:
                continue

            market_cap = _safe_float(quote.get("market_cap"))
            market_cap_jpy = None
            if market_cap is not None:
                market_cap_jpy = market_cap if is_jp else market_cap * usdjpy
                if avoid_mega_cap_local and market_cap_jpy > mega_cap_cutoff_jpy:
                    continue

            m = _compute_quant_metrics(sym)
            if not m.get("ok"):
                continue
            
            # --- APPLY NEWS FACTOR BIAS ---
            # Sentiment adds to alpha, Risk_Z penalizes it
            alpha = _safe_float(m.get("alpha_score"))
            if alpha is not None:
                alpha = alpha + (news_sentiment_bias * 1.5) - (news_risk_z * 0.5)
            
            if alpha is None or alpha < alpha_floor:
                continue

            affordability = 1.0 - min(
                1.0,
                abs(cost_per_lot_jpy - ideal_lot_cost_jpy) / max(ideal_lot_cost_jpy, 1.0),
            )
            affordability_score = 0.90 * affordability
            size_score = 0.0
            if market_cap_jpy is not None:
                if market_cap_jpy > 8.0e13:
                    size_score -= 0.65
                elif market_cap_jpy > 4.0e13:
                    size_score -= 0.25
                elif market_cap_jpy < 6.0e11:
                    size_score -= 0.10
                else:
                    size_score += 0.15
            if prefer_small_mid_cap_local and market_cap_jpy is not None and market_cap_jpy > 2.0e13:
                size_score -= 0.20
            if capital_base_jpy <= 600000 and is_jp:
                size_score += 0.08

            # D-03: Cross-sectional rank adjustment (momentum overlay)
            cs_percentile = cs_rank_map.get(sym)
            cs_rank_score = 0.0
            if cs_percentile is not None:
                if cs_percentile >= 0.75:
                    cs_rank_score = 0.30   # top quartile: strong momentum
                elif cs_percentile >= 0.50:
                    cs_rank_score = 0.10   # above median: mild tailwind
                elif cs_percentile < 0.25:
                    cs_rank_score = -0.30  # bottom quartile: momentum headwind
                else:
                    cs_rank_score = -0.10  # below median: mild drag

            selection_score = alpha + affordability_score + size_score + cs_rank_score
            lots = int(math.floor(max_pos_jpy / cost_per_lot_jpy))

            results_local.append({
                "symbol": sym,
                "market": "JP" if is_jp else "US",
                "signal": m.get("signal"),
                "score": alpha,
                "selection_score": selection_score,
                "risk": m.get("risk_state"),
                "price_orig": price,
                "price_jpy": price_jpy,
                "currency": quote.get("currency"),
                "lot_size": lot_size,
                "cost_per_lot_jpy": cost_per_lot_jpy,
                "market_cap": market_cap,
                "market_cap_jpy": market_cap_jpy,
                "affordability_score": affordability_score,
                "recommended_lots": lots,
                "recommended_shares": lots * lot_size,
                "estimated_cost_jpy": lots * cost_per_lot_jpy,
                "max_position_pct_used": round(max_pos_pct, 4),
                "alpha_floor_used": round(alpha_floor, 4),
                "cs_percentile": round(cs_percentile, 3) if cs_percentile is not None else None,
                "cs_rank_score": round(cs_rank_score, 2),
            })

        results_local.sort(
            key=lambda x: (x.get("selection_score", -999), x.get("score", -999)),
            reverse=True,
        )
        return results_local[:15], scanned_count, len(universe), market_key, max_pos_pct, alpha_floor

    import random

    base_strategies = {
        "strict": {
            "market": market_focus,
            "max_position_pct": base_max_pos_pct,
            "alpha_floor": base_alpha_floor,
            "avoid_mega_cap": avoid_mega_cap,
            "prefer_small_mid_cap": prefer_small_mid_cap,
        },
        "relax_position": {
            "market": market_focus,
            "max_position_pct": min(max(base_max_pos_pct, 0.30) + 0.07, 0.45),
            "alpha_floor": base_alpha_floor - 0.35,
            "avoid_mega_cap": avoid_mega_cap,
            "prefer_small_mid_cap": prefer_small_mid_cap,
        },
        "broaden_factors": {
            "market": "ALL" if auto_expand_market and market_focus != "ALL" else market_focus,
            "max_position_pct": min(max(base_max_pos_pct, 0.35) + 0.10, 0.55),
            "alpha_floor": base_alpha_floor - 0.75,
            "avoid_mega_cap": False,
            "prefer_small_mid_cap": False,
        },
        "fallback_wide": {
            "market": "ALL" if auto_expand_market and market_focus != "ALL" else market_focus,
            "max_position_pct": min(max(base_max_pos_pct, 0.40) + 0.15, 0.65),
            "alpha_floor": base_alpha_floor - 1.10,
            "avoid_mega_cap": False,
            "prefer_small_mid_cap": False,
        }
    }

    learning_key = None
    learning_store = {}
    mab_state = {}

    if enable_learning:
        learning_key = "|".join(["mab_v1", market_focus, goal_profile.get("risk_profile", "medium"), _capital_bucket(capital_base_jpy)])
        learning_store = _load_discovery_learning_store()
        mab_state = learning_store.get(learning_key) or {}
        
        for s_name in base_strategies.keys():
            if s_name not in mab_state:
                mab_state[s_name] = {"alpha": 1.0, "beta": 1.0}

    attempt_profiles = []
    if enable_learning:
        sampled_scores = {}
        for s_name, stats in mab_state.items():
            if s_name in base_strategies:
                a = max(1.0, stats.get("alpha", 1.0))
                b = max(1.0, stats.get("beta", 1.0))
                sampled_scores[s_name] = random.betavariate(a, b)
        
        sorted_strategy_names = sorted(sampled_scores.keys(), key=lambda k: sampled_scores[k], reverse=True)
        for s_name in sorted_strategy_names:
            profile = base_strategies[s_name].copy()
            profile["label"] = s_name
            attempt_profiles.append(profile)
    else:
        for k, v in base_strategies.items():
            p = v.copy()
            p["label"] = k
            attempt_profiles.append(p)

    while len(attempt_profiles) < max_attempts:
        prev = attempt_profiles[-1]
        attempt_profiles.append({
            "label": f"extended_{len(attempt_profiles)+1}",
            "market": "ALL" if auto_expand_market and market_focus != "ALL" else prev.get("market", market_focus),
            "max_position_pct": min(float(prev.get("max_position_pct", 0.50)) + 0.05, 0.65),
            "alpha_floor": max(float(prev.get("alpha_floor", -2.0)) - 0.25, -3.0),
            "avoid_mega_cap": False,
            "prefer_small_mid_cap": False,
        })

    best_results = []
    best_attempt_idx = 0
    best_profile = None
    best_market = market_focus
    best_max_pos_pct = base_max_pos_pct
    selected_alpha_floor = base_alpha_floor
    attempt_reports = []
    total_scanned = 0
    ran_attempts = 0

    for idx, profile in enumerate(attempt_profiles[:max_attempts], start=1):
        if _deadline_reached():
            print(f"[discovery] Time budget reached before attempt {idx}, stopping early.")
            break
        results_try, scanned_count, universe_size, market_used, max_pos_pct_used, alpha_floor_used = _run_attempt(profile)
        ran_attempts += 1
        total_scanned += scanned_count
        attempt_reports.append(
            {
                "attempt": idx,
                "label": profile.get("label", f"attempt_{idx}"),
                "market": market_used,
                "max_position_pct": round(max_pos_pct_used, 4),
                "alpha_floor": round(alpha_floor_used, 4),
                "avoid_mega_cap": bool(profile.get("avoid_mega_cap")),
                "prefer_small_mid_cap": bool(profile.get("prefer_small_mid_cap")),
                "universe_size": universe_size,
                "scanned_count": scanned_count,
                "candidate_count": len(results_try),
            }
        )

        if len(results_try) > len(best_results):
            best_results = results_try
            best_attempt_idx = idx
            best_profile = profile
            best_market = market_used
            best_max_pos_pct = max_pos_pct_used
            selected_alpha_floor = alpha_floor_used

        if len(results_try) >= min_candidates_target:
            break
        if not auto_evolve:
            break

    results = best_results
    results.sort(key=lambda x: (x.get("selection_score", -999), x.get("score", -999)), reverse=True)
    results = results[:15]

    position_plan = _build_position_plan(
        results,
        capital_base_jpy=capital_base_jpy,
        max_pos_pct=best_max_pos_pct,
        goal=goal_text,
        goal_profile=goal_profile,
        target_return_pct=target_return_pct,
        horizon_days=horizon_days,
    )

    if enable_learning and learning_key:
        decay_factor = 0.98
        for s_name in mab_state:
            mab_state[s_name]["alpha"] = max(1.0, mab_state[s_name]["alpha"] * decay_factor)
            mab_state[s_name]["beta"] = max(1.0, mab_state[s_name]["beta"] * decay_factor)
        
        if best_profile is not None:
            used_label = best_profile.get("label")
            if used_label in mab_state:
                if len(results) >= min_candidates_target:
                    mab_state[used_label]["alpha"] += 1.0
                else:
                    mab_state[used_label]["beta"] += 1.0

        learning_store[learning_key] = mab_state
        learning_store["updated_at"] = datetime.utcnow().isoformat() + "Z"
        _save_discovery_learning_store(learning_store)

    top_candidates_str = ""
    for idx, c in enumerate(results[:5]):
        sym = c.get("symbol")
        score = c.get("selection_score", 0)
        cost = c.get("estimated_cost_jpy", 0)
        top_candidates_str += f"- **{sym}**: 综合评分 {score:.2f}，建议配置约 {cost:,.0f} JPY\n"

    profile_label = best_profile.get('label') if best_profile else 'default'
    summary = (
        f"🎯 **选股与配置报告 (新闻因子集成版)**\n\n"
        f"**配置目标**: {goal_text or '稳健增长'} | **资金盘**: {capital_base_jpy:,.0f} JPY\n"
        f"**市场偏置**: 情绪评分 {news_sentiment_bias:+.2f} | 风险因子 {news_risk_z:.2f}\n\n"
        f"基于当前地缘政治（重点关注美国/中东）与市场情绪分析，系统完成了 {ran_attempts} 轮动态策略探索。"
        f"由于新闻因子判定当前环境为 {'偏多' if news_sentiment_bias > 0.2 else '偏空' if news_sentiment_bias < -0.2 else '震荡'}，"
        f"已自动对选股池进行了权重修正。\n\n"
        f"🏆 **核心推荐 Top 5**:\n{top_candidates_str if top_candidates_str else '- 暂无符合严格条件的标的。'}\n"
        f"💡 **仓位推演**: 预计总投资 {position_plan.get('planned_investment_jpy', 0):,.0f} JPY，保留现金流 {position_plan.get('cash_reserve_jpy', 0):,.0f} JPY。"
    )

    output = {
        "ok": True,
        "date": date_str,
        "market": best_market if results else market_focus,
        "market_explicit": market_explicit,
        "capital_base_jpy": capital_base_jpy,
        "capital_input_ccy": capital_input_ccy,
        "account_base_ccy": account_ccy if account_ccy in {"JPY", "USD"} else "JPY",
        "max_position_pct": best_max_pos_pct,
        "goal": goal_text or "balanced growth",
        "risk_profile": goal_profile.get("risk_profile"),
        "horizon_days": horizon_days,
        "target_return_pct": target_return_pct,
        "usdjpy_rate": usdjpy,
        "candidates": results,
        "position_plan": position_plan,
        "search_attempts": attempt_reports,
        "selected_attempt": best_attempt_idx,
        "min_candidates_target": min_candidates_target,
        "auto_evolve": auto_evolve,
        "auto_expand_market": auto_expand_market,
        "quick_mode": quick_mode,
        "time_budget_s": time_budget_s,
        "elapsed_s": round(time.time() - started_at, 2),
        "learning_enabled": enable_learning,
        "analysis": summary,
    }

    if run_id:
        record_fact(run_id, "quant", "discovery_result", output)
    return output

def compute_news_risk_factor(payload: dict):
    """
    Step 3B: Compute News Risk Factor using LLM.
    Scores sentiment, uncertainty, and severity, then writes to feature_daily.
    """
    symbol = str(payload.get("symbol", "NVDA")).upper()
    run_id = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()
    
    # 1. Fetch recent news for the symbol
    quote = _fetch_quote_facts(symbol)
    news = _merge_recent_news(quote, max_items=3)
    
    if not news:
        return {"ok": True, "symbol": symbol, "news_risk_raw": 0.0, "effective_news_risk_z": 0.0, "note": "No recent news found."}

    # Calculate lag to determine staleness
    import statistics
    from datetime import timezone
    try:
        from dateutil import parser
    except ImportError:
        pass
        
    lag_seconds_list = []
    now_utc = datetime.now(timezone.utc)
    for n in news:
        pub_str = n.get("published_at") or n.get("published_ts") or ""
        if pub_str:
            try:
                # Basic string handling since different sources return different formats
                if isinstance(pub_str, (int, float)):
                    pub_dt = datetime.fromtimestamp(pub_str, timezone.utc)
                else:
                    pub_dt = parser.parse(str(pub_str))
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                lag = (now_utc - pub_dt).total_seconds()
                lag_seconds_list.append(max(0, lag))
            except Exception:
                pass
                
    lag_p50 = statistics.median(lag_seconds_list) if lag_seconds_list else 0
    staleness_flag = "fresh"
    news_factor_weight = 1.0
    
    # T1 Stale threshold: 60 mins (3600s), T2 Expired threshold: 6 hours (21600s)
    if lag_p50 > 21600:
        staleness_flag = "expired"
        news_factor_weight = 0.2
    elif lag_p50 > 3600:
        staleness_flag = "stale"
        news_factor_weight = 0.5

    # 2. Use LLM to score the news
    news_text = "\n".join([f"- {n['title']} ({n['publisher']})" for n in news])
    system_prompt = """
    You are a quantitative news risk analyzer. Evaluate the following recent news headlines for a stock.
    Provide a JSON output ONLY with these three numerical scores (float):
    - "sentiment": between -1.0 (extremely negative) and 1.0 (extremely positive).
    - "uncertainty": between 0.0 (certain/clear impact) and 1.0 (highly uncertain/unclear impact).
    - "severity": between 0.0 (routine news) and 1.0 (major structural event, e.g. bankruptcy, huge lawsuit).
    """
    
    # Fallback default scores
    sentiment, uncertainty, severity = 0.0, 0.2, 0.1
    
    try:
        response = get_llm_response(system_prompt, news_text, model=QUANT_LLM_MODEL, timeout_s=min(18, LLM_TIMEOUT_S))
        if response:
            import re
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                scores = json.loads(json_match.group(0))
                sentiment = float(scores.get("sentiment", 0.0))
                uncertainty = float(scores.get("uncertainty", 0.2))
                severity = float(scores.get("severity", 0.1))
    except Exception as e:
        print(f"[NewsRisk] LLM scoring failed: {e}")

    # 3. Calculate raw risk score (as defined in design doc)
    # w1, w2, w3 weights
    w1, w2, w3 = 1.0, 0.5, 2.0
    news_risk_raw = w1 * max(0, -sentiment) + w2 * uncertainty + w3 * severity
    
    # 3.5 Apply staleness degradation
    effective_news_risk_z = news_risk_raw * news_factor_weight

    # 4. Save to feature_daily (SQLite)
    db_path = QUANT_DB_PATH
    saved = False
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            # Ensure table exists first
            import sys
            sys.path.insert(0, str(db_path.parent))
            import trade_schema
            trade_schema.ensure_trade_tables(conn)
            sys.path.pop(0)

            cur = conn.cursor()
            fact_ids_json = json.dumps([n.get("url", "unknown") for n in news])
            
            # Upsert
            cur.execute("""
                INSERT INTO feature_daily (asof, symbol, feature_name, value, source_fact_ids)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(asof, symbol, feature_name) 
                DO UPDATE SET value=excluded.value, source_fact_ids=excluded.source_fact_ids, created_at=CURRENT_TIMESTAMP
            """, (date_str, symbol, "news_risk_raw", effective_news_risk_z, fact_ids_json))
            
            conn.commit()
            conn.close()
            saved = True
        except Exception as e:
            print(f"[NewsRisk] DB save error: {e}")

    # 5. Trigger Browser Deep Dive if risk is high
    triggered_deep_dive = False
    if effective_news_risk_z > 2.0 or severity > 0.8:
        triggered_deep_dive = True
        # Enqueue deep analysis or screenshot via orchestrator if we had orchestrator access
        # Here we just record it as a fact for the brain to pick up
        pass

    result = {
        "ok": True,
        "symbol": symbol,
        "date": date_str,
        "scores": {
            "sentiment": sentiment,
            "uncertainty": uncertainty,
            "severity": severity
        },
        "news_risk_raw": news_risk_raw,
        "news_factor_weight": news_factor_weight,
        "effective_news_risk_z": effective_news_risk_z,
        "freshness": {
            "lag_p50_seconds": lag_p50,
            "staleness_flag": staleness_flag
        },
        "triggered_deep_dive": triggered_deep_dive,
        "saved_to_db": saved
    }

    if run_id:
        record_fact(run_id, "quant", "news_risk_factor", result)

    return result

def portfolio_set_account(payload: dict):
    """Set initial capital and currency in account_state."""
    capital = float(payload.get("starting_capital", 0))
    ccy = payload.get("ccy", "JPY")
    date_str = _now_date_str()
    
    db_path = QUANT_DB_PATH
    if not db_path.exists():
        return {"ok": False, "error": "Database not found."}
        
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            import sys
            sys.path.insert(0, str(db_path.parent))
            import trade_schema
            trade_schema.ensure_trade_tables(conn)
            sys.path.pop(0)
        except Exception:
            pass
        cur = conn.cursor()
        
        # Check if account exists, otherwise insert
        cur.execute("SELECT starting_capital FROM account_state ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        
        if row:
            cur.execute("""
                UPDATE account_state 
                SET starting_capital=?, cash_balance=?, equity=?, base_ccy=?, updated_at=CURRENT_TIMESTAMP
            """, (capital, capital, capital, ccy))
        else:
            cur.execute("""
                INSERT INTO account_state (asof, starting_capital, cash_balance, equity, base_ccy)
                VALUES (?, ?, ?, ?, ?)
            """, (date_str, capital, capital, capital, ccy))
            
        conn.commit()
        conn.close()
        return {"ok": True, "message": f"Successfully set account capital to {capital} {ccy}."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def portfolio_record_fill(payload: dict):
    """Record a manual stock purchase/sell."""
    symbol = _normalize_symbol_for_lookup(payload.get("symbol", ""))
    side = payload.get("side", "BUY").upper()
    qty = float(payload.get("qty", 0))
    price = float(payload.get("price", 0))
    date_str = _now_date_str()
    fill_id = uuid.uuid4().hex
    
    db_path = QUANT_DB_PATH
    if not db_path.exists():
        return {"ok": False, "error": "Database not found."}
        
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO fills (fill_id, run_id, asof, ts, symbol, side, qty, price, source)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, 'manual_discord')
        """, (fill_id, "manual", date_str, symbol, side, qty, price))
        
        # Deduct/Add from cash_balance in account_state
        if side == "BUY":
            cost = qty * price
            cur.execute("UPDATE account_state SET cash_balance = cash_balance - ?", (cost,))
        elif side == "SELL":
            revenue = qty * price
            cur.execute("UPDATE account_state SET cash_balance = cash_balance + ?", (revenue,))
            
        conn.commit()
        conn.close()
        return {"ok": True, "message": f"Recorded {side} of {qty} shares of {symbol} at {price}."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def web_search_and_browse(payload: dict):
    query = str(payload.get("query") or payload.get("text") or payload.get("question") or "").strip()
    if not query:
        return {"ok": False, "error": "missing query"}

    def _is_weather_query(q: str) -> bool:
        ql = q.lower()
        return any(k in ql for k in ["weather", "temperature", "forecast", "天气", "天気"])

    def _extract_weather_location(q: str) -> str:
        cleaned = re.sub(r"(today|now|current|weather|forecast|temperature|今天|现在|天气|天気)", " ", q, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        return cleaned or "Tokyo"

    def _fetch_weather_summary(location: str):
        try:
            g = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"},
                timeout=12,
            )
            g.raise_for_status()
            arr = (g.json() or {}).get("results") or []
            if not arr:
                return None
            first = arr[0]
            lat = first.get("latitude")
            lon = first.get("longitude")
            name = first.get("name") or location
            country = first.get("country") or ""
            w = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
                    "timezone": "auto",
                },
                timeout=12,
            )
            w.raise_for_status()
            cur = (w.json() or {}).get("current") or {}
            brief = (
                f"{name}, {country}: temp={cur.get('temperature_2m')}C, "
                f"feels={cur.get('apparent_temperature')}C, humidity={cur.get('relative_humidity_2m')}%, "
                f"wind={cur.get('wind_speed_10m')}km/h, weather_code={cur.get('weather_code')}"
            )
            return {
                "title": f"Current Weather - {name}",
                "body": brief,
                "href": "https://open-meteo.com/",
                "capture_url": f"https://wttr.in/{requests.utils.quote(name)}?lang=zh-cn",
            }
        except Exception:
            return None

    def _search_via_ddg_html(q: str, max_results: int = 5):
        items = []
        try:
            resp = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": q},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            resp.raise_for_status()
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select(".result")
            for row in rows:
                a = row.select_one(".result__a")
                if not a:
                    continue
                title = a.get_text(" ", strip=True)
                href = a.get("href") or ""
                snippet_node = row.select_one(".result__snippet")
                body = snippet_node.get_text(" ", strip=True) if snippet_node else ""
                if title or body or href:
                    items.append({"title": title, "body": body, "href": href, "capture_url": href})
                if len(items) >= max_results:
                    break
        except Exception:
            return []
        return items

    items = []
    if _is_weather_query(query):
        weather_item = _fetch_weather_summary(_extract_weather_location(query))
        if weather_item:
            items.append(weather_item)

    items.extend(_search_via_ddg_html(query, max_results=5))
    if not items:
        return {"ok": False, "error": "Search failed or returned no results.", "extracted_text": ""}

    snippets = []
    for i, res in enumerate(items[:6]):
        title = str(res.get("title") or "")
        body = str(res.get("body") or "")
        href = str(res.get("href") or "")
        snippets.append(f"[{i+1}] {title}\n{body}\nSource: {href}\n")

    capture_urls = []
    for it in items:
        u = str(it.get("capture_url") or "")
        if u and u not in capture_urls and u.startswith(("http://", "https://")):
            capture_urls.append(u)
    artifacts = _openclaw_capture_urls(capture_urls[:2], full_page=False, max_shots=2) if capture_urls else []

    extracted_text = "\n".join(snippets)
    return {
        "ok": True,
        "query": query,
        "extracted_text": extracted_text,
        "snippets": snippets,
        "artifacts": artifacts,
        "used_openclaw": bool(artifacts),
        "summary": f"Found {len(snippets)} search results for '{query}'."
    }

def _quick_news_brief(articles: List[dict], title: str = "市场快讯") -> str:
    if not articles:
        return "暂无可用新闻样本。"

    # --- Deduplication by title similarity ---
    unique_articles = []
    seen_titles = set()
    for a in articles:
        t = str(a.get("title") or "").strip()
        # Extract main core of the title (first 15 chars) to find hourly updates
        core = t[:15]
        if core in seen_titles: continue
        seen_titles.add(core)
        unique_articles.append(a)
    
    articles = unique_articles[:5]
    if not articles: return "未找到非重复的有效新闻。"

    domains = []
    headlines = []
    for a in articles:
        t = str(a.get("title") or "").strip()
        d = str(a.get("domain") or a.get("publisher") or "unknown").strip().lower()
        headlines.append(t)
        domains.append(d)

    domain_top = Counter(domains).most_common(3)
    top_domains = ", ".join([f"{k}({v})" for k, v in domain_top])

    text_blob = " ".join(headlines).lower()
    theme_map = {
        "宏观/利率": ["boj", "fed", "rate", "yield", "inflation", "cpi", "日银", "利率", "通胀"],
        "业绩/指引": ["earnings", "guidance", "forecast", "profit", "业绩", "财报", "盈利"],
        "科技/半导体": ["ai", "chip", "semiconductor", "nvidia", "tsmc", "半导体", "芯片"],
        "地缘/政策": ["middle east", "israel", "iran", "us policy", "election", "中东", "美国政策", "地缘"],
    }
    
    theme_hits = []
    for theme, keys in theme_map.items():
        hit = sum(1 for k in keys if k in text_blob)
        if hit > 0: theme_hits.append((theme, hit))
    
    theme_hits.sort(key=lambda x: x[1], reverse=True)
    themes = "、".join([t for t, _ in theme_hits[:3]]) if theme_hits else "目前主要受盘面指数波动驱动，暂无明显板块主线"

    return (
        f"1) {title}特征：{themes}。\n"
        f"2) 关键来源：{top_domains}。"
    )

def preclose_brief_jp(payload: dict):
    """
    Japanese market pre-close brief (15:15 JST).
    Provides quick insights before the 15:30 close.
    """
    run_id = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()
    
    freshness_hours = float(payload.get("freshness_hours", 12))

    # 1. Fetch fast path news (last 6h), then enforce freshness policy.
    # Use a more sophisticated query including geopolitics and major macro themes
    search_query = (
        "(japan OR tokyo OR nikkei) AND "
        "(geopolitics OR 'middle east' OR israel OR iran OR fed OR inflation OR 'us policy')"
    )
    raw_articles = _gdelt_doc_search(
        search_query,
        datetime.utcnow() - timedelta(hours=12), # Wider window for depth
        datetime.utcnow(),
        max_records=30,
    )
    articles, freshness = _apply_freshness_policy(raw_articles, max_age_hours=freshness_hours)

    if not articles:
        # Fallback to Google News with the same freshness gate.
        raw_articles = _fetch_news_from_google_rss("日経平均 OR 日本株", "市況 OR ニュース", max_items=10, lang="ja")
        articles, freshness = _apply_freshness_policy(raw_articles, max_age_hours=freshness_hours)

    items_text = ""
    for a in articles[:5]:
        title = a.get('title', 'Unknown')
        domain = a.get('domain', a.get('publisher', 'Unknown'))
        url = a.get('url', '')
        items_text += f"- {title} ({domain}) : {url}\n"

    if items_text:
        system = (
            "你是资深日股盘中策略分析师。请基于新闻标题生成中文盘尾情报长分析，结构必须包含："
            "1) 市场主线(宏观/行业/风格)；2) 情绪与风险偏好；3) 对收盘前30分钟资金行为的推演；"
            "4) 交易上可执行的关注清单(2-4条)；5) 风险提示。"
            "要求：专业、具体、逻辑连贯，避免空话，长度约220-450字。"
        )
        llm_summary = get_llm_response(system, items_text, model=QUANT_LLM_MODEL)
        if not str(llm_summary or "").strip() or len(str(llm_summary).strip()) < 60:
            llm_summary = _quick_news_brief(articles[:5], title="盘尾")
        freshness_line = (
            f"Freshness<= {int(freshness_hours)}h | fresh={freshness.get('fresh_count', 0)} | "
            f"dropped_stale={freshness.get('dropped_stale', 0)} | dropped_undated={freshness.get('dropped_undated', 0)}"
        )
        summary = f"【15:15 JST 盘尾情报】\n{freshness_line}\n{llm_summary}\n\n【原始快讯】\n{items_text}"
    else:
        summary = (
            "【15:15 JST 盘尾情报】\n"
            f"未找到满足 freshness<= {int(freshness_hours)}h 的可用新闻。"
            f"（剔除过旧: {freshness.get('dropped_stale', 0)}，无时间戳: {freshness.get('dropped_undated', 0)}）"
        )

    result = {
        "ok": True,
        "type": "preclose_brief_jp",
        "date": date_str,
        "freshness": freshness,
        "analysis": summary
    }
    
    if run_id:
        record_fact(run_id, "news", "preclose_brief", result)
        
    return result

def tdnet_close_flash(payload: dict):
    """
    Japanese market post-close flash (15:35 JST) with structured output and source links.
    E-01: Primary source is now _fetch_tdnet_announcements (Kabutan/Google-ja);
          GDELT retained as supplementary macro context only.
    """
    run_id = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()
    freshness_hours = float(payload.get("freshness_hours", 24))

    # E-01: Pull real TDnet/Kabutan announcements first
    tdnet_items = _fetch_tdnet_announcements(lookback_hours=freshness_hours)

    search_query = "決算 OR 発表 OR 地政学 OR 原油 OR 利下げ"
    raw_google = _fetch_news_from_google_rss(
        search_query,
        "日本株 OR 日経",
        max_items=15,
        lang="ja",
        resolve_original_links=True,
    )
    raw_gdelt = []
    try:
        gdelt_rows = _gdelt_doc_search(
            "japan OR nikkei OR tdnet OR earnings OR guidance OR oil OR geopolitics",
            datetime.utcnow() - timedelta(hours=max(12, int(freshness_hours))),
            datetime.utcnow(),
            max_records=25,
        )
        for g in gdelt_rows:
            title = str(g.get("title") or "").strip()
            url = str(g.get("url") or "").strip()
            if not title or not url:
                continue
            raw_gdelt.append({
                "title": title,
                "url": url,
                "publisher": str(g.get("domain") or g.get("source") or "").strip(),
                "published_at": str(g.get("seendate") or g.get("pubdate") or "").strip(),
                "source": "gdelt",
            })
    except Exception:
        raw_gdelt = []

    # Prefer direct-source links from GDELT, then merge Google RSS as fallback.
    merged_raw = []
    seen = set()
    for row in (raw_gdelt + raw_google):
        t = str(row.get("title") or "").strip().lower()
        u = str(row.get("url") or "").strip().lower()
        if not t or not u:
            continue
        key = (t, u)
        if key in seen:
            continue
        seen.add(key)
        merged_raw.append(row)

    articles, freshness = _apply_freshness_policy(merged_raw, max_age_hours=freshness_hours)
    # Ranking: direct links first, then newer items first.
    def _rank_tdnet_item(it: dict):
        is_google = "news.google." in str(it.get("url") or "").lower()
        dt = _parse_article_dt(it)
        ts = dt.timestamp() if dt is not None else 0.0
        return (1 if is_google else 0, -ts)
    articles.sort(key=_rank_tdnet_item)

    if not articles:
        return {"ok": True, "analysis": "【15:35 JST 盘后】未发现满足时效性的关键资讯。"}

    # Build LLM input from recent items.
    focus_items = articles[:10]
    items_text = "\n".join([f"- {a.get('title', 'Unknown')} : {a.get('url', '')}" for a in focus_items])

    system = (
        "你是一位顶尖量化分析师。请对盘后资讯输出深度研判。\n"
        "输出必须包含：\n"
        "1. [REPORT] （中文深度分析，至少120字）\n"
        "2. [FACTS] {\"sentiment\": -1~1, \"geo_risk\": 0~1, \"key_sectors\": [\"行业1\", \"行业2\"]}"
    )
    llm_raw = get_llm_response(system, items_text, model=QUANT_LLM_MODEL)

    report_text = ""
    fact_data = {"sentiment": 0.0, "geo_risk": 0.2, "key_sectors": []}
    degraded = False

    try:
        raw = str(llm_raw or "")
        m_report = re.search(r"\[REPORT\]\s*(.*?)(?:\n\s*\[FACTS\]|\Z)", raw, flags=re.S)
        if m_report:
            report_text = m_report.group(1).strip()

        m_facts = re.search(r"\[FACTS\]\s*(\{[\s\S]*?\})", raw, flags=re.S)
        if m_facts:
            parsed = json.loads(m_facts.group(1))
            if isinstance(parsed, dict):
                fact_data.update(parsed)
    except Exception:
        pass

    # Deterministic fallback when LLM report is unavailable/too short.
    if not report_text or len(report_text) < 80:
        degraded = True
        headlines = [str(a.get("title") or "").strip() for a in focus_items[:8]]
        merged = " ".join(headlines).lower()
        pos_hits = sum(merged.count(k) for k in ["上调", "增长", "超预期", "回购", "利好", "beat", "upgrade"])
        neg_hits = sum(merged.count(k) for k in ["下调", "下滑", "亏损", "风险", "预警", "miss", "downgrade"])
        geo_hits = sum(merged.count(k) for k in ["地缘", "中东", "冲突", "油价", "israel", "iran", "middle east"])

        denom = max(1, pos_hits + neg_hits)
        sentiment = max(-1.0, min(1.0, (pos_hits - neg_hits) / denom))
        geo_risk = max(0.0, min(1.0, 0.15 + 0.12 * geo_hits + 0.05 * neg_hits))
        fact_data["sentiment"] = round(float(sentiment), 2)
        fact_data["geo_risk"] = round(float(geo_risk), 2)
        report_text = _quick_news_brief(focus_items[:8], title="盘后")

    # Normalize facts for rendering.
    sentiment = max(-1.0, min(1.0, float(fact_data.get("sentiment", 0.0))))
    geo_risk = max(0.0, min(1.0, float(fact_data.get("geo_risk", 0.2))))
    sectors = fact_data.get("key_sectors")
    if not isinstance(sectors, list):
        sectors = []
    sector_text = "、".join([str(x).strip() for x in sectors if str(x).strip()][:3]) or "待观察"
    sentiment_label = "偏多" if sentiment > 0.2 else ("偏空" if sentiment < -0.2 else "中性")

    # Convert long report into concise bullet points for Discord readability.
    split_parts = re.split(r"[。\n；;!?！？]+", str(report_text))
    bullets = [p.strip() for p in split_parts if p.strip()]
    bullets = bullets[:3] if bullets else [str(report_text).strip()[:120]]

    # Preserve traceable source links.
    source_items = []
    for a in focus_items:
        title = str(a.get("title") or "Unknown").strip()
        url = str(a.get("url") or "").strip()
        publisher = str(a.get("publisher") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        google_url = url
        resolved_mode = "direct"
        # If still a Google News proxy URL, try a search-based direct-link fallback.
        if "news.google." in url.lower():
            guessed = _search_direct_url_by_title(title, publisher)
            if guessed:
                url = guessed
                resolved_mode = "resolved"
            else:
                # UX fallback: provide one-click search link to reach the original article.
                fallback_search = _build_article_search_url(title, publisher)
                if fallback_search:
                    url = fallback_search
                    resolved_mode = "search"
        source_items.append({
            "title": title,
            "url": url,
            "publisher": publisher,
            "url_google": google_url,
            "resolved_mode": resolved_mode,
        })
    source_items = source_items[:5]

    source_lines = []
    for idx, s in enumerate(source_items[:3], start=1):
        short_title = s["title"][:56] + ("..." if len(s["title"]) > 56 else "")
        tag = ""
        if s.get("resolved_mode") == "search":
            tag = " [原文检索]"
        elif s.get("resolved_mode") == "direct":
            tag = " [直链]"
        source_lines.append(f"{idx}) {short_title}{tag}\n{s['url']}")
    sources_block = "\n".join(source_lines) if source_lines else "暂无可用原文链接"

    if run_id:
        record_fact(run_id, "news", "macro_sentiment_impact", {"data": {"sentiment": sentiment, "geo_risk": geo_risk, "key_sectors": sectors}, "source": "tdnet"})

    card_data = {
        "title": "盘后研判报告",
        "subtitle": f"日本市场 | {date_str}",
        "metrics": [
            {"label": "情绪评分", "value": f"{sentiment:+.2f}"},
            {"label": "地缘风险", "value": f"{geo_risk*100:.0f}%"},
            {"label": "资讯新鲜度", "value": f"{freshness.get('fresh_count', 0)}条"},
        ],
        "highlights": str(report_text)[:500],
        "footer": "OpenClaw Nexus Quant Intelligence",
    }
    report_card = generate_report_card(card_data)

    safe_date = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(date_str))
    html_path = Path(f"/tmp/tdnet_close_flash_{safe_date}_{int(time.time())}.html")
    list_html = "".join([f"<li><a href='{html.escape(s['url'])}' target='_blank'>{html.escape(s['title'])}</a></li>" for s in source_items])
    html_doc = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>TDnet Close Flash {html.escape(str(date_str))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; line-height: 1.6; color: #1f2937; }}
    h1 {{ margin: 0 0 8px; }}
    .meta {{ color: #6b7280; margin-bottom: 12px; }}
    .card {{ background: #f8fafc; padding: 14px; border-radius: 8px; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>TDnet 盘后闪讯</h1>
  <div class="meta">Date: {html.escape(str(date_str))} | Fresh={int(freshness.get("fresh_count", 0))}</div>
  <div class="card"><strong>重点结论</strong><br>{'<br>'.join([html.escape(x) for x in bullets])}</div>
  <div class="card"><strong>风险温度</strong><br>情绪={sentiment:+.2f}（{sentiment_label}） | 地缘风险={geo_risk:.2f} | 关注方向={html.escape(sector_text)}</div>
  <div class="card"><strong>深度分析全文</strong><br>{html.escape(str(report_text)).replace(chr(10), '<br>')}</div>
  <div class="card"><strong>原文链接</strong><ol>{list_html}</ol></div>
</body>
</html>"""
    html_path.write_text(html_doc, encoding="utf-8")

    artifacts = [archive_file(html_path)]
    if isinstance(report_card, dict) and isinstance(report_card.get("artifacts"), list):
        artifacts.extend([a for a in report_card.get("artifacts", []) if isinstance(a, dict)])

    mode_note = "（AI深度研判暂不可用，已降级为规则分析）\n" if degraded else ""
    summary = (
        "【15:35 JST 盘后深度研判】\n\n"
        "【重点结论】\n"
        f"- {bullets[0] if len(bullets) > 0 else '暂无'}\n"
        f"- {bullets[1] if len(bullets) > 1 else '暂无'}\n"
        f"- {bullets[2] if len(bullets) > 2 else '暂无'}\n\n"
        "【风险温度】\n"
        f"- 情绪: {sentiment:+.2f} ({sentiment_label})\n"
        f"- 地缘风险: {geo_risk:.2f}\n"
        f"- 关注方向: {sector_text}\n\n"
        f"{mode_note}"
        "【原文链接 Top3】\n"
        f"{sources_block}"
    )

    return {
        "ok": True,
        "analysis": summary[:1800],
        "artifacts": artifacts,
        "source_links": source_items,
        "facts": {"sentiment": sentiment, "geo_risk": geo_risk, "key_sectors": sectors, "degraded": degraded},
    }

def _safe_workspace_path(rel_path: str) -> Path | None:
    rel = str(rel_path or "").replace("\\", "/").lstrip("/")
    if not rel:
        return None
    candidate = (WORKSPACE_ROOT / rel).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT.resolve())
        return candidate
    except Exception:
        return None

def _load_preview_blocklist() -> dict:
    candidates = [
        WORKSPACE_ROOT / "configs" / "preview" / "db_dependency_blocklist.json",
        Path("/app/configs/preview/db_dependency_blocklist.json"),
    ]
    for fp in candidates:
        try:
            if fp.exists():
                return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {"node": [], "python": []}

def _parse_requirements_names(req_path: Path) -> list[str]:
    out = []
    try:
        for raw in req_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            base = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower()
            if base:
                out.append(base)
    except Exception:
        return []
    return out

def _parse_pyproject_names(pyproject_path: Path) -> list[str]:
    out = []
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    project = data.get("project", {}) if isinstance(data, dict) else {}
    for item in project.get("dependencies", []) or []:
        base = re.split(r"[<>=!~\[]", str(item), maxsplit=1)[0].strip().lower()
        if base:
            out.append(base)
    optional = project.get("optional-dependencies", {}) if isinstance(project, dict) else {}
    if isinstance(optional, dict):
        for deps in optional.values():
            for item in deps or []:
                base = re.split(r"[<>=!~\[]", str(item), maxsplit=1)[0].strip().lower()
                if base:
                    out.append(base)
    return out

def _find_preview_project_root(payload: dict) -> tuple[Path | None, str]:
    candidates = []
    for value in payload.get("project_root_candidates", []) or []:
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    for value in payload.get("target_paths", []) or []:
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    project_type = str(payload.get("project_type") or "")
    if project_type == "webapp_crm":
        candidates.extend(["workspace/sandbox/crm_site", "ui"])
    elif project_type in ("generic_app", "single_file_html"):
        candidates.extend(["workspace/sandbox/app", "workspace/sandbox/project", "ui"])
    else:
        candidates.extend(["workspace/sandbox/app", "workspace/sandbox/project", "workspace/sandbox/crm_site", "ui"])
    seen = set()
    for rel in candidates:
        if rel in seen:
            continue
        seen.add(rel)
        abs_path = _safe_workspace_path(rel)
        if not abs_path or not abs_path.exists():
            continue
        return (abs_path if abs_path.is_dir() else abs_path.parent), rel
    return None, ""

def _scan_preview_eligibility(project_root: Path) -> dict:
    blocklist = _load_preview_blocklist()
    package_json = project_root / "package.json"
    requirements_txt = project_root / "requirements.txt"
    pyproject_toml = project_root / "pyproject.toml"
    result = {
        "project_type": "unknown",
        "status": "unknown",
        "blocked_dependencies": [],
        "manifest_path": "",
    }

    if package_json.exists():
        result["project_type"] = "node"
        result["manifest_path"] = str(package_json)
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            deps = set()
            for key in ["dependencies", "devDependencies"]:
                section = data.get(key, {})
                if isinstance(section, dict):
                    deps.update(str(name).lower() for name in section.keys())
            blocked = sorted(name for name in deps if name in set(blocklist.get("node", [])))
            result["blocked_dependencies"] = blocked
            result["status"] = "ineligible" if blocked else "eligible"
        except Exception as err:
            result["status"] = "unknown"
            result["scan_error"] = f"package.json parse failed: {err}"
        return result

    if requirements_txt.exists() or pyproject_toml.exists():
        result["project_type"] = "python"
        manifest = requirements_txt if requirements_txt.exists() else pyproject_toml
        result["manifest_path"] = str(manifest)
        deps = set()
        deps.update(_parse_requirements_names(requirements_txt) if requirements_txt.exists() else [])
        deps.update(_parse_pyproject_names(pyproject_toml) if pyproject_toml.exists() else [])
        blocked = sorted(name for name in deps if name in set(blocklist.get("python", [])))
        result["blocked_dependencies"] = blocked
        result["status"] = "ineligible" if blocked else "eligible"
        return result

    static_markers = list(project_root.glob("*.html")) + list(project_root.glob("*.css")) + list(project_root.glob("*.js"))
    if static_markers:
        result["project_type"] = "static"
        result["status"] = "eligible"
        result["manifest_path"] = str(static_markers[0])
        return result

    return result

def _render_extract_preview_url(service_payload: dict) -> str:
    if not isinstance(service_payload, dict):
        return ""
    for key in ["url", "preview_url", "serviceUrl"]:
        value = service_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = service_payload.get("serviceDetails")
    if isinstance(nested, dict):
        return _render_extract_preview_url(nested)
    service_obj = service_payload.get("service")
    if isinstance(service_obj, dict):
        return _render_extract_preview_url(service_obj)
    return ""

def _render_trigger_existing_service(service_id: str, api_base: str) -> dict:
    headers = {
        "Authorization": f"Bearer {RENDER_API_KEY}",
        "Content-Type": "application/json",
    }
    deploy_url = f"{api_base.rstrip('/')}/services/{service_id}/deploys"
    deploy_resp = requests.post(deploy_url, headers=headers, json={}, timeout=30)
    body = {}
    try:
        body = deploy_resp.json()
    except Exception:
        body = {"raw": deploy_resp.text[:400]}
    if deploy_resp.status_code < 200 or deploy_resp.status_code >= 300:
        return {"ok": False, "error": f"render deploy trigger failed: http {deploy_resp.status_code}", "response": body}

    service_resp = requests.get(f"{api_base.rstrip('/')}/services/{service_id}", headers=headers, timeout=30)
    service_body = {}
    try:
        service_body = service_resp.json()
    except Exception:
        service_body = {}
    return {
        "ok": True,
        "deploy_id": body.get("id") if isinstance(body, dict) else None,
        "preview_url": _render_extract_preview_url(service_body),
        "response": body,
    }

def _load_preview_registry() -> dict:
    try:
        if PREVIEW_REGISTRY_PATH.exists():
            data = json.loads(PREVIEW_REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                items = data.get("items", [])
                return {"items": items if isinstance(items, list) else []}
    except Exception:
        pass
    return {"items": []}

def _save_preview_registry(registry: dict):
    PREVIEW_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def _kill_preview_process(pid: int):
    if pid <= 0:
        return
    try:
        os.killpg(pid, 15)
    except Exception:
        try:
            os.kill(pid, 15)
        except Exception:
            pass

def _cleanup_expired_preview_runtimes(now_ts: float | None = None) -> dict:
    registry = _load_preview_registry()
    now_ts = float(now_ts or time.time())
    kept = []
    removed = []
    for item in registry.get("items", []):
        if not isinstance(item, dict):
            continue
        pid = int(item.get("pid") or 0)
        expires_at_ts = float(item.get("expires_at_ts") or 0)
        should_remove = False
        reason = ""
        if expires_at_ts and expires_at_ts <= now_ts:
            _kill_preview_process(pid)
            should_remove = True
            reason = "expired"
        elif pid and not _is_pid_alive(pid):
            should_remove = True
            reason = "dead_process"
        if should_remove:
            removed.append({
                "run_id": str(item.get("run_id") or ""),
                "port": int(item.get("port") or 0),
                "pid": pid,
                "reason": reason,
            })
        else:
            kept.append(item)
    registry["items"] = kept
    _save_preview_registry(registry)
    return {"removed": removed, "kept_count": len(kept)}

def _is_port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", int(port)))
            return True
    except OSError:
        return False

def _allocate_preview_port(registry: dict) -> int | None:
    active_ports = {
        int(item.get("port") or 0)
        for item in registry.get("items", [])
        if isinstance(item, dict) and int(item.get("port") or 0) > 0 and _is_pid_alive(int(item.get("pid") or 0))
    }
    for port in range(PREVIEW_PORT_START, PREVIEW_PORT_END + 1):
        if port in active_ports:
            continue
        if _is_port_available(port):
            return port
    return None

def _read_json_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _detect_local_preview_contract(project_root: Path, eligibility: dict) -> dict:
    project_type = str(eligibility.get("project_type") or "unknown")
    index_html = project_root / "index.html"
    package_json = project_root / "package.json"
    if index_html.exists():
        return {
            "mode": "static",
            "command": ["python", "-m", "http.server"],
            "command_source": "index.html",
            "working_dir": str(project_root),
        }

    if project_type == "node" and package_json.exists():
        package_data = _read_json_file(package_json)
        scripts = package_data.get("scripts", {}) if isinstance(package_data, dict) else {}
        if isinstance(scripts, dict):
            for script_name in ["preview", "start"]:
                if isinstance(scripts.get(script_name), str) and scripts.get(script_name).strip():
                    npm_cmd = shutil.which("npm") or shutil.which("npm.cmd")
                    if npm_cmd:
                        return {
                            "mode": "node_script",
                            "command": [npm_cmd, "run", script_name],
                            "command_source": f"package.json:scripts.{script_name}",
                            "working_dir": str(project_root),
                        }
        node_cmd = shutil.which("node")
        for entry in ["server.js", "app.js", "main.js"]:
            entry_path = project_root / entry
            if node_cmd and entry_path.exists():
                return {
                    "mode": "node_entry",
                    "command": [node_cmd, entry],
                    "command_source": entry,
                    "working_dir": str(project_root),
                }

    if project_type == "python":
        for entry in ["app.py", "server.py", "main.py"]:
            entry_path = project_root / entry
            if entry_path.exists():
                return {
                    "mode": "python_entry",
                    "command": ["python", entry],
                    "command_source": entry,
                    "working_dir": str(project_root),
                }

    return {"mode": "unknown", "command": [], "command_source": "", "working_dir": str(project_root)}

def _wait_for_preview_http(port: int, timeout_s: int) -> bool:
    deadline = time.time() + max(1, int(timeout_s))
    url = f"http://127.0.0.1:{int(port)}/"
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=1)
            if resp.status_code < 500:
                return True
        except Exception:
            time.sleep(0.5)
    return False

def _launch_local_preview(project_root: Path, payload: dict, preview_dir: Path | None, eligibility: dict) -> dict:
    cleanup = _cleanup_expired_preview_runtimes()
    registry = _load_preview_registry()
    contract = _detect_local_preview_contract(project_root, eligibility)
    if contract.get("mode") == "unknown":
        return {
            "ok": False,
            "error": "could not detect a deterministic local preview startup contract",
            "cleanup": cleanup,
            "contract": contract,
        }

    port = _allocate_preview_port(registry)
    if not port:
        return {
            "ok": False,
            "error": f"no preview port available in configured range {PREVIEW_PORT_START}-{PREVIEW_PORT_END}",
            "cleanup": cleanup,
            "contract": contract,
        }

    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = preview_dir / f"preview_stdout_{port}.log" if preview_dir else None
    stderr_path = preview_dir / f"preview_stderr_{port}.log" if preview_dir else None
    runtime_path = preview_dir / "preview_runtime.json" if preview_dir else None

    env = os.environ.copy()
    env["PORT"] = str(port)
    env["HOST"] = "0.0.0.0"
    env["PREVIEW_PORT"] = str(port)
    env["PREVIEW_HOST"] = PREVIEW_PUBLIC_HOST

    command = list(contract.get("command") or [])
    if contract.get("mode") == "static":
        command = command + [str(port), "--bind", "0.0.0.0"]

    stdout_handle = open(stdout_path, "a", encoding="utf-8") if stdout_path else subprocess.DEVNULL
    stderr_handle = open(stderr_path, "a", encoding="utf-8") if stderr_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(project_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
    finally:
        if hasattr(stdout_handle, "close"):
            stdout_handle.close()
        if hasattr(stderr_handle, "close"):
            stderr_handle.close()

    if not _wait_for_preview_http(port, PREVIEW_HEALTH_TIMEOUT_S):
        _kill_preview_process(proc.pid)
        return {
            "ok": False,
            "error": f"preview process failed readiness check on port {port}",
            "port": port,
            "pid": proc.pid,
            "cleanup": cleanup,
            "contract": contract,
        }

    preview_ttl_hours = max(1, int(payload.get("preview_ttl_hours") or 24))
    expires_at_ts = time.time() + preview_ttl_hours * 3600
    item = {
        "run_id": str(payload.get("run_id") or payload.get("workflow_run_id") or ""),
        "workflow_run_id": str(payload.get("workflow_run_id") or ""),
        "port": int(port),
        "pid": int(proc.pid),
        "mode": str(contract.get("mode") or ""),
        "project_root": str(project_root),
        "command": command,
        "command_source": str(contract.get("command_source") or ""),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "expires_at": datetime.utcfromtimestamp(expires_at_ts).isoformat() + "Z",
        "expires_at_ts": expires_at_ts,
    }
    registry_items = []
    for existing in registry.get("items", []):
        if isinstance(existing, dict) and str(existing.get("run_id") or "") == item["run_id"]:
            _kill_preview_process(int(existing.get("pid") or 0))
            continue
        registry_items.append(existing)
    registry_items.append(item)
    registry["items"] = registry_items
    _save_preview_registry(registry)
    if runtime_path:
        runtime_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "port": port,
        "pid": proc.pid,
        "preview_url": f"http://{PREVIEW_PUBLIC_HOST}:{port}",
        "preview_host": PREVIEW_PUBLIC_HOST,
        "mode": contract.get("mode"),
        "command_source": contract.get("command_source"),
        "cleanup": cleanup,
        "runtime_path": str(runtime_path) if runtime_path else "",
    }

def deploy_preview(payload: dict):
    artifact_root_rel = str(payload.get("artifact_root") or "").strip()
    artifact_root = _safe_workspace_path(artifact_root_rel) if artifact_root_rel else None
    if artifact_root:
        artifact_root.mkdir(parents=True, exist_ok=True)
    preview_dir = artifact_root / "preview" if artifact_root else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    project_root, project_root_source = _find_preview_project_root(payload)
    eligibility = _scan_preview_eligibility(project_root) if project_root else {
        "project_type": "unknown",
        "status": "unknown",
        "blocked_dependencies": [],
        "manifest_path": "",
    }

    result = {
        "ok": True,
        "preview_status": "skipped",
        "preview_url": None,
        "fallback_reason": "",
        "deployment_result_path": "preview/deployment_result.json" if preview_dir else "",
        "project_root": str(project_root) if project_root else "",
        "project_root_source": project_root_source,
        "eligibility": eligibility,
        "preview_provider": str(payload.get("preview_provider") or "local"),
        "preview_ttl_hours": int(payload.get("preview_ttl_hours") or 24),
        "render_deploy_mode": str(payload.get("render_deploy_mode") or ""),
        "render_service_id": str(payload.get("render_service_id") or ""),
    }

    if not project_root:
        result["fallback_reason"] = "project root not found under target_paths"
    elif eligibility.get("status") == "ineligible":
        blocked = ", ".join(eligibility.get("blocked_dependencies", []))
        result["preview_status"] = "ineligible"
        result["fallback_reason"] = f"preview blocked by database dependencies: {blocked}"
    elif eligibility.get("status") == "unknown":
        result["fallback_reason"] = "preview eligibility scan could not classify the project"
    else:
        preview_provider = str(payload.get("preview_provider") or "local").strip().lower()
        if preview_provider in ["", "local", "localhost"]:
            deploy = _launch_local_preview(project_root, payload, preview_dir, eligibility)
            if deploy.get("ok"):
                result["preview_status"] = "deployed"
                result["preview_url"] = deploy.get("preview_url") or None
                result["deployment"] = deploy
            else:
                result["preview_status"] = "failed"
                result["fallback_reason"] = str(deploy.get("error") or "local preview launch failed")
                result["deployment"] = deploy
        elif preview_provider == "render":
            deploy_mode = str(payload.get("render_deploy_mode") or "").strip().lower()
            render_service_id = str(payload.get("render_service_id") or "").strip()
            render_api_base = str(payload.get("render_api_base") or RENDER_API_BASE_URL).strip() or RENDER_API_BASE_URL
            if not RENDER_API_KEY:
                result["fallback_reason"] = "RENDER_API_KEY is missing"
            elif deploy_mode != "existing_service_redeploy":
                result["fallback_reason"] = "render_deploy_mode is not configured for a safe deploy path"
            elif not render_service_id:
                result["fallback_reason"] = "render_service_id is missing"
            else:
                deploy = _render_trigger_existing_service(render_service_id, render_api_base)
                if deploy.get("ok"):
                    result["preview_status"] = "deployed"
                    result["preview_url"] = deploy.get("preview_url") or None
                    result["deployment"] = deploy
                    if not result["preview_url"]:
                        result["fallback_reason"] = "deploy triggered but preview URL was not returned by Render"
                else:
                    result["preview_status"] = "failed"
                    result["fallback_reason"] = str(deploy.get("error") or "render deployment failed")
                    result["deployment"] = deploy
        else:
            result["fallback_reason"] = f"preview provider '{preview_provider}' is unsupported"

    if preview_dir:
        result_path = preview_dir / "deployment_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

# ─────────────────────────────────────────────────────────────
# SECTION: User Profile  (A-05)
# ─────────────────────────────────────────────────────────────
_USER_PROFILE_PATH = Path("/app/quant_trading/Project_optimized/user_profile.json")
_USER_PROFILE_DEFAULTS = {
    "capital_base_jpy": 400000,
    "max_position_pct": 0.35,
    "stop_loss_pct": 0.08,
    "take_profit_pct": 0.20,
    "max_positions": 5,
    "horizon_days": 20,
    "risk_profile": "medium",
    "preferred_markets": ["JP"],
    "avoid_sectors": [],
    "preferred_sectors": [],
    "notify_on_stop_loss": True,
    "notify_on_catalyst": True,
    "language": "zh",
}

def _load_user_profile() -> dict:
    try:
        if _USER_PROFILE_PATH.exists():
            data = json.loads(_USER_PROFILE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged = dict(_USER_PROFILE_DEFAULTS)
                merged.update(data)
                return merged
    except Exception:
        pass
    return dict(_USER_PROFILE_DEFAULTS)

def _save_user_profile(profile: dict):
    try:
        _USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _USER_PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[user_profile] save failed: {e}")

def portfolio_set_preference(payload: dict):
    """Update one or more user profile fields. E.g. {\"stop_loss_pct\": 0.10}"""
    profile = _load_user_profile()
    allowed_keys = set(_USER_PROFILE_DEFAULTS.keys())
    updated = []
    for k, v in payload.items():
        if k in allowed_keys:
            profile[k] = v
            updated.append(k)
    _save_user_profile(profile)
    return {"ok": True, "updated": updated, "profile": profile}


# ─────────────────────────────────────────────────────────────
# SECTION: Watchlist Alias Map  (B-04)
# ─────────────────────────────────────────────────────────────
_ALIAS_PATH = Path("/app/quant_trading/Project_optimized/watchlist_alias.json")

def _load_alias_map() -> dict:
    """Return {ticker: [alias1, alias2, ...]} mapping."""
    try:
        if _ALIAS_PATH.exists():
            data = json.loads(_ALIAS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def _build_reverse_alias(alias_map: dict) -> dict:
    """Return {alias_lower: ticker} reverse map."""
    rev = {}
    for ticker, aliases in alias_map.items():
        for a in aliases:
            rev[str(a).lower()] = ticker
        # Also map the 4-digit code (e.g. "5020" → "5020.T")
        code = ticker.replace(".T", "").replace(".t", "")
        rev[code] = ticker
    return rev


# ─────────────────────────────────────────────────────────────
# SECTION: Cost Price & Position Snapshot  (A-01, A-02)
# ─────────────────────────────────────────────────────────────
_DB_PATH = QUANT_DB_PATH

def _ensure_positions_snapshot_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asof TEXT NOT NULL,
            symbol TEXT NOT NULL,
            net_qty REAL NOT NULL,
            avg_cost REAL,
            current_price REAL,
            unrealized_pnl REAL,
            unrealized_pnl_pct REAL,
            stop_loss REAL,
            take_profit REAL,
            days_held INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(asof, symbol)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asof TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            alpha_score REAL,
            price_at_signal REAL,
            price_5d REAL,
            price_20d REAL,
            ret_5d REAL,
            ret_20d REAL,
            ic_contrib REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

def _calc_avg_cost(symbol: str) -> float | None:
    """Weighted-average cost for current open position (WAVG method)."""
    if not _DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT side, qty, price FROM fills WHERE UPPER(symbol)=? ORDER BY ts ASC, id ASC",
            (symbol.upper(),),
        )
        rows = cur.fetchall()
        conn.close()
        total_qty = 0.0
        total_cost = 0.0
        for side, qty, price in rows:
            qty = float(qty or 0)
            price = float(price or 0)
            if side.upper() == "BUY":
                total_cost += qty * price
                total_qty += qty
            elif side.upper() == "SELL":
                sell_qty = min(qty, total_qty)
                if total_qty > 0:
                    cost_per_share = total_cost / total_qty
                    total_cost -= sell_qty * cost_per_share
                total_qty -= sell_qty
                if total_qty <= 0:
                    total_qty = 0.0
                    total_cost = 0.0
        if total_qty <= 0:
            return None
        return total_cost / total_qty
    except Exception:
        return None

def _calc_days_held(symbol: str) -> int:
    """Days since first BUY that has net-positive open position."""
    if not _DB_PATH.exists():
        return 0
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT asof FROM fills WHERE UPPER(symbol)=? AND UPPER(side)='BUY' ORDER BY ts ASC LIMIT 1",
            (symbol.upper(),),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return 0
        first_date = datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date()
        return (datetime.utcnow().date() - first_date).days
    except Exception:
        return 0

def _write_positions_snapshot(positions: list[dict], date_str: str | None = None):
    """Upsert today's positions snapshot into DB."""
    if not _DB_PATH.exists():
        return
    asof = date_str or _now_date_str()
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        _ensure_positions_snapshot_table(conn)
        for p in positions:
            conn.execute("""
                INSERT INTO positions_snapshot
                    (asof, symbol, net_qty, avg_cost, current_price,
                     unrealized_pnl, unrealized_pnl_pct, stop_loss, take_profit, days_held)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(asof, symbol) DO UPDATE SET
                    net_qty=excluded.net_qty, avg_cost=excluded.avg_cost,
                    current_price=excluded.current_price,
                    unrealized_pnl=excluded.unrealized_pnl,
                    unrealized_pnl_pct=excluded.unrealized_pnl_pct,
                    stop_loss=excluded.stop_loss, take_profit=excluded.take_profit,
                    days_held=excluded.days_held
            """, (
                asof, p["symbol"], p.get("net_qty", 0),
                p.get("avg_cost"), p.get("current_price"),
                p.get("unrealized_pnl"), p.get("unrealized_pnl_pct"),
                p.get("stop_loss"), p.get("take_profit"),
                p.get("days_held", 0),
            ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[positions_snapshot] write failed: {e}")


# ─────────────────────────────────────────────────────────────
# SECTION: Signal Log  (D-01)
# ─────────────────────────────────────────────────────────────
def _log_signal(symbol: str, signal: str, alpha_score: float | None, price: float | None):
    """Append a signal record to signal_log for IC tracking."""
    if not _DB_PATH.exists():
        return
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        _ensure_positions_snapshot_table(conn)
        conn.execute("""
            INSERT INTO signal_log (asof, symbol, signal, alpha_score, price_at_signal)
            VALUES (?, ?, ?, ?, ?)
        """, (_now_date_str(), symbol.upper(), signal, alpha_score, price))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[signal_log] write failed: {e}")


# ─────────────────────────────────────────────────────────────
# SECTION: TDnet RSS Fetcher  (B-01)
# ─────────────────────────────────────────────────────────────
_TDNET_CATEGORY_MAP = {
    "上方修正": "earnings_revision_up",
    "下方修正": "earnings_revision_down",
    "増配":     "dividend_increase",
    "特別配当": "special_dividend",
    "自己株式": "buyback",
    "自社株":   "buyback",
    "業績修正": "earnings_revision",
    "決算短信": "earnings",
    "M&A":      "ma",
    "業務提携": "partnership",
    "不祥事":   "scandal",
    "行政処分": "regulatory",
    "減配":     "dividend_cut",
}
_TDNET_ALPHA_STRENGTH = {
    "earnings_revision_up":   5,
    "earnings_revision_down": 5,
    "dividend_increase":      4,
    "special_dividend":       4,
    "buyback":                4,
    "earnings":               3,
    "ma":                     3,
    "partnership":            2,
    "dividend_cut":           4,
    "scandal":                4,
    "regulatory":             4,
    "earnings_revision":      3,
}

def _classify_tdnet_announcement(title: str) -> str:
    for kw, cat in _TDNET_CATEGORY_MAP.items():
        if kw in title:
            return cat
    return "general"

def _fetch_tdnet_announcements(symbols: list | None = None, lookback_hours: float = 24.0) -> list[dict]:
    """
    Fetch TDnet announcements via multiple sources (priority order):
    1. Kabutan market news RSS (Japanese, reliable)
    2. Google News RSS with Japanese keywords
    3. GDELT fallback
    Returns list of {code, title, category, alpha_strength, published_at, url, source}
    """
    results = []
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

    # Source 1: Kabutan news RSS
    kabutan_urls = [
        "https://kabutan.jp/news/rss/?category=marketnews",
        "https://kabutan.jp/news/rss/?category=disclosure",
    ]
    for rss_url in kabutan_urls:
        try:
            resp = requests.get(rss_url, timeout=8,
                                headers={"User-Agent": "Mozilla/5.0 (compatible; NexusBot/1.0)"})
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.iter("item"):
                    title = (item.findtext("title") or "").strip()
                    link  = (item.findtext("link")  or "").strip()
                    pub   = (item.findtext("pubDate") or "").strip()
                    if not title:
                        continue
                    try:
                        pub_dt = parsedate_to_datetime(pub).replace(tzinfo=None) if pub else None
                    except Exception:
                        pub_dt = None
                    if pub_dt and pub_dt < cutoff:
                        continue
                    # Extract 4-digit code from title/link if present
                    code_match = re.search(r'[＜<](\d{4})[＞>]', title) or re.search(r'/stock/(\d{4})/', link)
                    code = (code_match.group(1) + ".T") if code_match else None
                    cat = _classify_tdnet_announcement(title)
                    results.append({
                        "code": code,
                        "title": title,
                        "category": cat,
                        "alpha_strength": _TDNET_ALPHA_STRENGTH.get(cat, 1),
                        "published_at": pub_dt.isoformat() if pub_dt else "",
                        "url": link,
                        "source": "kabutan",
                    })
        except Exception:
            pass
        if results:
            break

    # Source 2: Google News RSS (Japanese keywords)
    if not results:
        jp_keywords = ["決算 OR 増配 OR 自社株買い OR 業績修正 OR 上方修正 OR 下方修正"]
        for kw in jp_keywords:
            try:
                rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(kw)}&hl=ja&gl=JP&ceid=JP:ja"
                resp = requests.get(rss_url, timeout=8,
                                    headers={"User-Agent": "Mozilla/5.0 (compatible; NexusBot/1.0)"})
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.iter("item"):
                        title = (item.findtext("title") or "").strip()
                        link  = (item.findtext("link")  or "").strip()
                        pub   = (item.findtext("pubDate") or "").strip()
                        if not title:
                            continue
                        try:
                            pub_dt = parsedate_to_datetime(pub).replace(tzinfo=None) if pub else None
                        except Exception:
                            pub_dt = None
                        if pub_dt and pub_dt < cutoff:
                            continue
                        code_match = re.search(r'[＜<](\d{4})[＞>]', title)
                        code = (code_match.group(1) + ".T") if code_match else None
                        cat = _classify_tdnet_announcement(title)
                        results.append({
                            "code": code,
                            "title": title,
                            "category": cat,
                            "alpha_strength": _TDNET_ALPHA_STRENGTH.get(cat, 1),
                            "published_at": pub_dt.isoformat() if pub_dt else "",
                            "url": link,
                            "source": "google_news_ja",
                        })
            except Exception:
                pass

    # Filter by symbols if provided
    if symbols and results:
        alias_map = _load_alias_map()
        rev = _build_reverse_alias(alias_map)
        filtered = []
        for ann in results:
            # Direct code match
            if ann.get("code") and ann["code"] in symbols:
                filtered.append(ann)
                continue
            # Alias text match in title
            title_lower = ann["title"].lower()
            for alias_lower, ticker in rev.items():
                if ticker in symbols and alias_lower in title_lower:
                    ann["code"] = ann.get("code") or ticker
                    filtered.append(ann)
                    break
        return filtered

    return results[:40]  # cap at 40 items


# ─────────────────────────────────────────────────────────────
# SECTION: Announcement → Ticker Mapping  (B-02)
# ─────────────────────────────────────────────────────────────
def _match_announcement_to_tickers(announcement: dict, watchlist_symbols: list) -> list[str]:
    """Match an announcement to one or more watchlist tickers."""
    alias_map = _load_alias_map()
    rev = _build_reverse_alias(alias_map)
    matched = []

    # 1. Direct code field
    code = announcement.get("code")
    if code and code in watchlist_symbols:
        matched.append(code)

    # 2. Text search in title
    title_lower = str(announcement.get("title", "")).lower()
    for alias_lower, ticker in rev.items():
        if ticker in watchlist_symbols and alias_lower in title_lower and ticker not in matched:
            matched.append(ticker)

    return list(dict.fromkeys(matched))


# ─────────────────────────────────────────────────────────────
# SECTION: Cross-Sectional Rank Factor  (D-03)
# ─────────────────────────────────────────────────────────────
def _compute_cross_sectional_rank(symbols: list[str]) -> dict:
    """
    Compute 20d excess-return percentile rank for each symbol within the universe.
    Returns {symbol: percentile_rank_0_to_1}
    """
    scores = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(_normalize_symbol_for_lookup(sym))
            hist = ticker.history(period="3mo", interval="1d", auto_adjust=True)
            if hist is not None and not hist.empty and len(hist) >= 21:
                close = hist["Close"].dropna()
                ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
                scores[sym] = ret20
        except Exception:
            pass
    if not scores:
        return {}
    vals = sorted(scores.values())
    n = len(vals)
    result = {}
    for sym, score in scores.items():
        rank = vals.index(score)
        result[sym] = round(rank / max(n - 1, 1), 3)
    return result


# ─────────────────────────────────────────────────────────────
# SECTION: Portfolio Position Review  (A-03)
# ─────────────────────────────────────────────────────────────
def _build_enriched_positions() -> list[dict]:
    """Fetch net positions and enrich with avg_cost, current price, P&L."""
    raw = _get_current_positions_from_fills(max_symbols=20)
    if not raw:
        return []
    profile = _load_user_profile()
    stop_pct  = float(profile.get("stop_loss_pct",  0.08))
    take_pct  = float(profile.get("take_profit_pct", 0.20))
    enriched = []
    for pos in raw:
        sym = pos["symbol"]
        net_qty = float(pos.get("net_qty", 0))
        avg_cost = _calc_avg_cost(sym)
        days_held = _calc_days_held(sym)
        # Fetch current price
        current_price = None
        try:
            q = _fetch_quote_facts(sym)
            current_price = _safe_float(q.get("price"))
        except Exception:
            pass
        # P&L
        unrealized_pnl = None
        unrealized_pnl_pct = None
        if avg_cost and current_price:
            unrealized_pnl = (current_price - avg_cost) * net_qty
            unrealized_pnl_pct = (current_price / avg_cost - 1.0) * 100.0
        stop_loss   = round(avg_cost * (1 - stop_pct), 1)  if avg_cost else None
        take_profit = round(avg_cost * (1 + take_pct), 1) if avg_cost else None
        # Signal
        try:
            metrics = _compute_quant_metrics(sym)
            signal = metrics.get("signal", "N/A")
            alpha_score = metrics.get("alpha_score")
        except Exception:
            signal, alpha_score = "N/A", None
        enriched.append({
            "symbol": sym,
            "net_qty": net_qty,
            "avg_cost": round(avg_cost, 2) if avg_cost else None,
            "current_price": round(current_price, 2) if current_price else None,
            "unrealized_pnl": round(unrealized_pnl, 0) if unrealized_pnl is not None else None,
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 2) if unrealized_pnl_pct is not None else None,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "days_held": days_held,
            "signal": signal,
            "alpha_score": alpha_score,
            "market": pos.get("market", ""),
        })
    return enriched

def portfolio_position_review(payload: dict) -> dict:
    """
    Active position review: P&L, risk flags, per-position action suggestion.
    Tool: portfolio.position_review
    """
    run_id = payload.get("run_id")
    positions = _build_enriched_positions()
    profile   = _load_user_profile()
    account   = _get_account_state_snapshot()
    stop_pct  = float(profile.get("stop_loss_pct", 0.08))
    horizon   = int(profile.get("horizon_days", 20))

    if not positions:
        return {
            "ok": True,
            "positions": [],
            "summary": f"当前无持仓。可用资金: {account.get('starting_capital', 'N/A')} {account.get('base_ccy', 'JPY')}",
            "risk_flags": [],
            "suggestions": [],
        }

    # Aggregate stats
    total_cost_value = sum(
        (p["avg_cost"] or 0) * p["net_qty"] for p in positions if p["avg_cost"]
    )
    total_market_value = sum(
        (p["current_price"] or 0) * p["net_qty"] for p in positions if p["current_price"]
    )
    total_pnl = sum(p["unrealized_pnl"] or 0 for p in positions)
    total_pnl_pct = (total_pnl / total_cost_value * 100) if total_cost_value else 0
    cash = _safe_float(account.get("starting_capital", 0)) or 0
    position_utilization_pct = (total_market_value / (total_market_value + cash) * 100) if (total_market_value + cash) > 0 else 0

    # Risk flags
    risk_flags = []
    suggestions = []
    for p in positions:
        sym = p["symbol"]
        pnl_pct = p.get("unrealized_pnl_pct")
        sig = p.get("signal", "")
        days = p.get("days_held", 0)

        # Stop loss trigger
        if pnl_pct is not None and pnl_pct <= -stop_pct * 100:
            risk_flags.append(f"⚠️ {sym}: 浮亏 {pnl_pct:.1f}%，触及止损线（-{stop_pct*100:.0f}%）")
            suggestions.append({"symbol": sym, "action": "STOP_LOSS", "reason": f"浮亏超止损阈值 {pnl_pct:.1f}%"})
        elif sig in ("Underweight", "Strong Sell"):
            risk_flags.append(f"⚠️ {sym}: 信号已转为 {sig}")
            suggestions.append({"symbol": sym, "action": "REDUCE", "reason": f"信号转空: {sig}"})
        elif days > horizon:
            risk_flags.append(f"⏰ {sym}: 已持仓 {days} 天，超出目标周期 {horizon} 天")
            suggestions.append({"symbol": sym, "action": "REVIEW", "reason": f"持仓超期 {days}d > {horizon}d 目标"})
        elif sig in ("Strong Buy", "Overweight"):
            suggestions.append({"symbol": sym, "action": "HOLD", "reason": f"信号强势 ({sig})，继续持有"})
        else:
            suggestions.append({"symbol": sym, "action": "HOLD", "reason": "信号中性，暂无操作"})

    # LLM narrative
    pos_text = "\n".join(
        f"- {p['symbol']}: {p['net_qty']}股 | 成本{p['avg_cost']} | 现价{p['current_price']} | "
        f"浮盈{p['unrealized_pnl_pct']:.1f}% | 持{p['days_held']}天 | 信号:{p['signal']}"
        for p in positions if p.get("avg_cost")
    )
    flag_text = "\n".join(risk_flags) if risk_flags else "无风险预警"
    llm_summary = ""
    try:
        system = (
            "你是资深量化交易员。请基于以下持仓数据，用中文给出简洁的综合操作建议（3-5句话）。"
            "重点关注：1)整体仓位是否合理 2)有无需要立即处理的风险仓位 3)今日需关注的要点。"
            "直接给结论，不要废话。"
        )
        user_msg = f"持仓:\n{pos_text}\n\n风险预警:\n{flag_text}\n总浮盈:{total_pnl:.0f}JPY ({total_pnl_pct:.1f}%)"
        llm_summary = get_llm_response(system, user_msg, model=QUANT_LLM_MODEL) or ""
    except Exception:
        pass

    # Write snapshot
    _write_positions_snapshot(positions)

    # Log signals
    for p in positions:
        _log_signal(p["symbol"], p.get("signal", "N/A"), p.get("alpha_score"), p.get("current_price"))

    summary_header = (
        f"持仓 {len(positions)} 只 | 市值 {total_market_value:,.0f} JPY | "
        f"总浮盈 {total_pnl:+,.0f} JPY ({total_pnl_pct:+.1f}%) | 仓位使用率 {position_utilization_pct:.0f}%"
    )

    return {
        "ok": True,
        "positions": positions,
        "summary": summary_header,
        "total_market_value_jpy": round(total_market_value, 0),
        "total_unrealized_pnl_jpy": round(total_pnl, 0),
        "total_unrealized_pnl_pct": round(total_pnl_pct, 2),
        "position_utilization_pct": round(position_utilization_pct, 1),
        "risk_flags": risk_flags,
        "suggestions": suggestions,
        "llm_narrative": llm_summary,
    }


# ─────────────────────────────────────────────────────────────
# SECTION: Midday P&L Flash  (A-04)
# ─────────────────────────────────────────────────────────────
def portfolio_midday_pnl(payload: dict) -> dict:
    """Lightweight midday P&L flash. No LLM call. Target < 10s."""
    positions = _build_enriched_positions()
    if not positions:
        return {"ok": True, "message": "当前无持仓"}
    lines = ["【午间持仓快报 12:00 JST】"]
    total_pnl = 0.0
    for p in positions:
        sym   = p["symbol"]
        qty   = p["net_qty"]
        cost  = p.get("avg_cost")
        price = p.get("current_price")
        pnl   = p.get("unrealized_pnl") or 0
        pct   = p.get("unrealized_pnl_pct")
        total_pnl += pnl
        arrow = "▲" if (pct or 0) >= 0 else "▼"
        pct_str = f"{pct:+.1f}%" if pct is not None else "N/A"
        lines.append(f"  {sym}  {qty:.0f}股  ¥{price or 'N/A'}  {arrow}{pct_str}  ({pnl:+.0f}JPY)")
    lines.append(f"  ─────────────────────")
    lines.append(f"  总浮盈: {total_pnl:+,.0f} JPY")
    return {"ok": True, "positions": positions, "total_pnl_jpy": round(total_pnl, 0), "message": "\n".join(lines)}


# ─────────────────────────────────────────────────────────────
# SECTION: Event Alert  (B-03)
# ─────────────────────────────────────────────────────────────
def quant_event_alert(payload: dict) -> dict:
    """
    Cross-reference latest TDnet announcements with current positions.
    Generates targeted alerts for held securities.
    """
    lookback_hours = float(payload.get("lookback_hours", 24))
    positions = _build_enriched_positions()
    if not positions:
        return {"ok": True, "alerts": [], "message": "当前无持仓，无需预警"}

    held_symbols = [p["symbol"] for p in positions]
    announcements = _fetch_tdnet_announcements(symbols=held_symbols, lookback_hours=lookback_hours)

    pos_map = {p["symbol"]: p for p in positions}
    alerts = []
    for ann in announcements:
        matched = _match_announcement_to_tickers(ann, held_symbols)
        for sym in matched:
            pos = pos_map.get(sym, {})
            pnl_pct = pos.get("unrealized_pnl_pct")
            stop    = pos.get("stop_loss")
            cat     = ann.get("category", "general")
            strength = ann.get("alpha_strength", 1)
            # Positive or negative catalyst
            if cat in ("earnings_revision_up", "dividend_increase", "special_dividend", "buyback"):
                icon = "✅"
                action_hint = "催化剂利好，可考虑持有/加仓"
            elif cat in ("earnings_revision_down", "dividend_cut", "scandal", "regulatory"):
                icon = "⚠️"
                action_hint = f"利空公告，确认止损线 ¥{stop}，决策前勿冲动"
            else:
                icon = "📋"
                action_hint = "关注后续市场反应"
            pnl_str = f"{pnl_pct:+.1f}%" if pnl_pct is not None else "N/A"
            alerts.append({
                "symbol": sym,
                "icon": icon,
                "title": ann["title"],
                "category": cat,
                "alpha_strength": strength,
                "action_hint": action_hint,
                "current_pnl_pct": pnl_str,
                "stop_loss": stop,
                "url": ann.get("url", ""),
                "published_at": ann.get("published_at", ""),
            })

    if not alerts:
        return {"ok": True, "alerts": [], "message": "持仓标的无相关公告（过去%dh）" % int(lookback_hours)}

    # LLM narrative
    alert_text = "\n".join(
        f"[{a['icon']}] {a['symbol']}: {a['title']} ({a['category']}) → {a['action_hint']}"
        for a in alerts
    )
    llm_msg = ""
    try:
        system = "你是资深量化交易员。根据以下持仓相关公告，给出简洁中文操作建议（不超过150字）。"
        llm_msg = get_llm_response(system, alert_text, model=QUANT_LLM_MODEL) or ""
    except Exception:
        pass

    return {"ok": True, "alerts": alerts, "llm_narrative": llm_msg, "announcement_count": len(announcements)}


# ─────────────────────────────────────────────────────────────
# SECTION: Morning Brief  (C-01)
# ─────────────────────────────────────────────────────────────
def _fetch_market_overview() -> dict:
    """Fetch N225, Topix, VIX quick stats."""
    indices = {"N225": "^N225", "TOPIX": "1306.T", "VIX": "^VIX", "USDJPY": "JPY=X"}
    result = {}
    for name, sym in indices.items():
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info
            price = _safe_float(getattr(fi, "last_price", None) or fi.get("last_price"))
            prev  = _safe_float(getattr(fi, "previous_close", None) or fi.get("previous_close"))
            chg_pct = ((price / prev - 1) * 100) if price and prev else None
            result[name] = {"price": round(price, 2) if price else None,
                            "change_pct": round(chg_pct, 2) if chg_pct else None}
        except Exception:
            result[name] = {"price": None, "change_pct": None}
    return result

def portfolio_morning_brief(payload: dict) -> dict:
    """
    Daily morning brief (08:45 JST):
    Market overview + TDnet announcements + position status + today's focus.
    """
    run_id   = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()

    # 1. Market overview
    overview = _fetch_market_overview()

    # 2. TDnet announcements (yesterday + today)
    all_announcements = _fetch_tdnet_announcements(lookback_hours=20)
    positions = _build_enriched_positions()
    held_symbols = [p["symbol"] for p in positions]

    # Split: position-related vs general
    pos_announcements = []
    general_announcements = []
    for ann in all_announcements:
        matched = _match_announcement_to_tickers(ann, held_symbols)
        if matched:
            ann["matched_holdings"] = matched
            pos_announcements.append(ann)
        else:
            general_announcements.append(ann)

    # 3. Position snapshot
    total_pnl = sum(p.get("unrealized_pnl") or 0 for p in positions)

    # 4. Build context for LLM
    mkt_text = " | ".join(
        f"{k}: {v.get('price','N/A')} ({v.get('change_pct',0):+.1f}%)" if v.get("change_pct") is not None
        else f"{k}: N/A"
        for k, v in overview.items()
    )
    ann_text = "\n".join(f"- [{a['category']}] {a['title']}" for a in (pos_announcements + general_announcements[:5]))
    pos_text = "\n".join(
        f"- {p['symbol']}: {p['net_qty']}股 浮盈{p.get('unrealized_pnl_pct',0):+.1f}% 信号:{p.get('signal','N/A')}"
        for p in positions
    ) or "当前无持仓"

    focus_items = []
    try:
        system = (
            "你是资深日本股市量化交易员。基于今日市场环境和持仓状态，生成今日3个最重要的关注事项（每条不超过30字，纯中文）。"
            "直接输出3行，不要编号和多余格式。"
        )
        user_msg = f"市场: {mkt_text}\n公告:\n{ann_text}\n持仓:\n{pos_text}"
        llm_out = get_llm_response(system, user_msg, model=QUANT_LLM_MODEL) or ""
        focus_items = [l.strip("•-– ").strip() for l in llm_out.splitlines() if l.strip()][:3]
    except Exception:
        pass
    if not focus_items:
        focus_items = ["关注日元汇率变动对出口股影响", "注意N225期货与现货价差", "确认持仓止损线是否需要调整"]

    # Build readable output
    pos_lines = []
    for p in positions:
        pct = p.get("unrealized_pnl_pct")
        arrow = "▲" if (pct or 0) >= 0 else "▼"
        pos_lines.append(
            f"  {p['symbol']}  {p['net_qty']:.0f}股  ¥{p.get('current_price','N/A')}  "
            f"{arrow}{pct:+.1f}%  信号:{p.get('signal','N/A')}"
            if pct is not None else f"  {p['symbol']}  {p['net_qty']:.0f}股  信号:{p.get('signal','N/A')}"
        )

    output_lines = [
        f"【早间情报 {date_str} 08:45 JST】",
        "",
        "▌市场环境",
    ]
    for k, v in overview.items():
        chg = v.get("change_pct")
        output_lines.append(
            f"  {k}: {v.get('price','N/A')}  {chg:+.2f}%"
            if chg is not None else f"  {k}: N/A"
        )
    output_lines.append("")
    output_lines.append("▌TDnet公告（持仓相关）")
    if pos_announcements:
        for a in pos_announcements[:5]:
            output_lines.append(f"  [{a['icon'] if 'icon' in a else '📋'}] {','.join(a.get('matched_holdings',[]))}: {a['title']}")
    else:
        output_lines.append("  ✅ 持仓标的今日无重要公告")
    output_lines.append("")
    output_lines.append("▌持仓状态")
    if pos_lines:
        output_lines.extend(pos_lines)
        output_lines.append(f"  总浮盈: {total_pnl:+,.0f} JPY")
    else:
        output_lines.append("  当前无持仓")
    output_lines.append("")
    output_lines.append("▌今日3项关注")
    for i, f in enumerate(focus_items, 1):
        output_lines.append(f"  {i}. {f}")

    briefing_text = "\n".join(output_lines)

    return {
        "ok": True,
        "date": date_str,
        "market_overview": overview,
        "position_announcements": pos_announcements,
        "general_announcements": general_announcements[:10],
        "positions": positions,
        "focus_items": focus_items,
        "briefing": briefing_text,
    }


# ─────────────────────────────────────────────────────────────
# SECTION: Post-Close Review  (C-02)
# ─────────────────────────────────────────────────────────────
def portfolio_post_close(payload: dict) -> dict:
    """
    Post-close review (15:35 JST):
    Day's P&L change, TDnet announcements, tomorrow's plan.
    """
    run_id   = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()

    positions = _build_enriched_positions()
    all_announcements = _fetch_tdnet_announcements(lookback_hours=10)  # Today's announcements

    # Load yesterday's snapshot for comparison
    prev_snapshot = {}
    if _DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            _ensure_positions_snapshot_table(conn)
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol, current_price, unrealized_pnl_pct
                FROM positions_snapshot
                WHERE asof < ? ORDER BY asof DESC
            """, (date_str,))
            for row in cur.fetchall():
                if row[0] not in prev_snapshot:
                    prev_snapshot[row[0]] = {"price": row[1], "pnl_pct": row[2]}
            conn.close()
        except Exception:
            pass

    # Day change vs yesterday
    day_changes = []
    for p in positions:
        sym = p["symbol"]
        prev = prev_snapshot.get(sym, {})
        prev_price = prev.get("price")
        curr_price = p.get("current_price")
        day_chg_pct = ((curr_price / prev_price - 1) * 100) if (curr_price and prev_price) else None
        day_changes.append({
            "symbol": sym,
            "prev_price": prev_price,
            "current_price": curr_price,
            "day_change_pct": round(day_chg_pct, 2) if day_chg_pct is not None else None,
            "unrealized_pnl_pct": p.get("unrealized_pnl_pct"),
        })

    # Save today's snapshot
    _write_positions_snapshot(positions, date_str)

    # LLM: tomorrow's plan
    day_text = "\n".join(
        f"- {d['symbol']}: 今日{d['day_change_pct']:+.1f}% 总浮盈{d['unrealized_pnl_pct']:+.1f}%"
        if d['day_change_pct'] is not None else f"- {d['symbol']}: 价格N/A"
        for d in day_changes
    ) or "当前无持仓"
    ann_text = "\n".join(f"- {a['title']}" for a in all_announcements[:8]) or "今日无重要公告"

    tomorrow_plan = []
    try:
        system = (
            "你是资深量化交易员。基于今日持仓收益和公告，生成明日2-3条可执行计划（每条不超过40字，中文）。"
            "直接输出，不要编号和多余格式。"
        )
        user_msg = f"今日持仓表现:\n{day_text}\n\n今日公告:\n{ann_text}"
        llm_out = get_llm_response(system, user_msg, model=QUANT_LLM_MODEL) or ""
        tomorrow_plan = [l.strip("•-– ").strip() for l in llm_out.splitlines() if l.strip()][:3]
    except Exception:
        pass
    if not tomorrow_plan:
        tomorrow_plan = ["关注持仓标的开盘动向", "复核止损线是否需要调整"]

    output_lines = [f"【收盘复盘 {date_str} 15:35 JST】", ""]
    output_lines.append("▌今日持仓涨跌")
    for d in day_changes:
        chg = d.get("day_change_pct")
        pnl = d.get("unrealized_pnl_pct")
        arrow = "▲" if (chg or 0) >= 0 else "▼"
        output_lines.append(
            f"  {d['symbol']}  今日{arrow}{chg:+.1f}%  总浮盈{pnl:+.1f}%"
            if chg is not None and pnl is not None
            else f"  {d['symbol']}  数据不完整"
        )
    output_lines.extend(["", "▌今日重要公告"])
    if all_announcements:
        for a in all_announcements[:5]:
            output_lines.append(f"  [{a.get('category','?')}] {a['title']}")
    else:
        output_lines.append("  今日无重要公告")
    output_lines.extend(["", "▌明日计划"])
    for i, p in enumerate(tomorrow_plan, 1):
        output_lines.append(f"  {i}. {p}")

    return {
        "ok": True,
        "date": date_str,
        "day_changes": day_changes,
        "announcements": all_announcements[:10],
        "tomorrow_plan": tomorrow_plan,
        "review": "\n".join(output_lines),
    }


# ─────────────────────────────────────────────────────────────
# SECTION: Signal Backfill & IC Tracking  (D-02)
# ─────────────────────────────────────────────────────────────
def quant_signal_backfill(payload: dict) -> dict:
    """
    Backfill ret_5d / ret_20d for past signals and compute rolling IC.
    Run weekly to evaluate signal quality.
    """
    if not _DB_PATH.exists():
        return {"ok": False, "error": "DB not found"}
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        _ensure_positions_snapshot_table(conn)
        cur = conn.cursor()
        # Fetch signals with missing ret_20d
        cur.execute("""
            SELECT id, symbol, asof, price_at_signal, alpha_score
            FROM signal_log
            WHERE ret_20d IS NULL AND price_at_signal IS NOT NULL
        """)
        rows = cur.fetchall()
        updated = 0
        today = datetime.utcnow().date()
        for row_id, symbol, asof, price_at_signal, alpha_score in rows:
            try:
                signal_date = datetime.strptime(asof[:10], "%Y-%m-%d").date()
                days_passed = (today - signal_date).days
                if days_passed < 5:
                    continue  # Not enough time for 5d return
                ticker = yf.Ticker(_normalize_symbol_for_lookup(symbol))
                hist = ticker.history(start=signal_date.isoformat(), end=today.isoformat(),
                                      interval="1d", auto_adjust=True)
                if hist is None or hist.empty:
                    continue
                close = hist["Close"].dropna()
                p0 = float(price_at_signal)
                ret5  = (float(close.iloc[4]) / p0 - 1.0) if len(close) >= 5  else None
                ret20 = (float(close.iloc[19]) / p0 - 1.0) if len(close) >= 20 else None
                p5    = float(close.iloc[4])  if len(close) >= 5  else None
                p20   = float(close.iloc[19]) if len(close) >= 20 else None
                conn.execute("""
                    UPDATE signal_log
                    SET price_5d=?, price_20d=?, ret_5d=?, ret_20d=?
                    WHERE id=?
                """, (p5, p20, ret5, ret20, row_id))
                updated += 1
            except Exception:
                continue
        conn.commit()

        # Compute rolling IC (last 30 signals with ret_20d)
        cur.execute("""
            SELECT alpha_score, ret_20d FROM signal_log
            WHERE ret_20d IS NOT NULL AND alpha_score IS NOT NULL
            ORDER BY created_at DESC LIMIT 30
        """)
        ic_rows = cur.fetchall()
        ic_value = None
        icir = None
        if len(ic_rows) >= 5:
            scores = [r[0] for r in ic_rows]
            rets   = [r[1] for r in ic_rows]
            # Spearman rank correlation
            n = len(scores)
            rank_s = sorted(range(n), key=lambda i: scores[i])
            rank_r = sorted(range(n), key=lambda i: rets[i])
            rs = [0.0] * n
            rr = [0.0] * n
            for rank, idx in enumerate(rank_s):
                rs[idx] = rank
            for rank, idx in enumerate(rank_r):
                rr[idx] = rank
            mean_rs = sum(rs) / n
            mean_rr = sum(rr) / n
            num = sum((rs[i]-mean_rs)*(rr[i]-mean_rr) for i in range(n))
            den = (sum((rs[i]-mean_rs)**2 for i in range(n)) * sum((rr[i]-mean_rr)**2 for i in range(n))) ** 0.5
            ic_value = round(num/den, 4) if den else 0.0
            # Rolling IC series (chunks of 10)
            chunk_ics = []
            for i in range(0, n, 10):
                chunk = ic_rows[i:i+10]
                if len(chunk) < 5:
                    continue
                cs = [r[0] for r in chunk]
                cr = [r[1] for r in chunk]
                m_cs, m_cr = sum(cs)/len(cs), sum(cr)/len(cr)
                num_c = sum((cs[j]-m_cs)*(cr[j]-m_cr) for j in range(len(chunk)))
                den_c = ((sum((cs[j]-m_cs)**2 for j in range(len(chunk))) *
                          sum((cr[j]-m_cr)**2 for j in range(len(chunk))))**0.5)
                chunk_ics.append(num_c/den_c if den_c else 0.0)
            if len(chunk_ics) >= 2:
                ic_mean = sum(chunk_ics)/len(chunk_ics)
                ic_std  = statistics.stdev(chunk_ics)
                icir = round(ic_mean/ic_std, 3) if ic_std else None

        conn.close()
        valid = abs(ic_value) > 0.05 if ic_value is not None else False
        return {
            "ok": True,
            "backfilled": updated,
            "ic_signals_used": len(ic_rows),
            "rolling_ic": ic_value,
            "icir": icir,
            "signal_valid": valid,
            "verdict": "✅ 信号有效" if valid else "⚠️ 信号弱，建议检查因子配置",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────
# SECTION: Portfolio Risk  (D-04)
# ─────────────────────────────────────────────────────────────
def quant_portfolio_risk(payload: dict) -> dict:
    """
    Portfolio risk metrics: correlation matrix, portfolio vol, concentration flags.
    """
    positions = _build_enriched_positions()
    if len(positions) < 2:
        single = positions[0] if positions else None
        return {
            "ok": True,
            "message": "持仓不足2只，无需计算组合相关性",
            "single_position_vol": (
                round((_compute_quant_metrics(single["symbol"]) or {}).get("vol_20d_pct", 0), 2)
                if single else None
            ),
        }

    symbols = [p["symbol"] for p in positions]
    weights = []
    total_val = sum((p.get("current_price") or 0) * p["net_qty"] for p in positions)
    for p in positions:
        val = (p.get("current_price") or 0) * p["net_qty"]
        weights.append(val / total_val if total_val else 1.0 / len(positions))

    # Fetch 60-day returns
    returns_data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(_normalize_symbol_for_lookup(sym))
            hist = ticker.history(period="3mo", interval="1d", auto_adjust=True)
            if hist is not None and not hist.empty:
                close = hist["Close"].dropna()
                if len(close) >= 30:
                    returns_data[sym] = list(close.pct_change().dropna().tail(60))
        except Exception:
            pass

    valid_syms = [s for s in symbols if s in returns_data]
    if len(valid_syms) < 2:
        return {"ok": True, "message": "数据不足，无法计算相关矩阵", "symbols": symbols}

    # Correlation matrix
    n = len(valid_syms)
    min_len = min(len(returns_data[s]) for s in valid_syms)
    ret_mat = [[returns_data[s][i] for i in range(min_len)] for s in valid_syms]

    corr_matrix = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            ri, rj = ret_mat[i], ret_mat[j]
            mi, mj = sum(ri)/len(ri), sum(rj)/len(rj)
            num = sum((ri[k]-mi)*(rj[k]-mj) for k in range(len(ri)))
            di  = (sum((ri[k]-mi)**2 for k in range(len(ri)))**0.5)
            dj  = (sum((rj[k]-mj)**2 for k in range(len(rj)))**0.5)
            corr_matrix[i][j] = round(num/(di*dj), 3) if (di*dj) else (1.0 if i==j else 0.0)

    # Portfolio volatility (annualized)
    port_var = 0.0
    sym_vols = {}
    for i, si in enumerate(valid_syms):
        std_i = statistics.stdev(ret_mat[i]) if len(ret_mat[i]) > 1 else 0
        sym_vols[si] = round(std_i * (252**0.5) * 100, 2)
        for j, sj in enumerate(valid_syms):
            wi = weights[symbols.index(si)] if si in symbols else 1.0/n
            wj = weights[symbols.index(sj)] if sj in symbols else 1.0/n
            std_j = statistics.stdev(ret_mat[j]) if len(ret_mat[j]) > 1 else 0
            port_var += wi * wj * corr_matrix[i][j] * std_i * std_j
    port_vol = round((port_var**0.5) * (252**0.5) * 100, 2) if port_var > 0 else 0.0

    # Flags
    flags = []
    for i in range(n):
        for j in range(i+1, n):
            if corr_matrix[i][j] > 0.7:
                flags.append(f"⚠️ {valid_syms[i]} ↔ {valid_syms[j]} 相关性高 ({corr_matrix[i][j]:.2f})，考虑分散风险")
    for i, sym in enumerate(valid_syms):
        w = weights[symbols.index(sym)] if sym in symbols else 0
        if w > 0.4:
            flags.append(f"⚠️ {sym} 仓位集中度 {w*100:.0f}%，超过40%预警线")

    return {
        "ok": True,
        "symbols": valid_syms,
        "weights": [round(w, 3) for w in weights],
        "correlation_matrix": corr_matrix,
        "individual_vols_pct": sym_vols,
        "portfolio_vol_pct": port_vol,
        "flags": flags,
        "summary": f"组合年化波动率 {port_vol}% | 持仓{n}只 | {'有' if flags else '无'}集中度风险",
    }


# ─────────────────────────────────────────────────────────────
# SECTION: Watchlist Management  (E-04)
# ─────────────────────────────────────────────────────────────
def quant_watchlist_manage(payload: dict) -> dict:
    """
    Add / remove / list watchlist symbols.
    action: 'add' | 'remove' | 'list'
    symbol: ticker string (e.g. '5020.T')
    name: optional display name
    """
    action = str(payload.get("action", "list")).lower()
    wl = _load_watchlists()
    jp_syms = list(wl.get("jp_symbols", []))
    us_syms = list(wl.get("us_symbols", []))

    selected_path = Path("/app/quant_trading/Project_optimized/selected_tickers.json")

    if action == "list":
        return {"ok": True, "jp_symbols": jp_syms, "us_symbols": us_syms,
                "total": len(jp_syms) + len(us_syms)}

    symbol = str(payload.get("symbol", "")).strip().upper()
    if not symbol:
        return {"ok": False, "error": "symbol is required for add/remove"}

    is_jp = symbol.endswith(".T")

    if action == "add":
        target = jp_syms if is_jp else us_syms
        if symbol not in target:
            target.append(symbol)
            # Persist
            if is_jp:
                try:
                    data = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
                    data["symbols"] = target
                    selected_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as e:
                    return {"ok": False, "error": str(e)}
            return {"ok": True, "action": "added", "symbol": symbol, "total_jp": len(jp_syms), "total_us": len(us_syms)}
        return {"ok": True, "action": "already_exists", "symbol": symbol}

    if action == "remove":
        target = jp_syms if is_jp else us_syms
        if symbol in target:
            target.remove(symbol)
            if is_jp:
                try:
                    data = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
                    data["symbols"] = target
                    selected_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as e:
                    return {"ok": False, "error": str(e)}
            return {"ok": True, "action": "removed", "symbol": symbol}
        return {"ok": True, "action": "not_found", "symbol": symbol}

    return {"ok": False, "error": f"unknown action: {action}"}


TOOLS = {
    "dummy.echo": lambda p: p,
    "quant.fetch_price": fetch_stock_price,
    "quant.deep_analysis": deep_analysis,
    "quant.discovery_workflow": discovery_workflow,
    "quant.compute_news_risk_factor": compute_news_risk_factor,
    "quant.calc_limit_price": calc_limit_price_tool,
    "quant.run_optimized_pipeline": run_optimized_pipeline,
    "quant.signal_backfill": quant_signal_backfill,
    "quant.portfolio_risk": quant_portfolio_risk,
    "quant.event_alert": quant_event_alert,
    "quant.watchlist": quant_watchlist_manage,
    "portfolio.set_account": portfolio_set_account,
    "portfolio.record_fill": portfolio_record_fill,
    "portfolio.position_review": portfolio_position_review,
    "portfolio.midday_pnl": portfolio_midday_pnl,
    "portfolio.morning_brief": portfolio_morning_brief,
    "portfolio.post_close": portfolio_post_close,
    "portfolio.set_preference": portfolio_set_preference,
    "browser.screenshot": browser_screenshot,
    "ai.analyze": ai_analyze_report,
    "media.generate_report_card": generate_report_card,
    "news.daily_report": news_daily_report,
    "news.active_hot_search": news_active_hot_search,
    "news.preclose_brief_jp": preclose_brief_jp,
    "news.tdnet_close_flash": tdnet_close_flash,
    "news.tdnet_announcements": lambda p: {"ok": True, "announcements": _fetch_tdnet_announcements(
        symbols=p.get("symbols"), lookback_hours=float(p.get("lookback_hours", 24)))},
    "github.skills_daily_report": github_skills_daily_report,
    "web.search_and_browse": web_search_and_browse,
    "ops.deploy_preview": deploy_preview,
}

def main():
    print("[worker] Ready for multi-model logic.")

    def _parse_payload(raw):
        try:
            return json.loads(raw or "{}")
        except Exception:
            return {}

    def _emit_result(task_id, status, output=None, error=None, wf_id="", step_idx=""):
        msg = {
            "task_id": str(task_id or ""), 
            "status": str(status or ""), 
            "workflow_id": str(wf_id or ""), 
            "step_index": str(step_idx or "")
        }
        if output is not None:
            msg["output"] = json.dumps(output)
        if error:
            msg["error"] = str(error)
        r.xadd(STREAM_RESULT, msg)

    def _enqueue_retry(task_id, tool_name, run_id, payload, wf_id, step_idx, retry_count):
        r.xadd(
            STREAM_TASK,
            {
                "task_id": str(task_id or ""),
                "run_id": str(run_id or ""),
                "tool_name": str(tool_name or ""),
                "payload": json.dumps(payload or {}),
                "workflow_id": str(wf_id or ""),
                "step_index": str(step_idx or ""),
                "retry_count": str(retry_count or 0),
            },
        )

    def _send_to_dlq(task_id, tool_name, payload, error, retry_count):
        r.xadd(
            STREAM_DLQ,
            {
                "task_id": str(task_id or ""),
                "tool_name": str(tool_name or ""),
                "payload": json.dumps(payload or {}),
                "error": str(error or ""),
                "retry_count": str(retry_count or 0),
                "failed_at": datetime.utcnow().isoformat() + "Z",
            },
        )
        record_event(task_id, "task.dlq", {"tool_name": tool_name, "error": str(error), "retry_count": retry_count})

    def _handle_message(msg_id, obj, reclaimed=False):
        task_id = obj.get("task_id")
        tool_name = obj.get("tool_name")
        run_id = obj.get("run_id")
        wf_id = obj.get("workflow_id") or ""
        step_idx = obj.get("step_index") or ""
        retry_count = int(obj.get("retry_count") or 0)
        payload = _parse_payload(obj.get("payload", "{}"))

        if not task_id or not tool_name:
            r.xack(STREAM_TASK, GROUP, msg_id)
            return

        if reclaimed:
            record_event(task_id, "task.reclaimed", {"task_id": task_id, "consumer": CONSUMER, "message_id": msg_id})

        record_execution_audit(run_id, task_id, "worker-quant", "task_start", {
            "tool_name": tool_name,
            "workflow_id": wf_id,
            "step_index": step_idx,
        })
        _emit_result(task_id, "claimed", wf_id=wf_id, step_idx=step_idx)

        try:
            payload_ok, payload_errors = validate_single_agent_payload_guardrails(payload)
            if not payload_ok:
                raise RuntimeError(f"SINGLE_AGENT_GUARDRAILS_INVALID: {'; '.join(payload_errors)}")
            handler = TOOLS.get(tool_name)
            if not handler:
                raise RuntimeError(f"Unknown tool: {tool_name}")
            _emit_result(task_id, "tool_call", output={
                "tool_name": tool_name,
                "input_summary": str(payload.get("task_prompt") or payload.get("prompt") or payload.get("symbol") or "")[:240],
            }, wf_id=wf_id, step_idx=step_idx)
            record_execution_audit(run_id, task_id, "worker-quant", "tool_call", {
                "tool_name": tool_name,
                "input": payload,
            })
            started_at = time.time()
            output = handler(payload)
            if not isinstance(output, dict):
                output = {"ok": True, "raw": output}
            status = "succeeded" if output.get("ok", True) else "failed"
            output = {
                **output,
                "worker_result": build_worker_result_envelope(
                    run_id=run_id,
                    worker_name="worker-quant",
                    ok=output.get("ok", True),
                    status=status,
                    output=output,
                    error=output.get("error"),
                    duration_ms=int((time.time() - started_at) * 1000),
                    tool_calls=int(output.get("tool_calls", 0) or 0),
                    permission_decisions=output.get("permission_decisions") if isinstance(output.get("permission_decisions"), list) else [],
                    bounded_validation=output.get("bounded_validation") if isinstance(output.get("bounded_validation"), list) and len(output.get("bounded_validation")) > 0 else [{
                        "name": "result_shape",
                        "ok": isinstance(output, dict),
                        "detail": "dict_result_emitted" if isinstance(output, dict) else "non_dict_result",
                    }],
                    evidence_id=str(payload.get("evidence_id", "") or ""),
                    replay_tag=str(payload.get("replay_tag", "") or ""),
                ),
            }
            if is_single_agent_payload(payload):
                result_ok, result_errors = validate_single_agent_worker_result_guardrails(
                    output.get("worker_result"),
                    require_single_agent=True,
                )
                if not result_ok:
                    raise RuntimeError(f"SINGLE_AGENT_GUARDRAILS_INVALID: {'; '.join(result_errors)}")
            record_execution_audit(run_id, task_id, "worker-quant", "tool_result", {
                "tool_name": tool_name,
                "ok": output.get("ok", True),
                "worker_result": output.get("worker_result"),
            })
            _emit_result(task_id, "tool_result", output={
                "tool_name": tool_name,
                "ok": output.get("ok", True),
                "summary": str(output.get("briefing") or output.get("analysis") or output.get("summary") or output.get("error") or "")[:240],
            }, wf_id=wf_id, step_idx=step_idx)
            record_execution_audit(run_id, task_id, "worker-quant", "task_complete", {
                "worker_result": output.get("worker_result"),
            })
            _emit_result(task_id, status, output=output, wf_id=wf_id, step_idx=step_idx)
            r.xack(STREAM_TASK, GROUP, msg_id)
        except Exception as e:
            err_text = str(e)
            record_execution_audit(run_id, task_id, "worker-quant", "task_error", {
                "tool_name": tool_name,
                "error": err_text,
                "retry_count": retry_count,
            })
            if retry_count < MAX_RETRIES:
                _enqueue_retry(task_id, tool_name, run_id, payload, wf_id, step_idx, retry_count + 1)
                record_event(task_id, "task.retrying", {"error": err_text, "retry_count": retry_count + 1})
                r.xack(STREAM_TASK, GROUP, msg_id)
            else:
                _send_to_dlq(task_id, tool_name, payload, err_text, retry_count)
                _emit_result(task_id, "failed", error=err_text, wf_id=wf_id, step_idx=step_idx)
                r.xack(STREAM_TASK, GROUP, msg_id)

    def _reclaim_pending():
        if not hasattr(r, "xautoclaim"):
            return []
        try:
            res = r.xautoclaim(
                STREAM_TASK,
                GROUP,
                CONSUMER,
                min_idle_time=VISIBILITY_TIMEOUT_S * 1000,
                start_id="0-0",
                count=10,
            )
            if not res:
                return []
            _next_id, messages = res[0], res[1]
            return messages or []
        except Exception as e:
            print(f"[worker] reclaim failed: {e}")
            return []

    # --- Ensure Consumer Group exists ---
    try:
        r.xgroup_create(STREAM_TASK, GROUP, id="0", mkstream=True)
    except Exception: pass # Already exists
    try:
        r.xgroup_create(STREAM_TASK_CODING, GROUP, id="0", mkstream=True)
    except Exception: pass

    last_reclaim = 0.0
    while True:
        try:
            now = time.time()
            if now - last_reclaim >= RECLAIM_INTERVAL_S:
                for msg_id, kv in _reclaim_pending():
                    _handle_message(msg_id, dict(kv), reclaimed=True)
                last_reclaim = now

            res = r.xreadgroup(GROUP, CONSUMER, streams={STREAM_TASK: ">"}, count=1, block=5000)
            if not res:
                continue
            msg_id, kv = res[0][1][0]
            _handle_message(msg_id, dict(kv), reclaimed=False)
        except Exception as e:
            print(f"[worker] task loop failed: {e}")
            traceback.print_exc()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
