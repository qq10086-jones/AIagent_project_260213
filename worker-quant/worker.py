import os, json, time, subprocess, hashlib, base64, io, re, traceback
import html
import xml.etree.ElementTree as ET
from pathlib import Path
import redis
import yfinance as yf
import boto3
from datetime import datetime, timedelta
import urllib.request
import urllib.error
from PIL import Image, ImageDraw
import requests
from jinja2 import Template
import uuid
import psycopg2
import sqlite3

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# --- Env Config ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_TASK = os.getenv("STREAM_TASK", "stream:task")
STREAM_RESULT = os.getenv("STREAM_RESULT", "stream:result")
GROUP = os.getenv("GROUP", "cg:workers")
CONSUMER = os.getenv("CONSUMER", "worker-quant-1")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "nexus")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "nexuspassword")

# --- LLM Config ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower() # ollama, openai, dashscope, gemini
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.113:11434")
OLLAMA_CHAT_API = f"{OLLAMA_BASE}/api/chat"

# Cloud Provider Configs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DASH_SCOPE_API_KEY = os.getenv("DASH_SCOPE_API_KEY", os.getenv("QWEN_API_KEY"))
DASH_SCOPE_BASE_URL = os.getenv("DASH_SCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

QUANT_LLM_MODEL = os.getenv("QUANT_LLM_MODEL", "deepseek-r1:1.5b")
CODE_LLM_MODEL = os.getenv("CODE_LLM_MODEL", "glm-4.7-flash:latest")

r = redis.from_url(REDIS_URL, decode_responses=True)

def get_llm_response(system: str, user: str, model: str | None = None, provider: str | None = None, timeout_s: int = 120) -> str:
    """Unified LLM client supporting multiple providers."""
    prov = (provider or LLM_PROVIDER).lower()
    model = model or QUANT_LLM_MODEL
    
    # Strip provider prefix if present (e.g. "ollama/deepseek-r1:1.5b")
    if isinstance(model, str) and "/" in model:
        p_prefix, m_name = model.split("/", 1)
        if p_prefix in ["ollama", "openai", "dashscope", "gemini"]:
            prov = p_prefix
            model = m_name

    if prov == "ollama":
        data = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False
        }).encode("utf-8")
        req = urllib.request.Request(OLLAMA_CHAT_API, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw_res = json.loads(resp.read().decode("utf-8"))
                return raw_res.get("message", {}).get("content", "")
        except Exception as e:
            print(f"[LLM] Ollama error: {e}")
            return ""
            
    elif prov in ["openai", "dashscope", "gemini"]:
        # Standard OpenAI compatible API call
        api_key = OPENAI_API_KEY if prov == "openai" else DASH_SCOPE_API_KEY
        base_url = OPENAI_BASE_URL if prov == "openai" else DASH_SCOPE_BASE_URL
        if prov == "gemini":
            # Gemini typically uses its own SDK or specific endpoint, but let's assume compatible if requested via env
            base_url = os.getenv("GEMINI_BASE_URL", base_url)
            api_key = os.getenv("GEMINI_API_KEY", api_key)

        if not api_key:
            print(f"[LLM] Provider {prov} requested but API key is missing.")
            return ""

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
                return get_llm_response(system, user, provider="ollama")
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

# --- AI Analysis with Fallback Request ---
def ai_analyze_report(payload: dict):
    model = payload.get("model") or QUANT_LLM_MODEL
    report_path = Path("/app/quant_trading/Project_optimized/reports/strategy_report.html")
    
    content = "Summary of backtest results unavailable."
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()[:6000] # 上下文控制

    system = "You are a senior quantitative analyst. Provide 3 sharp insights based on the HTML report provided."
    user = f"Here is the report content:\n{content}"
    
    analysis = get_llm_response(system, user, model=model)
    if analysis:
        return {"ok": True, "model_used": model, "analysis": analysis}
    else:
        return {"ok": False, "error": "LLM_FAILURE", "suggestion": "Check API keys or Ollama connectivity."}


def fetch_stock_price(payload: dict):
    symbol = str(payload.get("symbol", "NVDA")).upper()
    quote = _fetch_quote_facts(symbol)
    return {"symbol": symbol, "price": quote.get("price"), "ts": time.time(), "company_name": quote.get("company_name")}

def _quote_url(symbol: str) -> str:
    sym = str(symbol or "").upper()
    if sym.endswith(".T"):
        return f"https://finance.yahoo.co.jp/quote/{sym}"
    return f"https://finance.yahoo.com/quote/{sym}"

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
    sym = str(symbol or "").upper()
    ticker = yf.Ticker(sym)
    company_name = None
    currency = None
    price = None
    source = "yfinance"
    recent_news = []

    try:
        info = ticker.info or {}
        company_name = info.get("longName") or info.get("shortName")
        currency = info.get("currency")
    except Exception:
        pass

    try:
        fi = ticker.fast_info or {}
        price = _safe_float(fi.get("last_price"))
        currency = currency or fi.get("currency")
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
        "prev_close": prev_close,
        "change_pct": change_pct,
        "quote_url": _quote_url(sym),
        "recent_news": recent_news,
        "source": source,
    }

def _fetch_ss6_signal(symbol: str) -> dict:
    """Fetch professional regression signal from ss6_sqlite database if available."""
    sym = str(symbol or "").upper()
    db_path = Path("/app/quant_trading/Project_optimized/japan_market.db")
    if not db_path.exists():
        return {"found": False}
    
    try:
        conn = sqlite3.connect(str(db_path))
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
    sym = str(symbol or "").upper()
    ticker = yf.Ticker(sym)
    
    # NEW: Fetch professional signal first
    ss6 = _fetch_ss6_signal(sym)
    
    try:
        # Use a bit more data for better SMA/RSI stability
        hist = ticker.history(period="1y", interval="1d", auto_adjust=False)
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

def _fetch_news_from_google_rss(symbol: str, company_name: str | None, max_items: int = 5) -> list[dict]:

    query_terms = [symbol]
    if company_name:
        query_terms.append(company_name)
    query_terms.append("stock")
    q = " ".join([x for x in query_terms if x])
    url = "https://news.google.com/rss/search"
    try:
        resp = requests.get(url, params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}, timeout=20)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        out = []
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("source") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            if title and link:
                out.append({
                    "title": title,
                    "url": link,
                    "publisher": pub,
                    "published_at": pub_date,
                    "source": "google_news_rss",
                })
        return out
    except Exception:
        return []

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

def _merge_recent_news(quote: dict, max_items: int = 5) -> list[dict]:
    news = list((quote or {}).get("recent_news") or [])
    symbol = str((quote or {}).get("symbol") or "")
    company_name = (quote or {}).get("company_name")
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
        key = (title.lower(), url.lower())
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
            
    # --- Generate SS7 News Overlay CSV ---
    csv_path = base_dir / "news_overlay.csv"
    db_path = base_dir / "japan_market.db"
    has_news = False
    try:
        if db_path.exists():
            import csv
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT asof, symbol, value FROM feature_daily WHERE feature_name='news_risk_raw'")
            rows = cur.fetchall()
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(["date", "ticker", "sent", "weight", "conf"])
                for r in rows:
                    # Invert risk: High risk -> Negative sentiment
                    w.writerow([r[0], r[1], -float(r[2]), 1.0, 1.0])
                    has_news = True
            conn.close()
    except Exception as e:
        print(f"Failed to generate news CSV: {e}")
    
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = str(base_dir) + ":" + my_env.get("PYTHONPATH", "")
    
    if has_news:
        my_env["SS6_NEWS_ON"] = "1"
        my_env["SS6_NEWS_CSV"] = "news_overlay.csv"
        
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

def _openclaw_capture_urls(urls: list[str], full_page: bool = True, max_shots: int = 3):
    artifacts = []
    if not urls:
        return artifacts
    urls = urls[:max_shots]
    start = _openclaw_run("browser.start", {})
    if not start.get("ok", False):
        return artifacts
    for u in urls:
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
                artifacts.append({
                    "name": f"browser_screenshot_{len(artifacts)+1}.png",
                    "object_key": archived.get("object_key"),
                    "sha256": archived.get("sha256"),
                    "size": 0,
                    "mime": "image/png",
                })
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
  <title>Daily Market News Report {{ date }}</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #111; }
    h1 { margin-bottom: 6px; }
    .meta { color: #555; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; padding: 12px; border-radius: 8px; }
    .item { margin-bottom: 12px; }
    .item-title { font-weight: bold; }
    .tag { display: inline-block; padding: 2px 6px; background: #eef; margin-right: 6px; border-radius: 4px; font-size: 12px; }
    .small { color: #666; font-size: 12px; }
    img { max-width: 100%; }
  </style>
</head>
<body>
  <h1>Daily Market News Report</h1>
  <div class="meta">Date: {{ date }} | Items: {{ items|length }} | Lookback: {{ lookback_hours }}h</div>
  <div class="grid">
    <div class="card">
      <h3>Summary</h3>
      {% if summary_lines %}
        <ul>
          {% for s in summary_lines %}
            <li>{{ s }}</li>
          {% endfor %}
        </ul>
      {% else %}
        <div class="small">No summary generated.</div>
      {% endif %}
      <h3>Highlights</h3>
      {% for it in items[:10] %}
      <div class="item">
        <div class="item-title"><a href="{{ it.url }}">{{ it.title }}</a></div>
        <div class="small">{{ it.source }} | {{ it.seendate }} | {{ it.lang }}</div>
        {% if it.tickers %}
          <div>
            {% for t in it.tickers %}
              <span class="tag">{{ t }}</span>
            {% endfor %}
          </div>
        {% endif %}
        {% if it.snippet %}
          <div class="small">{{ it.snippet }}</div>
        {% endif %}
      </div>
      {% endfor %}
    </div>
    <div class="card">
      <h3>Charts</h3>
      {% if charts.tickers %}<img src="{{ charts.tickers }}" />{% endif %}
      {% if charts.sources %}<img src="{{ charts.sources }}" />{% endif %}
    </div>
  </div>
  <div class="card" style="margin-top:16px;">
    <h3>All Items</h3>
    <ol>
      {% for it in items %}
        <li class="item">
          <div class="item-title"><a href="{{ it.url }}">{{ it.title }}</a></div>
          <div class="small">{{ it.source }} | {{ it.seendate }} | {{ it.lang }}</div>
        </li>
      {% endfor %}
    </ol>
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
    symbol = str(payload.get("symbol", "NVDA")).upper()
    run_id = payload.get("run_id")
    quote = _fetch_quote_facts(symbol)
    metrics = _compute_quant_metrics(symbol)
    news = _merge_recent_news(quote, max_items=5)
    quote["recent_news"] = news
    analysis = _grounded_quant_summary(quote)

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
        "price": quote.get("price"),
        "currency": quote.get("currency"),
        "quote_url": quote.get("quote_url"),
        "recent_news": news,
        "metrics": metrics,
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

def discovery_workflow(payload: dict):
    """
    Stage 3: Screening high-potential stocks based on intelligence.
    This bridges news discovery with quantitative screening.
    """
    run_id = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()
    
    # 1. Load watchlists and intelligence
    wl = _load_watchlists()
    all_known = wl.get("jp_symbols", []) + wl.get("us_symbols", [])
    
    # 2. Trigger a fresh news scan if requested, or use existing facts
    # For now, we simulate discovery from the internal DB
    candidates = []
    try:
        db_path = Path("/app/quant_trading/Project_optimized/japan_market.db")
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            # Find symbols with high news sentiment or frequent mentions in the last 24h
            # This is a placeholder for a more complex SQL query
            cur.execute("SELECT DISTINCT symbol FROM orders WHERE asof >= ?", (date_str,))
            rows = cur.fetchall()
            candidates = [r[0] for r in rows if r[0] in all_known]
            conn.close()
    except Exception as e:
        print(f"Discovery DB Error: {e}")

    # 3. If no candidates from DB, fallback to top watchlist
    if not candidates:
        candidates = all_known[:10]

    # 4. Perform lightweight screening (Stage 3 & 4)
    results = []
    for sym in candidates[:5]: # Limit to top 5 to avoid timeout
        m = _compute_quant_metrics(sym)
        if m.get("ok") and m.get("alpha_score", 0) > 1.0:
            results.append({
                "symbol": sym,
                "signal": m.get("signal"),
                "score": m.get("alpha_score"),
                "risk": m.get("risk_state")
            })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    analysis = f"Discovery complete for {date_str}. Found {len(results)} high-potential candidates."
    if results:
        analysis += " Top candidate: " + results[0]["symbol"]

    if run_id:
        record_fact(run_id, "quant", "discovery_result", {
            "date": date_str,
            "candidates": results,
            "summary": analysis
        })

    return {
        "ok": True,
        "date": date_str,
        "candidates": results,
        "analysis": analysis
    }

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
        response = get_llm_response(system_prompt, news_text, model=QUANT_LLM_MODEL)
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
    db_path = Path("/app/quant_trading/Project_optimized/japan_market.db")
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
    
    db_path = Path("/app/quant_trading/Project_optimized/japan_market.db")
    if not db_path.exists():
        return {"ok": False, "error": "Database not found."}
        
    try:
        conn = sqlite3.connect(str(db_path))
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
    symbol = payload.get("symbol", "").upper()
    side = payload.get("side", "BUY").upper()
    qty = float(payload.get("qty", 0))
    price = float(payload.get("price", 0))
    date_str = _now_date_str()
    fill_id = uuid.uuid4().hex
    
    db_path = Path("/app/quant_trading/Project_optimized/japan_market.db")
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

def preclose_brief_jp(payload: dict):
    """
    Japanese market pre-close brief (15:15 JST).
    Provides quick insights before the 15:30 close.
    """
    run_id = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()
    
    # 1. Fetch fast path news (last 6h)
    articles = _gdelt_doc_search("japan OR tokyo OR nikkei OR stock", datetime.utcnow() - timedelta(hours=6), datetime.utcnow(), max_records=15)
    
    if not articles:
        # Fallback to Google News
        articles = _fetch_news_from_google_rss("Japan market", "Nikkei", max_items=5)
        
    items_text = ""
    for a in articles[:5]:
        title = a.get('title', 'Unknown')
        domain = a.get('domain', a.get('publisher', 'Unknown'))
        url = a.get('url', '')
        items_text += f"- {title} ({domain}) : {url}\n"

    if items_text:
        system = "You are a quantitative news analyst summarizing pre-close market conditions for the Tokyo Stock Exchange. Produce a professional, concise Chinese summary highlighting key market drivers, sentiment, and any notable events from the provided headlines. Make it look like a professional brief."
        llm_summary = get_llm_response(system, items_text, model=QUANT_LLM_MODEL)
        summary = f"【15:15 JST 盘尾情报】\n{llm_summary}\n\n【原始快讯】\n{items_text}"
    else:
        summary = "【15:15 JST 盘尾情报】\n未发现重大近期事件。\n状态：准备接收 action_model 反馈。"

    result = {
        "ok": True,
        "type": "preclose_brief_jp",
        "date": date_str,
        "analysis": summary
    }
    
    if run_id:
        record_fact(run_id, "news", "preclose_brief", result)
        
    return result

def tdnet_close_flash(payload: dict):
    """
    Japanese market post-close flash (15:35 JST) focusing on TDnet announcements.
    """
    run_id = payload.get("run_id")
    date_str = payload.get("date") or _now_date_str()
    
    # Placeholder for TDnet scraping logic
    # We fallback to Google News looking for earnings/announcements
    articles = _fetch_news_from_google_rss("Japan earnings OR Tokyo announcement", "finance", max_items=5)
    
    items_text = ""
    for a in articles[:3]:
        title = a.get('title', 'Unknown')
        url = a.get('url', '')
        items_text += f"- {title} : {url}\n"
        
    if items_text:
        system = "You are an analyst reviewing post-close corporate announcements for the Tokyo Stock Exchange. Summarize the following headlines in a concise, professional Chinese flash report."
        llm_summary = get_llm_response(system, items_text, model=QUANT_LLM_MODEL)
        summary = f"【15:35 JST 盘后闪讯】\n{llm_summary}\n\n【监测到以下公告/新闻】\n{items_text}"
    else:
        summary = "【15:35 JST 盘后闪讯】\n监测窗口（15:00-15:35）未发现高严重度公告。"

    result = {
        "ok": True,
        "type": "tdnet_close_flash",
        "date": date_str,
        "analysis": summary
    }
    
    if run_id:
        record_fact(run_id, "news", "tdnet_flash", result)
        
    return result

TOOLS = {
    "dummy.echo": lambda p: p,
    "quant.fetch_price": fetch_stock_price,
    "quant.deep_analysis": deep_analysis,
    "quant.discovery_workflow": discovery_workflow,
    "quant.compute_news_risk_factor": compute_news_risk_factor,
    "quant.run_optimized_pipeline": run_optimized_pipeline,
    "portfolio.set_account": portfolio_set_account,
    "portfolio.record_fill": portfolio_record_fill,
    "browser.screenshot": browser_screenshot,
    "ai.analyze": ai_analyze_report,
    "media.generate_report_card": generate_report_card,
    "news.daily_report": news_daily_report,
    "news.preclose_brief_jp": preclose_brief_jp,
    "news.tdnet_close_flash": tdnet_close_flash,
    "github.skills_daily_report": github_skills_daily_report,
}

def main():
    print("[worker] Ready for multi-model logic.")
    while True:
        try:
            res = r.xreadgroup(GROUP, CONSUMER, streams={STREAM_TASK: ">"}, count=1, block=5000)
            if not res: continue
            msg_id, kv = res[0][1][0]
            obj = dict(kv)
            task_id = obj.get("task_id")
            wf_id, step_idx = obj.get("workflow_id"), obj.get("step_index")
            r.xadd(STREAM_RESULT, {"task_id": task_id, "status": "claimed", "workflow_id": wf_id or "", "step_index": step_idx or ""})
            
            tool_name = obj.get("tool_name")
            payload = json.loads(obj.get("payload", "{}"))
            handler = TOOLS.get(tool_name)
            output = handler(payload)
            
            # 如果 output 里包含建议升级模型的信号，我们可以特殊处理状态
            status = "succeeded" if output.get("ok", True) else "failed"
            res_msg = {"task_id": task_id, "status": status, "output": json.dumps(output), "workflow_id": wf_id or "", "step_index": step_idx or ""}
            r.xadd(STREAM_RESULT, res_msg)
            r.xack(STREAM_TASK, GROUP, msg_id)
        except Exception as e:
            print(f"[worker] task failed: {e}")
            traceback.print_exc()
            if 'task_id' in locals(): r.xadd(STREAM_RESULT, {"task_id": task_id, "status": "failed", "error": str(e)})
        time.sleep(0.1)

if __name__ == "__main__":
    main()
