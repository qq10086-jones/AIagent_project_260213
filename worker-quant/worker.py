import os, json, time, subprocess, hashlib, base64, io, re
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

# 使用 host.docker.internal 或宿主机 IP
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.113:11434")
OLLAMA_CHAT_API = f"{OLLAMA_BASE}/api/chat"
QUANT_LLM_MODEL = os.getenv("QUANT_LLM_MODEL", "deepseek-r1:32b")
CODE_LLM_MODEL = os.getenv("CODE_LLM_MODEL", "glm-4.7-flash:latest")

r = redis.from_url(REDIS_URL, decode_responses=True)

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
    return {"name": file_path.name, "object_key": object_key, "sha256": file_hash, "size": len(data), "mime": f"image/{ext}" if ext in ["png","jpg"] else "text/plain"}

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

def _ollama_chat(system: str, user: str, model: str | None = None, timeout_s: int = 120) -> str:
    model = model or QUANT_LLM_MODEL
    if isinstance(model, str) and model.startswith("ollama/"):
        model = model.split("/", 1)[1]
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
    # Accept both "ollama/<model>" and raw "<model>" for Ollama direct API
    if isinstance(model, str) and model.startswith("ollama/"):
        model = model.split("/", 1)[1]
    report_path = Path("/app/quant_trading/Project_optimized/reports/strategy_report.html")
    
    content = "Summary of backtest results unavailable."
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()[:6000] # 上下文控制

    # 构造 Ollama Chat 消息
    messages = [
        {"role": "system", "content": "You are a senior quantitative analyst. Provide 3 sharp insights based on the HTML report provided."},
        {"role": "user", "content": f"Here is the report content:\n{content}"}
    ]
    
    data = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False
    }).encode("utf-8")
    
    req = urllib.request.Request(OLLAMA_CHAT_API, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw_res = json.loads(resp.read().decode("utf-8"))
            analysis = raw_res.get("message", {}).get("content", "No analysis generated.")
            return {"ok": True, "model_used": model, "analysis": analysis}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"ok": False, "error": "MODEL_NOT_FOUND", "suggestion": f"Model {model} not found or API path incorrect. Please check Ollama."}
        return {"ok": False, "error": f"HTTP_{e.code}", "suggestion": "Local resource exhausted. Suggest upgrading to Cloud API (Gemini/Claude)."}
    except Exception as e:
        return {"ok": False, "error": str(e), "suggestion": "Connection failed. Is Ollama running with OLLAMA_HOST=0.0.0.0?"}

def fetch_stock_price(payload: dict):
    symbol = payload.get("symbol", "NVDA")
    ticker = yf.Ticker(symbol)
    return {"symbol": symbol, "price": ticker.fast_info.get("last_price"), "ts": time.time()}

def run_optimized_pipeline(payload: dict):
    base_dir = Path("/app/quant_trading/Project_optimized")
    reports_dir = base_dir / "reports"
    if reports_dir.exists():
        for f in reports_dir.glob("*"):
            try: f.unlink()
            except: pass
    
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = str(base_dir) + ":" + my_env.get("PYTHONPATH", "")
    res = subprocess.run(["python", "run_pipeline.py", "--config", "config.yaml"], cwd=str(base_dir), capture_output=True, text=True, timeout=600, env=my_env)
    
    artifacts = []
    if reports_dir.exists():
        for f in reports_dir.glob("*"):
            if f.is_file(): artifacts.append(archive_file(f))
    return {"ok": res.returncode == 0, "artifacts": artifacts, "stdout": res.stdout[-1000:]}

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

    summary_lines = []
    use_llm = str(os.getenv("NEWS_USE_LLM", "1")).lower() in ["1", "true", "yes"]
    if use_llm and items:
        titles = "\n".join([f"- {it['title']}" for it in items[:20]])
        system = "You summarize market news in Chinese. Provide 5 concise bullet points."
        out = _ollama_chat(system, titles, model=QUANT_LLM_MODEL, timeout_s=120)
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

    return {"ok": True, "date": date_str, "count": len(items), "artifacts": artifacts}

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
        out = _ollama_chat(system, prompt, model=CODE_LLM_MODEL, timeout_s=180)
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

TOOLS = {
    "dummy.echo": lambda p: p,
    "quant.fetch_price": fetch_stock_price,
    "quant.run_optimized_pipeline": run_optimized_pipeline,
    "ai.analyze": ai_analyze_report,
    "media.generate_report_card": generate_report_card,
    "news.daily_report": news_daily_report,
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
            if 'task_id' in locals(): r.xadd(STREAM_RESULT, {"task_id": task_id, "status": "failed", "error": str(e)})
        time.sleep(0.1)

if __name__ == "__main__":
    main()
