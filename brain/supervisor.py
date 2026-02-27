import os
import json
import requests
import time
import psycopg2
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from state import AgentState

ORCHESTRATOR_URL = "http://nexus-orchestrator:3000"
# --- LLM Config ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "dashscope").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_CHAT_API = f"{OLLAMA_BASE_URL}/api/chat"
LOCAL_WRITER_MODEL = os.getenv("QUANT_LLM_MODEL", "deepseek-r1:1.5b")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")

def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "db"),
        user=os.getenv("PGUSER", "nexus"),
        password=os.getenv("PGPASSWORD", "nexus"),
        database=os.getenv("PGDATABASE", "nexus")
    )

def poll_for_fact(run_id, agent_name, timeout=120):
    """Wait for a specific agent to write a result into the DB."""
    start_time = time.time()
    print(f"[Brain] Polling for {agent_name} results (run_id: {run_id})...")
    while time.time() - start_time < timeout:
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT payload_json FROM fact_items WHERE run_id = %s AND agent_name = %s ORDER BY created_at DESC LIMIT 1",
                (run_id, agent_name),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                print(f"[Brain] Found results for {agent_name}!")
                payload = row[0]
                if isinstance(payload, dict):
                    return payload
                try:
                    return json.loads(payload)
                except Exception:
                    return {"raw": str(payload)}
        except Exception as e:
            print(f"[Brain] DB polling error for {agent_name}: {e}")
        time.sleep(5)
    print(f"[Brain] Timeout waiting for {agent_name}")
    return None

def get_llm(state: AgentState = None):
    """Returns a LangChain-compatible LLM based on LLM_PROVIDER."""
    prov = LLM_PROVIDER
    if prov == "ollama":
        # Using ChatOpenAI with Ollama's OpenAI-compatible endpoint if available, 
        # or just fallback to the manual call_local_writer in writer_agent_node.
        # For now, let's stick to the current logic but make it provider-aware.
        return None 
    
    if prov in ["dashscope", "openai"]:
        api_key = os.getenv("QWEN_API_KEY") if prov == "dashscope" else os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1") if prov == "dashscope" else os.getenv("OPENAI_BASE_URL")
        
        default_model = QWEN_MODEL if prov == "dashscope" else os.getenv("OPENAI_MODEL", "gpt-4o")
        model = state.get("qwen_model", default_model) if state else default_model

        if not api_key:
            return None
            
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=60
        )
    return None

def _browser_enabled_for_symbol(symbol: str) -> bool:
    mode = str(os.getenv("ENABLE_BROWSER_FACTS", "auto")).lower()
    if mode in ["1", "true", "yes", "on"]:
        return True
    if mode in ["0", "false", "no", "off"]:
        return False
    return str(symbol or "").upper().endswith(".T")

def _build_grounded_narrative(state: AgentState) -> str:
    symbol = state.get("symbol", "unknown")
    facts = state.get("facts", [])
    quant_data = None
    browser_data = None
    for f in facts:
        if f.get("agent") == "quant":
            quant_data = f.get("data", {})
        if f.get("agent") == "browser":
            browser_data = f.get("data", {})

    company = (quant_data or {}).get("company_name") or "Data unavailable"
    price = (quant_data or {}).get("price")
    currency = (quant_data or {}).get("currency") or ""
    quote_url = (quant_data or {}).get("quote_url") or "Data unavailable"
    source = (quant_data or {}).get("source") or "Data unavailable"
    news = (quant_data or {}).get("recent_news") or []
    metrics = (quant_data or {}).get("metrics") or {}
    browser_url = (browser_data or {}).get("url") or "Data unavailable"

    if price is None:
        price_text = "Data unavailable"
    else:
        price_text = f"{price} {currency}".strip()

    lines = [
        f"{symbol} 分析报告",
        "",
        "1) 标的概览",
        f"- symbol: {symbol}",
        f"- company: {company}",
        f"- quote_source: {source}",
        "",
        "2) 市场事实",
        f"- latest_price: {price_text}",
        f"- quote_url: {quote_url}",
        f"- browser_evidence_url: {browser_url}",
    ]

    # Normalize headings to avoid encoding artifacts.
    lines = [
        f"{symbol} 分析报告",
        "",
        "1) 标的概览",
        f"- symbol: {symbol}",
        f"- company: {company}",
        f"- quote_source: {source}",
        "",
        "2) 市场事实",
        f"- latest_price: {price_text}",
        f"- quote_url: {quote_url}",
        f"- browser_evidence_url: {browser_url}",
    ]

    if len(lines) >= 9:
        lines[0] = f"{symbol} Analysis Report"
        lines[2] = "1) Overview"
        lines[7] = "2) Market Facts"

    if metrics:
        lines.extend([
            f"- signal: {metrics.get('signal', 'N/A')}",
            f"- alpha_score: {metrics.get('alpha_score', 'N/A')}",
            f"- ret_20d_pct: {metrics.get('ret_20d_pct', 'N/A')}",
            f"- rsi14: {metrics.get('rsi14', 'N/A')}",
            f"- vol_20d_pct: {metrics.get('vol_20d_pct', 'N/A')}",
        ])

    lines.extend([
        "",
        "3) 最近新闻",
    ])

    if lines:
        lines[-1] = "3) Recent News"

    if news:
        for idx, item in enumerate(news[:5], start=1):
            title = item.get("title") or "Data unavailable"
            url = item.get("url") or "Data unavailable"
            lines.append(f"- {idx}. {title} ({url})")
    else:
        lines.append("- Data unavailable")

    lines.extend([
        "",
        "4) 结论",
        "- 以上结论来自已抓取数据、浏览器证据与量化特征计算。",
        "- 已生成模板化完整报告（Markdown/HTML），可在 Discord 作为附件查看。",
    ])
    if len(lines) >= 4:
        lines[-3] = "4) Conclusion"
        lines[-2] = "- Conclusions are based on collected facts, browser evidence, and computed metrics."
        lines[-1] = "- A full Markdown/HTML report has been generated for Discord attachment."
    return "\n".join(lines)

def call_local_writer(prompt: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_CHAT_API,
            json={
                "model": LOCAL_WRITER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=90,
        )
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[Brain] Local writer failed: {e}")
        return ""

def trigger_tool(tool_name: str, payload: dict, run_id: str):
    idem_key = f"{run_id}:{tool_name}:{payload.get('symbol', 'na')}"
    resp = requests.post(
        f"{ORCHESTRATOR_URL}/execute-tool",
        json={
            "tool_name": tool_name,
            "payload": payload,
            "run_id": run_id,
            "idempotency_key": idem_key,
        },
        timeout=20,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"execute-tool failed {resp.status_code}: {resp.text[:200]}")
    return resp.json()

def _extract_tool_payload(state: AgentState, tool_name: str) -> dict:
    raw = state.get("tool_payload") or {}
    if not isinstance(raw, dict):
        return {}
    allow_map = {
        "quant.discovery_workflow": {
            "capital_base_jpy", "capital_base", "max_position_pct", "market",
            "goal", "risk_profile", "target_return_pct", "horizon_days",
            "avoid_mega_cap", "prefer_small_mid_cap"
        },
        "quant.deep_analysis": {
            "symbol", "capital_base_jpy", "capital_base", "capital_headroom_pct",
            "max_position_pct", "mode"
        },
    }
    allowed = allow_map.get(tool_name, set())
    return {k: v for k, v in raw.items() if k in allowed}

def supervisor_node(state: AgentState):
    """
    Core Logic: Task Decomposition & Routing.
    This is where OpenClaw Nexus decides which agent takes which tool.
    """
    mode = state.get("mode", "analysis")
    existing = [f['agent'] for f in state.get('facts', [])]
    
    if mode == "discovery":
        # Discovery Workflow Pipeline: Intel -> Screening -> Report
        if "intel" not in existing:
            return {"next_step": "discovery"}
        if "screener" not in existing:
            return {"next_step": "screening"}
        return {"next_step": "writer"}
    else:
        # Single Symbol Analysis Pipeline: Quant -> Browser -> Report
        enable_browser = _browser_enabled_for_symbol(state.get("symbol", ""))
        if "quant" not in existing:
            return {"next_step": "quant"}
        if enable_browser and "browser" not in existing:
            return {"next_step": "browser"}
        return {"next_step": "writer"}

def discovery_agent_node(state: AgentState):
    """Stage 1: Intelligence Discovery (Macro/Global News)."""
    run_id = state.get('run_id')
    print(f"--- [Agent: Intelligence] Gathering Market Data ---")
    try:
        trigger_tool("news.daily_report", {"run_id": run_id, "lookback_hours": 24}, run_id)
    except Exception as e:
        print(f"Discovery trigger failed: {e}")

    result = poll_for_fact(run_id, "news", timeout=240)
    return {"facts": state.get("facts", []) + [{"agent": "intel", "data": result or {}}]}

def screening_agent_node(state: AgentState):
    """Stage 2: Screener & Regression Modeling (ss6_sqlite)."""
    run_id = state.get('run_id')
    print(f"--- [Agent: Screener] Applying Professional Model Scoring ---")
    
    # Identify candidates from intel or default to watchlist
    intel_data = {}
    for f in state.get("facts", []):
        if f.get("agent") == "intel":
            intel_data = f.get("data", {})
            break
            
    # Use discovery tool (renamed to batch_score for semantic clarity)
    try:
        payload = {"run_id": run_id}
        payload.update(_extract_tool_payload(state, "quant.discovery_workflow"))
        trigger_tool("quant.discovery_workflow", payload, run_id)
    except Exception as e:
        print(f"Screening trigger failed: {e}")

    result = poll_for_fact(run_id, "quant", timeout=240)
    return {"facts": state.get("facts", []) + [{"agent": "screener", "data": result or {}}]}

def quant_agent_node(state: AgentState):
    # Existing analysis logic...
    run_id = state.get('run_id')
    symbol = state.get('symbol')
    print(f"--- [Agent: Quant] Real-time Analysis for {symbol} ---")
    try:
        payload = {"symbol": symbol, "run_id": run_id}
        payload.update(_extract_tool_payload(state, "quant.deep_analysis"))
        payload["symbol"] = payload.get("symbol") or symbol
        trigger_tool("quant.deep_analysis", payload, run_id)
    except Exception as e:
        return {"facts": state.get("facts", []) + [{"agent": "quant", "data": {"error": str(e)}}]}
    result = poll_for_fact(run_id, "quant", timeout=90)
    return {"facts": state.get("facts", []) + [{"agent": "quant", "data": result or {}}]}

def browser_agent_node(state: AgentState):
    run_id = state.get('run_id')
    symbol = state.get('symbol')
    print(f"--- Triggering Real Browser for {symbol} ---")

    try:
        trigger_tool("browser.screenshot", {"symbol": symbol, "run_id": run_id}, run_id)
    except Exception as e:
        print(f"[Brain] browser trigger failed: {e}")
        return {"facts": state.get("facts", []) + [{"agent": "browser", "data": {"error": str(e), "url": "Data unavailable"}}]}

    result = poll_for_fact(run_id, "browser", timeout=45)
    if result:
        return {"facts": state.get("facts", []) + [{"agent": "browser", "data": result}]}
    return {"facts": state.get("facts", []) + [{"agent": "browser", "data": {"error": "timeout", "url": "Data unavailable"}}]}

def writer_agent_node(state: AgentState):
    mode = state.get("mode", "analysis")
    grounded = _build_grounded_narrative(state)
    report_markdown = None
    report_html_object_key = None
    
    # Extract quant artifacts if present
    for f in state.get("facts", []):
        if f.get("agent") == "quant":
            quant_data = f.get("data", {}) or {}
            report_markdown = quant_data.get("report_markdown")
            report_html_object_key = quant_data.get("report_html_object_key")
            break

    if mode == "discovery":
        screener_fact = {}
        for f in state.get("facts", []):
            if f.get("agent") == "screener":
                screener_fact = f.get("data", {})
                break
        candidates = screener_fact.get("candidates", [])
        if candidates:
            lines = ["# 优质股票推荐报告 (Discovery Report)", ""]
            for idx, c in enumerate(candidates, 1):
                alpha_val = c.get("score")
                try:
                    alpha_text = f"{float(alpha_val):.3f}"
                except Exception:
                    alpha_text = "N/A"
                lines.append(
                    f"{idx}. **{c.get('symbol', 'N/A')}** | Signal: {c.get('signal', 'N/A')} | "
                    f"Alpha: {alpha_text} | Risk: {c.get('risk', c.get('risk_state', 'unknown'))}"
                )
            plan = screener_fact.get("position_plan") or {}
            if plan:
                lines.append("")
                lines.append("## 建仓规划")
                lines.append(f"- 目标: {plan.get('goal', 'N/A')}")
                lines.append(f"- 资金: {plan.get('capital_base_jpy', 'N/A')} JPY")
                lines.append(f"- 分批节奏: {plan.get('entry_style', 'N/A')}")
            lines.append("\n以上标的经 intelligence 发现并由 ss6_sqlite 专业模型评分验证。")
            grounded = "\n".join(lines)
            
    # Use LLM for final polish if enabled
    use_llm_writer = str(os.getenv("USE_LLM_WRITER", "1")).lower() in ["1", "true", "yes"]
    if not use_llm_writer:
        out = {"narrative": grounded}
        if report_markdown: out["report_markdown"] = report_markdown
        if report_html_object_key: out["report_html_object_key"] = report_html_object_key
        return out

    facts_str = json.dumps(state.get("facts", []), indent=2)
    prompt = f"""
    You are a professional investment analyst.
    Review these RAW FACTS collected by our agents:
    {facts_str}

    TASK: Rewrite facts into a concise Chinese report.
    STRICT:
    - Do not invent any data or news.
    - Missing fields must be 'Data unavailable'.
    - Output <= 1200 Chinese characters.
    """
    llm = get_llm(state)
    if llm is not None:
        try:
            response = llm.invoke(prompt)
            content = str(response.content).strip()
            if content:
                out = {"narrative": content}
                if report_markdown: out["report_markdown"] = report_markdown
                if report_html_object_key: out["report_html_object_key"] = report_html_object_key
                return out
        except Exception as e:
            print(f"[Brain] LLM writer failed, fallback to raw grounded text: {e}")

    out = {"narrative": grounded}
    if report_markdown: out["report_markdown"] = report_markdown
    if report_html_object_key: out["report_html_object_key"] = report_html_object_key
    return out


# Re-build Graph
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("discovery", discovery_agent_node)
builder.add_node("screening", screening_agent_node)
builder.add_node("quant", quant_agent_node)
builder.add_node("browser", browser_agent_node)
builder.add_node("writer", writer_agent_node)

builder.set_entry_point("supervisor")

# Route based on next_step
builder.add_conditional_edges("supervisor", lambda x: x["next_step"], {
    "discovery": "discovery",
    "screening": "screening",
    "quant": "quant",
    "browser": "browser",
    "writer": "writer"
})

builder.add_edge("discovery", "supervisor")
builder.add_edge("screening", "supervisor")
builder.add_edge("quant", "supervisor")
builder.add_edge("browser", "supervisor")
builder.add_edge("writer", END)
brain_graph = builder.compile()
