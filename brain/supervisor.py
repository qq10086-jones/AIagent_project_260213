import os
import json
import requests
import time
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from state import AgentState
from runtime_config import (
    LLM_PROVIDER_DEFAULT,
    OLLAMA_BASE_URL_DEFAULT,
    LOCAL_MODEL_DEFAULT,
    QWEN_MODEL_DEFAULT,
    QWEN_BASE_URL_DEFAULT,
    LLM_TIMEOUT_SEC_DEFAULT,
)

ORCHESTRATOR_URL = "http://orchestrator:3000"
# --- LLM Config ---
LLM_PROVIDER = LLM_PROVIDER_DEFAULT
OLLAMA_BASE_URL = OLLAMA_BASE_URL_DEFAULT
OLLAMA_CHAT_API = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_GENERATE_API = f"{OLLAMA_BASE_URL}/api/generate"
LOCAL_WRITER_MODEL = LOCAL_MODEL_DEFAULT
QWEN_MODEL = QWEN_MODEL_DEFAULT

def poll_for_fact(run_id, agent_name, timeout=120, tool_name=None):
        """Wait for a specific agent (and optionally a specific tool) via orchestrator HTTP."""
        start_time = time.time()
        print(f"[Brain] Polling for {agent_name} {tool_name or ''} results (run_id: {run_id})...")
        while time.time() - start_time < timeout:
            try:
                resp = requests.get(
                    f"{ORCHESTRATOR_URL}/brain/facts/latest",
                    params={
                        "run_id": run_id,
                        "agent_name": agent_name,
                        "tool_name": tool_name or "",
                    },
                    timeout=10,
                )
                if resp.status_code == 404:
                    time.sleep(2)
                    continue
                if resp.status_code >= 300:
                    raise RuntimeError(f"brain facts gateway failed {resp.status_code}: {resp.text[:200]}")
                data = resp.json() or {}
                fact = data.get("fact") or {}
                payload = fact.get("payload")
                if payload is not None:
                    if isinstance(payload, dict):
                        return payload
                    try:
                        return json.loads(payload)
                    except Exception:
                        return {"raw": str(payload)}
            except Exception as e:
                print(f"Fact poll error: {e}")
            time.sleep(2)
        return None

def get_llm(state: AgentState = None):
    """Returns a LangChain-compatible LLM based on LLM_PROVIDER."""
    model_preference = str((state or {}).get("model_preference", "")).lower()
    if model_preference in ["local_small", "local_large"]:
        return None

    prov = LLM_PROVIDER
    if model_preference == "api" and prov == "ollama":
        prov = "dashscope" if os.getenv("QWEN_API_KEY") else "openai"
    if prov == "ollama":
        # Using ChatOpenAI with Ollama's OpenAI-compatible endpoint if available, 
        # or just fallback to the manual call_local_writer in writer_agent_node.
        # For now, let's stick to the current logic but make it provider-aware.
        return None 
    
    if prov in ["dashscope", "openai"]:
        api_key = os.getenv("QWEN_API_KEY") if prov == "dashscope" else os.getenv("OPENAI_API_KEY")
        base_url = QWEN_BASE_URL_DEFAULT if prov == "dashscope" else os.getenv("OPENAI_BASE_URL")
        
        default_model = QWEN_MODEL if prov == "dashscope" else os.getenv("OPENAI_MODEL", "gpt-4o")
        model = state.get("qwen_model", default_model) if state else default_model

        if not api_key:
            return None
            
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=LLM_TIMEOUT_SEC_DEFAULT
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
    return call_local_writer_with_model(prompt, LOCAL_WRITER_MODEL)

def call_local_writer_with_model(prompt: str, model_name: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_CHAT_API,
            json={
                "model": model_name or LOCAL_WRITER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=90,
        )
        if resp.status_code == 404:
            gen_resp = requests.post(
                OLLAMA_GENERATE_API,
                json={
                    "model": model_name or LOCAL_WRITER_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=90,
            )
            data = gen_resp.json()
            return data.get("response", "").strip()
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
            "avoid_mega_cap", "prefer_small_mid_cap", "quick_mode",
            "time_budget_s", "max_attempts", "min_candidates",
            "auto_evolve", "auto_expand_market", "enable_learning"
        },
        "quant.deep_analysis": {
            "symbol", "capital_base_jpy", "capital_base", "capital_headroom_pct",
            "max_position_pct", "mode"
        },
    }
    allowed = allow_map.get(tool_name, set())
    return {k: v for k, v in raw.items() if k in allowed}

def coder_agent_node(state: AgentState):
    """
    Coding Agent: Handles autonomous code modification and execution.
    Interacts via orchestrator's /execute-tool and worker-coder.
    """
    run_id = state.get('run_id')
    
    # Robustly extract task prompt from messages
    task_prompt = "Fix bugs in the codebase."
    if state.get('messages'):
        last_msg = state.get('messages')[-1]
        if isinstance(last_msg, dict):
            task_prompt = last_msg.get('content', task_prompt)
        else:
            task_prompt = getattr(last_msg, 'content', task_prompt)
            
    print(f"--- [Agent: Coder] Starting Task: {task_prompt[:50]}... ---")
    
    try:
        prompt = f"""
        You are a senior developer. Task: {task_prompt}
        Generate a SEARCH/REPLACE block for the fix.
        Format:
        FILE: [relative_path]
        <<<<<<< SEARCH
        [lines]
        =======
        [lines]
        >>>>>>> REPLACE
        
        Then suggest a test command.
        COMMAND: [cmd]
        """
        
        llm_response = call_local_writer_with_model(prompt, state.get("local_model", LOCAL_WRITER_MODEL))
        
        # Robust extraction
        import re
        
        file_path = None
        file_match = re.search(r"(?:FILE|File|Target File):\s*([^\s\n\r]+)", llm_response)
        if file_match:
            file_path = file_match.group(1).strip()
            
        edit_blocks = []
        matches = re.findall(r"(<<<<<<< SEARCH[\s\S]+?>{6,7}\s*REPLACE)", llm_response)
        if matches:
            edit_blocks = matches
            
        cmd = None
        cmd_match = re.search(r"(?:COMMAND|Command|Test Command):\s*(.+)", llm_response)
        if cmd_match:
            cmd = cmd_match.group(1).strip()
            
        results = []
        if file_path and edit_blocks:
            full_patch = "\n\n".join(edit_blocks)
            print(f"[Coder] Triggering patch for {file_path}")
            trigger_tool("coding.patch", {
                "file_path": file_path,
                "edit_block": full_patch
            }, run_id)
            patch_result = poll_for_fact(run_id, "coder", timeout=60, tool_name="coding.patch")
            if patch_result:
                results.append(patch_result.get("output", {}))
            
        if cmd:
            print(f"[Coder] Triggering test command: {cmd}")
            trigger_tool("coding.execute", {
                "command": cmd
            }, run_id)
            cmd_result = poll_for_fact(run_id, "coder", timeout=15, tool_name="coding.execute")
            if cmd_result:
                results.append(cmd_result.get("output", {}))
            
        if not results:
            print(f"[Coder] Warning: No actions extracted from LLM response. Content: {llm_response[:100]}...")
            
        return {"facts": state.get("facts", []) + [{"agent": "coder", "data": {"results": results, "llm_raw": llm_response, "parsed": {"file": file_path, "blocks_count": len(edit_blocks), "cmd": cmd}}}]}
    except Exception as e:
        print(f"[Brain] Coder agent failed: {e}")
        return {"facts": state.get("facts", []) + [{"agent": "coder", "data": {"error": str(e)}}]}

def _is_single_agent_mode(state: AgentState) -> bool:
    """
    判断是否走 single_agent 路径。
    满足以下任一条件即路由到 single_agent：
      1. state.mode == "single_agent"（显式指定）
      2. task_envelope.decision == "single_agent"（来自 OpenClaw 路由决策）
    适用场景（参见设计文档 7.1）：
      - coding bug fix（预估影响 < 3 个文件）
      - quant 查询分析（非批量 pipeline）
      - 简单 web 研究
    """
    if str(state.get("mode", "")).strip() == "single_agent":
        return True
    task_envelope = state.get("task_envelope")
    if isinstance(task_envelope, dict) and str(task_envelope.get("decision", "")).strip() == "single_agent":
        return True
    return False


def supervisor_node(state: AgentState):
    """
    Core Logic: Task Decomposition & Routing.
    """
    # single_agent 路径：轻量微工作流，绕过完整 DAG pipeline
    if _is_single_agent_mode(state):
        return {"next_step": "single_agent_dispatch"}

    mode = state.get("mode", "analysis")
    existing = [f['agent'] for f in state.get('facts', [])]

    if mode == "coding":
        if "coder" not in existing:
            return {"next_step": "coder"}
        return {"next_step": "writer"}
    elif mode == "discovery":
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
    discovery_payload = _extract_tool_payload(state, "quant.discovery_workflow")
    is_fast_discovery = bool(discovery_payload.get("quick_mode")) or (
        "capital_base_jpy" in discovery_payload and state.get("mode") == "discovery"
    )
    if is_fast_discovery:
        return {"facts": state.get("facts", []) + [{"agent": "intel", "data": {"skipped": True, "reason": "quick_mode"}}]}

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
        if "capital_base_jpy" in payload and "quick_mode" not in payload:
            payload["quick_mode"] = True
            payload["time_budget_s"] = min(int(payload.get("time_budget_s") or 75), 75)
            payload["max_attempts"] = min(int(payload.get("max_attempts") or 2), 2)
            payload["min_candidates"] = min(int(payload.get("min_candidates") or 2), 2)
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

    local_model = state.get("local_model", LOCAL_WRITER_MODEL)
    local_content = call_local_writer_with_model(prompt, local_model)
    if local_content:
        out = {"narrative": local_content}
        if report_markdown: out["report_markdown"] = report_markdown
        if report_html_object_key: out["report_html_object_key"] = report_html_object_key
        return out

    out = {"narrative": grounded}
    if report_markdown: out["report_markdown"] = report_markdown
    if report_html_object_key: out["report_html_object_key"] = report_html_object_key
    return out


def single_agent_dispatch_node(state: AgentState):
    """
    single_agent 微工作流分发节点（BR-01 / SA-01）。

    设计原则（参见设计文档 7.1–7.3）：
    - single_agent 不是 DAG bypass，是轻量但有质量门控的执行路径
    - 适用场景：coding bug fix（< 3 文件）、quant 查询、简单 web 研究
    - 必须携带 evidence_id + replay_tag（来自 task_envelope 或 run_id 衍生）
    - 输出 decision: single_agent，由 OpenClaw 负责验证四项门控后执行

    注意：该节点不直接执行任务，只做路由分发决策输出，
    实际执行由 worker-coder / worker-quant 完成，审计由 AuditHooks 覆盖。
    """
    run_id = str(state.get("run_id") or "")
    task_envelope = state.get("task_envelope") if isinstance(state.get("task_envelope"), dict) else {}

    # evidence_id 优先从 task_envelope 取；兜底从 run_id 衍生
    evidence_id = str(task_envelope.get("evidence_id") or "").strip()
    if not evidence_id and run_id:
        evidence_id = f"sa-{run_id[:16]}"

    # replay_tag：任务输入参数快照摘要，用于重放
    replay_tag = str(task_envelope.get("replay_tag") or "").strip()
    if not replay_tag:
        import hashlib
        tag_source = json.dumps({
            "run_id": run_id,
            "mode": state.get("mode", ""),
            "tool_name": task_envelope.get("tool_name", ""),
        }, sort_keys=True, ensure_ascii=False)
        replay_tag = hashlib.sha256(tag_source.encode()).hexdigest()[:16]

    # 根据 mode/task_envelope 判断路由目标 worker
    mode = str(state.get("mode", "")).strip()
    tool_name = str(task_envelope.get("tool_name", "")).strip()
    if mode in ("coding", "coding_single") or tool_name.startswith("coding."):
        target_worker = "worker-coder"
    elif mode in ("quant", "quant_single") or tool_name.startswith("quant."):
        target_worker = "worker-quant"
    else:
        target_worker = "worker-coder"  # 默认 coding

    print(f"[Brain:single_agent_dispatch] run_id={run_id} evidence_id={evidence_id} "
          f"target_worker={target_worker} tool_name={tool_name or '(none)'}")

    return {
        "narrative": "",
        "decision": "single_agent",
        "single_agent_meta": {
            "evidence_id": evidence_id,
            "replay_tag": replay_tag,
            "target_worker": target_worker,
            "tool_name": tool_name,
            "run_id": run_id,
        },
    }


# Re-build Graph
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("coder", coder_agent_node)
builder.add_node("discovery", discovery_agent_node)
builder.add_node("screening", screening_agent_node)
builder.add_node("quant", quant_agent_node)
builder.add_node("browser", browser_agent_node)
builder.add_node("writer", writer_agent_node)
builder.add_node("single_agent_dispatch", single_agent_dispatch_node)

builder.set_entry_point("supervisor")

# Route based on next_step
builder.add_conditional_edges("supervisor", lambda x: x["next_step"], {
    "coder": "coder",
    "discovery": "discovery",
    "screening": "screening",
    "quant": "quant",
    "browser": "browser",
    "writer": "writer",
    "single_agent_dispatch": "single_agent_dispatch",
})

builder.add_edge("coder", "supervisor")
builder.add_edge("discovery", "supervisor")
builder.add_edge("screening", "supervisor")
builder.add_edge("quant", "supervisor")
builder.add_edge("browser", "supervisor")
builder.add_edge("writer", END)
builder.add_edge("single_agent_dispatch", END)
brain_graph = builder.compile()
