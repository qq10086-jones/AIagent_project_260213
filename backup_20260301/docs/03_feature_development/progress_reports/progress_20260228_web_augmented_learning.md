# OpenClaw Nexus Progress Update

- **Date**: 2026-02-28
- **Status**: Milestone Reached - "The Learning & Connected Nexus"
- **Scope**: Integrated Learning Loop, Adaptive Inference, and Web-Augmented Chat (ReAct)

## Today's Achievements

### 1. Zero-Budget Learning Loop (MVP-1 & MVP-2)
- **Hard Rule Engine**: Implemented post-processing interception for local LLM replies. Automatically detects project context (OpenClaw/Quant) and triggers a **single-pass rewrite** if hard constraints (e.g., PowerShell usage, Core Algorithm changes) are violated.
- **Feedback-to-Rule/SOP**: 
    - Created a data foundation in Postgres (`projects`, `rules`, `mem_items`, `traces`).
    - **Discord Integration**: Hooked into Discord reactions (💯, 👍, 👎). 
        - 👎 triggers automatic "Soft Rule" generation based on user reason.
        - 💯/👍 triggers automatic "SOP/Memory" creation.
- **Context Builder**: Implemented dynamic prompt injection. Local LLM now reads the top 5 relevant rules and top 3 SOPs from the database before every reply, allowing NEXUS to "remember" your preferences and past mistakes.

### 2. Stability & UX Refinement
- **Adaptive Token Strategy**: Implemented a "Smart Fallback" for `num_predict`. 
    - Defaults to **4096 tokens** for deep reasoning.
    - Automatically downgrades to **2048 tokens** if latency exceeds 90s or a timeout occurs, preventing system hangs.
- **Conversational Memory**: Added native support for multi-turn dialogue. Orchestrator now fetches the **last 6 messages** from the Discord channel to provide full context to the local model.
- **Output Sanitation**: Fixed regex filters that were over-aggressively masking valid reasoning-heavy replies.

### 3. Web-Augmented Chat (Phase 1: Light Track)
- **Search Intent Detection**: Upgraded the NLP Router to recognize queries requiring real-time data (weather, stocks, news).
- **Worker Search Plugin**: Integrated `ddgs` (DuckDuckGo) into the Python Worker.
- **ReAct Loop (Synthesis)**: Orchestrator now intercepts search results, injects them into a "Synthesis Prompt," and forces the LLM to provide a natural language summary with **source citations** instead of a dry task card.

## System State
- **Orchestrator**: Operational with Learning APIs and Discord Reaction listeners.
- **Worker-Quant**: Operational with `ddgs` search capabilities.
- **Database**: Schema updated with learning and trace tables.
- **Local LLM**: DeepSeek-R1:32b is now "connected" and "learning."

## Next Steps
1. **Phase 2 Web**: Deep integration with **OpenClaw** for full-page browsing and JS-heavy site extraction.
2. **Preference Learning**: Implement the Bradley-Terry preference model for A/B testing of templates.
3. **Observation**: Monitor the quality of auto-generated rules from Discord reactions.

## Incremental Update (2026-02-28, Quant Routing & News Quality)

### 4. Quant Routing Reliability (Model-Switch Safe)
- Added **rule-based forced tool fallback** in orchestrator when LLM intent parsing returns chat/low-confidence/no-tool.
- For geo-impact requests (e.g., Middle East tensions -> Japan market), routing now reliably enters workflow mode and dispatches:
  - `news.active_hot_search`
  - `quant.discovery_workflow`
- Verified via `/chat` regression that the same Chinese prompt now consistently produces workflow tasks instead of generic chat completion.

### 5. Market/Currency Alignment Fix (JP-first)
- Fixed mismatch where JP news context could still yield US candidates/USD-oriented recommendations.
- Geo-impact JP route now sends explicit payload:
  - `market = JP`
  - `auto_expand_market = false`
- In `discovery_workflow`, explicit user market is now treated as a strong constraint (no unintended auto expansion).
- Added account/position-aware defaults to improve market and capital context consistency.

### 6. Holdings-Priority Global Hot News
- Upgraded active hot news collection to a **global source mix** (macro/geopolitics/energy + multi-language RSS/GDELT).
- When holdings exist, system now enriches and prioritizes holdings-linked news.
- Added fallback freshness widening for holdings-linked news when strict 24h window is sparse.
- Normalized JP numeric symbols (e.g., `9432` -> `9432.T`) to avoid quote/news lookup failures.

### 7. Blank Screenshot Artifact Mitigation
- Diagnosed issue of occasional blank screenshot attachments from browser capture path.
- Added blank-image heuristic filter (near-solid bright/dark frame detection) before returning screenshot artifacts.
- Added observable diagnostics in `news.active_hot_search` output:
  - `artifact_capture_stats` with `requested`, `attempted`, `archived_ok`, `blank_filtered`, `kept`, `start_ok`, `skipped_reason`.
- Smoke test confirmed filtering behavior (blank frames filtered out, no useless image attachment emitted).

### 8. Documentation & Ops
- Added formal quant design document:
  - `docs/01_design/quant/quant_design_latest.md`
- Restarted and validated core services for this patch set:
  - `nexus-orchestrator`
  - `nexus-worker-quant`
  - `brain`
