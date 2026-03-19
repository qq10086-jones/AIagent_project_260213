# OpenBB Intelligence Integration Design

## 1. Overview
The goal of this project is to integrate the OpenBB Platform (v4) into the `worker-quant` service of the Nexus project. This replaces the fragile BeautifulSoup/RSS scraping logic with an industrial-grade, multi-source financial news and data aggregation API, improving the quality of sentiment analysis and alpha generation.

## 2. Architecture Changes
- **Dependency Layer**: Add `openbb` to `worker-quant/requirements.txt`.
- **Data Fetching Layer (`worker.py`)**: 
  - Introduce `_fetch_news_from_openbb` as the primary news retrieval method.
  - Implement a fallback mechanism: If OpenBB fails or returns insufficient data, fallback to `_fetch_news_from_quote_page` (Yahoo Finance) and `_fetch_news_from_google_rss`.
- **Data Normalization**: Map OpenBB's Unified Data Model (UDM) into the existing worker schema (`title`, `url`, `publisher`, `published_at`, `source`).

## 3. Risk Mitigation & QA Strategy
- **Rate Limiting / Authentication**: OpenBB relies on third-party providers. Default providers without API keys have strict rate limits. **Requirement**: Inject API keys via environment variables (e.g., `OPENBB_FMP_KEY`).
- **Synchronous Blocking**: The `worker.py` runs a message loop. Synchronous network calls to OpenBB must have explicit timeouts to prevent worker starvation.
- **Dependency Conflicts**: OpenBB is heavy. Need to ensure compatibility with existing `pandas`, `numpy`, and `yfinance` versions.
- **Deduplication**: Ensure identical news articles fetched from both OpenBB and the fallback mechanisms are deduplicated before LLM sentiment scoring.

## 4. Task List
- [x] Phase 1.1: Inject `openbb` into `requirements.txt`.
- [x] Phase 1.2: Implement `_fetch_news_from_openbb` in `worker.py`.
- [x] Phase 1.3: Update `_merge_recent_news` to use OpenBB with fallback.
- [x] Phase 2.1: Add explicit HTTP timeouts and retry logic to OpenBB calls.
- [x] Phase 2.2: Configure OpenBB provider API keys (FMP, Benzinga, etc.) via Docker `ENV`.
- [x] Phase 2.3: Validate deduplication logic in `_merge_recent_news` handles OpenBB format.
- [ ] Phase 2.4: Run a full container build (`docker-compose build`) to check for Python dependency conflicts.
- [ ] Phase 3.1: Write unit tests verifying the fallback mechanism triggers correctly when OpenBB is offline.