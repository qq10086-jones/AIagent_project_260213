CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  tool_name TEXT NOT NULL,
  status TEXT NOT NULL,
  risk_level TEXT NOT NULL DEFAULT 'low',
  payload_json TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS event_log (
  id BIGSERIAL PRIMARY KEY,
  task_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_log_task_id ON event_log(task_id);

CREATE TABLE IF NOT EXISTS assets (
  asset_id BIGSERIAL PRIMARY KEY,
  task_id TEXT NOT NULL,
  object_key TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  mime_type TEXT,
  file_size BIGINT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS workflows (
  workflow_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  definition_json TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- --- Phase 0: MAS Support ---

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  client_msg_id TEXT UNIQUE,
  user_id TEXT,
  status TEXT NOT NULL,
  input_text TEXT,
  cost_ledger_json TEXT DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fact_items (
  fact_id TEXT PRIMARY KEY,
  run_id TEXT REFERENCES runs(run_id),
  agent_name TEXT NOT NULL,
  kind TEXT NOT NULL, -- e.g., 'price', 'news_summary', 'financial_ratio'
  payload_json TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY,
  url TEXT,
  captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  screenshot_ref_json TEXT, -- {object_key, sha256}
  extracted_text TEXT,
  content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS links (
  fact_id TEXT REFERENCES fact_items(fact_id),
  evidence_id TEXT REFERENCES evidence(evidence_id),
  PRIMARY KEY (fact_id, evidence_id)
);

ALTER TABLE tasks ADD COLUMN IF NOT EXISTS run_id TEXT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS idempotency_key TEXT UNIQUE;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS workflow_id TEXT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS step_index INT;

