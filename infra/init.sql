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
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS result_json TEXT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS error_code TEXT;

CREATE TABLE IF NOT EXISTS workflow_runs (
  workflow_run_id TEXT PRIMARY KEY,
  run_id TEXT,
  workflow_id TEXT NOT NULL,
  project_type TEXT NOT NULL,
  status TEXT NOT NULL,
  current_step_index INT NOT NULL DEFAULT 0,
  last_checkpoint_id TEXT,
  resume_token TEXT,
  input_json TEXT NOT NULL DEFAULT '{}',
  error_code TEXT,
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS workflow_steps (
  id BIGSERIAL PRIMARY KEY,
  workflow_run_id TEXT NOT NULL,
  step_index INT NOT NULL,
  step_id TEXT NOT NULL,
  role_name TEXT,
  tool_name TEXT,
  gate_name TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  task_id TEXT,
  risk_level TEXT,
  approval_required BOOLEAN NOT NULL DEFAULT FALSE,
  approval_reasons_json TEXT NOT NULL DEFAULT '[]',
  checkpoint_id TEXT,
  result_json TEXT,
  error_code TEXT,
  started_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(workflow_run_id, step_index)
);

CREATE TABLE IF NOT EXISTS workflow_checkpoints (
  checkpoint_id TEXT PRIMARY KEY,
  workflow_run_id TEXT NOT NULL,
  step_index INT NOT NULL,
  step_id TEXT NOT NULL,
  task_id TEXT,
  workspace_hash TEXT NOT NULL,
  artifact_refs_json TEXT NOT NULL DEFAULT '[]',
  checkpoint_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workflow_runs_run_id ON workflow_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_run ON workflow_steps(workflow_run_id, step_index);
CREATE INDEX IF NOT EXISTS idx_workflow_cp_run ON workflow_checkpoints(workflow_run_id, step_index);

