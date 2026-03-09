/**
 * Bootstrap schema helpers for orchestrator startup.
 */

/**
 * @param {import('pg').Pool} pool
 */
export async function ensureOrchestratorSchema(pool) {
  await pool.query("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS workflow_id TEXT");
  await pool.query("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS step_index INT");
  await pool.query("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS result_json TEXT");
  await pool.query("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS error_code TEXT");

  await pool.query(
    `CREATE TABLE IF NOT EXISTS workflow_runs(
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
    )`
  );
  await pool.query(
    `CREATE TABLE IF NOT EXISTS workflow_steps(
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
    )`
  );
  await pool.query(
    `CREATE TABLE IF NOT EXISTS workflow_checkpoints(
      checkpoint_id TEXT PRIMARY KEY,
      workflow_run_id TEXT NOT NULL,
      step_index INT NOT NULL,
      step_id TEXT NOT NULL,
      task_id TEXT,
      workspace_hash TEXT NOT NULL,
      artifact_refs_json TEXT NOT NULL DEFAULT '[]',
      checkpoint_json TEXT NOT NULL DEFAULT '{}',
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )`
  );
  await pool.query("CREATE INDEX IF NOT EXISTS idx_workflow_runs_run_id ON workflow_runs(run_id)");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_workflow_steps_run ON workflow_steps(workflow_run_id, step_index)");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_workflow_cp_run ON workflow_checkpoints(workflow_run_id, step_index)");

  await pool.query("CREATE TABLE IF NOT EXISTS projects(project_id TEXT PRIMARY KEY, name TEXT, profile_json TEXT, updated_at TIMESTAMPTZ DEFAULT NOW())");
  await pool.query("CREATE TABLE IF NOT EXISTS rules(rule_id TEXT PRIMARY KEY, project_id TEXT, scope TEXT, rule_type TEXT, rule_json TEXT, weight INT, updated_at TIMESTAMPTZ DEFAULT NOW())");
  await pool.query("CREATE TABLE IF NOT EXISTS mem_items(mem_id TEXT PRIMARY KEY, project_id TEXT, type TEXT, content TEXT, tags TEXT, alpha INT DEFAULT 1, beta INT DEFAULT 1, created_at TIMESTAMPTZ DEFAULT NOW())");
  await pool.query("CREATE TABLE IF NOT EXISTS traces(trace_id TEXT PRIMARY KEY, project_id TEXT, task_type TEXT, context_digest TEXT, action_json TEXT, metrics_json TEXT, feedback_json TEXT, created_at TIMESTAMPTZ DEFAULT NOW())");

  // WS-30-01: Routing Decision Audit Log (Design Doc v4.0 Section 5.3)
  await pool.query(
    `CREATE TABLE IF NOT EXISTS routing_decision_log (
      id BIGSERIAL PRIMARY KEY,
      log_id TEXT NOT NULL UNIQUE,
      run_id TEXT NOT NULL,
      workflow_run_id TEXT,
      workflow_id TEXT,
      router_mode TEXT,
      dynamic_routing_enabled BOOLEAN,
      classifier_version TEXT,
      classifier_confidence NUMERIC(6,4),
      classifier_confidence_band TEXT,
      classifier_work_shape TEXT,
      classifier_domain_lead TEXT,
      classifier_parallel_candidate BOOLEAN,
      classifier_model_tier TEXT,
      classifier_deny_or_degrade_reason TEXT,
      feature_snapshot_ref TEXT,
      routing_decision_source TEXT NOT NULL,
      final_execution_decision TEXT NOT NULL,
      safety_override_result TEXT,
      decision_json TEXT NOT NULL DEFAULT '{}',
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )`
  );
  await pool.query("CREATE INDEX IF NOT EXISTS idx_rdl_run_id ON routing_decision_log(run_id)");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_rdl_workflow_run_id ON routing_decision_log(workflow_run_id)");

  // WS-30-02: Waterfall Trace / Latency Attribution (Design Doc v4.0 § 7)
  await pool.query(
    `CREATE TABLE IF NOT EXISTS waterfall_stage_log (
      id BIGSERIAL PRIMARY KEY,
      run_id TEXT NOT NULL,
      workflow_run_id TEXT,
      stage TEXT NOT NULL,
      started_at TIMESTAMPTZ NOT NULL,
      ended_at TIMESTAMPTZ,
      duration_ms INTEGER,
      metadata_json TEXT NOT NULL DEFAULT '{}',
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )`
  );
  await pool.query("CREATE INDEX IF NOT EXISTS idx_wsl_run_id ON waterfall_stage_log(run_id)");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_wsl_workflow_run_id ON waterfall_stage_log(workflow_run_id)");
  await pool.query("CREATE INDEX IF NOT EXISTS idx_wsl_stage ON waterfall_stage_log(stage, started_at DESC)");
}
