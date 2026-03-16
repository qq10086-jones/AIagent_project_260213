/**
 * discord_progress_manager.js
 * 
 * Manages the "latest state snapshot" and debounced editing of Discord messages
 * for real-time workflow progress bars.
 */

// In-memory state map for M12 Phase A (SQLite/Redis later if needed for multi-instance)
const snapshots = new Map();

/**
 * Ensures a snapshot exists for the run_id.
 */
function getOrInitSnapshot(run_id, channel_id, message_id) {
  if (!snapshots.has(run_id)) {
    snapshots.set(run_id, {
      run_id,
      channel_id,
      message_id,
      current_step: "unknown",
      current_step_index: 0,
      total_steps: 0,
      current_status: "WAITING",
      action_summary: "Initializing...",
      last_heartbeat_ts: Date.now(),
      dirty: true,
      debounceTimer: null
    });
  }
  return snapshots.get(run_id);
}

/**
 * Format the progress bar text
 */
function formatProgressMessage(snapshot) {
  const { current_step, current_step_index, total_steps, current_status, action_summary, last_heartbeat_ts } = snapshot;
  
  let header = `🛠️ **项目制作中**`;
  if (current_status === "COMPLETED") header = `🚀 **项目开发完成！**`;
  if (current_status === "FAILED") header = `❌ **项目执行失败**`;

  let bar = "";
  if (total_steps > 0) {
    const filled = Math.min(total_steps, Math.max(0, current_step_index));
    const empty = total_steps - filled;
    bar = `[${"=".repeat(filled)}${" ".repeat(empty)}] (${filled}/${total_steps})`;
  } else {
    bar = `[⏳] (排队中...)`;
  }

  let statusIcon = "⏳";
  if (current_status === "RUNNING") statusIcon = "🏃";
  if (current_status === "COMPLETED") statusIcon = "✅";
  if (current_status === "FAILED") statusIcon = "❌";
  if (current_status === "RETRYING") statusIcon = "⚠️";

  const ago = Math.floor((Date.now() - last_heartbeat_ts) / 1000);
  const heartbeatText = current_status === "RUNNING" || current_status === "RETRYING" ? ` | Last activity: ${ago}s ago` : "";

  return `${header}\n${statusIcon} ${bar} \`${current_step}\`\n💬 ${action_summary}${heartbeatText}`;
}

/**
 * Handle incoming events and update snapshot
 */
export function handleWorkflowEvent(discordClient, eventType, payload) {
  const { workflow_run_id, channel_id, message_id, step_id, step_index, step_count, action_summary, error_message, result_url } = payload;
  
  if (!workflow_run_id || !channel_id || !message_id) return;

  const snapshot = getOrInitSnapshot(workflow_run_id, channel_id, message_id);
  snapshot.last_heartbeat_ts = Date.now();
  snapshot.dirty = true;

  if (eventType === "workflow.started") {
    snapshot.total_steps = step_count || 0;
    snapshot.current_status = "RUNNING";
    snapshot.action_summary = "Workflow started...";
  } else if (eventType === "step.started") {
    snapshot.current_step = step_id;
    snapshot.current_step_index = (step_index || 0) + 1;
    snapshot.current_status = "RUNNING";
    snapshot.action_summary = action_summary || `Running step ${step_id}...`;
  } else if (eventType === "step.progress") {
    snapshot.action_summary = action_summary || snapshot.action_summary;
    snapshot.current_status = "RUNNING";
  } else if (eventType === "step.completed") {
    snapshot.current_step = step_id;
    snapshot.current_step_index = (step_index || 0) + 1;
    // Don't change current_status here, keep it RUNNING for the next step, or let workflow.completed handle it
    snapshot.action_summary = `Completed ${step_id}`;
  } else if (eventType === "step.failed") {
    snapshot.current_status = "FAILED";
    snapshot.action_summary = `Error: ${error_message || "Unknown error"}`;
  } else if (eventType === "workflow.completed") {
    snapshot.current_status = "COMPLETED";
    snapshot.action_summary = "All steps finished successfully.";
    if (result_url) {
      snapshot.action_summary += `\n\n🔗 **Live Preview:** ${result_url}`;
    }
  } else if (eventType === "workflow.failed") {
    snapshot.current_status = "FAILED";
    snapshot.action_summary = `Workflow Failed: ${error_message || "Unknown error"}`;
  }

  // Trigger render
  requestRender(discordClient, snapshot, eventType === "workflow.completed" || eventType === "workflow.failed");
}

function requestRender(discordClient, snapshot, forceImmediate) {
  if (forceImmediate) {
    if (snapshot.debounceTimer) {
      clearTimeout(snapshot.debounceTimer);
      snapshot.debounceTimer = null;
    }
    doRender(discordClient, snapshot);
    return;
  }

  if (!snapshot.debounceTimer) {
    snapshot.debounceTimer = setTimeout(() => {
      snapshot.debounceTimer = null;
      doRender(discordClient, snapshot);
    }, 3000); // 3 second debounce
  }
}

async function doRender(discordClient, snapshot) {
  if (!snapshot.dirty) return;
  snapshot.dirty = false;

  try {
    const channel = await discordClient.channels.fetch(snapshot.channel_id).catch(() => null);
    if (!channel) return;
    const message = await channel.messages.fetch(snapshot.message_id).catch(() => null);
    if (!message) return;

    const newContent = formatProgressMessage(snapshot);
    if (message.content !== newContent) {
      await message.edit(newContent);
    }
  } catch (err) {
    // Basic 429 backoff would go here, but discord.js handles standard 429s automatically
    console.warn(`[ProgressManager] Failed to edit message for run ${snapshot.run_id}:`, err.message);
  }
}
