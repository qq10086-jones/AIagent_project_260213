/**
 * task_watchdog.js
 *
 * Periodic watchdog that times out stale running and queued tasks,
 * optionally routing them to a dead-letter queue.
 * Extracted from index.js as part of WS-11-05 decomposition.
 */

/**
 * @param {{ pool, redis, workflowEngine, normalizeResultPayload, recordEvent,
 *           listStaleRunningTasks, listStaleQueuedTasks, failRunningTaskIfStillRunning,
 *           failQueuedTaskIfStillQueued, enqueueToStream, groupTask,
 *           runningTimeoutSec, queuedTimeoutSec, intervalMs, autoDlq,
 *           streamTaskDlq }} deps
 */
export function createTaskWatchdog({
  pool,
  redis,
  workflowEngine,
  normalizeResultPayload,
  recordEvent,
  listStaleRunningTasks,
  listStaleQueuedTasks,
  failRunningTaskIfStillRunning,
  failQueuedTaskIfStillQueued,
  enqueueToStream,
  groupTask,
  runningTimeoutSec,
  queuedTimeoutSec,
  intervalMs,
  autoDlq,
  streamTaskDlq,
}) {
  async function start() {
    console.log(
      `[watchdog] enabled interval=${intervalMs}ms running_timeout=${runningTimeoutSec}s queued_timeout=${queuedTimeoutSec}s auto_dlq=${autoDlq}`
    );

    while (true) {
      try {
        const staleRunning = await listStaleRunningTasks(pool, runningTimeoutSec);

        for (const row of staleRunning) {
          const timeoutError = "TASK_TIMEOUT";
          const timeoutPayload = normalizeResultPayload(
            "failed",
            { error: `task timeout after ${runningTimeoutSec}s`, watchdog: true },
            timeoutError
          );
          await failRunningTaskIfStillRunning(pool, row.task_id, timeoutError, timeoutPayload);
          await recordEvent(row.task_id, "task.timeout", {
            task_id: row.task_id,
            run_id: row.run_id,
            timeout_sec: runningTimeoutSec,
            tool_name: row.tool_name,
          });

          if (autoDlq) {
            await enqueueToStream(redis, streamTaskDlq, groupTask, {
              task_id: row.task_id,
              run_id: row.run_id || "",
              tool_name: row.tool_name || "",
              payload: row.payload_json || "{}",
              error_code: timeoutError,
            });
            await recordEvent(row.task_id, "task.dlq.enqueued", { stream: streamTaskDlq, reason: timeoutError });
          }

          await workflowEngine
            .handleTaskTerminal({
              task_id: row.task_id,
              status: "failed",
              output: { error: `task timeout after ${runningTimeoutSec}s`, watchdog: true },
              error_code: timeoutError,
            })
            .catch(err => console.warn(`[watchdog] workflow timeout propagation failed: ${err.message}`));
        }

        const staleQueued = await listStaleQueuedTasks(pool, queuedTimeoutSec);

        for (const row of staleQueued) {
          const staleError = "TASK_QUEUED_STALE";
          const stalePayload = normalizeResultPayload(
            "failed",
            { error: `task queued stale after ${queuedTimeoutSec}s`, watchdog: true },
            staleError
          );
          await failQueuedTaskIfStillQueued(pool, row.task_id, staleError, stalePayload);
          await recordEvent(row.task_id, "task.queued.stale", {
            task_id: row.task_id,
            run_id: row.run_id,
            timeout_sec: queuedTimeoutSec,
            tool_name: row.tool_name,
          });

          if (autoDlq) {
            await enqueueToStream(redis, streamTaskDlq, groupTask, {
              task_id: row.task_id,
              run_id: row.run_id || "",
              tool_name: row.tool_name || "",
              payload: row.payload_json || "{}",
              error_code: staleError,
            });
            await recordEvent(row.task_id, "task.dlq.enqueued", { stream: streamTaskDlq, reason: staleError });
          }

          await workflowEngine
            .handleTaskTerminal({
              task_id: row.task_id,
              status: "failed",
              output: { error: `task queued stale after ${queuedTimeoutSec}s`, watchdog: true },
              error_code: staleError,
            })
            .catch(err => console.warn(`[watchdog] workflow queued-stale propagation failed: ${err.message}`));
        }
      } catch (err) {
        console.warn(`[watchdog] loop error: ${err.message}`);
      }
      await new Promise(r => setTimeout(r, intervalMs));
    }
  }

  return { start };
}
