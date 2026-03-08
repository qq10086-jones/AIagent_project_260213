/**
 * cron_scheduler.js
 *
 * Cron job registrations for scheduled reports and JP market briefs.
 * Extracted from index.js as part of WS-11-05 decomposition.
 */

import cron from "node-cron";
import { v4 as uuidv4 } from "uuid";

/**
 * @param {{ ensureRun, enqueueTask, runToContext, autoReportChannelId, autoReportTimezone }} deps
 */
export function registerCronSchedules({ ensureRun, enqueueTask, runToContext, autoReportChannelId, autoReportTimezone }) {
  // Daily report at 16:00 in configured timezone
  cron.schedule(
    "0 16 * * *",
    async () => {
      if (!autoReportChannelId) return;
      const dailyDate = new Date().toISOString().slice(0, 10);
      const dailyTasks = [
        { tool: "news.daily_report", payload: { max_items: 20, date: dailyDate } },
        { tool: "quant.run_optimized_pipeline", payload: { date: dailyDate } },
      ];
      for (const t of dailyTasks) {
        const run_id = uuidv4();
        const context = { run_id, channelId: autoReportChannelId, startTime: Date.now(), lang: "zh", closeRunOnTaskResult: true };
        runToContext.set(run_id, context);
        await ensureRun(run_id, {
          client_msg_id: `cron-${dailyDate}-${t.tool}-${run_id.slice(0, 8)}`,
          user_id: "system-cron",
          status: "running",
          input_text: `daily:${t.tool}`,
        });
        await enqueueTask({ tool_name: t.tool, payload: t.payload, run_id, idempotency_key: `${dailyDate}:${t.tool}`, context });
      }
    },
    { timezone: autoReportTimezone }
  );

  // JP Market Pre-Close Brief at 15:15 JST (Mon-Fri)
  cron.schedule(
    "15 15 * * 1-5",
    async () => {
      if (!autoReportChannelId) return;
      const dailyDate = new Date().toISOString().slice(0, 10);
      const tool = "news.preclose_brief_jp";
      const run_id = uuidv4();
      const context = { run_id, channelId: autoReportChannelId, startTime: Date.now(), lang: "zh", closeRunOnTaskResult: true };
      runToContext.set(run_id, context);
      await ensureRun(run_id, {
        client_msg_id: `cron-${dailyDate}-${tool}-${run_id.slice(0, 8)}`,
        user_id: "system-cron",
        status: "running",
        input_text: `cron:${tool}`,
      });
      await enqueueTask({ tool_name: tool, payload: { date: dailyDate, type: "preclose" }, run_id, idempotency_key: `${dailyDate}:${tool}`, context });
    },
    { timezone: "Asia/Tokyo" }
  );

  // JP Market TDnet Close Flash at 15:35 JST (Mon-Fri)
  cron.schedule(
    "35 15 * * 1-5",
    async () => {
      if (!autoReportChannelId) return;
      const dailyDate = new Date().toISOString().slice(0, 10);
      const tool = "news.tdnet_close_flash";
      const run_id = uuidv4();
      const context = { run_id, channelId: autoReportChannelId, startTime: Date.now(), lang: "zh", closeRunOnTaskResult: true };
      runToContext.set(run_id, context);
      await ensureRun(run_id, {
        client_msg_id: `cron-${dailyDate}-${tool}-${run_id.slice(0, 8)}`,
        user_id: "system-cron",
        status: "running",
        input_text: `cron:${tool}`,
      });
      await enqueueTask({ tool_name: tool, payload: { date: dailyDate, type: "postclose_flash" }, run_id, idempotency_key: `${dailyDate}:${tool}`, context });
    },
    { timezone: "Asia/Tokyo" }
  );
}
