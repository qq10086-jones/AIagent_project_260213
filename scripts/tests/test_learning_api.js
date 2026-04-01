async function runTest() {
  console.log("=== Testing MVP-1 Learning & Trace API ===");

  try {
    // 1. Create a Trace
    console.log("[1] Creating a new trace...");
    const traceRes = await fetch("http://localhost:3000/traces", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: "quant",
        task_type: "debug",
        context_digest: "User asked for a quant strategy, but we used a prohibited library.",
        action_json: { template: "step-by-step", steps: ["install prohibited-lib", "run strategy"] }
      })
    });
    const traceData = await traceRes.json();
    console.log("Trace API Response:", traceData);
    if (!traceData.ok) throw new Error("Failed to create trace");
    
    const traceId = traceData.trace_id;

    // 2. Submit Negative Feedback to auto-generate a Rule
    console.log("[2] Submitting negative feedback to generate a rule...");
    const badFeedbackRes = await fetch(`http://localhost:3000/traces/${traceId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feedback: "❌",
        reason: "用了禁止的量化库，需要遵守刚性约束。"
      })
    });
    const badFeedbackData = await badFeedbackRes.json();
    console.log("Feedback API Response:", badFeedbackData);

    // 3. Create another Trace for Positive Feedback
    console.log("[3] Creating another trace for success scenario...");
    const traceRes2 = await fetch("http://localhost:3000/traces", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: "openclaw",
        task_type: "deploy",
        context_digest: "User asked to deploy with correct standard cmd.",
        action_json: { cmd: "npm run build && npm start" }
      })
    });
    const traceId2 = (await traceRes2.json()).trace_id;

    // 4. Submit Positive Feedback to auto-generate a Memory/SOP
    console.log(`[4] Submitting positive feedback for trace ${traceId2}...`);
    const goodFeedbackRes = await fetch(`http://localhost:3000/traces/${traceId2}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feedback: "✅",
        reason: "完美遵守规则，给出的命令可以直接执行。"
      })
    });
    const goodFeedbackData = await goodFeedbackRes.json();
    console.log("Feedback API Response:", goodFeedbackData);

    console.log("=== All Tests Completed Successfully! ===");

  } catch (err) {
    console.error("[Error during test]:", err.message);
  }
}

runTest();
