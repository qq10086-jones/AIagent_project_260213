console.log("Starting test_coding.js...");

async function runTests() {
  const baseUrl = "http://localhost:3000";
  const runId = (globalThis.crypto && globalThis.crypto.randomUUID)
    ? globalThis.crypto.randomUUID()
    : `test-run-${Date.now()}`;

  try {
    // 1. Test /execute-tool -> coding.patch
    console.log("Testing /execute-tool (coding.patch)...");
    const editBlock = `<<<<<<< SEARCH\nHello World\n=======\nHello Coding Agent\n>>>>>>> REPLACE`;
    const patchRes = await fetch(`${baseUrl}/execute-tool`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        run_id: runId, 
        tool_name: "coding.patch",
        payload: {
          task_id: "test_task_1",
          file_path: "__write_test__.txt",
          edit_block: editBlock
        }
      })
    });
    const patchData = await patchRes.json();
    console.log("Patch response:", patchData);

    // 2. Test /execute-tool -> coding.execute
    console.log("\nTesting /execute-tool (coding.execute)...");
    const execRes = await fetch(`${baseUrl}/execute-tool`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: runId,
        tool_name: "coding.execute",
        payload: {
          task_id: "test_task_1",
          command: "echo Command Executed!"
        }
      })
    });
    const execData = await execRes.json();
    console.log("Execute response:", execData);

    console.log("\nAll tests completed!");
  } catch (err) {
    console.error("Test failed:", err.message);
  }
}

runTests();
