import http from "http";

const TOTAL_REQUESTS = 50;
const CONCURRENCY_DELAY_MS = 2000; // 2 seconds between injections to respect rate limits

const tasks = [
  "Build a new dashboard system with a frontend UI and backend API integration, [LoadTest] project workflow",
  "Implement a multi-agent orchestration platform with a web dashboard and backend services, [LoadTest] architecture system",
  "Refactor the frontend and backend architecture to support parallel execution, [LoadTest] full project",
  "Develop a payment gateway system with a UI payment form and Stripe API integration, [LoadTest] complex workflow",
  "Create a documentation portal with a React frontend and a CMS backend, [LoadTest] project system"
];

function injectTask(index) {
  return new Promise((resolve, reject) => {
    const raw_input = tasks[index % tasks.length] + ` (seq: ${index})`;
    const payload = JSON.stringify({ raw_input });

    const options = {
      hostname: "localhost",
      port: 3000,
      path: "/vnext/dispatch",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(payload)
      }
    };

    const req = http.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.ok) {
            console.log(`[${index + 1}/${TOTAL_REQUESTS}] Injected: ${raw_input} -> RunID: ${parsed.run_id}`);
            resolve(parsed);
          } else {
            console.error(`[${index + 1}/${TOTAL_REQUESTS}] Failed: ${parsed.error}`);
            resolve(null);
          }
        } catch (e) {
          console.error(`[${index + 1}/${TOTAL_REQUESTS}] Parse Error: ${data.substring(0, 100)}`);
          resolve(null);
        }
      });
    });

    req.on("error", (e) => {
      console.error(`[${index + 1}/${TOTAL_REQUESTS}] Connection Error: ${e.message}`);
      resolve(null);
    });

    req.write(payload);
    req.end();
  });
}

async function main() {
  console.log(`🚀 Starting Live Traffic Injection: ${TOTAL_REQUESTS} tasks...`);
  console.log(`Using models: Control=qwen-flash, Coder=qwen3-coder-plus, Quant=qwen-plus`);
  
  for (let i = 0; i < TOTAL_REQUESTS; i++) {
    await injectTask(i);
    if (i < TOTAL_REQUESTS - 1) {
      await new Promise(r => setTimeout(resolve => r(), CONCURRENCY_DELAY_MS));
    }
  }
  
  console.log("\n✅ All tasks injected. System is processing in background.");
  console.log("Check http://localhost:8501 or routing_decision_log for progress.");
}

main();
