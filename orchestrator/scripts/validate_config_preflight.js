import { assertConfigPreflight } from "../src/config_preflight.js";
import { pathToFileURL } from "url";
import { ORCHESTRATOR_ROOT, resolveRepoPath } from "./_paths.js";

export function main() {
  const result = assertConfigPreflight({
    runtimeConfigPath: process.env.RUNTIME_CONFIG_PATH || resolveRepoPath("configs", "runtime", "runtime_defaults.json"),
    workspaceRoot: ORCHESTRATOR_ROOT,
  });
  console.log("config preflight ok");
  console.log(JSON.stringify({
    ok: result.ok,
    items: result.items.map((item) => ({
      id: item.id,
      path: item.path.replace(/\\/g, "/"),
      exists: item.exists,
    })),
  }, null, 2));
  return {
    reportPath: "",
    report: {
      ok: result.ok,
      items: result.items.map((item) => ({
        id: item.id,
        path: item.path.replace(/\\/g, "/"),
        exists: item.exists,
      })),
    },
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    main();
  } catch (err) {
    console.error(err?.message || err);
    process.exit(1);
  }
}
