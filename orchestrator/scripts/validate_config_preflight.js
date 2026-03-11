import { assertConfigPreflight } from "../src/config_preflight.js";

function main() {
  const result = assertConfigPreflight({
    runtimeConfigPath: process.env.RUNTIME_CONFIG_PATH || "configs/runtime/runtime_defaults.json",
    workspaceRoot: process.cwd(),
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
}

try {
  main();
} catch (err) {
  console.error(err?.message || err);
  process.exit(1);
}
