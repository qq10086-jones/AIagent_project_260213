import { normalizeInputRequest } from "./input_normalizer.js";
import { routeTaskRequest } from "./brain_router.js";
import { assertRouteContractResponse } from "./contract_validator.js";

export function buildRouteContractResponse({ body = {}, analyzerResult = null, registry }) {
  const normalized = normalizeInputRequest(body || {});
  if (!normalized.raw_input) {
    const err = new Error("raw_input/message is required");
    err.code = "BAD_REQUEST";
    throw err;
  }

  const routed = routeTaskRequest({
    ...normalized,
    analyzerResult,
    registry,
  });

  return assertRouteContractResponse({
    ok: true,
    normalized,
    analyzer_result: analyzerResult,
    ...routed,
  });
}
