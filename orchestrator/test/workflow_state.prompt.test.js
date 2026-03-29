import test from "node:test";
import assert from "node:assert/strict";

import { buildStepPrompt } from "../src/domain/workflow_state.js";

test("buildStepPrompt includes concrete arch interface contract guidance", () => {
  const prompt = buildStepPrompt({
    run: { workflow_id: "coding_team_v0", project_type: "webapp_crm" },
    stepDef: { id: "arch_design", role: "architect" },
    input: { goal: "Build a minimal CRM web app" },
    payload: { artifact_root: "artifacts/release/test-run" },
    promptScript: {
      script_id: "architect.system_spec.v1",
      llm_role: "architect",
      role: "architect",
      artifact_type: "system_spec",
      validation: { required_sections: ["module_breakdown", "interfaces", "dependency_choices", "risk_notes"] },
      system_prompt: "Generate a system specification with module boundaries, interfaces, dependency choices, implementation sequence, and risk notes. In plan/interfaces.md, define every concrete HTTP endpoint, internal RPC, or event as its own markdown heading such as '## GET /api/customers' or '## Event: customer.created'. Under each heading, include request shape, response shape or payload shape, and auth requirement. In plan/arch.md, keep the Interfaces section as a short summary that points to plan/interfaces.md rather than replacing it.",
    },
  });

  assert.match(prompt, /Define all API endpoints or internal interfaces in plan\/interfaces\.md/);
  assert.match(prompt, /## GET \/api\/customers/);
  assert.match(prompt, /request shape, response shape or payload shape, and auth requirement/i);
  assert.match(prompt, /plan\/arch\.md, keep the Interfaces section as a short summary that points to plan\/interfaces\.md/i);
});

test("buildStepPrompt adds static-site guardrails for single_file_html arch step", () => {
  const prompt = buildStepPrompt({
    run: { workflow_id: "coding_team_v0", project_type: "single_file_html" },
    stepDef: { id: "arch_design", role: "architect" },
    input: { goal: "Build a simple landing page with hero, CTA, feature cards, and FAQ" },
    payload: { artifact_root: "artifacts/release/test-run" },
    promptScript: {
      script_id: "architect.system_spec.v1",
      llm_role: "architect",
      role: "architect",
      artifact_type: "system_spec",
      validation: { required_sections: ["module_breakdown", "interfaces", "dependency_choices", "risk_notes"] },
      system_prompt: "Generate a system specification.",
    },
  });

  assert.match(prompt, /static single-file HTML landing page/i);
  assert.match(prompt, /## GET \//);
  assert.match(prompt, /## Event: faq\.toggle/i);
  assert.match(prompt, /Do not write vague 'N\/A' placeholders/i);
});
