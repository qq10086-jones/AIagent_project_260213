/**
 * step_artifact_contract.js
 *
 * Step-level artifact contract: handoff schema definitions and per-step validator
 * dispatch for the coding_team_v0 workflow.
 *
 * Extracted from coding_service.js to keep that file within the 800-line budget
 * and to give this concern a clear home. Validators are dynamically resolved from
 * the orchestrator source tree (supports both Docker and local layouts).
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));

function resolveOrchestratorModule(relPath) {
    const safeRel = String(relPath || "").replace(/\\/g, "/").replace(/^\/+/, "");
    const candidates = [
        path.resolve(MODULE_DIR, "..", "workspace", "orchestrator", "src", safeRel),
        path.resolve(MODULE_DIR, "..", "orchestrator", "src", safeRel),
    ];
    for (const abs of candidates) {
        if (fs.existsSync(abs)) {
            return pathToFileURL(abs).href;
        }
    }
    throw new Error(`Unable to resolve orchestrator module '${safeRel}' from step_artifact_contract.js`);
}

const {
    validatePmOutput,
    validateArchitectOutput,
    validateReleaseOutput,
} = await import(resolveOrchestratorModule("coding_team_validators.js"));

export const {
    validateCodingTeamHandoff,
} = await import(resolveOrchestratorModule("coding_team_handoff_validators.js"));

const {
    validateImplementationDelta,
} = await import(resolveOrchestratorModule("domain/workflow_step_validator.js"));

/**
 * Returns the handoff contract for a given step, or null if no handoff is
 * required (e.g. release_pack). Used by salvageWorkflowArtifactFailure to
 * determine which artifacts must be scaffolded and validated.
 */
export function getWorkflowStepHandoff(stepId) {
    switch (String(stepId || "")) {
        case "pm_spec":
            return {
                from_step: "pm_spec",
                required_artifacts: ["handoff/pm_to_architect.json"],
                required_sections: [],
                typed_handoff: {
                    file: "handoff/pm_to_architect.json",
                    required_fields: ["from_step", "to_steps", "scope_summary", "artifacts", "acceptance.criteria"],
                },
            };
        case "arch_design":
            return {
                from_step: "arch_design",
                required_artifacts: ["plan/workplan.json", "handoff/architect_to_impl.json"],
                required_sections: [],
                typed_handoff: {
                    file: "handoff/architect_to_impl.json",
                    required_fields: ["from_step", "to_steps", "modules", "interfaces", "decisions", "risks"],
                },
            };
        case "impl_be":
            return {
                from_step: "impl_be",
                required_artifacts: ["impl/be_changes/server.js", "impl/be_changes/package.json", "impl/be_notes.md", "handoff/be_to_fe.json"],
                required_sections: ["api_contracts", "shared_types", "scope_constraints"],
                typed_handoff: {
                    file: "handoff/be_to_fe.json",
                    required_fields: ["from_step", "to_step", "be_changes_path", "api_contracts", "shared_types", "scope_constraints"],
                },
            };
        case "impl_fe":
            return {
                from_step: "impl_fe",
                required_artifacts: ["impl/be_changes/server.js", "impl/fe_changes/public/index.html", "impl/fe_changes/public/app.js", "impl/be_notes.md", "impl/fe_notes.md", "handoff/impl_to_qa.json"],
                required_sections: [],
                typed_handoff: {
                    file: "handoff/impl_to_qa.json",
                    required_fields: ["from_steps", "to_step", "be_changes_path", "fe_changes_path", "run_instructions", "known_limitations"],
                },
            };
        default:
            return null;
    }
}

/**
 * Phase 3 (P3-1): Step dependency graph metadata.
 * Each step declares its upstream dependencies and parallel eligibility.
 * Used by dag_scheduler and workflow_parallelization_policy to determine
 * which steps can run concurrently.
 *
 * parallel_eligible is deny-by-default (false) — only impl_be/impl_fe
 * are eligible, and only when arch_design declares be_fe_independent: true.
 */
export const STEP_DEPENDENCY_GRAPH = {
    pm_spec: {
        depends_on: [],
        parallel_eligible: false,
        parallel_group: null,
    },
    arch_design: {
        depends_on: ["pm_spec"],
        parallel_eligible: false,
        parallel_group: null,
    },
    impl_be: {
        depends_on: ["arch_design"],
        parallel_eligible: true,
        parallel_group: "implementation",
    },
    impl_fe: {
        depends_on: ["arch_design"],
        parallel_eligible: true,
        parallel_group: "implementation",
    },
    smoke_test: {
        depends_on: ["impl_be", "impl_fe"],
        parallel_eligible: false,
        parallel_group: null,
    },
    qa_verify: {
        depends_on: ["impl_be", "impl_fe"],
        parallel_eligible: false,
        parallel_group: null,
    },
    release_pack: {
        depends_on: ["qa_verify"],
        parallel_eligible: false,
        parallel_group: null,
    },
    deploy_preview: {
        depends_on: ["release_pack"],
        parallel_eligible: false,
        parallel_group: null,
    },
};

/**
 * Get the dependency metadata for a step.
 * @param {string} stepId
 * @returns {{ depends_on: string[], parallel_eligible: boolean, parallel_group: string|null }}
 */
export function getStepDependencyMeta(stepId) {
    return STEP_DEPENDENCY_GRAPH[String(stepId || "")] || {
        depends_on: [],
        parallel_eligible: false,
        parallel_group: null,
    };
}

/**
 * Dispatches to the appropriate per-step artifact validator.
 * Returns { checked: bool, ok: bool, ... } consistent with orchestrator validator contracts.
 */
export function validateWorkflowStepArtifacts({ workspaceRoot, artifactRoot, stepId }) {
    switch (String(stepId || "")) {
        case "pm_spec":
            return validatePmOutput({ workspaceRoot, artifactRoot });
        case "arch_design":
            return validateArchitectOutput({ workspaceRoot, artifactRoot });
        case "release_pack":
            return validateReleaseOutput({ workspaceRoot, artifactRoot });
        case "impl_be":
        case "impl_fe":
            return validateImplementationDelta({
                run: { workflow_id: "coding_team_v0" },
                stepId: String(stepId),
                output: {},
                payload: { artifact_root: artifactRoot },
                workspaceRoot,
            });
        default:
            return { checked: false, ok: true };
    }
}
