# Coding Team Reference Demo

Path: `workspace/sandbox/crm_site`

## Delivery Status
- This workspace is the current **reference delivery demo** for the Nexus coding team
- The current reference scenario is a **Customer CRM** sample served by `impl/be_changes/server.js`
- `npm start` launches the reference backend and serves the reference frontend from `impl/be_changes/public/`
- Root-level `index.html` and `app.js` are legacy sandbox assets and are not the primary demo entrypoint for delivery
- The CRM slice is being used as the acceptance sample for the delivery chain, not as the product boundary of Nexus

## Scope
- Validate the end-to-end coding-team delivery chain on a concrete full-stack sample
- Prove `pm_spec -> arch_design -> impl_be -> impl_fe -> qa_verify -> release_pack`
- Provide one runnable sample that can be demonstrated, verified, and packaged for release

## Current Reference Scenario
- Current scenario: Customer CRM
- Why this scenario: it is the most complete full-stack sample currently present in the repo
- What it proves: backend API delivery, frontend delivery, handoff contracts, runtime smoke, QA handoff, and release packaging
- What it does not imply: Nexus is not limited to CRM projects

## Run

```bash
cd workspace/sandbox/crm_site
npm install
npm start
```

Then open:
- `http://localhost:3000`
- Login with `admin / admin123`

## Notes
- Persistence may use SQLite or JSON fallback depending on local runtime availability
- The active shipped sample is the CRM reference app under `impl/`
- Root-level sandbox files are retained for repo history and verification compatibility

## Delivery Artifacts
- Release notes: `release/release_notes.md`
- Release manifest: `release/artifact_manifest.json`
- Delivery runbook: `release/README.md`
- Startup scripts: `release/start.sh`, `release/start.ps1`
- Runtime smoke evidence: `smoke/smoke_result.json`
