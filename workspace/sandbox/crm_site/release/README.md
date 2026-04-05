# Reference Demo Runbook

This folder describes the current deliverable package for `workspace/sandbox/crm_site`.

## Positioning

- This package is the current reference demo for the Nexus coding-team delivery pipeline
- The current reference scenario is a Customer CRM sample
- The CRM sample is an acceptance vehicle for the platform, not the product boundary of Nexus

## Start

```bash
cd workspace/sandbox/crm_site
npm install
node impl/be_changes/server.js
```

Browser URL: http://localhost:3000

Login credentials:
- Username: `admin`
- Password: `admin123`

## Verification

```bash
cd workspace/sandbox/crm_site
npm run release:verify
```

Runtime smoke evidence is written to `smoke/smoke_result.json`.

## Delivered Files

- `impl/be_changes/server.js`
- `impl/be_changes/package.json`
- `impl/be_changes/public/index.html`
- `impl/be_notes.md`
- `handoff/be_to_fe.json`
- `handoff/impl_to_qa.json`
- `smoke/smoke_result.json`

## Notes

- The active demo is the Customer CRM reference implementation under `impl/`.
- Root-level sandbox files are retained for repo history and verification compatibility.
