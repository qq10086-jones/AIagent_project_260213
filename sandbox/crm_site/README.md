# Document Release Hub

Path: `sandbox/crm_site`

## Scope
- Discord intake inbox for document requests
- Template-driven document issuance
- Automatic controlled document numbering
- Release ledger with full revision history
- Expansion path for complaints, CAPA, and ticket claiming

## Current Model
- Intake source is currently simulated by a local Discord-form entry
- Templates reference Excel master files by path / ID
- One click can create a document record and its first release-history entry
- Existing documents can be revised with tracked revision bumps

## Run

```bash
cd sandbox/crm_site
npm start
```

Then open:
- `http://localhost:3000`

## Notes
- Persistence is file-based in `data/store.json`
- This is the first operational scaffold for document control, not the final ERP/QMS integration
- Next planned modules: complaint record system, ticket intake / claiming, real Excel template generation
