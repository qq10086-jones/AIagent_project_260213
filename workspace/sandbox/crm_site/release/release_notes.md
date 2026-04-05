# Release Notes

## Delivered Demo

This package delivers the current runnable reference demo for Nexus coding-team validation.
The current reference scenario is a Customer CRM sample.
The backend is served by `impl/be_changes/server.js` and the frontend is served from `impl/be_changes/public/`.

## Included Capability

- Session login with demo credentials `admin / admin123`
- CSRF token issuance for mutating requests
- Customer list, detail, create, and update flows
- Search and pagination support on the customer list API
- File-backed fallback persistence when SQLite native loading is unavailable

## Verification Evidence

- `npm run lint` passed
- `npm run build` passed
- `npm run test` passed
- Runtime smoke evidence recorded in `smoke/smoke_result.json`

## Known Limits

- Demo credentials are hardcoded for internal validation only
- Delete support exists in the backend contract, but this delivery package is positioned as a reviewable CRM MVP
- Root-level sandbox assets remain in the workspace but are not the delivery entrypoint
- This sample demonstrates the delivery pipeline on one scenario and should not be interpreted as the full scope of Nexus-supported project types
