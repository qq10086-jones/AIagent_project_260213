# CRM Pro Demo Site

Path: `sandbox/crm_site`

## Features
- Login and session state
- Role-based access control (`admin` / `sales` / `viewer`)
- Customer list with search + stage filter
- Customer detail panel with stage transitions
- Create/Edit/Delete (permission-controlled)
- Activity audit log
- Local learning insights (derived from pipeline + activity)
- Local persistence via `localStorage`

## Demo Accounts
- `admin / admin123`
- `sales / sales123`
- `viewer / viewer123`
- `tech / tech123` (complaint management module only)

## Run
From project root:

```bash
cd sandbox/crm_site
python -m http.server 8088
```

Then open:
- `http://localhost:8088`

You can also double-click `index.html` directly for a quick preview.
