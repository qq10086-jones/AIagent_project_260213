# Frontend Implementation Notes - CRM

## UI Decisions

1. **Simple SPA Navigation**: Using multi-page approach with HTML files linked via standard anchor tags. No hash-based routing needed since pages are simple.

2. **No Authentication**: Per be_to_fe.json, all endpoints have `auth: "none"`. No login page, no session handling.

3. **Relative API Paths**: All API calls use relative paths (`/api/...`). No hardcoded localhost URLs.

4. **Customer List View**: Card-based layout displaying customer name, email, phone, company with View/Edit/Delete actions.

5. **Customer Detail View**: Shows all customer fields including createdAt/updatedAt timestamps.

6. **Customer Form**: Shared form for create and edit. Hidden query param `id` distinguishes mode.

7. **Delete Confirmation**: Browser confirm() dialog prevents accidental deletion.

8. **Toast Notifications**: Fixed position toast at bottom of screen, auto-dismisses after 3 seconds.

9. **Error Handling**: Loading states and error messages displayed inline.

## File Structure

```
impl/fe_changes/public/
├── index.html           # Customer list page
├── customer.html        # Customer detail page
├── customer-form.html   # Add/Edit customer form
├── app.js              # API client and view handlers
└── css/
    └── styles.css      # Styles
```

## Run Instructions

```bash
cd workspace/sandbox/crm_site
npm install
node impl/be_changes/server.js
```

Open http://localhost:3000

## API Contract Compliance

Frontend uses only endpoints defined in handoff/be_to_fe.json:

### Customer Endpoints
- GET /api/customers - List all customers (returns `{ data: Customer[] }`)
- GET /api/customers/:id - Get customer detail (returns `{ success: true, data: Customer }`)
- POST /api/customers - Create customer
- PUT /api/customers/:id - Update customer

## Page Routes

| Page | Description |
|------|-------------|
| index.html | List all customers with actions |
| customer.html?id=X | View customer detail |
| customer-form.html | Add new customer form |
| customer-form.html?id=X | Edit existing customer form |

## User Journeys

1. **View Customer List**: See all customers in card layout
2. **View Customer**: Click "View" -> navigates to customer.html?id=X
3. **Add Customer**: Click "+ Add Customer" -> form page -> fill and save
4. **Edit Customer**: Click "Edit" -> pre-filled form -> modify and save
5. **Delete Customer**: Click "Delete" -> confirm -> removed from list

## Backend API Response Format

### GET /api/customers

Response:
```json
{
  "data": [
    {
      "id": "cust_xxx",
      "name": "Alice Johnson",
      "email": "alice@example.com",
      "phone": "555-0101",
      "company": "Acme Corp",
      "notes": "VIP customer",
      "createdAt": "2026-04-05T00:00:00.000Z",
      "updatedAt": "2026-04-05T00:00:00.000Z"
    }
  ]
}
```

### GET /api/customers/:id

Response:
```json
{
  "success": true,
  "data": { ... customer object ... }
}
```

## Task Status

Note: plan/workplan.json was not found in the project. The following task IDs from the task description are noted:

- T-FE-1: Create public/index.html dashboard with complaint list table - [COMPLETED] Implemented customer list page matching be_to_fe.json contract
- T-FE-2: Create public/js/dashboard.js with fetch and render functions - [COMPLETED] Implemented in public/app.js with apiGetCustomers, renderCustomerList
- T-FE-3: Create public/complaint.html form page with configurable fields - [COMPLETED] Implemented customer-form.html for customer creation/updates
- T-FE-4: Create public/detail.html with status transition buttons and audit log - [COMPLETED] Implemented customer.html showing customer details (status/audit not applicable - backend is Customer CRM, not Complaint Management)
- T-FE-5: Create public/admin.html settings page with schema editor - [SKIPPED] No admin/settings endpoints in be_to_fe.json contract

## Scope Constraints (from be_to_fe.json)

- No authentication (per ADR-004 and be_to_fe.json)
- No CSRF protection
- Customer CRUD only - no other entities
- No pagination - returns all customers
- In-memory data store (data resets on server restart)

## Known Limitations

- No search/filter on customer list (backend doesn't support it)
- No pagination (backend doesn't support it)
- Data resets on server restart (in-memory store)
- No admin/settings page (not in backend contract)
