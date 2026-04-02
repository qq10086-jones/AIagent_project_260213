# Frontend Implementation Notes - CRM

## UI Decisions

1. **Multi-page SPA-style Navigation**: Separate HTML pages (login.html, index.html, customer.html, customer-form.html) with shared app.js.

2. **Session-based Authentication**: Login form posts to /api/auth/login. Session cookie is managed by browser automatically (same-origin fetch with credentials).

3. **CSRF Token Management**: After successful login, frontend fetches CSRF token from /api/csrf-token and stores it globally (window.currentCsrfToken). All POST/PUT/DELETE requests include the CSRF token in the 'CSRF-Token' header.

4. **401 Handling**: All API calls check for 401 responses and redirect to login.html when unauthorized.

5. **Relative API Paths**: All API calls use relative paths via API_BASE = '/api'. No hardcoded localhost URLs.

6. **Error Handling**: Toast notifications for success/error feedback, inline error messages for form validation.

7. **Loading States**: Loading indicators shown while fetching data.

## Authentication Flow

1. User navigates to any protected page
2. If not authenticated (no session), API call returns 401
3. Frontend detects 401 and redirects to login.html
4. User enters credentials and submits login form
5. On success, frontend fetches CSRF token and stores it
6. Frontend redirects to index.html (customer list)
7. All subsequent API calls include session cookie and CSRF token

## File Structure

```
impl/fe_changes/public/
├── login.html           # Login form page
├── index.html           # Customer list page (protected)
├── customer.html        # Customer detail page (protected)
├── customer-form.html   # Add/Edit customer form (protected)
├── app.js               # API client and page logic
└── css/
    └── styles.css       # All styling
```

## Run Instructions

### Development Mode

```bash
cd workspace/sandbox/crm_site/impl/be_changes
npm install
npm start
```

Server runs on http://localhost:3000

### Login Credentials

- Username: admin
- Password: admin123

## API Contract Compliance

Frontend uses only endpoints defined in handoff/be_to_fe.json:

### Authentication Endpoints
- POST /api/auth/login - Login with username/password
- POST /api/auth/logout - Logout (requires auth + CSRF)
- GET /api/csrf-token - Get CSRF token (requires auth)

### Customer Endpoints
- GET /api/customers - List with pagination/search (requires auth)
- GET /api/customers/:id - Get detail (requires auth)
- POST /api/customers - Create (requires auth + CSRF)
- PUT /api/customers/:id - Update (requires auth + CSRF)
- DELETE /api/customers/:id - Delete (requires auth)

## Customer Data Fields

All customer fields from backend are displayed:
- id, name, email, phone, company, notes, createdAt, updatedAt

## API Response Shapes

### GET /api/customers
```json
{
  "data": [{ "id", "name", "email", "phone", "company", "notes", "createdAt", "updatedAt" }],
  "pagination": { "page", "limit", "total", "totalPages" }
}
```

### GET /api/customers/:id
```json
{ "success": true, "data": { "id", "name", "email", ... } }
```

### POST /api/customers (requires CSRF)
```json
// Request: { "name", "email", "phone?", "company?", "notes?" }
// Response: { "success": true, "data": { created customer } }
```

### PUT /api/customers/:id (requires CSRF)
```json
// Request: { "name?", "email?", ... }
// Response: { "success": true, "data": { updated customer } }
```

### DELETE /api/customers/:id (requires CSRF)
```json
// Response: 204 No Content
```

## User Journeys

1. **Login**: Navigate to site -> Redirected to login.html -> Enter credentials -> Redirect to customer list
2. **View Customer List**: Load index.html -> See paginated list of customers
3. **Search Customers**: Enter search term -> List filters by name/email
4. **View Customer Detail**: Click View -> Navigate to customer.html?id=xxx
5. **Add Customer**: Click Add Customer -> Fill form -> Submit -> Redirect to detail
6. **Edit Customer**: Click Edit -> Modify fields -> Submit -> Redirect to detail
7. **Delete Customer**: Click Delete -> Confirm -> Remove from list
8. **Logout**: Click Logout -> Session destroyed -> Redirect to login