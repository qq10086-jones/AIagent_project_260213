# Frontend Implementation Notes - CRM

## UI Decisions

1. **Hash-Based Routing**: Implemented client-side routing using window.location.hash. Routes: `#/customers` (list), `#/customers/:id` (detail), `#/customers/new` (add form), `#/customers/:id/edit` (edit form).

2. **XSS Protection**: Using DOMPurify library loaded from CDN to sanitize all user-generated content before rendering. The `sanitize()` function is used for customer data, and `escapeHtml()` for table cell content.

3. **Session-based Authentication**: Login posts to /api/auth/login with credentials: admin/admin123. Session cookie managed by browser via fetch with `credentials: 'include'`.

4. **CSRF Token Management**: After login, frontend fetches CSRF token from /api/csrf-token. Token stored globally and included in all POST/PUT/DELETE requests via 'CSRF-Token' header.

5. **Customer List View**: Table layout with columns: Name, Email, Phone, Company, Actions. Search input with 300ms debounce. Pagination with Previous/Next buttons and page info.

6. **Customer Detail View**: Displays all customer fields with edit/delete buttons. Uses hash-based navigation links.

7. **Add/Edit Form**: Shared form for create and update. Hidden id field distinguishes mode. Client-side validation for required fields (name, email).

8. **Delete Confirmation Modal**: Modal overlay prevents accidental deletion. Shows customer name before confirming delete action.

9. **Toast Notifications**: Success/error feedback using fixed position toast that auto-dismisses after 3 seconds.

10. **Relative API Paths**: All API calls use relative paths (/api/...). No hardcoded localhost URLs.

11. **Responsive Design**: CSS media query for mobile viewports (max-width: 600px) adjusts table font size, padding, and stacks detail rows.

## File Structure

```
impl/fe_changes/public/
├── index.html    # Complete HTML shell with inline CSS and DOMPurify CDN
└── app.js        # Hash-based router, API client, view handlers
```

## Run Instructions

```bash
cd workspace/sandbox/crm_site
npm install
node impl/be_changes/server.js
```

Open http://localhost:3000 and login with admin/admin123.

## API Contract Compliance

Frontend uses only endpoints defined in handoff/be_to_fe.json:

### Authentication Endpoints
- POST /api/auth/login - Login with username/password
- GET /api/csrf-token - Get CSRF token (requires auth)
- POST /api/auth/logout - Logout (requires auth)

### Customer Endpoints
- GET /api/customers - List with pagination (page, limit, search) and auth
- GET /api/customers/:id - Get detail (requires auth)
- POST /api/customers - Create (requires auth + CSRF)
- PUT /api/customers/:id - Update (requires auth + CSRF)
- DELETE /api/customers/:id - Delete (requires auth + CSRF)

## Hash Routes

| Route | View | Description |
|-------|------|-------------|
| #/customers | CustomerListView | List customers with search and pagination |
| #/customers/:id | CustomerDetailView | Show customer details |
| #/customers/new | CustomerFormView (add mode) | Add new customer form |
| #/customers/:id/edit | CustomerFormView (edit mode) | Edit existing customer form |

## User Journeys

1. **Login**: Enter admin/admin123 -> Session established, CSRF token loaded -> Redirects to #/customers
2. **View Customer List**: See table of customers with search bar and pagination
3. **Search Customers**: Type in search -> List filters by name/email with 300ms debounce
4. **View Customer Detail**: Click View button -> Navigates to #/customers/:id
5. **Add Customer**: Click "+ Add Customer" or navigate to #/customers/new -> Form shown -> Fill and Save -> Redirects to customer detail
6. **Edit Customer**: Click Edit button or navigate to #/customers/:id/edit -> Form pre-populated -> Modify and Save -> Redirects to detail
7. **Delete Customer**: Click Delete -> Modal confirmation -> Confirm -> Customer removed, redirects to list
8. **Pagination**: Use Previous/Next buttons to navigate pages
9. **Logout**: Click Logout -> Session destroyed -> Back to login

## DOMPurify Integration

DOMPurify is loaded from CDN: `https://unpkg.com/dompurify@3.0.6/dist/purify.min.js`

Used for sanitizing customer data (name, email, phone, company, notes) before rendering to prevent XSS attacks. Script tags in customer data will render as text, not execute.