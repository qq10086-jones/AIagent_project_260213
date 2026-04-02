# Backend Implementation Notes

## Decisions & Assumptions

1. **Express.js Server**: Using Express as the backend framework per package.json dependencies
2. **SQLite Database**: Using better-sqlite3 for persistent storage with customers and sessions tables
3. **Session-based Authentication**: Using express-session with csurf for CSRF protection
4. **JSON API**: All endpoints return JSON responses with consistent `{ success, data, error }` structure
5. **Customer Entity**: Core domain model with id, name, email, phone, company, notes, createdAt, updatedAt fields
6. **Seeded Data**: Three sample customers are pre-loaded for testing if database is empty

## API Endpoints (Customer CRM)

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login` | POST | Login with username/password |
| `/api/auth/logout` | POST | Logout and destroy session |
| `/api/csrf-token` | GET | Get CSRF token for form submissions |

### Customers (requires authentication)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/customers` | GET | List customers with pagination and search |
| `/api/customers/:id` | GET | Get customer detail by ID |
| `/api/customers` | POST | Create a new customer |
| `/api/customers/:id` | PUT | Update an existing customer |
| `/api/customers/:id` | DELETE | Delete a customer |

## Customer API Details

### GET /api/customers

Query parameters:
- `page` (integer, default: 1) - Page number
- `limit` (integer, default: 20) - Items per page
- `search` (string, optional) - Search by name or email

Response:
```json
{
  "data": [{ "id", "name", "email", "phone", "company", "notes", "createdAt", "updatedAt" }],
  "pagination": { "page", "limit", "total", "totalPages" }
}
```

### GET /api/customers/:id

Response (200): Customer object
Response (401): `{ "success": false, "error": "Unauthorized" }`
Response (404): `{ "success": false, "error": "Customer not found" }`

### POST /api/customers

Request body:
```json
{
  "name": "string (required)",
  "email": "string (required)",
  "phone": "string (optional)",
  "company": "string (optional)",
  "notes": "string (optional)"
}
```

Response (201): Created customer object
Response (400): `{ "success": false, "error": "name and email are required" }`
Response (401): `{ "success": false, "error": "Unauthorized" }`

### PUT /api/customers/:id

Request body: Partial customer object
Response (200): Updated customer object
Response (401): `{ "success": false, "error": "Unauthorized" }`
Response (404): `{ "success": false, "error": "Customer not found" }`

### DELETE /api/customers/:id

Response (204): No content
Response (401): `{ "success": false, "error": "Unauthorized" }`
Response (404): `{ "success": false, "error": "Customer not found" }`

## Authentication API Details

### POST /api/auth/login

Request body:
```json
{
  "username": "string (required)",
  "password": "string (required)"
}
```

Response (200):
```json
{
  "success": true,
  "data": { "userId": "user_1", "username": "admin" }
}
```
Default credentials: username=admin, password=admin123

Response (400): `{ "success": false, "error": "username and password are required" }`
Response (401): `{ "success": false, "error": "Invalid credentials" }`

### POST /api/auth/logout

Response (200): `{ "success": true }`
Response (401): `{ "success": false, "error": "Unauthorized" }`

## Static Files

- Server serves static files from `impl/be_changes/public/`
- GET / returns public/index.html
- All other non-API paths return index.html (SPA fallback)

## Run Instructions

```bash
cd workspace/sandbox/crm_site/impl/be_changes
npm install
npm start
```

Server runs on `http://localhost:3000` (or PORT env var if set)

## Scope Constraints

- Customer CRUD API only (no other entities)
- SQLite persistence (data survives server restart)
- Session-based authentication (no JWT)
- CSRF protection on POST/PUT/DELETE endpoints
- No pagination beyond limit parameter (max 100 recommended)
- No customer relationships or associated entities

## Database Schema

### customers table
- id: TEXT PRIMARY KEY
- name: TEXT NOT NULL
- email: TEXT NOT NULL
- phone: TEXT DEFAULT ''
- company: TEXT DEFAULT ''
- notes: TEXT DEFAULT ''
- createdAt: TEXT NOT NULL (ISO8601)
- updatedAt: TEXT NOT NULL (ISO8601)

### sessions table
- id: TEXT PRIMARY KEY
- userId: TEXT NOT NULL
- username: TEXT NOT NULL
- createdAt: TEXT NOT NULL (ISO8601)
- expiresAt: TEXT NOT NULL (ISO8601)

## Shared Types

Customer entity fields:
- `id`: string (UUID prefixed with "cust_")
- `name`: string
- `email`: string
- `phone`: string
- `company`: string
- `notes`: string
- `createdAt`: ISO8601 timestamp
- `updatedAt`: ISO8601 timestamp