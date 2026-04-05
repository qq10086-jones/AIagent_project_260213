# Backend Implementation Notes

## Task Status

- T-BE-1: Initialize Node.js project with Express 4.x dependency - [COMPLETED] Express 4.18.2 specified in package.json
- T-BE-2: Create server.js with in-memory customer store using Map - [COMPLETED] Using Map data structure for customers
- T-BE-3: Implement GET /api/customers returning customer array - [COMPLETED] Returns { data: customers[] }
- T-BE-4: Implement GET /api/customers/:id returning single customer - [COMPLETED] Returns 404 if not found
- T-BE-5: Implement POST /api/customers with validation - [COMPLETED] Returns 400 for invalid data

## Architecture Decisions Followed

- ADR-003 In-Memory Data Store: Using JavaScript Map for customer storage, no SQLite
- ADR-004 No Authentication: All endpoints are public, no session or CSRF protection

## Decisions & Assumptions

1. **Express.js Server**: Using Express as the backend framework per package.json dependencies
2. **In-Memory Data Store**: Using JavaScript Map for O(1) customer lookups, no SQLite
3. **No Authentication**: Per ADR-004, all endpoints are public
4. **Input Validation**: Using express-validator for request validation on customer endpoints
5. **JSON API**: All endpoints return JSON responses with consistent structure
6. **Customer Entity**: Core domain model with id, name, email, phone, company, notes, createdAt, updatedAt fields
7. **Seeded Data**: Three sample customers pre-loaded on startup

## API Endpoints

### Customers

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/customers` | GET | List all customers |
| `/api/customers/:id` | GET | Get customer detail by ID |
| `/api/customers` | POST | Create a new customer |
| `/api/customers/:id` | PUT | Update an existing customer |

### Static Files

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve static index.html |
| `/public/*` | GET | Serve static assets |

## API Response Format

### GET /api/customers

Response (200):
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

Response (200): Customer object
Response (404): `{ "success": false, "error": "Customer not found" }`

### POST /api/customers

Request body:
```json
{
  "name": "string (required, non-empty)",
  "email": "string (required, valid email format)",
  "phone": "string (optional)",
  "company": "string (optional)",
  "notes": "string (optional)"
}
```

Response (201): Created customer object
Response (400): `{ "success": false, "errors": [{ "msg": "...", "path": "..." }] }`

### PUT /api/customers/:id

Request body: Partial customer object
Response (200): Updated customer object
Response (400): `{ "success": false, "errors": [...] }`
Response (404): `{ "success": false, "error": "Customer not found" }`

## Static Files

- Server serves static files from `impl/be_changes/public/`
- GET / returns public/index.html

## Run Instructions

```bash
cd impl/be_changes
npm install
npm start
```

Server runs on http://localhost:3000 (or PORT env var if set)

## Scope Constraints

- Customer CRUD API only (no other entities)
- In-memory data store (data resets on server restart)
- No authentication (per ADR-004)
- No CSRF protection (not needed without auth)
- Input validation via express-validator on POST/PUT
- No pagination (returns all customers)
- No customer relationships or associated entities

## Shared Types

```typescript
interface Customer {
  id: string;           // UUID prefixed with "cust_"
  name: string;
  email: string;
  phone: string;
  company: string;
  notes: string;
  createdAt: string;     // ISO8601 timestamp
  updatedAt: string;     // ISO8601 timestamp
}

interface CustomerCreateRequest {
  name: string;          // required
  email: string;         // required, valid email
  phone?: string;
  company?: string;
  notes?: string;
}

interface CustomerUpdateRequest {
  name?: string;
  email?: string;
  phone?: string;
  company?: string;
  notes?: string;
}
```
