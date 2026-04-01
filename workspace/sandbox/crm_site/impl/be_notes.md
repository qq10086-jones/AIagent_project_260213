# Backend Implementation Notes

## Decisions & Assumptions

1. **Express.js Server**: Using Express as the backend framework per package.json dependencies
2. **In-Memory Data Store**: Customer data stored in-memory Map; no external database required
3. **JSON API**: All endpoints return JSON responses with consistent `{ success, data, error }` structure
4. **Customer Entity**: Core domain model with id, name, email, phone, company, notes, createdAt, updatedAt fields
5. **Seeded Data**: Three sample customers are pre-loaded for testing

## API Endpoints (Customer CRM)

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

### PUT /api/customers/:id

Request body: Partial customer object
Response (200): Updated customer object
Response (404): `{ "success": false, "error": "Customer not found" }`

### DELETE /api/customers/:id

Response (204): No content
Response (404): `{ "success": false, "error": "Customer not found" }`

## Run Instructions

```bash
cd workspace/sandbox/crm_site
npm install
npm start
```

Server runs on `http://localhost:3000`

## Scope Constraints

- Customer CRUD API only
- In-memory storage (data lost on server restart)
- No authentication/authorization
- No pagination beyond limit parameter
- No data validation beyond required fields
- No customer relationships or associated entities

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