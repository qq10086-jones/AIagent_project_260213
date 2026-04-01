# Backend Implementation Notes

## Implementation Decisions

### Architecture
- Express.js server with in-memory customer store (Map-based)
- RESTful API design following backend/frontend separation principle (ADR-001)
- Explicit API contracts for customer flows (ADR-002)

### Data Model
- Customer entity with fields: id, name, email, phone, company, notes, createdAt, updatedAt
- ID format: `cust_<uuid>` prefix for customer entities
- Timestamps stored as ISO 8601 format

### API Endpoints
- `GET /api/customers` - List customers with pagination and search
- `GET /api/customers/:id` - Get single customer by ID
- `POST /api/customers` - Create new customer (requires name and email)
- `PUT /api/customers/:id` - Update existing customer
- `DELETE /api/customers/:id` - Delete customer
- `GET /health` - Health check endpoint

### Validation
- Email format validation using regex on create and update
- Required field validation (name, email) on create
- 404 returned for non-existent customer lookups

### Scope Constraints (NOT Implemented)
- No database persistence (in-memory only, data lost on restart)
- No authentication/authorization
- No pagination cursor-based navigation (uses offset/limit only)
- No customer filtering by company or other fields beyond search
- No bulk operations
- No soft delete / recovery

## Run Instructions

### Install dependencies
```bash
cd workspace/sandbox/project/impl/be_changes
npm init -y
npm install express cors
```

### Run server
```bash
node server.js
```
Server starts on http://localhost:3000

### Test endpoints
```bash
# List customers
curl http://localhost:3000/api/customers

# Get single customer
curl http://localhost:3000/api/customers/cust_<id>

# Create customer
curl -X POST http://localhost:3000/api/customers \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@example.com"}'

# Update customer
curl -X PUT http://localhost:3000/api/customers/cust_<id> \
  -H "Content-Type: application/json" \
  -d '{"company":"New Company"}'

# Delete customer
curl -X DELETE http://localhost:3000/api/customers/cust_<id>
```