# Module Breakdown

## Modules
- auth: handles login and session management
- crm: contact and deal management

## Interfaces
- POST /login
- GET /contacts

## Dependency Choices
- PostgreSQL for primary persistence
- Redis for session cache

## Risk Notes
- Auth migration requires staged rollout
- Session token rotation on deploy