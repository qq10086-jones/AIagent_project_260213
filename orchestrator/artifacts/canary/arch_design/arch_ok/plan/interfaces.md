# API Interfaces

## POST /login
Request: `{ username, password }`
Response: `{ token, expires_at }`

## GET /contacts
Response: `[{ id, name, email }]`

## POST /contacts
Request: `{ name, email, phone? }`