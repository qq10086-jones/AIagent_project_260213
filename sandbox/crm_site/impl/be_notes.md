# Backend Implementation Notes

## Decisions & Assumptions

1. **Express.js Server**: Using Express as the backend framework per package.json dependencies
2. **Static File Serving**: Serving the SPA from the same Express server for simplicity
3. **In-Memory Data**: Content stored in-memory; no database required for promotional site
4. **JSON API**: All endpoints return JSON responses with consistent structure
5. **Fashion Brand Content**: Updated all content to match the fashion brand (Elegance Fashion) theme

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/content` | GET | Get all landing page content |
| `/api/hero` | GET | Get hero section data |
| `/api/features` | GET | Get product selling points |
| `/api/faqs` | GET | Get FAQ items |
| `/api/contact` | GET | Get contact info |
| `/api/contact` | POST | Submit contact form |
| `/api/icons` | GET | Get SVG icons |
| `/api/story` | GET | Get brand story |
| `/api/reviews` | GET | Get all user reviews |
| `/api/reviews/latest` | GET | Get latest 3 reviews |
| `/api/reviews` | POST | Submit a new review |

## Run Instructions

```bash
cd sandbox/crm_site
npm install
npm start
```

Server runs on `http://localhost:3000`

## Content Structure

### Hero Section
```json
{
  "title": "Elegance Fashion",
  "subtitle": "Discover timeless elegance and modern style...",
  "ctaText": "Explore Collection"
}
```

### Features (Product Selling Points)
```json
[
  { "icon": "quality", "title": "Premium Quality", "description": "..." },
  { "icon": "design", "title": "Modern Design", "description": "..." },
  { "icon": "sustainability", "title": "Sustainable Fashion", "description": "..." }
]
```

### Brand Story
```json
{
  "title": "Our Story",
  "subtitle": "A Legacy of Elegance",
  "content": [
    { "heading": "Founded in 2010", "paragraph": "..." },
    { "heading": "Our Philosophy", "paragraph": "..." },
    { "heading": "Craftsmanship", "paragraph": "..." }
  ],
  "timeline": [
    { "year": "2010", "event": "Founded in Shanghai" },
    ...
  ]
}
```

### User Reviews
```json
[
  {
    "id": 1,
    "name": "Sarah Chen",
    "rating": 5,
    "comment": "Absolutely love the quality!",
    "date": "2024-01-15"
  }
]
```

## Contact Form Integration

The POST `/api/contact` endpoint accepts:
```json
{
  "name": "string",
  "email": "string",
  "message": "string"
}
```

Returns success/error response for frontend to display.

## Reviews Submission

The POST `/api/reviews` endpoint accepts:
```json
{
  "name": "string",
  "rating": "number (1-5)",
  "comment": "string"
}
```

Returns success message and the created review.

## Scope Constraints

- Static file serving for promotional site
- Landing page content API endpoints
- Contact form submission
- Brand story endpoint
- User reviews endpoint
- NOT in scope: User authentication, database persistence, email sending, shopping cart, product catalog
