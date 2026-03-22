# Frontend Implementation Notes - Coffee Shop Website

## UI Scope

### Coffee Shop Website Components

1. **Hero Section (首页主视觉)**
   - Brand name "Bean & Brew" with coffee shop tagline
   - CTA button "Reserve a Table" for reservations
   - Smooth scroll to reservation section on click

2. **Menu Display (菜单展示)**
   - Three categories: Espresso Drinks, Handcrafted Beverages, Bakery
   - Item name, price, and description for each
   - Grid layout for menu items

3. **Business Hours (营业时间)**
   - Weekday and weekend hours display
   - Icon-enhanced visual presentation

4. **Store Address (门店地址)**
   - Full address display in contact section
   - Phone number and email

5. **Reservation Button (预约按钮)**
   - Placeholder with "Coming Soon" message
   - Calls to phone number for actual bookings
   - Scroll to reservation section from hero CTA

6. **Features Section (门店卖点)**
   - Premium Beans, Expert Baristas, Eco-Friendly
   - Feature cards with icons

7. **Brand Story (门店介绍)**
   - History with timeline (2018-2024)
   - Content sections

8. **User Reviews (用户评价)**
   - Latest 3 reviews in card format
   - Review submission form

9. **FAQ Section (常见问题)**
   - Accordion-style FAQ items
   - Topics: dairy-free options, reservations, Wi-Fi, pet-friendly

10. **Contact Section**
    - Form with name, email, message fields
    - Contact info display

## API Consumption

- **Contract Source**: `handoff/be_to_fe.json`
- **Consumed Endpoints**:
  - `GET /api/hero` - Hero section content
  - `GET /api/features` - Store selling points
  - `GET /api/story` - Brand story with timeline
  - `GET /api/reviews/latest` - Latest 3 customer reviews
  - `GET /api/faqs` - FAQ items
  - `GET /api/contact` - Contact information
  - `POST /api/contact` - Contact form submission
  - `POST /api/reviews` - Review submission

- **Fallback**: Static data used when API unavailable (LANDING_CONTENT)

## Run Instructions

### Development Mode

```bash
cd sandbox/crm_site
npm install
npm start
```

The server will start on http://localhost:3000

### Verification

```bash
node --check sandbox/crm_site/impl/fe_changes/app.js
```

### File Structure

```
sandbox/crm_site/
├── impl/
│   └── fe_changes/
│       └── app.js          # Frontend implementation
├── index.html              # Entry HTML
├── styles.css              # All styling
└── app.js                  # Root app.js (loads fe_changes)
```

## Assumptions

1. Backend implements all API endpoints as defined in be_to_fe.json contract
2. HTML structure has required elements with matching CSS classes
3. Contact form provides user feedback on submission
4. Reviews section displays latest 3 reviews by default
5. Reservation system is placeholder only (actual booking not in backend scope)
6. Menu data is hardcoded but could be fetched from API
7. Static fallback data available for offline/demo mode
