# Expense Tracker — Interface Specification

## REST API
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/expenses | List expenses (query: ?month=&category=) |
| POST | /api/expenses | Create expense |
| PUT | /api/expenses/:id | Update expense |
| DELETE | /api/expenses/:id | Delete expense |
| GET | /api/summary/monthly | Monthly aggregation |
| GET | /api/export/csv | CSV download |

## Data Model
```sql
CREATE TABLE expenses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  amount REAL NOT NULL,
  description TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'Other',
  date TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```