import express from 'express';
import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const db = new Database(path.join(__dirname, 'expenses.db'));
db.exec(`CREATE TABLE IF NOT EXISTS expenses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  amount REAL NOT NULL,
  description TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'Other',
  date TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
)`);

// GET /api/expenses
app.get('/api/expenses', (req, res) => {
  const { month, category } = req.query;
  let sql = 'SELECT * FROM expenses WHERE 1=1';
  const params = [];
  if (month) { sql += ' AND strftime("%Y-%m", date) = ?'; params.push(month); }
  if (category) { sql += ' AND category = ?'; params.push(category); }
  sql += ' ORDER BY date DESC';
  res.json(db.prepare(sql).all(...params));
});

// POST /api/expenses
app.post('/api/expenses', (req, res) => {
  const { amount, description, category = 'Other', date } = req.body;
  if (!amount || !description || !date) return res.status(400).json({ error: 'Missing required fields' });
  const result = db.prepare('INSERT INTO expenses (amount, description, category, date) VALUES (?, ?, ?, ?)').run(amount, description, category, date);
  res.status(201).json({ id: result.lastInsertRowid, amount, description, category, date });
});

// PUT /api/expenses/:id
app.put('/api/expenses/:id', (req, res) => {
  const { amount, description, category, date } = req.body;
  const existing = db.prepare('SELECT * FROM expenses WHERE id = ?').get(req.params.id);
  if (!existing) return res.status(404).json({ error: 'Not found' });
  db.prepare('UPDATE expenses SET amount=?, description=?, category=?, date=? WHERE id=?')
    .run(amount ?? existing.amount, description ?? existing.description, category ?? existing.category, date ?? existing.date, req.params.id);
  res.json({ ok: true });
});

// DELETE /api/expenses/:id
app.delete('/api/expenses/:id', (req, res) => {
  const result = db.prepare('DELETE FROM expenses WHERE id = ?').run(req.params.id);
  if (result.changes === 0) return res.status(404).json({ error: 'Not found' });
  res.json({ ok: true });
});

// GET /api/summary/monthly
app.get('/api/summary/monthly', (req, res) => {
  const rows = db.prepare(`
    SELECT strftime('%Y-%m', date) as month, category, SUM(amount) as total
    FROM expenses GROUP BY month, category ORDER BY month DESC
  `).all();
  const grouped = {};
  for (const row of rows) {
    if (!grouped[row.month]) grouped[row.month] = { month: row.month, total: 0, by_category: {} };
    grouped[row.month].total += row.total;
    grouped[row.month].by_category[row.category] = row.total;
  }
  res.json(Object.values(grouped));
});

// GET /api/export/csv
app.get('/api/export/csv', (req, res) => {
  const { from, to } = req.query;
  let sql = 'SELECT * FROM expenses WHERE 1=1';
  const params = [];
  if (from) { sql += ' AND date >= ?'; params.push(from); }
  if (to) { sql += ' AND date <= ?'; params.push(to); }
  sql += ' ORDER BY date DESC';
  const rows = db.prepare(sql).all(...params);
  const header = 'id,amount,description,category,date,created_at';
  const csvRows = rows.map(r => [r.id, r.amount, `"${String(r.description).replace(/"/g, '""')}"`, r.category, r.date, r.created_at].join(','));
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', 'attachment; filename="expenses.csv"');
  res.send([header, ...csvRows].join('\n'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Expense Tracker running on http://localhost:${PORT}`));
