import express from "express";
import cors from "cors";
import session from "express-session";
import csurf from "csurf";
import Database from "better-sqlite3";
import { randomUUID } from "crypto";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const appRoot = resolve(__dirname, "..", "..");
const dbPath = join(appRoot, "data", "crm.db");

const db = new Database(dbPath);

db.exec(`
  CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT DEFAULT '',
    company TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    createdAt TEXT NOT NULL,
    updatedAt TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    userId TEXT NOT NULL,
    username TEXT NOT NULL,
    createdAt TEXT NOT NULL,
    expiresAt TEXT NOT NULL
  );
`);

const insertCustomer = db.prepare(`
  INSERT INTO customers (id, name, email, phone, company, notes, createdAt, updatedAt)
  VALUES (@id, @name, @email, @phone, @company, @notes, @createdAt, @updatedAt)
`);

const seedCustomers = [
  { name: '张三', email: 'zhangsan@example.com', phone: '13800138000', company: '示例公司A', notes: '' },
  { name: '李四', email: 'lisi@example.com', phone: '13900139000', company: '示例公司B', notes: '' },
  { name: '王五', email: 'wangwu@example.com', phone: '13700137000', company: '示例公司C', notes: '' },
];

const existingCustomers = db.prepare("SELECT COUNT(*) as count FROM customers").get();
if (existingCustomers.count === 0) {
  const now = new Date().toISOString();
  for (const c of seedCustomers) {
    insertCustomer.run({
      id: `cust_${randomUUID()}`,
      name: c.name,
      email: c.email,
      phone: c.phone,
      company: c.company,
      notes: c.notes,
      createdAt: now,
      updatedAt: now,
    });
  }
}

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.use(session({
  secret: process.env.SESSION_SECRET || 'crm-dev-secret-change-in-production',
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: false,
    httpOnly: true,
    maxAge: 24 * 60 * 60 * 1000,
  },
}));

const csrfProtection = csurf({ cookie: false });

function validateSession(req, _res, next) {
  if (!req.session || !req.session.userId) {
    return next();
  }
  const sessionId = req.session.sessionId;
  if (!sessionId) {
    return next();
  }
  const session = db.prepare("SELECT * FROM sessions WHERE id = ? AND expiresAt > ?").get(sessionId, new Date().toISOString());
  if (!session) {
    req.session.destroy();
    return next();
  }
  req.session.user = { id: session.userId, username: session.username };
  next();
}

function generateCsrfToken(req, _res, next) {
  req.csrfToken = req.csrfToken;
  next();
}

function requireAuth(req, res, next) {
  if (!req.session || !req.session.userId) {
    return res.status(401).json({ success: false, error: 'Unauthorized' });
  }
  next();
}

function loggerMiddleware(req, _res, next) {
  console.log(`${req.method} ${req.path}`);
  next();
}

app.use(loggerMiddleware);
app.use(validateSession);
app.use(generateCsrfToken);

app.get('/api/csrf-token', (req, res) => {
  res.json({ csrfToken: req.csrfToken });
});

app.post('/api/auth/login', (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ success: false, error: 'username and password are required' });
  }
  if (username === 'admin' && password === 'admin123') {
    const sessionId = `sess_${randomUUID()}`;
    const now = new Date();
    const expiresAt = new Date(now.getTime() + 24 * 60 * 60 * 1000);
    db.prepare("INSERT INTO sessions (id, userId, username, createdAt, expiresAt) VALUES (?, ?, ?, ?, ?)").run(
      sessionId, 'user_1', username, now.toISOString(), expiresAt.toISOString()
    );
    req.session.sessionId = sessionId;
    req.session.userId = 'user_1';
    req.session.username = username;
    return res.json({ success: true, data: { userId: 'user_1', username } });
  }
  res.status(401).json({ success: false, error: 'Invalid credentials' });
});

app.post('/api/auth/logout', requireAuth, (req, res) => {
  const sessionId = req.session.sessionId;
  if (sessionId) {
    db.prepare("DELETE FROM sessions WHERE id = ?").run(sessionId);
  }
  req.session.destroy();
  res.json({ success: true });
});

app.get('/api/customers', requireAuth, (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 20;
  const search = (req.query.search || '').toLowerCase();

  let query = "SELECT * FROM customers";
  let countQuery = "SELECT COUNT(*) as count FROM customers";
  const params = [];

  if (search) {
    query += " WHERE LOWER(name) LIKE ? OR LOWER(email) LIKE ?";
    countQuery += " WHERE LOWER(name) LIKE ? OR LOWER(email) LIKE ?";
    params.push(`%${search}%`, `%${search}%`);
  }

  query += " ORDER BY createdAt DESC LIMIT ? OFFSET ?";
  params.push(limit, (page - 1) * limit);

  const customers = db.prepare(query).all(...params);
  const { count } = db.prepare(countQuery).get(...(search ? [`%${search}%`, `%${search}%`] : []));
  const total = count;

  res.json({
    data: customers,
    pagination: { page, limit, total, totalPages: Math.ceil(total / limit) }
  });
});

app.get('/api/customers/:id', requireAuth, (req, res) => {
  const customer = db.prepare("SELECT * FROM customers WHERE id = ?").get(req.params.id);
  if (!customer) {
    return res.status(404).json({ success: false, error: 'Customer not found' });
  }
  res.json({ success: true, data: customer });
});

app.post('/api/customers', requireAuth, csrfProtection, (req, res) => {
  const { name, email, phone, company, notes } = req.body;
  if (!name || !email) {
    return res.status(400).json({ success: false, error: 'name and email are required' });
  }
  const now = new Date().toISOString();
  const customer = {
    id: `cust_${randomUUID()}`,
    name,
    email,
    phone: phone || '',
    company: company || '',
    notes: notes || '',
    createdAt: now,
    updatedAt: now,
  };
  insertCustomer.run(customer);
  res.status(201).json({ success: true, data: customer });
});

app.put('/api/customers/:id', requireAuth, csrfProtection, (req, res) => {
  const existing = db.prepare("SELECT * FROM customers WHERE id = ?").get(req.params.id);
  if (!existing) {
    return res.status(404).json({ success: false, error: 'Customer not found' });
  }
  const { name, email, phone, company, notes } = req.body;
  const updated = {
    ...existing,
    name: name !== undefined ? name : existing.name,
    email: email !== undefined ? email : existing.email,
    phone: phone !== undefined ? phone : existing.phone,
    company: company !== undefined ? company : existing.company,
    notes: notes !== undefined ? notes : existing.notes,
    updatedAt: new Date().toISOString(),
  };
  db.prepare(`
    UPDATE customers SET name = @name, email = @email, phone = @phone, company = @company, notes = @notes, updatedAt = @updatedAt
    WHERE id = @id
  `).run(updated);
  res.json({ success: true, data: updated });
});

app.delete('/api/customers/:id', requireAuth, (req, res) => {
  const existing = db.prepare("SELECT * FROM customers WHERE id = ?").get(req.params.id);
  if (!existing) {
    return res.status(404).json({ success: false, error: 'Customer not found' });
  }
  db.prepare("DELETE FROM customers WHERE id = ?").run(req.params.id);
  res.status(204).send();
});

app.use(express.static(join(__dirname, 'public')));

app.get('/', (_req, res) => {
  res.sendFile(join(__dirname, 'public', 'index.html'));
});

app.use((req, res) => {
  res.status(404).json({ success: false, error: 'Not found' });
});

app.use((err, req, res, _next) => {
  console.error(err.stack);
  res.status(500).json({ success: false, error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`CRM Backend listening on http://localhost:${PORT}`);
});

export default app;