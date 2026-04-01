import express from 'express';
import cors from 'cors';
import { randomUUID } from 'crypto';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const customerStore = new Map();

function createCustomer(data) {
  const now = new Date().toISOString();
  const customer = {
    id: `cust_${randomUUID()}`,
    name: String(data.name || ''),
    email: String(data.email || ''),
    phone: String(data.phone || ''),
    company: String(data.company || ''),
    notes: String(data.notes || ''),
    createdAt: now,
    updatedAt: now,
  };
  customerStore.set(customer.id, customer);
  return customer;
}

createCustomer({ name: 'Alice Chen', email: 'alice@example.com', phone: '13800138000', company: 'Acme Corp' });
createCustomer({ name: 'Bob Wang', email: 'bob@example.com', phone: '13900139000', company: 'TechStart Inc' });
createCustomer({ name: 'Carol Liu', email: 'carol@example.com', phone: '13700137000', company: 'Global Solutions' });

app.get('/api/customers', (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 20;
  const search = (req.query.search || '').toLowerCase();

  let result = Array.from(customerStore.values());
  if (search) {
    result = result.filter(c =>
      c.name.toLowerCase().includes(search) ||
      c.email.toLowerCase().includes(search) ||
      c.company.toLowerCase().includes(search)
    );
  }

  result.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

  const total = result.length;
  const totalPages = Math.ceil(total / limit);
  const offset = (page - 1) * limit;

  res.json({
    data: result.slice(offset, offset + limit),
    pagination: { page, limit, total, totalPages }
  });
});

app.get('/api/customers/:id', (req, res) => {
  const customer = customerStore.get(req.params.id);
  if (!customer) {
    return res.status(404).json({ success: false, error: 'Customer not found' });
  }
  res.json({ success: true, data: customer });
});

app.post('/api/customers', (req, res) => {
  const { name, email } = req.body;
  if (!name || !email) {
    return res.status(400).json({ success: false, error: 'name and email are required' });
  }
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({ success: false, error: 'invalid email format' });
  }
  const customer = createCustomer(req.body);
  res.status(201).json({ success: true, data: customer });
});

app.put('/api/customers/:id', (req, res) => {
  const customer = customerStore.get(req.params.id);
  if (!customer) {
    return res.status(404).json({ success: false, error: 'Customer not found' });
  }
  const { email } = req.body;
  if (email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ success: false, error: 'invalid email format' });
    }
  }
  const updated = {
    ...customer,
    ...req.body,
    id: customer.id,
    createdAt: customer.createdAt,
    updatedAt: new Date().toISOString(),
  };
  customerStore.set(updated.id, updated);
  res.json({ success: true, data: updated });
});

app.delete('/api/customers/:id', (req, res) => {
  if (!customerStore.has(req.params.id)) {
    return res.status(404).json({ success: false, error: 'Customer not found' });
  }
  customerStore.delete(req.params.id);
  res.status(204).send();
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`Customer API server listening on http://localhost:${PORT}`);
});

export default app;