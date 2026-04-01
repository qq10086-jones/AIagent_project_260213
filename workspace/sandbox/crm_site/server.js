const express = require('express');
const path = require('path');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const customers = [
  { id: 'c1', name: 'Ada Lovelace', email: 'ada@example.com' },
  { id: 'c2', name: 'Grace Hopper', email: 'grace@example.com' },
];

function findCustomer(id) {
  return customers.find((customerRecord) => customerRecord.id === id);
}

app.get('/api/customers', (_req, res) => {
  res.json({ customers });
});

app.get('/api/customers/:id', (req, res) => {
  const customer = findCustomer(req.params.id);
  if (!customer) return res.status(404).json({ error: 'not_found' });
  return res.json(customer);
});

app.post('/api/customers', (req, res) => {
  const next = {
    id: `c${customers.length + 1}`,
    name: String(req.body?.name || 'New Customer'),
    email: String(req.body?.email || 'new@example.com'),
  };
  customers.push(next);
  res.status(201).json(next);
});

app.put('/api/customers/:id', (req, res) => {
  const customer = findCustomer(req.params.id);
  if (!customer) return res.status(404).json({ error: 'not_found' });
  customer.name = String(req.body?.name || customer.name);
  customer.email = String(req.body?.email || customer.email);
  return res.json(customer);
});

app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const port = Number(process.env.PORT || 3000);
app.listen(port, () => {
  console.log(`crm-live-ready:${port}`);
});
