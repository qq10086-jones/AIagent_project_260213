import express from "express";
import cors from "cors";
import { randomUUID } from "crypto";
import { body, validationResult } from "express-validator";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const customers = new Map();

const seedCustomers = [
  { name: "Alice Johnson", email: "alice@example.com", phone: "555-0101", company: "Acme Corp", notes: "VIP customer" },
  { name: "Bob Smith", email: "bob@example.com", phone: "555-0102", company: "Tech Inc", notes: "" },
  { name: "Carol White", email: "carol@example.com", phone: "555-0103", company: "Global Co", notes: "" },
];

for (const c of seedCustomers) {
  const id = `cust_${randomUUID()}`;
  const now = new Date().toISOString();
  customers.set(id, { id, ...c, createdAt: now, updatedAt: now });
}

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

function validateRequest(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ success: false, errors: errors.array() });
  }
  next();
}

app.get("/api/customers", (req, res) => {
  const all = Array.from(customers.values());
  res.json({ data: all });
});

app.get("/api/customers/:id", (req, res) => {
  const customer = customers.get(req.params.id);
  if (!customer) {
    return res.status(404).json({ success: false, error: "Customer not found" });
  }
  res.json({ success: true, data: customer });
});

const customerValidation = [
  body("name").trim().notEmpty().withMessage("name is required"),
  body("email").trim().isEmail().withMessage("email must be a valid email"),
  body("phone").optional().trim(),
  body("company").optional().trim(),
  body("notes").optional().trim(),
];

app.post("/api/customers", customerValidation, validateRequest, (req, res) => {
  const { name, email, phone, company, notes } = req.body;
  const now = new Date().toISOString();
  const customer = {
    id: `cust_${randomUUID()}`,
    name: name.trim(),
    email: email.trim().toLowerCase(),
    phone: (phone || "").trim(),
    company: (company || "").trim(),
    notes: (notes || "").trim(),
    createdAt: now,
    updatedAt: now,
  };
  customers.set(customer.id, customer);
  res.status(201).json({ success: true, data: customer });
});

app.put("/api/customers/:id", [
  body("name").optional().trim().notEmpty().withMessage("name cannot be empty"),
  body("email").optional().trim().isEmail().withMessage("email must be a valid email"),
  body("phone").optional().trim(),
  body("company").optional().trim(),
  body("notes").optional().trim(),
], validateRequest, (req, res) => {
  const existing = customers.get(req.params.id);
  if (!existing) {
    return res.status(404).json({ success: false, error: "Customer not found" });
  }
  const { name, email, phone, company, notes } = req.body;
  const updated = {
    ...existing,
    name: name !== undefined ? name.trim() : existing.name,
    email: email !== undefined ? email.trim().toLowerCase() : existing.email,
    phone: phone !== undefined ? phone.trim() : existing.phone,
    company: company !== undefined ? company.trim() : existing.company,
    notes: notes !== undefined ? notes.trim() : existing.notes,
    updatedAt: new Date().toISOString(),
  };
  customers.set(updated.id, updated);
  res.json({ success: true, data: updated });
});

app.use(express.static(join(__dirname, "public")));

app.get("/", (_req, res) => {
  res.sendFile(join(__dirname, "public", "index.html"));
});

app.use((req, res) => {
  res.status(404).json({ success: false, error: "Not found" });
});

app.listen(PORT, () => {
  console.log(`CRM Backend listening on http://localhost:${PORT}`);
});

export default app;
