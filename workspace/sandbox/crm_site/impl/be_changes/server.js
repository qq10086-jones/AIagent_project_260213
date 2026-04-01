import express from "express";
import cors from "cors";
import fs from "fs";
import { randomUUID } from "crypto";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const appRoot = resolve(__dirname, "..", "..");
const dataPath = join(appRoot, "data", "store.json");

function readStore() {
  return JSON.parse(fs.readFileSync(dataPath, "utf8"));
}

function writeStore(store) {
  fs.writeFileSync(dataPath, JSON.stringify(store, null, 2), "utf8");
}

function withPreviews(store) {
  const year = new Date().getFullYear();
  const templates = (store.templates || []).map((template) => {
    const count = (store.documents || []).filter((doc) => doc.templateId === template.id && String(doc.docNumber || "").includes(String(year))).length;
    return {
      ...template,
      nextNumberPreview: `${template.codePrefix}-${year}-${String(count + 1).padStart(3, "0")}`,
    };
  });
  return { ...store, templates };
}

function nowIso() {
  return new Date().toISOString();
}

function summarizeFromContent(content) {
  const clean = String(content || "").trim();
  if (!clean) return "未命名请求";
  return clean.length > 28 ? `${clean.slice(0, 28)}...` : clean;
}

function nextDocNumber(store, template) {
  const year = new Date().getFullYear();
  const matches = (store.documents || []).filter((doc) => doc.templateId === template.id && String(doc.docNumber || "").includes(`${year}`));
  return `${template.codePrefix}-${year}-${String(matches.length + 1).padStart(3, "0")}`;
}

function nextRevision(currentRevision) {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const idx = alphabet.indexOf(String(currentRevision || "A").toUpperCase());
  if (idx < 0 || idx >= alphabet.length - 1) return "Z";
  return alphabet[idx + 1];
}

function makeHistory(action, document, payload) {
  const labelMap = { issued: "首版发行", revised: "修订发行" };
  return {
    id: `hist_${randomUUID()}`,
    documentId: document.id,
    docNumber: document.docNumber,
    revision: document.revision,
    action,
    actionLabel: labelMap[action] || action,
    actor: payload.actor || document.owner,
    source: payload.source || document.sourceLabel || "system",
    distribution: document.distribution,
    changeSummary: payload.changeSummary,
    timestamp: nowIso(),
  };
}

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(appRoot));

app.get("/api/bootstrap", (_req, res) => {
  res.json({ success: true, data: withPreviews(readStore()) });
});

app.post("/api/intake/discord", (req, res) => {
  const store = readStore();
  const content = String(req.body?.content || "").trim();
  if (!content) {
    return res.status(400).json({ success: false, error: "content is required" });
  }
  const request = {
    id: `req_${randomUUID()}`,
    source: "discord",
    requester: String(req.body?.requester || "discord-user"),
    channel: String(req.body?.channel || "#doc-control"),
    summary: summarizeFromContent(content),
    content,
    status: "new",
    suggestedTemplateId: req.body?.suggestedTemplateId || null,
    createdAt: nowIso(),
  };
  store.intakeRequests.unshift(request);
  writeStore(store);
  res.json({ success: true, data: request });
});

app.post("/api/documents/generate", (req, res) => {
  const store = readStore();
  const payload = req.body || {};
  const template = (store.templates || []).find((item) => item.id === payload.templateId);
  if (!template) {
    return res.status(400).json({ success: false, error: "unknown templateId" });
  }
  if (!payload.title || !payload.department || !payload.owner || !payload.effectiveDate) {
    return res.status(400).json({ success: false, error: "title, department, owner, effectiveDate are required" });
  }

  const document = {
    id: `doc_${randomUUID()}`,
    templateId: template.id,
    templateName: template.name,
    docNumber: nextDocNumber(store, template),
    revision: "A",
    title: String(payload.title).trim(),
    department: String(payload.department).trim(),
    owner: String(payload.owner).trim(),
    effectiveDate: String(payload.effectiveDate).trim(),
    distribution: Array.isArray(payload.distribution) ? payload.distribution : [],
    status: "released",
    sourceLabel: payload.sourceRequestId ? "Discord Intake" : "Manual Issue",
    sourceRequestId: payload.sourceRequestId || null,
    excelTemplateRef: String(payload.excelTemplateRef || template.excelTemplateRef || ""),
    payloadSummary: String(payload.payloadSummary || ""),
    updatedAt: nowIso(),
  };

  const history = makeHistory("issued", document, {
    actor: document.owner,
    source: document.sourceLabel,
    changeSummary: String(payload.changeSummary || `首版发行 ${document.title}`),
  });

  store.documents.unshift(document);
  store.releaseHistory.unshift(history);
  if (document.sourceRequestId) {
    store.intakeRequests = (store.intakeRequests || []).map((item) => (
      item.id === document.sourceRequestId ? { ...item, status: "issued" } : item
    ));
  }
  writeStore(store);
  res.json({ success: true, data: { document, history } });
});

app.post("/api/documents/:id/revise", (req, res) => {
  const store = readStore();
  const payload = req.body || {};
  const target = (store.documents || []).find((item) => item.id === req.params.id);
  if (!target) {
    return res.status(404).json({ success: false, error: "document not found" });
  }
  if (!payload.actor || !payload.effectiveDate || !payload.changeSummary) {
    return res.status(400).json({ success: false, error: "actor, effectiveDate, changeSummary are required" });
  }

  target.revision = nextRevision(target.revision);
  target.effectiveDate = String(payload.effectiveDate).trim();
  target.updatedAt = nowIso();
  target.status = "released";

  const history = makeHistory("revised", target, {
    actor: String(payload.actor).trim(),
    source: target.sourceLabel || "Revision Desk",
    changeSummary: String(payload.changeSummary).trim(),
  });

  store.releaseHistory.unshift(history);
  writeStore(store);
  res.json({ success: true, data: { document: target, history } });
});

app.get("*", (_req, res) => {
  res.sendFile(join(appRoot, "index.html"));
});

const customerStore = new Map();

function createCustomer(data) {
  const now = new Date().toISOString();
  const customer = {
    id: `cust_${randomUUID()}`,
    name: data.name || '',
    email: data.email || '',
    phone: data.phone || '',
    company: data.company || '',
    notes: data.notes || '',
    createdAt: now,
    updatedAt: now,
  };
  customerStore.set(customer.id, customer);
  return customer;
}

createCustomer({ name: '张三', email: 'zhangsan@example.com', phone: '13800138000', company: '示例公司A' });
createCustomer({ name: '李四', email: 'lisi@example.com', phone: '13900139000', company: '示例公司B' });
createCustomer({ name: '王五', email: 'wangwu@example.com', phone: '13700137000', company: '示例公司C' });

app.get('/api/customers', (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 20;
  const search = (req.query.search || '').toLowerCase();

  let result = Array.from(customerStore.values());
  if (search) {
    result = result.filter(c =>
      c.name.toLowerCase().includes(search) ||
      c.email.toLowerCase().includes(search)
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
  const customer = createCustomer(req.body);
  res.status(201).json({ success: true, data: customer });
});

app.put('/api/customers/:id', (req, res) => {
  const customer = customerStore.get(req.params.id);
  if (!customer) {
    return res.status(404).json({ success: false, error: 'Customer not found' });
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

app.listen(PORT, () => {
  console.log(`Document Release Hub listening on http://localhost:${PORT}`);
});

export default app;
