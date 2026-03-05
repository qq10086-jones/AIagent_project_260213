const STORAGE_KEY = "nova_crm_pro_v1";
const SESSION_KEY = "nova_crm_session_v1";

const USERS = [
  { username: "admin", password: "admin123", role: "admin" },
  { username: "sales", password: "sales123", role: "sales" },
  { username: "viewer", password: "viewer123", role: "viewer" },
  { username: "tech", password: "tech123", role: "tech" },
];

const ROLE_PERMS = {
  admin: { create: true, edit: true, delete: true, changeStage: true },
  sales: { create: true, edit: true, delete: false, changeStage: true },
  viewer: { create: false, edit: false, delete: false, changeStage: false },
  tech: { create: false, edit: false, delete: false, changeStage: false },
};

const stageOrder = ["lead", "qualified", "proposal", "won", "lost"];

function buildInitialCustomers() {
  return [
    {
      id: crypto.randomUUID(),
      name: "Wei Lin",
      company: "Hikari Retail",
      email: "linwei@hikari.example",
      phone: "+81-90-1223-4455",
      status: "qualified",
      deal: 180000,
      notes: "Priority on loyalty module integration.",
      owner: "sales",
      updatedAt: new Date().toISOString(),
    },
    {
      id: crypto.randomUUID(),
      name: "Jun Wang",
      company: "Northstar Foods",
      email: "wjun@northstar.example",
      phone: "+86-139-2233-9988",
      status: "lead",
      deal: 95000,
      notes: "Sensitive to pricing; asks for clear support SLA.",
      owner: "sales",
      updatedAt: new Date().toISOString(),
    },
  ];
}

const state = {
  db: loadDb(),
  user: loadSession(),
  selectedId: null,
  editingId: null,
  query: "",
  statusFilter: "all",
};

const el = {
  authView: document.getElementById("authView"),
  appView: document.getElementById("appView"),
  loginForm: document.getElementById("loginForm"),
  loginUser: document.getElementById("loginUser"),
  loginPass: document.getElementById("loginPass"),
  authError: document.getElementById("authError"),

  sessionText: document.getElementById("sessionText"),
  logoutBtn: document.getElementById("logoutBtn"),
  newCustomerBtn: document.getElementById("newCustomerBtn"),

  customerList: document.getElementById("customerList"),
  customerCount: document.getElementById("customerCount"),
  searchInput: document.getElementById("searchInput"),
  statusFilter: document.getElementById("statusFilter"),
  detailEmpty: document.getElementById("detailEmpty"),
  detailView: document.getElementById("detailView"),
  activityList: document.getElementById("activityList"),
  activityCount: document.getElementById("activityCount"),
  insightList: document.getElementById("insightList"),

  complaintPanel: document.getElementById("complaintPanel"),
  complaintCount: document.getElementById("complaintCount"),
  complaintForm: document.getElementById("complaintForm"),
  complaintCustomer: document.getElementById("complaintCustomer"),
  complaintPriority: document.getElementById("complaintPriority"),
  complaintTitle: document.getElementById("complaintTitle"),
  complaintDesc: document.getElementById("complaintDesc"),
  complaintList: document.getElementById("complaintList"),

  modal: document.getElementById("modal"),
  modalTitle: document.getElementById("modalTitle"),
  form: document.getElementById("customerForm"),
  closeModalBtn: document.getElementById("closeModalBtn"),
  cancelBtn: document.getElementById("cancelBtn"),
  nameInput: document.getElementById("nameInput"),
  companyInput: document.getElementById("companyInput"),
  emailInput: document.getElementById("emailInput"),
  phoneInput: document.getElementById("phoneInput"),
  statusInput: document.getElementById("statusInput"),
  dealInput: document.getElementById("dealInput"),
  notesInput: document.getElementById("notesInput"),
};

function loadDb() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed.customers) && Array.isArray(parsed.activities)) {
        if (!Array.isArray(parsed.complaints)) parsed.complaints = [];
        return parsed;
      }
    } catch {}
  }
  const db = { customers: buildInitialCustomers(), activities: [], complaints: [] };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(db));
  return db;
}

function saveDb() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.db));
}

function loadSession() {
  const raw = localStorage.getItem(SESSION_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (parsed?.username && parsed?.role) return parsed;
  } catch {}
  return null;
}

function saveSession(user) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(user));
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

function perms() {
  return ROLE_PERMS[state.user?.role || "viewer"] || ROLE_PERMS.viewer;
}

function stageLabel(stage) {
  if (stage === "lead") return "Lead";
  if (stage === "qualified") return "Qualified";
  if (stage === "proposal") return "Proposal";
  if (stage === "won") return "Won";
  if (stage === "lost") return "Lost";
  return stage;
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatCurrency(amount) {
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "CNY",
    maximumFractionDigits: 0,
  }).format(amount || 0);
}

function logActivity(type, customerId, message) {
  state.db.activities.unshift({
    id: crypto.randomUUID(),
    type,
    customerId,
    message,
    actor: state.user?.username || "system",
    ts: new Date().toISOString(),
  });
  state.db.activities = state.db.activities.slice(0, 120);
  saveDb();
}

function visibleCustomers() {
  let rows = [...state.db.customers];
  const role = state.user?.role || "viewer";
  if (role === "sales") rows = rows.filter((c) => c.owner === "sales");
  if (state.statusFilter !== "all") rows = rows.filter((c) => c.status === state.statusFilter);
  if (state.query) {
    const q = state.query.toLowerCase();
    rows = rows.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        c.company.toLowerCase().includes(q) ||
        c.email.toLowerCase().includes(q)
    );
  }
  rows.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  return rows;
}

function renderAuthState() {
  if (!state.user) {
    el.authView.classList.remove("hidden");
    el.appView.classList.add("hidden");
    el.complaintPanel.classList.add("hidden");
    return;
  }
  el.authView.classList.add("hidden");
  el.appView.classList.remove("hidden");
  el.sessionText.textContent = `User: ${state.user.username} | Role: ${state.user.role}`;
  el.newCustomerBtn.disabled = !perms().create;
  if (state.user.role === "tech") el.complaintPanel.classList.remove("hidden");
  else el.complaintPanel.classList.add("hidden");
  renderAll();
}

function renderList() {
  const rows = visibleCustomers();
  el.customerCount.textContent = `${rows.length} records`;
  el.customerList.innerHTML = "";
  if (rows.length === 0) {
    el.customerList.innerHTML = '<li class="empty-state">No matching customer.</li>';
    return;
  }
  rows.forEach((customer) => {
    const li = document.createElement("li");
    li.className = `customer-item ${state.selectedId === customer.id ? "active" : ""}`;
    li.innerHTML = `
      <div class="row-main">
        <strong>${escapeHtml(customer.name)}</strong>
        <span class="tag ${customer.status}">${stageLabel(customer.status)}</span>
      </div>
      <div class="sub">${escapeHtml(customer.company)} | ${escapeHtml(customer.email)}</div>
      <div class="sub">Deal: ${formatCurrency(customer.deal)} | Owner: ${escapeHtml(customer.owner)}</div>
    `;
    li.addEventListener("click", () => {
      state.selectedId = customer.id;
      renderDetail();
    });
    el.customerList.appendChild(li);
  });
}

function detailActionButtons(customer) {
  const p = perms();
  const canEdit = p.edit;
  const canDelete = p.delete;
  const canStage = p.changeStage;
  const next = nextStage(customer.status);
  const prev = prevStage(customer.status);
  return `
    <div class="form-actions">
      <button class="btn btn-ghost" id="editBtn" ${canEdit ? "" : "disabled"}>Edit</button>
      <button class="btn btn-ghost" id="deleteBtn" ${canDelete ? "" : "disabled"}>Delete</button>
      <button class="btn btn-ghost" id="stageBackBtn" ${canStage && prev ? "" : "disabled"}>Stage -</button>
      <button class="btn btn-primary" id="stageNextBtn" ${canStage && next ? "" : "disabled"}>Stage +</button>
    </div>
  `;
}

function renderDetail() {
  const customer = state.db.customers.find((c) => c.id === state.selectedId);
  if (!customer) {
    el.detailEmpty.classList.remove("hidden");
    el.detailView.classList.add("hidden");
    el.detailView.innerHTML = "";
    return;
  }

  el.detailEmpty.classList.add("hidden");
  el.detailView.classList.remove("hidden");
  el.detailView.innerHTML = `
    <div class="detail-card">
      <div class="row-main">
        <strong>${escapeHtml(customer.name)}</strong>
        <span class="tag ${customer.status}">${stageLabel(customer.status)}</span>
      </div>
      <div class="detail-grid">
        <div class="detail-item"><span>Company</span><strong>${escapeHtml(customer.company)}</strong></div>
        <div class="detail-item"><span>Email</span><strong>${escapeHtml(customer.email)}</strong></div>
        <div class="detail-item"><span>Phone</span><strong>${escapeHtml(customer.phone)}</strong></div>
        <div class="detail-item"><span>Deal</span><strong>${formatCurrency(customer.deal)}</strong></div>
        <div class="detail-item"><span>Owner</span><strong>${escapeHtml(customer.owner)}</strong></div>
        <div class="detail-item"><span>Updated</span><strong>${new Date(customer.updatedAt).toLocaleString()}</strong></div>
      </div>
    </div>
    <div class="detail-card">
      <div class="detail-item"><span>Notes</span><strong>${escapeHtml(customer.notes || "-")}</strong></div>
    </div>
    ${detailActionButtons(customer)}
  `;

  const editBtn = document.getElementById("editBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const stageNextBtn = document.getElementById("stageNextBtn");
  const stageBackBtn = document.getElementById("stageBackBtn");
  if (editBtn) editBtn.addEventListener("click", () => openModal(customer.id));
  if (deleteBtn) deleteBtn.addEventListener("click", () => removeCustomer(customer.id));
  if (stageNextBtn) stageNextBtn.addEventListener("click", () => moveStage(customer.id, +1));
  if (stageBackBtn) stageBackBtn.addEventListener("click", () => moveStage(customer.id, -1));
}

function renderActivities() {
  const role = state.user?.role || "viewer";
  const activities = role === "admin"
    ? state.db.activities
    : state.db.activities.filter((a) => a.actor === state.user.username || a.actor === "system");
  el.activityCount.textContent = `${activities.length} events`;
  el.activityList.innerHTML = "";
  if (activities.length === 0) {
    el.activityList.innerHTML = '<li class="empty-state">No activities yet.</li>';
    return;
  }
  activities.slice(0, 20).forEach((a) => {
    const customer = state.db.customers.find((c) => c.id === a.customerId);
    const li = document.createElement("li");
    li.className = "activity-item";
    li.innerHTML = `
      <strong>${escapeHtml(a.type)} | ${escapeHtml(a.actor)}</strong>
      <div class="sub">${escapeHtml(a.message)}</div>
      <div class="sub">${customer ? escapeHtml(customer.name) : "N/A"} | ${new Date(a.ts).toLocaleString()}</div>
    `;
    el.activityList.appendChild(li);
  });
}

function buildInsights() {
  const byStage = { lead: 0, qualified: 0, proposal: 0, won: 0, lost: 0 };
  state.db.customers.forEach((c) => { byStage[c.status] = (byStage[c.status] || 0) + 1; });
  const total = state.db.customers.length || 1;
  const winRate = Math.round((byStage.won / total) * 100);
  const proposalRisk = byStage.proposal > byStage.won + 1;
  const noteCoverage = Math.round(
    (state.db.customers.filter((c) => String(c.notes || "").trim().length > 8).length / total) * 100
  );

  const list = [];
  list.push({
    title: "Pipeline Mix",
    detail: `Lead:${byStage.lead} Qualified:${byStage.qualified} Proposal:${byStage.proposal} Won:${byStage.won} Lost:${byStage.lost}`,
  });
  list.push({
    title: "Current Win Rate",
    detail: `${winRate}% based on current local dataset.`,
  });
  if (proposalRisk) {
    list.push({
      title: "Risk Alert",
      detail: "Too many proposal-stage deals are not converting. Prioritize follow-up cadence.",
    });
  }
  if (noteCoverage < 70) {
    list.push({
      title: "Data Quality",
      detail: `Only ${noteCoverage}% customers have meaningful notes. Add context for better follow-up.`,
    });
  } else {
    list.push({
      title: "Data Quality",
      detail: `Notes coverage is healthy at ${noteCoverage}%.`,
    });
  }
  const recentCreates = state.db.activities.filter((a) => a.type === "customer_created").length;
  if (recentCreates >= 3) {
    list.push({
      title: "Learning Suggestion",
      detail: "New lead volume is high. Consider adding auto-assignment rules by industry.",
    });
  }
  return list;
}

function renderInsights() {
  const items = buildInsights();
  el.insightList.innerHTML = "";
  items.forEach((i) => {
    const li = document.createElement("li");
    li.className = "insight-item";
    li.innerHTML = `<strong>${escapeHtml(i.title)}</strong><div class="sub">${escapeHtml(i.detail)}</div>`;
    el.insightList.appendChild(li);
  });
}

function renderComplaintOptions() {
  if (!el.complaintCustomer) return;
  el.complaintCustomer.innerHTML = "";
  const rows = [...state.db.customers].sort((a, b) => a.name.localeCompare(b.name));
  rows.forEach((c) => {
    const option = document.createElement("option");
    option.value = c.id;
    option.textContent = `${c.name} - ${c.company}`;
    el.complaintCustomer.appendChild(option);
  });
}

function renderComplaints() {
  if (state.user?.role !== "tech") return;
  const tickets = [...(state.db.complaints || [])].sort(
    (a, b) => new Date(b.updatedAt) - new Date(a.updatedAt)
  );
  el.complaintCount.textContent = `${tickets.length} tickets`;
  el.complaintList.innerHTML = "";
  if (tickets.length === 0) {
    el.complaintList.innerHTML = '<li class="empty-state">No complaints yet.</li>';
    return;
  }
  tickets.forEach((t) => {
    const customer = state.db.customers.find((c) => c.id === t.customerId);
    const li = document.createElement("li");
    li.className = "activity-item";
    li.innerHTML = `
      <div class="row-main">
        <strong>${escapeHtml(t.title)}</strong>
        <span class="ticket-status ${t.status}">${escapeHtml(t.status)}</span>
      </div>
      <div class="sub">Priority: ${escapeHtml(t.priority)} | Customer: ${escapeHtml(customer?.name || "N/A")}</div>
      <div class="sub">${escapeHtml(t.description)}</div>
      <div class="form-actions">
        <button class="btn btn-ghost" data-ticket="${t.id}" data-action="toggle">
          ${t.status === "open" ? "Mark Resolved" : "Reopen"}
        </button>
      </div>
    `;
    el.complaintList.appendChild(li);
  });
}

function handleComplaintSubmit(evt) {
  evt.preventDefault();
  if (state.user?.role !== "tech") return;
  const customerId = el.complaintCustomer.value;
  const title = el.complaintTitle.value.trim();
  const description = el.complaintDesc.value.trim();
  const priority = el.complaintPriority.value;
  if (!customerId || !title || !description) return;
  state.db.complaints.unshift({
    id: crypto.randomUUID(),
    customerId,
    title,
    description,
    priority,
    status: "open",
    createdBy: state.user.username,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });
  logActivity("complaint_created", customerId, `Complaint ticket created: ${title}`);
  saveDb();
  el.complaintForm.reset();
  renderComplaintOptions();
  renderComplaints();
}

function toggleComplaintStatus(ticketId) {
  if (state.user?.role !== "tech") return;
  const ticket = state.db.complaints.find((x) => x.id === ticketId);
  if (!ticket) return;
  ticket.status = ticket.status === "open" ? "resolved" : "open";
  ticket.updatedAt = new Date().toISOString();
  logActivity(
    "complaint_status_changed",
    ticket.customerId,
    `Complaint "${ticket.title}" -> ${ticket.status}`
  );
  saveDb();
  renderComplaints();
}

function renderAll() {
  renderList();
  renderDetail();
  renderActivities();
  renderInsights();
  renderComplaintOptions();
  renderComplaints();
}

function resetForm() {
  el.form.reset();
  el.statusInput.value = "lead";
  el.dealInput.value = "0";
}

function openModal(customerId = null) {
  if (!perms().create && !perms().edit) return;
  state.editingId = customerId;
  resetForm();
  if (customerId) {
    const c = state.db.customers.find((x) => x.id === customerId);
    if (!c) return;
    el.modalTitle.textContent = "Edit Customer";
    el.nameInput.value = c.name;
    el.companyInput.value = c.company;
    el.emailInput.value = c.email;
    el.phoneInput.value = c.phone;
    el.statusInput.value = c.status;
    el.dealInput.value = c.deal;
    el.notesInput.value = c.notes || "";
  } else {
    el.modalTitle.textContent = "New Customer";
  }
  el.modal.classList.remove("hidden");
}

function closeModal() {
  el.modal.classList.add("hidden");
  state.editingId = null;
}

function saveCustomer(evt) {
  evt.preventDefault();
  const canCreate = perms().create;
  const canEdit = perms().edit;
  if (!canCreate && !canEdit) return;

  const payload = {
    name: el.nameInput.value.trim(),
    company: el.companyInput.value.trim(),
    email: el.emailInput.value.trim(),
    phone: el.phoneInput.value.trim(),
    status: el.statusInput.value,
    deal: Number(el.dealInput.value || 0),
    notes: el.notesInput.value.trim(),
    updatedAt: new Date().toISOString(),
  };
  if (!payload.name || !payload.company || !payload.email) return;

  if (state.editingId) {
    if (!canEdit) return;
    state.db.customers = state.db.customers.map((c) =>
      c.id === state.editingId ? { ...c, ...payload } : c
    );
    logActivity("customer_updated", state.editingId, `Updated ${payload.name}`);
  } else {
    if (!canCreate) return;
    const created = {
      id: crypto.randomUUID(),
      owner: state.user.username,
      ...payload,
    };
    state.db.customers.push(created);
    state.selectedId = created.id;
    logActivity("customer_created", created.id, `Created ${created.name}`);
  }
  saveDb();
  closeModal();
  renderAll();
}

function removeCustomer(id) {
  if (!perms().delete) return;
  const customer = state.db.customers.find((c) => c.id === id);
  if (!customer) return;
  const ok = confirm(`Delete customer "${customer.name}"?`);
  if (!ok) return;
  state.db.customers = state.db.customers.filter((c) => c.id !== id);
  if (state.selectedId === id) state.selectedId = null;
  logActivity("customer_deleted", id, `Deleted ${customer.name}`);
  saveDb();
  renderAll();
}

function nextStage(stage) {
  const idx = stageOrder.indexOf(stage);
  if (idx < 0 || idx >= stageOrder.length - 1) return null;
  return stageOrder[idx + 1];
}

function prevStage(stage) {
  const idx = stageOrder.indexOf(stage);
  if (idx <= 0) return null;
  return stageOrder[idx - 1];
}

function moveStage(customerId, delta) {
  if (!perms().changeStage) return;
  const customer = state.db.customers.find((c) => c.id === customerId);
  if (!customer) return;
  const idx = stageOrder.indexOf(customer.status);
  const nextIdx = idx + delta;
  if (nextIdx < 0 || nextIdx >= stageOrder.length) return;
  customer.status = stageOrder[nextIdx];
  customer.updatedAt = new Date().toISOString();
  logActivity("stage_changed", customerId, `Moved ${customer.name} to ${stageLabel(customer.status)}`);
  saveDb();
  renderAll();
}

function handleLogin(evt) {
  evt.preventDefault();
  const username = el.loginUser.value.trim();
  const password = el.loginPass.value;
  const found = USERS.find((u) => u.username === username && u.password === password);
  if (!found) {
    el.authError.textContent = "Invalid credentials.";
    el.authError.classList.remove("hidden");
    return;
  }
  state.user = { username: found.username, role: found.role };
  saveSession(state.user);
  el.authError.classList.add("hidden");
  renderAuthState();
}

function handleLogout() {
  state.user = null;
  clearSession();
  state.selectedId = null;
  renderAuthState();
}

function bindEvents() {
  el.loginForm.addEventListener("submit", handleLogin);
  el.logoutBtn.addEventListener("click", handleLogout);
  el.searchInput.addEventListener("input", (e) => {
    state.query = e.target.value.trim();
    renderList();
  });
  el.statusFilter.addEventListener("change", (e) => {
    state.statusFilter = e.target.value;
    renderList();
  });
  el.newCustomerBtn.addEventListener("click", () => openModal(null));
  el.closeModalBtn.addEventListener("click", closeModal);
  el.cancelBtn.addEventListener("click", closeModal);
  el.form.addEventListener("submit", saveCustomer);
  el.modal.addEventListener("click", (e) => {
    if (e.target === el.modal) closeModal();
  });
  if (el.complaintForm) {
    el.complaintForm.addEventListener("submit", handleComplaintSubmit);
  }
  if (el.complaintList) {
    el.complaintList.addEventListener("click", (e) => {
      const target = e.target;
      if (!(target instanceof HTMLElement)) return;
      const action = target.getAttribute("data-action");
      const ticketId = target.getAttribute("data-ticket");
      if (action === "toggle" && ticketId) toggleComplaintStatus(ticketId);
    });
  }
}

bindEvents();
renderAuthState();
