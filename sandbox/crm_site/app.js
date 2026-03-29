const state = {
  store: null,
  selectedRequestId: null,
  selectedDocumentId: null,
  activeTemplateId: null,
};

const els = {
  metrics: document.getElementById("metrics"),
  inbox: document.getElementById("intakeInbox"),
  ledger: document.getElementById("documentLedger"),
  history: document.getElementById("releaseTimeline"),
  templates: document.getElementById("templateCatalog"),
  roadmap: document.getElementById("roadmap"),
  generatorForm: document.getElementById("generatorForm"),
  intakeForm: document.getElementById("intakeForm"),
  revisionForm: document.getElementById("revisionForm"),
  selectedRequest: document.getElementById("selectedRequest"),
  releaseFeedback: document.getElementById("releaseFeedback"),
  revisionFeedback: document.getElementById("revisionFeedback"),
  systemNotice: document.getElementById("systemNotice"),
};

function fmtDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function text(value) {
  return String(value ?? "").replace(/[&<>"]/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
  }[char]));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || !data.success) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data.data;
}

function getRequestById(id) {
  return (state.store?.intakeRequests || []).find((item) => item.id === id) || null;
}

function getDocumentById(id) {
  return (state.store?.documents || []).find((item) => item.id === id) || null;
}

function getTemplateById(id) {
  return (state.store?.templates || []).find((item) => item.id === id) || null;
}

function updateSystemNotice(message, tone = "info") {
  if (!els.systemNotice) return;
  els.systemNotice.textContent = message;
  els.systemNotice.className = `system-notice ${tone}`;
}

function renderMetrics() {
  const totalDocs = state.store.documents.length;
  const releasedDocs = state.store.documents.filter((doc) => doc.status === "released").length;
  const openRequests = state.store.intakeRequests.filter((item) => item.status !== "issued").length;
  const historyCount = state.store.releaseHistory.length;
  const cards = [
    { label: "模板", value: state.store.templates.length, note: "Excel / 文档模板台账" },
    { label: "待处理请求", value: openRequests, note: "Discord intake inbox" },
    { label: "在册文档", value: totalDocs, note: `${releasedDocs} 份已发行` },
    { label: "发行记录", value: historyCount, note: "完整可追溯历史" },
  ];
  els.metrics.innerHTML = cards.map((card) => `
    <article class="metric-card">
      <span class="metric-label">${card.label}</span>
      <strong class="metric-value">${card.value}</strong>
      <span class="metric-note">${card.note}</span>
    </article>
  `).join("");
}

function renderTemplates() {
  els.templates.innerHTML = state.store.templates.map((template) => `
    <article class="template-card ${template.id === state.activeTemplateId ? "active" : ""}" data-template-id="${template.id}">
      <div class="template-card-top">
        <div>
          <strong>${text(template.name)}</strong>
          <p>${text(template.codePrefix)} / ${text(template.category)}</p>
        </div>
        <span class="template-badge">${text(template.defaultDepartment)}</span>
      </div>
      <p>${text(template.description)}</p>
      <div class="template-meta">
        <span>字段: ${template.fields.length}</span>
        <span>下一个编号: ${text(template.nextNumberPreview)}</span>
      </div>
    </article>
  `).join("");
}

function renderInbox() {
  const rows = [...state.store.intakeRequests].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  els.inbox.innerHTML = rows.map((item) => `
    <article class="inbox-card ${item.id === state.selectedRequestId ? "active" : ""}" data-request-id="${item.id}">
      <div class="inbox-head">
        <span class="template-badge">${text(item.source.toUpperCase())}</span>
        <span class="status-pill ${text(item.status)}">${text(item.status)}</span>
      </div>
      <strong>${text(item.summary)}</strong>
      <p>${text(item.content)}</p>
      <div class="inbox-meta">
        <span>${text(item.requester)}</span>
        <span>${text(item.channel)}</span>
        <span>${fmtDate(item.createdAt)}</span>
      </div>
    </article>
  `).join("");
}

function renderSelectedRequest() {
  const request = getRequestById(state.selectedRequestId);
  if (!request) {
    els.selectedRequest.innerHTML = `
      <div class="empty-card">
        <strong>等待选择 intake 请求</strong>
        <p>从左侧 Discord inbox 选一条请求，表单会自动带入模板建议和主题。</p>
      </div>
    `;
    return;
  }
  els.selectedRequest.innerHTML = `
    <div class="selected-request-card">
      <div class="selected-request-top">
        <strong>${text(request.summary)}</strong>
        <span class="status-pill ${text(request.status)}">${text(request.status)}</span>
      </div>
      <p>${text(request.content)}</p>
      <div class="selected-request-grid">
        <span>来源人: ${text(request.requester)}</span>
        <span>频道: ${text(request.channel)}</span>
        <span>建议模板: ${text(request.suggestedTemplateId || "-")}</span>
        <span>创建时间: ${fmtDate(request.createdAt)}</span>
      </div>
    </div>
  `;
}

function renderLedger() {
  const docs = [...state.store.documents].sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  els.ledger.innerHTML = docs.map((doc) => `
    <article class="ledger-card ${doc.id === state.selectedDocumentId ? "active" : ""}" data-document-id="${doc.id}">
      <div class="ledger-head">
        <div>
          <strong>${text(doc.docNumber)}</strong>
          <span class="revision-chip">Rev ${text(doc.revision)}</span>
        </div>
        <span class="status-pill ${text(doc.status)}">${text(doc.status)}</span>
      </div>
      <h3>${text(doc.title)}</h3>
      <p>${text(doc.templateName)} / ${text(doc.department)} / ${text(doc.owner)}</p>
      <div class="ledger-meta">
        <span>生效: ${text(doc.effectiveDate)}</span>
        <span>分发: ${text((doc.distribution || []).join(", "))}</span>
      </div>
      <div class="ledger-footer">
        <span>来源: ${text(doc.sourceLabel || "manual")}</span>
        <span>${fmtDate(doc.updatedAt)}</span>
      </div>
    </article>
  `).join("");
}

function renderHistory() {
  const rows = [...state.store.releaseHistory].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  els.history.innerHTML = rows.map((item) => `
    <article class="timeline-item">
      <div class="timeline-top">
        <strong>${text(item.docNumber)} / Rev ${text(item.revision)}</strong>
        <span>${fmtDate(item.timestamp)}</span>
      </div>
      <p>${text(item.actionLabel)} · ${text(item.changeSummary)}</p>
      <div class="timeline-meta">
        <span>操作者: ${text(item.actor)}</span>
        <span>分发: ${text((item.distribution || []).join(", "))}</span>
        <span>来源: ${text(item.source)}</span>
      </div>
    </article>
  `).join("");
}

function renderRoadmap() {
  els.roadmap.innerHTML = state.store.roadmap.map((item) => `
    <article class="roadmap-card">
      <div class="roadmap-top">
        <strong>${text(item.name)}</strong>
        <span class="status-pill ${text(item.status)}">${text(item.status)}</span>
      </div>
      <p>${text(item.description)}</p>
      <span class="roadmap-note">${text(item.nextStep)}</span>
    </article>
  `).join("");
}

function syncGeneratorForm() {
  const templateSelect = document.getElementById("templateId");
  const request = getRequestById(state.selectedRequestId);
  const doc = getDocumentById(state.selectedDocumentId);
  templateSelect.innerHTML = state.store.templates.map((template) => `
    <option value="${template.id}" ${template.id === state.activeTemplateId ? "selected" : ""}>
      ${text(template.name)} (${text(template.codePrefix)})
    </option>
  `).join("");

  const template = getTemplateById(state.activeTemplateId || request?.suggestedTemplateId || state.store.templates[0]?.id);
  if (!template) return;
  state.activeTemplateId = template.id;
  const form = els.generatorForm;
  if (!form.dataset.seeded || form.dataset.seeded !== `${state.selectedRequestId || ""}:${state.selectedDocumentId || ""}:${template.id}`) {
    form.templateId.value = template.id;
    form.title.value = doc?.title || request?.summary || "";
    form.department.value = doc?.department || template.defaultDepartment || "";
    form.owner.value = doc?.owner || request?.requester || "Document Control";
    form.effectiveDate.value = doc?.effectiveDate || new Date().toISOString().slice(0, 10);
    form.distribution.value = (doc?.distribution || template.defaultDistribution || []).join(", ");
    form.excelTemplateRef.value = doc?.excelTemplateRef || template.excelTemplateRef || "";
    form.changeSummary.value = doc ? "" : (request ? `根据 Discord 请求创建首版文件：${request.summary}` : "创建首版文件");
    form.payloadSummary.value = doc?.payloadSummary || request?.content || "";
    form.dataset.seeded = `${state.selectedRequestId || ""}:${state.selectedDocumentId || ""}:${template.id}`;
  }
  document.getElementById("templateGuidance").innerHTML = `
    <strong>${text(template.name)}</strong>
    <p>${text(template.description)}</p>
    <div class="guidance-row">
      <span>编号前缀: ${text(template.codePrefix)}</span>
      <span>模板文件: ${text(template.excelTemplateRef)}</span>
    </div>
    <div class="guidance-row">
      <span>默认部门: ${text(template.defaultDepartment)}</span>
      <span>下一个编号: ${text(template.nextNumberPreview)}</span>
    </div>
  `;
}

function syncRevisionForm() {
  const doc = getDocumentById(state.selectedDocumentId);
  const label = document.getElementById("revisionTarget");
  if (!doc) {
    els.revisionForm.classList.add("is-disabled");
    els.revisionForm.querySelectorAll("input, textarea, button").forEach((node) => { node.disabled = true; });
    label.textContent = "先从文档台账中选择一份文件";
    return;
  }
  els.revisionForm.classList.remove("is-disabled");
  els.revisionForm.querySelectorAll("input, textarea, button").forEach((node) => { node.disabled = false; });
  els.revisionForm.documentId.value = doc.id;
  els.revisionForm.actor.value = doc.owner;
  els.revisionForm.effectiveDate.value = new Date().toISOString().slice(0, 10);
  label.textContent = `${doc.docNumber} / 当前 Rev ${doc.revision}`;
}

function renderAll() {
  renderMetrics();
  renderTemplates();
  renderInbox();
  renderSelectedRequest();
  renderLedger();
  renderHistory();
  renderRoadmap();
  syncGeneratorForm();
  syncRevisionForm();
}

async function refresh() {
  state.store = await api("/api/bootstrap");
  if (!state.activeTemplateId) state.activeTemplateId = state.store.templates[0]?.id || null;
  if (!state.selectedRequestId) state.selectedRequestId = state.store.intakeRequests[0]?.id || null;
  renderAll();
}

async function handleIntakeSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const payload = {
    requester: form.requester.value.trim() || "discord-user",
    channel: form.channel.value.trim() || "#doc-control",
    content: form.content.value.trim(),
    suggestedTemplateId: form.suggestedTemplateId.value || null,
  };
  if (!payload.content) {
    updateSystemNotice("请先输入 Discord 请求内容。", "error");
    return;
  }
  await api("/api/intake/discord", { method: "POST", body: JSON.stringify(payload) });
  form.reset();
  updateSystemNotice("新的 Discord intake 请求已写入待办队列。", "success");
  await refresh();
}

async function handleGeneratorSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const payload = {
    templateId: form.templateId.value,
    title: form.title.value.trim(),
    department: form.department.value.trim(),
    owner: form.owner.value.trim(),
    effectiveDate: form.effectiveDate.value,
    distribution: form.distribution.value.split(",").map((item) => item.trim()).filter(Boolean),
    excelTemplateRef: form.excelTemplateRef.value.trim(),
    payloadSummary: form.payloadSummary.value.trim(),
    changeSummary: form.changeSummary.value.trim(),
    sourceRequestId: state.selectedRequestId || null,
  };
  if (!payload.templateId || !payload.title || !payload.department || !payload.owner || !payload.effectiveDate) {
    updateSystemNotice("发行前请补齐模板、标题、部门、责任人和生效日期。", "error");
    return;
  }
  const result = await api("/api/documents/generate", { method: "POST", body: JSON.stringify(payload) });
  state.selectedDocumentId = result.document.id;
  els.releaseFeedback.textContent = `发行成功：${result.document.docNumber} / Rev ${result.document.revision}`;
  els.releaseFeedback.className = "feedback success";
  updateSystemNotice(`已生成并发行 ${result.document.docNumber} / Rev ${result.document.revision}。`, "success");
  await refresh();
}

async function handleRevisionSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const payload = {
    actor: form.actor.value.trim(),
    effectiveDate: form.effectiveDate.value,
    changeSummary: form.changeSummary.value.trim(),
  };
  if (!form.documentId.value || !payload.actor || !payload.effectiveDate || !payload.changeSummary) {
    updateSystemNotice("修订前请填写操作者、生效日期和变更摘要。", "error");
    return;
  }
  const result = await api(`/api/documents/${form.documentId.value}/revise`, { method: "POST", body: JSON.stringify(payload) });
  state.selectedDocumentId = result.document.id;
  els.revisionFeedback.textContent = `已创建新修订：${result.document.docNumber} / Rev ${result.document.revision}`;
  els.revisionFeedback.className = "feedback success";
  updateSystemNotice(`文档 ${result.document.docNumber} 已修订到 Rev ${result.document.revision}。`, "success");
  await refresh();
}

function bindEvents() {
  els.templates.addEventListener("click", (event) => {
    const card = event.target.closest("[data-template-id]");
    if (!card) return;
    state.activeTemplateId = card.dataset.templateId;
    els.generatorForm.dataset.seeded = "";
    renderAll();
  });

  els.inbox.addEventListener("click", (event) => {
    const card = event.target.closest("[data-request-id]");
    if (!card) return;
    state.selectedRequestId = card.dataset.requestId;
    state.selectedDocumentId = null;
    els.generatorForm.dataset.seeded = "";
    renderAll();
  });

  els.ledger.addEventListener("click", (event) => {
    const card = event.target.closest("[data-document-id]");
    if (!card) return;
    state.selectedDocumentId = card.dataset.documentId;
    els.generatorForm.dataset.seeded = "";
    renderAll();
  });

  els.intakeForm.addEventListener("submit", (event) => {
    handleIntakeSubmit(event).catch((error) => updateSystemNotice(error.message || "写入 intake 失败。", "error"));
  });
  els.generatorForm.addEventListener("submit", (event) => {
    handleGeneratorSubmit(event).catch((error) => updateSystemNotice(error.message || "发行失败。", "error"));
  });
  els.revisionForm.addEventListener("submit", (event) => {
    handleRevisionSubmit(event).catch((error) => updateSystemNotice(error.message || "修订失败。", "error"));
  });
  document.getElementById("templateId").addEventListener("change", (event) => {
    state.activeTemplateId = event.target.value;
    els.generatorForm.dataset.seeded = "";
    renderAll();
  });
}

async function init() {
  bindEvents();
  await refresh();
  updateSystemNotice("Discord intake 已接通到本地发行工作台。你可以直接投喂请求，然后一键建档并登记发行历史。", "info");
}

init().catch((error) => updateSystemNotice(error.message || "系统初始化失败。", "error"));

export { init };
