async function fetchJson(url, options) {
  const response = await fetch(url, options);
  return response.json();
}

function renderCustomers(customers) {
  const list = document.getElementById('customer-list');
  list.innerHTML = customers.map((customer) => `<li data-id="${customer.id}">${customer.name} <span>${customer.email}</span></li>`).join('');
}

function renderDetail(customer) {
  const detail = document.getElementById('customer-detail');
  detail.textContent = `${customer.name} <${customer.email}>`;
}

async function boot() {
  const payload = await fetchJson('/api/customers');
  const customers = Array.isArray(payload.customers) ? payload.customers : [];
  renderCustomers(customers);
  if (customers[0]) renderDetail(customers[0]);
}

document.getElementById('customer-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const name = form.elements.name.value.trim();
  const email = form.elements.email.value.trim();
  const created = await fetchJson('/api/customers', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email }),
  });
  renderDetail(created);
  const payload = await fetchJson('/api/customers');
  renderCustomers(payload.customers || []);
  form.reset();
});

boot();
