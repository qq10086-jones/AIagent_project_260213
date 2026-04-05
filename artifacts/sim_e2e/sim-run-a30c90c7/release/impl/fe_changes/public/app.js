const API = '/api';
const CATEGORIES = { Food: '#ef4444', Transport: '#3b82f6', Housing: '#f59e0b', Entertainment: '#8b5cf6', Health: '#10b981', Other: '#6b7280' };
let barChart, pieChart;

async function fetchJson(url) { const r = await fetch(url); return r.json(); }

async function loadExpenses() {
  const expenses = await fetchJson(API + '/expenses');
  const list = document.getElementById('expenseList');
  list.innerHTML = expenses.slice(0, 20).map(e =>
    `<tr class="border-b hover:bg-gray-50">
      <td class="p-2">${e.date}</td><td>${e.description}</td>
      <td><span class="px-2 py-1 rounded text-xs text-white" style="background:${CATEGORIES[e.category] || '#6b7280'}">${e.category}</span></td>
      <td class="text-right font-mono">$${Number(e.amount).toFixed(2)}</td>
      <td><button onclick="deleteExpense(${e.id})" class="text-red-500 hover:text-red-700">x</button></td>
    </tr>`
  ).join('');
  document.getElementById('txCount').textContent = expenses.length;
  const thisMonth = new Date().toISOString().slice(0, 7);
  const monthExpenses = expenses.filter(e => e.date.startsWith(thisMonth));
  const total = monthExpenses.reduce((s, e) => s + Number(e.amount), 0);
  document.getElementById('monthTotal').textContent = '$' + total.toFixed(2);
  const byCat = {};
  monthExpenses.forEach(e => { byCat[e.category] = (byCat[e.category] || 0) + Number(e.amount); });
  const topCat = Object.entries(byCat).sort((a, b) => b[1] - a[1])[0];
  document.getElementById('topCategory').textContent = topCat ? topCat[0] : '—';
}

async function loadCharts() {
  const summary = await fetchJson(API + '/summary/monthly');
  const months = summary.map(s => s.month).reverse().slice(-6);
  const totals = months.map(m => summary.find(s => s.month === m)?.total || 0);
  const ctx1 = document.getElementById('barChart').getContext('2d');
  if (barChart) barChart.destroy();
  barChart = new Chart(ctx1, { type: 'bar', data: { labels: months, datasets: [{ label: 'Monthly Total', data: totals, backgroundColor: '#6366f1' }] }, options: { responsive: true } });
  const latest = summary[0];
  if (latest) {
    const cats = Object.keys(latest.by_category);
    const vals = Object.values(latest.by_category);
    const ctx2 = document.getElementById('pieChart').getContext('2d');
    if (pieChart) pieChart.destroy();
    pieChart = new Chart(ctx2, { type: 'pie', data: { labels: cats, datasets: [{ data: vals, backgroundColor: cats.map(c => CATEGORIES[c] || '#6b7280') }] }, options: { responsive: true } });
  }
}

async function deleteExpense(id) {
  await fetch(API + '/expenses/' + id, { method: 'DELETE' });
  loadExpenses(); loadCharts();
}

document.getElementById('expenseForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  await fetch(API + '/expenses', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount: +document.getElementById('amount').value, description: document.getElementById('description').value,
      category: document.getElementById('category').value, date: document.getElementById('date').value }) });
  e.target.reset(); loadExpenses(); loadCharts();
});

document.getElementById('exportBtn').addEventListener('click', () => {
  window.open(API + '/export/csv', '_blank');
});

loadExpenses(); loadCharts();
