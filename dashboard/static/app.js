async function loadDashboard() {
  try {
    const response = await fetch('/api/dashboard');
    if (!response.ok) {
      throw new Error('No run data');
    }
    const data = await response.json();
    renderSummary(data.summary);
    renderRewardChart(data.reward_trace, data.confidence_trace);
    renderRepairs(data.repair_events);
    renderPlanner(data.planner_evolution);
    renderStability(data.stability_history, data.stability_metrics);
    renderFailures(data.repeated_failures);
    renderMemory(data.memory_summary);
    renderBehavior(data.behavior_predictions);
    renderSkills(data.skills, data.skill_repairs);
    renderDomViewer(data.dom_assets);
  } catch (error) {
    document.getElementById('summary-content').textContent =
      'No recent autonomy runs found. Run run_autonomy_demo.py first.';
    console.error(error);
  }
}

function renderSummary(summary = {}) {
  const status = summary.completed ? 'PASS' : 'CHECK';
  const html = `
    <div class="summary-row">
      <div><strong>Goal:</strong> ${summary.goal ?? 'n/a'}</div>
      <div><strong>Status:</strong> ${status} (${summary.reason ?? 'pending'})</div>
      <div><strong>Duration:</strong> ${(summary.duration_seconds ?? 0).toFixed(1)}s</div>
      <div><strong>Steps:</strong> ${summary.step_count ?? 'n/a'}</div>
    </div>`;
  document.getElementById('summary-content').innerHTML = html;
}

function renderRewardChart(rewardTrace = [], confidence = []) {
  const ctx = document.getElementById('rewardChart').getContext('2d');
  const labels = rewardTrace.map((entry, idx) => entry.step_id ?? `step_${idx + 1}`);
  const rewards = rewardTrace.map((entry) => entry.reward ?? 0);
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Reward',
          data: rewards,
          borderColor: '#38bdf8',
          tension: 0.3,
        },
        {
          label: 'Confidence',
          data: confidence,
          borderColor: '#fbbf24',
          borderDash: [6, 6],
          tension: 0.3,
        },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: '#cbd5f5' } } },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: '#1f2937' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: '#1f2937' } },
      },
    },
  });
}

function renderRepairs(repairs = []) {
  const list = document.getElementById('repairs');
  if (!repairs.length) {
    list.innerHTML = '<li>No repairs recorded.</li>';
    return;
  }
  list.innerHTML = repairs
    .map((event, idx) => `<li>${idx + 1}. ${event.patch?.type ?? 'repair'} — ${event.reason ?? 'auto'}</li>`)
    .join('');
}

function renderPlanner(events = []) {
  const pre = document.getElementById('planner-log');
  if (!events.length) {
    pre.textContent = 'No planner evolution records.';
    return;
  }
  pre.textContent = JSON.stringify(events, null, 2);
}

function renderStability(history = [], metrics = {}) {
  const ctx = document.getElementById('stabilityChart').getContext('2d');
  const labels = history.map((entry) => entry.timestamp ?? 'run');
  const rewards = history.map((entry) => entry.metrics?.avg_reward ?? 0);
  const confidence = history.map((entry) => entry.metrics?.avg_confidence ?? 0);
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Avg Reward',
          data: rewards,
          backgroundColor: 'rgba(56, 189, 248, 0.6)',
        },
        {
          label: 'Avg Confidence',
          data: confidence,
          backgroundColor: 'rgba(251, 191, 36, 0.6)',
        },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: '#cbd5f5' } } },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: '#1f2937' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: '#1f2937' } },
      },
    },
  });
}

function renderFailures(clusters = []) {
  const tbody = document.querySelector('#failure-table tbody');
  if (!clusters.length) {
    tbody.innerHTML = '<tr><td colspan="2">No repeated failures 🎉</td></tr>';
    return;
  }
  tbody.innerHTML = clusters
    .map((cluster) => `<tr><td>${cluster.reason}</td><td>${cluster.count}</td></tr>`)
    .join('');
}

function renderMemory(summary = {}) {
  const container = document.getElementById('memory-stats');
  if (!Object.keys(summary).length) {
    container.textContent = 'Memory summary unavailable.';
    return;
  }
  container.innerHTML = `
    <p>Entries: ${summary.entries ?? 0}</p>
    <p>Avg Confidence: ${(summary.avg_confidence ?? 0).toFixed(2)}</p>
    <p>Avg Difficulty: ${(summary.avg_difficulty ?? 0).toFixed(2)}</p>`;
}

function renderBehavior(predictions = []) {
  const list = document.getElementById('behavior-list');
  if (!predictions.length) {
    list.innerHTML = '<li>No predictions captured.</li>';
    return;
  }
  list.innerHTML = predictions
    .map((prediction) => {
      const diff = typeof prediction.difficulty === 'number' ? prediction.difficulty.toFixed(2) : 'n/a';
      return `<li>${prediction.step_id ?? prediction.fingerprint}: diff ${diff}, bias ${prediction.selector_bias ?? 'css'}, repair ${prediction.likely_repair ? 'likely' : 'no'}</li>`;
    })
    .join('');
}

function renderSkills(events = [], repairs = []) {
  const eventsContainer = document.getElementById('skill-events');
  const repairsContainer = document.getElementById('skill-repairs');
  if (!events.length) {
    eventsContainer.innerHTML = '<p>No skill plugins fired.</p>';
  } else {
    eventsContainer.innerHTML = `
      <h3>Triggered Skills</h3>
      <ul>${events
        .map(
          (event) =>
            `<li>${event.name}: ${event.subgoals} subgoals, ${event.repairs} repairs (step ${event.step_id ?? 'n/a'})</li>`
        )
        .join('')}</ul>`;
  }
  if (!repairs.length) {
    repairsContainer.innerHTML = '<p>No repair suggestions.</p>';
  } else {
    repairsContainer.innerHTML = `
      <h3>Repair Suggestions</h3>
      <ul>${repairs
        .map((entry) => `<li>${entry.action ?? entry.name ?? 'repair'} — ${entry.reason ?? 'no reason'}</li>`)
        .join('')}</ul>`;
  }
}

function renderDomViewer(assets = []) {
  const container = document.getElementById('dom-viewer');
  if (!assets.length) {
    container.textContent = 'No DOM snapshots captured for this run.';
    return;
  }
  container.innerHTML = assets
    .map((asset) => {
      const domPreview = asset.dom_snapshot ? asset.dom_snapshot.substring(0, 1200) : 'No DOM captured.';
      const screenshot = asset.screenshot_path
        ? `<img src="${asset.screenshot_path}" alt="${asset.step_id} screenshot" />`
        : '<em>No screenshot available.</em>';
      return `
        <div class="dom-card">
          <strong>${asset.step_id}</strong> — ${asset.action ?? 'action'}
          <pre>${domPreview}</pre>
          ${screenshot}
        </div>`;
    })
    .join('');
}

window.addEventListener('DOMContentLoaded', loadDashboard);
