"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const pct = (value) => (value == null || Number.isNaN(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(1)}%`);
const num = (value, digits = 2) => (value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits));
const label = (value) => String(value ?? "unknown").replaceAll("_", " ");

const state = {
  data: null,
  tasks: new Map(),
  datasets: [],
  heroMode: "recap",
  heroIndex: 0,
  driftTask: "",
  inspect: { task: "", baseline: "" },
  run: { task: "", baseline: "" },
  architecture: "recall",
};

function baselines() {
  return Object.keys(state.data?.summary ?? {});
}

function rowsFor(baseline) {
  return (state.data?.per_task ?? []).filter((row) => row.baseline === baseline);
}

function rowFor(taskId, baseline) {
  return (state.data?.per_task ?? []).find(
    (row) => row.task_id === taskId && row.baseline === baseline
  );
}

function selectableStats(baseline) {
  const selectableIds = new Set(
    (state.data?.tasks ?? [])
      .filter((task) => task.should_select === true)
      .map((task) => task.task_id)
  );
  const rows = rowsFor(baseline).filter((row) => selectableIds.has(row.task_id));
  return {
    n: selectableIds.size,
    success: rows.filter((row) => row.task_success === true).length,
  };
}

function bestBaseline() {
  const entries = Object.entries(state.data?.summary ?? {});
  if (!entries.length) return null;
  return entries.sort((a, b) => {
    const success = (b[1].task_success_rate ?? 0) - (a[1].task_success_rate ?? 0);
    if (Math.abs(success) > 1e-9) return success;
    return (a[1].harmful_transfer_rate ?? 1) - (b[1].harmful_transfer_rate ?? 1);
  })[0][0];
}

function silentCandidates() {
  return (state.data?.per_task ?? []).filter((row) => {
    const task = state.tasks.get(row.task_id);
    return task?.silent_drift || String(task?.scenario_id ?? "").includes("silent");
  });
}

function heroTask() {
  const candidates = silentCandidates();
  if (!candidates.length) return null;
  const grouped = new Map();
  for (const row of candidates) {
    if (!grouped.has(row.task_id)) grouped.set(row.task_id, []);
    grouped.get(row.task_id).push(row);
  }
  const relevanceBaselines = [
    "bm25",
    "hybrid_rrf",
    "vector",
    "hybrid_rerank",
    "hybrid_rrf_version_filter",
  ];
  const ranked = Array.from(grouped.entries())
    .map(([taskId, rows]) => {
      const naive =
        relevanceBaselines
          .map((name) => rows.find((row) => row.baseline === name))
          .find(Boolean) ?? null;
      const recap =
        rows.find((row) => row.baseline === "outcome_aware") ??
        rows.find((row) => row.baseline === bestBaseline()) ??
        null;
      const score =
        (recap?.task_success === true ? 1 : 0) -
        (naive?.task_success === true ? 1 : 0) +
        (naive?.harmful_transfer === true ? 1 : 0);
      return { taskId, rows, naive, recap, score };
    })
    .sort((a, b) => b.score - a.score || a.taskId.localeCompare(b.taskId));
  const chosen = ranked[state.heroIndex % Math.max(ranked.length, 1)] ?? ranked[0];
  if (!chosen) return null;
  return {
    task: state.tasks.get(chosen.taskId),
    naive: chosen.naive,
    recap: chosen.recap,
    hasContrast: Boolean(chosen.naive && chosen.recap),
  };
}

function populateDatasetSelect() {
  const select = $("#dataset-select");
  if (!select) return;
  const datasets = state.datasets;
  select.innerHTML = datasets.length
    ? datasets
        .map((meta, index) => {
          const mode = meta.agent_mode === "deterministic" ? "deterministic" : "live model";
          return `<option value="${index}">${meta.suite ?? "?"} · ${meta.split ?? "?"} · ${mode} · ${meta.model || meta.agent || "agent"}</option>`;
        })
        .join("")
    : `<option value="0">${state.data?.meta?.suite ?? "v3"} · ${state.data?.meta?.split ?? "?"} · recorded</option>`;
  select.value = "0";
}

async function loadDataset(index) {
  const source = `replay/dataset-${index}.json`;
  const response = await fetch(source, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${source}`);
  state.data = await response.json();
  state.heroIndex = 0;
  state.driftTask = "";
  state.inspect = { task: "", baseline: "" };
  state.run = { task: "", baseline: "" };
  renderAll();
}

function setProvenance() {
  const meta = state.data?.meta ?? {};
  const inference = meta.inference ?? {};
  const mode =
    meta.agent_mode === "deterministic"
      ? "deterministic replay (not live-model evidence)"
      : `live-model replay (${meta.agent_mode || "agent"})`;
  const pieces = [
    "REPLAY OF RECORDED RUNS",
    mode,
    `suite ${meta.suite ?? "?"}`,
    `split ${meta.split ?? "?"}`,
    `agent ${meta.agent ?? "?"}`,
    `model ${meta.model || "none"}`,
    `embedder ${meta.embedder || "unknown"}${meta.embedder_provider ? ` (${meta.embedder_provider})` : ""}`,
    `reranker ${meta.reranker || "default"}`,
    `generated ${meta.generated_at || "?"}`,
    `dataset ${meta.dataset_hash || "?"}`,
    `timeout ${inference.read_timeout_seconds ?? inference.timeout_seconds ?? "n/a"}s`,
  ];
  $("#provenance").textContent = pieces.join(" · ");
  $("#footer-source").textContent = meta.source || "report";
  $("#limitations-source").textContent = meta.note || "";
}

function renderHero() {
  const hero = heroTask();
  if (!hero) {
    $("#hero-naive-result").textContent = "No silent scenario";
    $("#hero-recap-result").textContent = "No silent scenario";
    return;
  }
  const { task, naive, recap } = hero;
  const active = state.heroMode === "recap" ? recap : naive;
  if (!active) {
    $("#hero-explain").textContent =
      "This replay artifact does not contain both a relevance-only baseline and the outcome-aware policy for a silent-drift task. Pick a report with the full baseline ladder.";
    $("#hero-naive-result").textContent = "Not in artifact";
    $("#hero-recap-result").textContent = "Not in artifact";
    return;
  }
  $("#hero-task").textContent = task?.task_id ?? active.task_id;
  const naiveSuccess = naive?.task_success === true;
  $("#hero-naive-result").textContent = naive
    ? naiveSuccess
      ? "Looks successful"
      : "Wrong / incomplete"
    : "Not in artifact";
  $("#hero-naive-detail").textContent = naive
    ? `${label(naive.baseline)} · ${pct(naive.task_success ? 1 : 0)} success · ${naive.tool_calls ?? "?"} calls`
    : "this report has no relevance-only baseline";
  const recapSuccess = recap?.task_success === true;
  $("#hero-recap-result").textContent = recap
    ? recapSuccess
      ? "Verified success"
      : "Safe failure"
    : "Not in artifact";
  $("#hero-recap-detail").textContent = recap
    ? `${label(recap.baseline)} · selected ${recap.selected_memory_ids?.length ?? 0} · ${recap.tool_calls ?? "?"} calls`
    : "this report has no outcome-aware baseline";
  $$(".compare-option").forEach((button) => {
    button.classList.toggle("active", button.dataset.hero === state.heroMode);
    button.setAttribute("aria-pressed", String(button.dataset.hero === state.heroMode));
  });
  const rejected = (recap?.retrieval_rejections ?? []).slice(0, 2).map((item) => item.reason);
  const detail =
    state.heroMode === "recap"
      ? recapSuccess
        ? `The policy selected a compatible strategy and the external postcondition passed. Rejections: ${rejected.join(", ") || "none"}.`
        : `No safe strategy passed the gates. Failing closed is the correct action here; it is not counted as a task success. Rejections: ${rejected.join(", ") || "none"}.`
      : naiveSuccess
        ? "The run completed, but success here is measured externally. Check harmful transfer before calling it safe."
        : `The completed-looking call still produced the wrong record set. Final error: ${naive?.final_error || "none reported"}.`;
  $("#hero-explain").textContent = detail;
  $("#hero-swap").textContent = state.heroMode === "recap" ? "Show stale-memory baseline →" : "Show RECAP policy →";
}

function renderTiers() {
  const tiers = state.data?.by_tier ?? {};
  for (const tier of ["A", "B", "C"]) {
    const stats = tiers[tier] ?? {};
    const noMemory = stats.no_memory?.task_success_rate;
    const outcome = stats.outcome_aware?.task_success_rate;
    const text =
      noMemory == null && outcome == null
        ? "not part of this split"
        : `no memory ${pct(noMemory)} · outcome-aware ${pct(outcome)}`;
    $(`#tier-${tier.toLowerCase()}`).textContent = text;
  }
  const gate = state.data?.acceptance;
  if (!gate) {
    $("#gate").textContent = "No v3 acceptance gate in this artifact (legacy/v2 suite).";
    return;
  }
  $("#gate").textContent = `Tier C acceptance gate · no_memory ${gate.successes}/${gate.n_tasks} (${pct(gate.success_rate)}) · threshold ≤ ${pct(gate.threshold)} · ${gate.passed ? "PASS" : "FAIL — redesign the suite, do not report it"}`;
  $("#gate").style.borderLeftColor = gate.passed ? "var(--ok)" : "var(--danger)";
}

function renderDrift() {
  const select = $("#drift-scenario");
  const candidates = [];
  const seen = new Set();
  for (const row of silentCandidates()) {
    const task = state.tasks.get(row.task_id);
    if (!task || seen.has(task.task_id)) continue;
    seen.add(task.task_id);
    candidates.push(task);
  }
  if (!candidates.length) {
    select.innerHTML = "";
    $("#drift-comparison").innerHTML = "<p class='hint'>No silent-drift rows in this artifact.</p>";
    return;
  }

  const rankTask = (task) => {
    const rows = (state.data.per_task ?? []).filter((row) => row.task_id === task.task_id);
    const naive =
      rows.find((row) => row.baseline === "bm25") ??
      rows.find((row) => row.baseline === "hybrid_rrf") ??
      rows.find((row) => row.baseline === "no_memory");
    const recap =
      rows.find((row) => row.baseline === "outcome_aware") ??
      rows.find((row) => row.baseline === bestBaseline());
    let score =
      (recap?.task_success === true ? 1 : 0) -
      (naive?.task_success === true ? 1 : 0) +
      (naive?.harmful_transfer === true ? 1 : 0);
    // Prefer a real policy win over a correct abstention on a negative control.
    if (task.should_select === false) score -= 10;
    if (recap?.task_success === true && naive?.task_success !== true) score += 2;
    return { task, score, rows, naive, recap };
  };

  const ranked = candidates.map(rankTask).sort(
    (a, b) => b.score - a.score || a.task.task_id.localeCompare(b.task.task_id)
  );
  const byScenario = new Map();
  for (const item of ranked) {
    const key = item.task.scenario_id || item.task.task_id;
    if (!byScenario.has(key)) byScenario.set(key, item);
  }
  select.innerHTML = Array.from(byScenario.values())
    .map(
      ({ task }) =>
        `<option value="${task.task_id}">${label(task.scenario_id || task.task_id)} \u00b7 ${label(
          task.failure_mode
        )} \u00b7 ${task.should_select ? "memory required" : "must abstain"}</option>`
    )
    .join("");
  const preferred = ranked[0]?.task;
  if (!preferred) return;
  if (!state.driftTask || !select.querySelector(`option[value="${state.driftTask}"]`)) {
    state.driftTask = preferred.task_id;
  }
  select.value = state.driftTask;

  const task = state.tasks.get(state.driftTask);
  const rows = (state.data.per_task ?? []).filter((row) => row.task_id === state.driftTask);
  const naive =
    rows.find((row) => row.baseline === "bm25") ??
    rows.find((row) => row.baseline === "hybrid_rrf") ??
    rows.find((row) => row.baseline === "no_memory");
  const recap =
    rows.find((row) => row.baseline === "outcome_aware") ??
    rows.find((row) => row.baseline === bestBaseline());
  $("#drift-hint").textContent = task
    ? `Task ${task.task_id} \u00b7 tier ${task.tier} \u00b7 expected ${
        task.should_select === false ? "safe abstention" : label(task.failure_mode)
      }`
    : "";

  const verdict = (row) => {
    if (!row) return { good: false, text: "No row in artifact" };
    if (task?.should_select === false) {
      return row.abstained === true || (row.selected_memory_ids ?? []).length === 0
        ? { good: true, text: "Correct safe abstention" }
        : { good: false, text: "Unsafe memory applied" };
    }
    if (row.task_success === true) return { good: true, text: "Externally verified" };
    if (row.incompatible_selected > 0 || row.harmful_transfer === true) {
      return { good: false, text: "Unsafe transfer" };
    }
    if (row.abstained === true) return { good: false, text: "False abstention" };
    return { good: false, text: "Wrong / incomplete" };
  };

  const card = (title, row) => {
    if (!row) return `<article class="compare-card"><h4>${title}</h4><p>No row.</p></article>`;
    const stateOf = verdict(row);
    const selected = (row.selected_memory_ids ?? []).join(", ") || "none";
    return `<article class="compare-card">
      <h4>${title} \u00b7 ${label(row.baseline)}</h4>
      <div class="verdict ${stateOf.good ? "good" : "bad"}">${stateOf.text}</div>
      <p>Selected: <code>${selected}</code></p>
      <p>Harmful transfer: ${row.harmful_transfer ? "yes" : "no"} \u00b7 incompatible: ${
        row.incompatible_selected ?? 0
      } \u00b7 calls: ${row.tool_calls ?? "?"}</p>
      <p>${row.final_error || "No final error recorded."}</p>
    </article>`;
  };
  $("#drift-comparison").innerHTML = card("Relevance-first", naive) + card("Outcome-aware", recap);
}

function renderStats() {
  const summary = state.data?.summary ?? {};
  const best = bestBaseline();
  const bestStats = summary[best] ?? {};
  const noMemory = summary.no_memory ?? {};
  const outcome = summary.outcome_aware ?? summary[best] ?? {};
  const gate = state.data?.acceptance;
  const cards = [
    ["Best success", pct(bestStats.task_success_rate), `${label(best)} · cluster CI ${formatCi(bestStats.task_success_cluster_ci95)}`],
    ["No-memory control", pct(noMemory.task_success_rate), "no retrieval at all"],
    ["Outcome-aware harm", pct(outcome.harmful_transfer_rate), "verified wrong result after applying memory"],
    ["False abstention", pct(outcome.false_abstention_rate), gate ? `Tier C gate ${gate.passed ? "passed" : "failed"}` : "higher means missed help"],
  ];
  $("#stat-row").innerHTML = cards
    .map(([name, value, detail]) => `<div class="stat"><span class="value">${value}</span><span class="label">${name}</span><span class="detail">${detail}</span></div>`)
    .join("");
}

function formatCi(ci) {
  if (!Array.isArray(ci) || ci.length !== 2) return "—";
  return `[${pct(ci[0])}, ${pct(ci[1])}]`;
}

function renderBars() {
  const summary = state.data?.summary ?? {};
  const entries = Object.entries(summary).sort((a, b) => (b[1].task_success_rate ?? 0) - (a[1].task_success_rate ?? 0));
  $("#success-bars").innerHTML = entries
    .map(([name, stats]) => {
      const value = stats.task_success_rate ?? 0;
      const ci = formatCi(stats.task_success_cluster_ci95);
      return `<div class="bar-row" title="${label(name)}: ${pct(value)} · cluster 95% CI ${ci} · ${stats.n_scenarios ?? "?"} scenarios">
        <span class="bar-label">${label(name)}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${Math.max(0, Math.min(100, value * 100))}%"></span></span>
        <span class="bar-value">${pct(value)}</span>
      </div>`;
    })
    .join("");
}

function renderScatter() {
  const svg = $("#scatter");
  const summary = state.data?.summary ?? {};
  const entries = Object.entries(summary);
  const pad = { left: 42, right: 16, top: 14, bottom: 34 };
  const width = 420;
  const height = 300;
  const x = (value) => pad.left + value * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - value * (height - pad.top - pad.bottom);
  const grid = [0, 0.25, 0.5, 0.75, 1];
  let markup = "";
  for (const tick of grid) {
    markup += `<line class="scatter-axis" x1="${x(tick)}" y1="${pad.top}" x2="${x(tick)}" y2="${height - pad.bottom}" opacity="0.35"/>`;
    markup += `<line class="scatter-axis" x1="${pad.left}" y1="${y(tick)}" x2="${width - pad.right}" y2="${y(tick)}" opacity="0.35"/>`;
    markup += `<text class="scatter-tick" x="${x(tick)}" y="${height - pad.bottom + 14}" text-anchor="middle">${pct(tick)}</text>`;
    markup += `<text class="scatter-tick" x="${pad.left - 8}" y="${y(tick) + 3}" text-anchor="end">${pct(tick)}</text>`;
  }
  markup += `<text class="scatter-tick" x="${width / 2}" y="${height - 6}" text-anchor="middle">harmful transfer →</text>`;
  markup += `<text class="scatter-tick" x="12" y="${height / 2}" transform="rotate(-90 12 ${height / 2})" text-anchor="middle">success →</text>`;
  entries.forEach(([name, stats], index) => {
    const harm = stats.harmful_transfer_rate ?? 0;
    const success = stats.task_success_rate ?? 0;
    const cx = x(harm);
    const cy = y(success);
    const dy = index % 2 === 0 ? -8 : 13;
    markup += `<g class="scatter-point"><title>${label(name)}: success ${pct(success)}, harm ${pct(harm)}</title>
      <circle cx="${cx}" cy="${cy}" r="5"/>
      <text x="${cx + 7}" y="${cy + dy}">${label(name)}</text></g>`;
  });
  svg.innerHTML = markup;
}

function renderSummaryTable() {
  const summary = state.data?.summary ?? {};
  const rows = Object.entries(summary).sort((a, b) => (b[1].task_success_rate ?? 0) - (a[1].task_success_rate ?? 0));
  $("#summary-body").innerHTML = rows
    .map(([name, stats]) => `<tr>
      <td>${label(name)}</td>
      <td>${pct(stats.task_success_rate)}</td>
      <td>${formatCi(stats.task_success_ci95)}</td>
      <td>${num(stats.recall_at_3, 3)}</td>
      <td>${num(stats.mrr, 3)}</td>
      <td>${pct(stats.harmful_transfer_rate)}</td>
      <td>${pct(stats.false_abstention_rate)}</td>
      <td>${num(stats.mean_tool_calls, 2)}</td>
      <td>${pct(stats.incompatible_selection_rate)}</td>
    </tr>`)
    .join("");
}

function renderFamily() {
  const select = $("#family-baseline");
  const names = baselines();
  if (!select.options.length) {
    select.innerHTML = names.map((name) => `<option value="${name}">${label(name)}</option>`).join("");
  }
  if (!select.value || !names.includes(select.value)) select.value = bestBaseline() ?? names[0] ?? "";
  const baseline = select.value;
  const byFamily = state.data?.by_task_family ?? {};
  const rows = Object.entries(byFamily)
    .map(([family, statsByBaseline]) => [family, statsByBaseline[baseline]])
    .filter(([, stats]) => stats);
  $("#family-body").innerHTML = rows.length
    ? rows
        .map(([family, stats]) => `<tr>
          <td>${label(family)}</td>
          <td>${pct(stats.task_success_rate)}</td>
          <td>${pct(stats.harmful_transfer_rate)}</td>
          <td>${pct(stats.false_abstention_rate)}</td>
          <td>${pct(stats.hard_negative_outrank_rate)}</td>
        </tr>`)
        .join("")
    : "<tr><td colspan='5'>No family breakdown in this artifact.</td></tr>";
}

function renderInspectorControls() {
  const taskSelect = $("#inspect-task");
  const baselineSelect = $("#inspect-baseline");
  const taskIds = Array.from(new Set((state.data?.per_task ?? []).map((row) => row.task_id))).sort();
  taskSelect.innerHTML = taskIds.map((id) => `<option value="${id}">${id}</option>`).join("");
  baselineSelect.innerHTML = baselines().map((name) => `<option value="${name}">${label(name)}</option>`).join("");
  if (!state.inspect.task || !taskIds.includes(state.inspect.task)) state.inspect.task = taskIds[0] ?? "";
  if (!state.inspect.baseline || !baselines().includes(state.inspect.baseline)) state.inspect.baseline = bestBaseline() ?? baselines()[0] ?? "";
  taskSelect.value = state.inspect.task;
  baselineSelect.value = state.inspect.baseline;

  const runTask = $("#run-task");
  const runBaseline = $("#run-baseline");
  runTask.innerHTML = taskSelect.innerHTML;
  runBaseline.innerHTML = baselineSelect.innerHTML;
  if (!state.run.task || !taskIds.includes(state.run.task)) state.run.task = state.inspect.task;
  if (!state.run.baseline || !baselines().includes(state.run.baseline)) state.run.baseline = state.inspect.baseline;
  runTask.value = state.run.task;
  runBaseline.value = state.run.baseline;
}

function renderInspector() {
  const row = rowFor(state.inspect.task, state.inspect.baseline);
  if (!row) {
    $("#inspector-content").innerHTML = "<p class='hint'>No row for this task/baseline combination.</p>";
    return;
  }
  const task = state.tasks.get(row.task_id) ?? {};
  const rejections = row.retrieval_rejections ?? [];
  const selected = (row.selected_memory_ids ?? []).join(", ") || "none";
  const relevant = (row.relevant_memory_ids ?? []).join(", ") || "none";
  const details = [
    ["task", row.task_id],
    ["tier", `${task.tier ?? "?"} · ${label(task.failure_mode ?? row.task_family)}`],
    ["baseline", label(row.baseline)],
    ["status", label(row.retrieval_status ?? row.outcome)],
    ["success", row.task_success ? "true (external postcondition)" : "false"],
    ["selected", selected],
    ["gold", relevant],
    ["tool calls", row.tool_calls],
    ["retries", row.retries],
    ["final error", row.final_error || "none"],
    ["run", row.run_id || "not recorded"],
  ];
  $("#inspector-content").innerHTML = `
    <article class="inspect-card">
      <h3>Decision</h3>
      <dl class="kv">${details.map(([key, value]) => `<dt>${key}</dt><dd>${String(value)}</dd>`).join("")}</dl>
    </article>
    <article class="inspect-card">
      <h3>Rejected candidates</h3>
      <p class="hint">${rejections.length} candidate(s) carry an auditable reason. The selected memory is not repeated here.</p>
      <ul class="rejections">${rejections
        .slice(0, 80)
        .map((item) => `<li><span class="reason">${label(item.reason)}</span><span><code>${item.memory_id}</code>${item.detail ? ` · ${item.detail}` : ""}</span></li>`)
        .join("") || "<li>No rejections persisted for this row.</li>"}</ul>
    </article>`;
}

function renderLifecycle() {
  const steps = [
    ["Observe", "A failure window is recorded, not inferred from a chat message."],
    ["Compile", "A typed repair candidate with actions and preconditions."],
    ["Validate", "Schema, side effects, and replay checks must pass."],
    ["Promote", "A separate gate writes evidence and an audit event."],
    ["Recall", "Field-scoped compatibility and preconditions run before selection."],
    ["Shift", "Suspect, quarantine, or retire when the world changes."],
  ];
  $("#lifecycle-flow").innerHTML = steps
    .map(([name, detail]) => `<div class="lifecycle-node"><strong>${name}</strong><span>${detail}</span></div>`)
    .join("");
  const store = state.data?.store ?? {};
  const chips = [
    ["gold fixtures", store.gold_memories],
    ["synthetic decoys", store.synthetic_fixtures],
    ["total store", store.total],
    ["provenance", store.provenance],
  ];
  $("#store-strip").innerHTML = chips
    .filter(([, value]) => value != null)
    .map(([name, value]) => `<span class="chip">${name}: ${value}</span>`)
    .join("");
}

function renderTimeline() {
  const row = rowFor(state.run.task, state.run.baseline);
  if (!row) {
    $("#timeline").innerHTML = "<div class='timeline-step'>No row selected.</div>";
    return;
  }
  const task = state.tasks.get(row.task_id) ?? {};
  const steps = [
    ["01", "Task start", `${task.instruction ?? row.task_id} · tier ${task.tier ?? "?"} · tenant ${task.tenant ?? "?"}`],
    ["02", "Memory snapshot", "The run pins the memory set before retrieval; later changes cannot leak into this result."],
    ["03", "Retrieval", `${label(row.retrieval_status)} · selected: ${(row.selected_memory_ids ?? []).join(", ") || "none"} · gold: ${(row.relevant_memory_ids ?? []).join(", ") || "none"}`],
    ["04", "Tool loop", `${row.tool_calls ?? "?"} tool calls · ${row.retries ?? "?"} retries · outcome ${label(row.outcome)}`],
    ["05", "Postcondition", row.task_success ? "Passed against ground-truth records, not the model's claim." : `Failed: ${row.final_error || "no detail"}`],
    ["06", "Attribution", row.used_memory_id ? `Memory ${row.used_memory_id} was applied.` : "No memory was applied."],
  ];
  $("#timeline").innerHTML = steps
    .map(([number, title, detail]) => `<div class="timeline-step"><span class="step-no">${number}</span><div><strong>${title}</strong><div class="step-detail">${detail}</div></div></div>`)
    .join("");
}

function citation(text) {
  return `<span class="citation">${text}</span>`;
}

function renderAsk() {
  const summary = state.data?.summary ?? {};
  const best = bestBaseline();
  const gate = state.data?.acceptance;
  const questions = [
    {
      q: "Does memory actually help?",
      answer: () => {
        const bestStats = summary[best] ?? {};
        const noMemory = summary.no_memory ?? {};
        const bestSelectable = selectableStats(best);
        const noSelectable = selectableStats("no_memory");
        const live = state.data?.meta?.agent_mode !== "deterministic";
        const caveat = live
          ? "This is a live-model artifact in a deterministic mock environment. The model sometimes selected the correct memory but failed to execute its strategy, so a harmful-transfer count here is not the same as wrong memory selection."
          : "This is a deterministic pipeline artifact, not live-model evidence.";
        return `<strong>${label(best)}</strong> records ${pct(bestStats.task_success_rate)} task success versus ${pct(noMemory.task_success_rate)} for no memory. On the ${bestSelectable.n} selectable tasks, excluding the no-safe-memory controls, it is <strong>${bestSelectable.success}/${bestSelectable.n}</strong> versus <strong>${noSelectable.success}/${noSelectable.n}</strong>. ${caveat} ${citation("summary.task_success_rate + tasks.should_select")}`;
      },
    },
    {
      q: "Is the benchmark non-discriminative?",
      answer: () => {
        const values = Object.values(summary).map((stats) => `${stats.task_success_rate}/${stats.harmful_transfer_rate}/${stats.recall_at_3}`);
        const unique = new Set(values).size;
        return `${unique === 1 ? "All baselines are identical, so the suite is <strong>not discriminative</strong>." : `The baselines differ: <strong>${unique}</strong> distinct success/harm/recall triples.`} ${citation("summary")}`;
      },
    },
    {
      q: "What happens on Tier C without memory?",
      answer: () => {
        if (!gate) return "This artifact has no v3 Tier C gate.";
        return `no_memory succeeds on <strong>${gate.successes}/${gate.n_tasks}</strong> Tier C tasks (${pct(gate.success_rate)}); the suite requires ≤ ${pct(gate.threshold)} and currently <strong>${gate.passed ? "passes" : "fails"}</strong>. ${citation("acceptance")}`;
      },
    },
    {
      q: "Which baseline transfers harm?",
      answer: () => {
        const entries = Object.entries(summary).sort((a, b) => (b[1].harmful_transfer_rate ?? 0) - (a[1].harmful_transfer_rate ?? 0));
        const [name, stats] = entries[0] ?? ["none", {}];
        return `The highest harmful-transfer rate is <strong>${label(name)}</strong> at ${pct(stats.harmful_transfer_rate)}. Outcome-aware is ${pct(summary.outcome_aware?.harmful_transfer_rate)}. ${citation("summary.harmful_transfer_rate")}`;
      },
    },
    {
      q: "Why does RECAP abstain?",
      answer: () => {
        const outcome = summary.outcome_aware ?? {};
        return `Abstention is a first-class outcome: outcome-aware false abstention is ${pct(outcome.false_abstention_rate)}, while abstention precision is ${pct(outcome.abstention_precision)} and recall is ${pct(outcome.abstention_recall)}. A stale memory is rejected by field-scoped compatibility, not by a whole-schema mismatch. ${citation("summary.abstention_*")}`;
      },
    },
  ];
  $("#ask-buttons").innerHTML = questions.map((item, index) => `<button type="button" data-question="${index}">${item.q}</button>`).join("");
  $$("#ask-buttons button").forEach((button) => {
    button.addEventListener("click", () => {
      $("#answer").innerHTML = questions[Number(button.dataset.question)].answer();
    });
  });
}

function renderArchitecture() {
  const items = [
    {
      id: "recall",
      name: "Recall engine",
      short: "hybrid retrieve → fuse → gate → select",
      detail: "Elasticsearch returns lexical and vector branches. RRF fuses them, applicability and field-scoped compatibility gate candidates, and utility policy selects or returns no_safe_memory. Source: <code>src/recap/recall/engine.py</code>.",
    },
    {
      id: "postgres",
      name: "Postgres",
      short: "source of truth + outbox",
      detail: "Memories, versions, run events, snapshots, decisions, and audit events live here. The outbox makes indexing retryable; Elasticsearch is derived state. Source: <code>src/recap/infrastructure/db/</code>.",
    },
    {
      id: "validation",
      name: "Compile & validate",
      short: "typed repair, promotion gate",
      detail: "A failure becomes a candidate only through typed actions, preconditions, side-effect checks, and replay validation. Promotion is separate from validation. Source: <code>src/recap/repair_compiler/</code> and <code>src/recap/validation/</code>.",
    },
    {
      id: "eval",
      name: "v3 evaluation",
      short: "tiers, silent drift, clustered CIs",
      detail: "Tier A controls the harness, Tier B is probe-discoverable, Tier C needs memory. Scenario-clustered bootstrap intervals and a no-memory acceptance gate prevent a flattering but broken suite. Source: <code>src/recap/evals/</code>.",
    },
    {
      id: "api",
      name: "FastAPI",
      short: "runs, memories, retrieval, Ask",
      detail: "The API exposes the same services the benchmark uses. Live mode is optional on this page; replay mode never needs a backend. Source: <code>src/recap/api/</code>.",
    },
    {
      id: "elasticsearch",
      name: "Elasticsearch",
      short: "BM25 + kNN + filters",
      detail: "Concrete versioned indices sit behind read/write aliases. Chunk and memory-card granules are indexed together for chunk ablations. Source: <code>src/recap/infrastructure/elasticsearch/</code>.",
    },
    {
      id: "workers",
      name: "Outbox worker",
      short: "index repair with backoff",
      detail: "A real daemon drains pending index events, retries with backoff, and shuts down gracefully. Start it with <code>recap search worker</code>. Source: <code>src/recap/infrastructure/outbox.py</code> and <code>src/recap/cli.py</code>.",
    },
    {
      id: "web",
      name: "This app",
      short: "static replay, optional live",
      detail: "The hosted site reads exported report JSON. A recruiter needs no Docker, database, or model. Source: <code>web/</code> and <code>scripts/export_replay.py</code>.",
    },
  ];
  $("#architecture-grid").innerHTML = items
    .map((item) => `<button class="arch-card ${state.architecture === item.id ? "active" : ""}" type="button" data-arch="${item.id}"><strong>${item.name}</strong><span>${item.short}</span></button>`)
    .join("");
  const active = items.find((item) => item.id === state.architecture) ?? items[0];
  $("#architecture-grid").insertAdjacentHTML("afterend", "");
  let detail = $("#arch-detail");
  if (!detail) {
    detail = document.createElement("div");
    detail.id = "arch-detail";
    detail.className = "arch-detail";
    $("#architecture-grid").after(detail);
  }
  detail.innerHTML = `<strong>${active.name}</strong> — ${active.detail}`;
  $$(".arch-card").forEach((button) => {
    button.addEventListener("click", () => {
      state.architecture = button.dataset.arch;
      renderArchitecture();
    });
  });
}

function renderAll() {
  state.tasks = new Map((state.data?.tasks ?? []).map((task) => [task.task_id, task]));
  setProvenance();
  renderHero();
  renderTiers();
  renderDrift();
  renderStats();
  renderBars();
  renderScatter();
  renderSummaryTable();
  renderFamily();
  renderInspectorControls();
  renderInspector();
  renderLifecycle();
  renderTimeline();
  renderAsk();
  renderArchitecture();
}

function wireEvents() {
  $("#theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    localStorage.setItem("recap-theme", next);
  });
  $$(".compare-option").forEach((button) => {
    button.addEventListener("click", () => {
      state.heroMode = button.dataset.hero;
      renderHero();
    });
  });
  $("#hero-swap").addEventListener("click", () => {
    state.heroMode = state.heroMode === "recap" ? "naive" : "recap";
    renderHero();
  });
  $("#dataset-select").addEventListener("change", async (event) => {
    try {
      await loadDataset(Number(event.target.value));
    } catch (error) {
      showError(error?.stack || String(error));
    }
  });
  $("#drift-scenario").addEventListener("change", (event) => {
    state.driftTask = event.target.value;
    renderDrift();
  });
  $("#family-baseline").addEventListener("change", renderFamily);
  $("#inspect-task").addEventListener("change", (event) => {
    state.inspect.task = event.target.value;
    renderInspector();
  });
  $("#inspect-baseline").addEventListener("change", (event) => {
    state.inspect.baseline = event.target.value;
    renderInspector();
  });
  $("#run-task").addEventListener("change", (event) => {
    state.run.task = event.target.value;
    renderTimeline();
  });
  $("#run-baseline").addEventListener("change", (event) => {
    state.run.baseline = event.target.value;
    renderTimeline();
  });
}

function showError(message) {
  $("#provenance").textContent = "Replay data unavailable — serve this directory over HTTP and run scripts/export_replay.py.";
  $("#main").innerHTML = `<section class="section"><h2>Replay data unavailable</h2><p class="section-note">${message}</p><p>From the repository root:</p><pre><code>python scripts/export_replay.py --input benchmark/reports/benchmark_test_*.json --out web/replay\npython -m http.server 8000 --directory web</code></pre></section>`;
}

async function init() {
  const saved = localStorage.getItem("recap-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  try {
    const [response, catalog] = await Promise.all([
      fetch("replay/benchmark.json", { cache: "no-store" }),
      fetch("replay/datasets.json", { cache: "no-store" }),
    ]);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    state.datasets = catalog.ok ? (await catalog.json()).datasets ?? [] : [];
    if (!state.datasets.length) state.datasets = [state.data.meta];
    populateDatasetSelect();
    wireEvents();
    renderAll();
  } catch (error) {
    showError(String(error));
  }
}

init();
