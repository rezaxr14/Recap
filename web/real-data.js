"use strict";
// Renders committed BFCL reports. Numbers are copied from the JSON, never recomputed.
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const num = (x) => (typeof x === "number" ? x.toFixed(3) : "–");
const withCi = (v, c) => (Array.isArray(c) ? `${num(v)} [${num(c[0])}, ${num(c[1])}]` : num(v));

const RETRIEVAL_ROWS = [
  ["bm25", "BM25"],
  ["vector", "Dense"],
  ["hybrid_rrf_w1", "RRF, equal weight"],
  ["hybrid_rrf_w2", "RRF, dense ×2"],
  ["hybrid_rrf_w3", "RRF, dense ×3"],
];
const TRANSFER_ROWS = [
  ["no_memory", "No memory"],
  ["retry_only", "Retry only"],
  ["bm25", "BM25 memory"],
  ["vector", "Dense memory"],
  ["hybrid_rrf", "RRF memory"],
  ["function_keyed", "Keyed by tool name"],
  ["oracle_memory", "Oracle"],
];

function table(head, rows) {
  const th = head.map((h) => `<th scope="col">${esc(h)}</th>`).join("");
  const tr = rows.map((r) => `<tr>${r.map((c, i) => `<${i ? "td" : "th scope=\"row\""}>${c}</${i ? "td" : "th"}>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
}

function renderRetrieval(report) {
  const rows = RETRIEVAL_ROWS.filter(([k]) => report.results[k]).map(([k, label]) => {
    const t = report.results[k].test;
    const a = report.results[k].abstention || {};
    const name = esc(label) + (k === report.selected_variant ? " (selected on dev)" : "");
    return [name, withCi(t.hit_at_1, t.hit_at_1_ci95), num(t.hit_at_3), num(t.reciprocal_rank), num(t.auc_vs_irrelevance), a.identified === false ? "n/a (threshold 0)" : num(a.test_control_abstention)];
  });
  return table(["Method", "hit@1 (95% CI)", "hit@3", "MRR@10", "AUC vs irrelevance", "Control abstention at 95% coverage"], rows);
}

function renderTransfer(report) {
  const rows = TRANSFER_ROWS.filter(([k]) => report.results[k]).map(([k, label]) => {
    const r = report.results[k];
    return [esc(label), `${r.successes}/${r.n_scenarios}`, withCi(r.success_rate, r.success_ci95), r.retrieval_hit_at_1 === undefined ? "n/a" : num(r.retrieval_hit_at_1)];
  });
  return table(["Method", "Successes", "Success rate (95% CI)", "Memory hit@1"], rows);
}

function provenance(ret, tr) {
  const e = ret.embedding;
  return `BFCL v3 · ${ret.labeled_tasks} labeled queries (dev ${ret.dev_labeled} / test ${ret.test_labeled}) · ${ret.control_tasks} irrelevance controls · ${ret.function_pool} functions · ${tr.scenarios} transfer scenarios · embedder ${e.model} (${e.dimension}-d) · generated ${String(ret.generated_at).slice(0, 10)}`;
}

function renderDelta(current, previous, kind) {
  const rows = (kind === "retrieval" ? RETRIEVAL_ROWS : TRANSFER_ROWS)
    .filter(([k]) => current.results[k] && previous.results[k])
    .map(([k, label]) => {
      const now = kind === "retrieval" ? current.results[k].test.hit_at_1 : current.results[k].success_rate;
      const before = kind === "retrieval" ? previous.results[k].test.hit_at_1 : previous.results[k].success_rate;
      const delta = now - before;
      return [esc(label), num(before), num(now), `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`];
    });
  return table(["Method", "v1", "current", "change"], rows);
}

function pairedNote(report) {
  const stats = (report.paired_vs_vector || {}).hybrid_rrf_w3_minus_vector;
  if (!stats) return "";
  return `Paired bootstrap on the same queries (test): RRF ×3 minus dense = ${stats.observed >= 0 ? "+" : ""}${stats.observed.toFixed(3)} [${stats.cluster_ci95[0].toFixed(3)}, ${stats.cluster_ci95[1].toFixed(3)}]; excludes zero: ${stats.excludes_zero}.`;
}

async function load(emb) {
  const get = async (name) => {
    const res = await fetch(`data/${name}-${emb}.json`);
    if (!res.ok) throw new Error(`${name}-${emb}.json: HTTP ${res.status}`);
    return res.json();
  };
  const getOptional = async (name) => {
    const res = await fetch(`data/${name}-${emb}-v1.json`);
    return res.ok ? res.json() : null;
  };
  const getRecap = async () => {
    const res = await fetch(`data/real-data-bfcl-recap-${emb}.json`);
    return res.ok ? res.json() : null;
  };
  const getLive = async () => {
    const res = await fetch(`data/real-data-bfcl-live-llama-3-2-3b-instruct-${emb}.json`);
    return res.ok ? res.json() : null;
  };
  const [ret, tr, retV1, trV1, recap, live] = await Promise.all([
    get("real-data-bfcl"),
    get("real-data-bfcl-transfer"),
    getOptional("real-data-bfcl"),
    getOptional("real-data-bfcl-transfer"),
    getRecap(),
    getLive(),
  ]);
  return { ret, tr, retV1, trV1, recap, live };
}

const RECAP_ARMS = [
  ["bm25", "BM25"],
  ["vector", "Dense"],
  ["hybrid_rrf", "RRF"],
  ["rrf_typed", "RRF + typed context"],
  ["rrf_version_filter", "RRF + environment-id filter"],
  ["fingerprint_filter", "RRF + fingerprint filter"],
  ["function_keyed", "Keyed by tool name"],
  ["schema_hash", "Schema-hash match"],
  ["full_recap", "Full RECAP policy"],
];

const LIVE_ARMS = [
  ["no_memory", "No memory applied"],
  ["retry_only", "Retry only (model sees the error)"],
  ["bm25", "BM25 memory"],
  ["vector", "Dense memory"],
  ["rrf_typed", "RRF + typed context"],
  ["full_recap", "Full RECAP policy"],
];

function renderRecapSafe(report) {
  const conditions = ["in_pool", "drift_phase1", "drift_phase3"].filter((c) => report.conditions[c]);
  const rows = RECAP_ARMS.filter(([arm]) =>
    conditions.every((condition) => report.conditions[condition].results[arm]),
  ).map(([arm, label]) => [
    esc(label),
    ...conditions.map((condition) => num(report.conditions[condition].results[arm].call_valid_rate)),
  ]);
  return table(["Arm", ...conditions.map((c) => `${c}: call valid`)], rows);
}

function renderRecapNoSafe(report) {
  const conditions = ["held_out", "sibling", "drift_phase2"].filter((c) => report.conditions[c]);
  const rows = RECAP_ARMS.filter(([arm]) =>
    conditions.every((condition) => report.conditions[condition].results[arm]),
  ).map(([arm, label]) => [
    esc(label),
    ...conditions.flatMap((condition) => {
      const stats = report.conditions[condition].results[arm];
      return [num(stats.false_apply_rate), num(stats.abstain_rate)];
    }),
  ]);
  const head = [
    "Arm",
    ...conditions.flatMap((c) => [`${c}: false apply`, `${c}: abstain`]),
  ];
  return table(head, rows);
}

function renderRecapAbstention(report) {
  const rows = Object.entries(report.abstention)
    .filter(([arm]) => arm !== "no_memory" && arm !== "retry_only")
    .map(([arm, stats]) => [
      esc(arm),
      num(stats.precision),
      num(stats.recall),
      `${stats.missed_abstentions}/${stats.n_no_safe}`,
      `${stats.unneeded_abstentions}/${stats.n_safe}`,
    ]);
  return table(
    ["Arm", "Precision", "Recall", "Missed (no-safe selected)", "Unneeded (safe abstained)"],
    rows,
  );
}

function renderRecapPaired(report) {
  const arms = [
    "bm25",
    "vector",
    "hybrid_rrf",
    "rrf_typed",
    "rrf_version_filter",
    "fingerprint_filter",
    "function_keyed",
    "schema_hash",
  ];
  const signed = (x) => `${x >= 0 ? "+" : ""}${x.toFixed(3)}`;
  const rows = [];
  for (const subset of ["safe", "no_safe"]) {
    for (const arm of arms) {
      const stats = ((report.paired || {})[subset] || {})[arm];
      if (!stats) continue;
      const [lo, hi] = stats.cluster_ci95;
      rows.push([
        subset === "safe" ? "Safe" : "No-safe",
        esc(arm),
        signed(stats.observed),
        `[${signed(lo)}, ${signed(hi)}]`,
        stats.excludes_zero ? "yes" : "no",
      ]);
    }
  }
  return table(["Subset", "Baseline", "full_recap − baseline", "95% CI", "Excludes zero"], rows);
}

function renderRecap(report) {
  if (!report) {
    return "<p class=\"section-note\">No real-stack report is deployed for this embedder.</p>";
  }
  const drift = report.drift;
  const examples = (drift.examples || [])
    .slice(0, 3)
    .map((e) => `\`${e.function}\`: ${e.first_rename.join(" -> ")} then ${e.second_rename.join(" -> ")}, supersedes ${e.superseded ? "confirmed" : "not confirmed"}`)
    .join("; ");
  return `
    <p class="section-note">${report.funnel.scenarios} scenarios, ${report.funnel.memories_promoted} memories promoted (${report.funnel.validations_passed} validations passed), ${report.funnel.sibling_pairs ?? "n/a"} sibling pairs, utility threshold ${Number(report.utility_threshold).toFixed(2)}.</p>
    <h3>Safe conditions: a matching memory exists</h3>
    <div class="table-wrap">${renderRecapSafe(report)}</div>
    <h3>No-safe conditions: the right answer is to abstain</h3>
    <div class="table-wrap">${renderRecapNoSafe(report)}</div>
    <h3>Abstention precision and recall</h3>
    <div class="table-wrap">${renderRecapAbstention(report)}</div>
    <h3>Paired bootstrap against the full policy</h3>
    <div class="table-wrap">${renderRecapPaired(report)}</div>
    <p class="section-note">Positive means the full policy did better: call-valid rate on safe rows, abstention rate on no-safe rows. Intervals are 95% cluster bootstraps over functions; a zero-inclusive interval means the data do not separate the arms.</p>
    <p class="section-note">Drift: ${drift.functions} functions, ${drift.superseded} supersessions confirmed, ${drift.active_after} active memories after supersession. ${examples ? `Examples: ${esc(examples)}.` : ""}</p>
  `;
}

function renderLiveArms(report, subsetKey) {
  const subset = (report.eval.subsets || {})[subsetKey] || {};
  const rows = LIVE_ARMS.filter(([arm]) => subset[arm]).map(([arm, label]) => {
    const stats = subset[arm];
    return [
      esc(label),
      String(stats.n),
      num(stats.call_valid_rate),
      num(stats.execution_repair_rate),
      num(stats.selected_rate),
      num(stats.abstain_rate),
      num(stats.harmful_apply_rate),
    ];
  });
  return table(
    ["Arm", "n", "Call valid", "Repaired failed call", "Selected", "Abstained", "Harmful apply"],
    rows,
  );
}

function renderLivePaired(report) {
  const subset = (report.paired || {}).safe_failed || {};
  const harm = report.paired_harm || {};
  const conditions = report.condition_paired || {};
  const signed = (x) => `${x >= 0 ? "+" : ""}${x.toFixed(3)}`;
  const rows = [];
  for (const [arm, stats] of Object.entries(subset)) {
    const [lo, hi] = stats.cluster_ci95;
    rows.push(["Safe failure: call valid", esc(arm), signed(stats.observed), `[${signed(lo)}, ${signed(hi)}]`, stats.excludes_zero ? "yes" : "no"]);
  }
  for (const [arm, stats] of Object.entries(harm)) {
    const [lo, hi] = stats.cluster_ci95;
    rows.push(["No safe memory: harmful apply", esc(arm), signed(stats.observed), `[${signed(lo)}, ${signed(hi)}]`, stats.excludes_zero ? "yes" : "no"]);
  }
  for (const [arm, stats] of Object.entries(conditions)) {
    const [lo, hi] = stats.cluster_ci95;
    rows.push(["In pool − held out: call valid", esc(arm), signed(stats.observed), `[${signed(lo)}, ${signed(hi)}]`, stats.excludes_zero ? "yes" : "no"]);
  }
  return table(["Comparison", "Arm", "Delta", "95% CI", "Excludes zero"], rows);
}

function renderLiveConditions(report) {
  const rows = report.eval.per_row || [];
  const conditions = [...new Set(rows.map((row) => row.condition))].sort();
  const arms = ["no_memory", "rrf_typed", "full_recap"];
  const out = [];
  for (const condition of conditions) {
    for (const arm of arms) {
      const block = rows.filter((row) => row.condition === condition && row.arm === arm);
      if (!block.length) continue;
      const mean = (key) => block.reduce((sum, row) => sum + (row[key] ? 1 : 0), 0) / block.length;
      out.push([
        `\`${condition}\``,
        `\`${arm}\``,
        String(block.length),
        num(mean("call_valid")),
        num(mean("selected")),
        num(mean("harmful_apply")),
      ]);
    }
  }
  return table(["Condition", "Arm", "n", "Call valid", "Selected", "Harmful apply"], out);
}

function renderLive(report) {
  if (!report) {
    return "<p class=\"section-note\">No live-model failure report is deployed.</p>";
  }
  const d = report.discovery.summary;
  const f = report.funnel;
  return `
    <p class="section-note">${esc(report.model.model)} at ${esc(report.model.base_url)} · discovery temperature ${num(report.decoding.discovery_temperature)} · eval temperature ${num(report.decoding.eval_temperature)} · seeds ${report.decoding.seeds.join(", ")} · embedder ${esc(report.embedding.model)} (${report.embedding.dimension}-d).</p>
    <p class="section-note">${f.dev_calls} dev calls produced ${d.execution_failures} execution failures (${d.type_failures} declared-type mismatches) and ${d.selection_failures} selection failures. ${f.memories_promoted} memories were recorded, compiled, sandbox-validated, and promoted. ${f.eval_first_calls} held-out model calls were made under ${f.eval_conditions} conditions.</p>
    <h3>Held-out calls where the function's memory was in the pool and the first call failed</h3>
    <div class="table-wrap">${renderLiveArms(report, "safe_failed")}</div>
    <h3>Held-out calls where that memory was excluded and the first call failed</h3>
    <div class="table-wrap">${renderLiveArms(report, "no_safe_failed")}</div>
    <h3>With vs without the function's memory</h3>
    <div class="table-wrap">${renderLiveConditions(report)}</div>
    <h3>Paired cluster bootstrap against the full policy</h3>
    <div class="table-wrap">${renderLivePaired(report)}</div>
    <p class="section-note">Positive call-valid means the full policy repaired more failed calls. Negative harmful-apply means the full policy applied fewer inapplicable memories. Intervals are 95% function-level cluster bootstraps.</p>
  `;
}

async function show(emb) {
  const $ = (s) => document.querySelector(s);
  try {
    const { ret, tr, retV1, trV1, recap, live } = await load(emb);
    $("#provenance").textContent = provenance(ret, tr);
    $("#retrieval-table").innerHTML = renderRetrieval(ret);
    $("#transfer-table").innerHTML = renderTransfer(tr);
    $("#retrieval-note").textContent = `Test split, n=${ret.test_labeled}. RRF k=${ret.rrf_k}. Selection rule: ${ret.selection_rule}.`;
    $("#transfer-note").textContent = `${tr.scenarios} paired scenarios. ${tr.injected_failure}. Success means a schema-valid second call.`;
    $("#recap-body").innerHTML = renderRecap(recap);
    $("#recap-note").textContent = recap
      ? `The only report here that exercises the real record → compile → validate → promote → retrieve stack. ${recap.embedding.model} (${recap.embedding.dimension}-d), generated ${String(recap.generated_at).slice(0, 10)}.`
      : "The real-stack experiment has not been run for this embedder.";
    $("#live-body").innerHTML = renderLive(live);
    $("#live-note").textContent = live
      ? "Real model failures, not injected renames. The local model's own invalid calls become the failure evidence; every repair is compiled and promoted by the real stack."
      : "The live-model failure experiment has not been run.";
    const delta = [];
    if (retV1) delta.push(`<h3>Retrieval, test hit@1</h3>${renderDelta(ret, retV1, "retrieval")}<p class="section-note">${esc(pairedNote(ret))}</p>`);
    if (trV1) delta.push(`<h3>Transfer, task success</h3>${renderDelta(tr, trV1, "transfer")}`);
    $("#delta-tables").innerHTML = delta.join("") || "<p class=\"section-note\">Pre-fix reports are not deployed.</p>";
    const limits = [...ret.limits, ...tr.limits, ...(recap ? recap.limits : []), ...(live ? live.limits : [])];
    $("#limits-list").innerHTML = limits.map((l) => `<li>${esc(l)}</li>`).join("");
  } catch (err) {
    $("#provenance").textContent = `Report data unavailable: ${err.message}. Serve over HTTP and copy docs/evidence/real-data-bfcl-*.json into web/data/.`;
  }
}

if (typeof document !== "undefined") {
  const sel = document.querySelector("#emb");
  sel.addEventListener("change", () => show(sel.value));
  show(sel.value);
}
if (typeof module !== "undefined") module.exports = { renderRetrieval, renderTransfer, provenance, renderDelta, pairedNote, renderRecap, renderRecapPaired, renderLive, renderLiveArms, renderLiveConditions, renderLivePaired };
