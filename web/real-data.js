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
    return [name, withCi(t.hit_at_1, t.hit_at_1_ci95), num(t.hit_at_3), num(t.reciprocal_rank), num(t.auc_vs_irrelevance), num(a.test_control_abstention)];
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

async function load(emb) {
  const get = async (name) => {
    const res = await fetch(`data/${name}-${emb}.json`);
    if (!res.ok) throw new Error(`${name}-${emb}.json: HTTP ${res.status}`);
    return res.json();
  };
  const [ret, tr] = await Promise.all([get("real-data-bfcl"), get("real-data-bfcl-transfer")]);
  return { ret, tr };
}

async function show(emb) {
  const $ = (s) => document.querySelector(s);
  try {
    const { ret, tr } = await load(emb);
    $("#provenance").textContent = provenance(ret, tr);
    $("#retrieval-table").innerHTML = renderRetrieval(ret);
    $("#transfer-table").innerHTML = renderTransfer(tr);
    $("#retrieval-note").textContent = `Test split, n=${ret.test_labeled}. RRF k=${ret.rrf_k}. Selection rule: ${ret.selection_rule}.`;
    $("#transfer-note").textContent = `${tr.scenarios} paired scenarios. ${tr.injected_failure}. Success means a schema-valid second call.`;
    $("#limits-list").innerHTML = [...ret.limits, ...tr.limits].map((l) => `<li>${esc(l)}</li>`).join("");
  } catch (err) {
    $("#provenance").textContent = `Report data unavailable: ${err.message}. Serve over HTTP and copy docs/evidence/real-data-bfcl-*.json into web/data/.`;
  }
}

if (typeof document !== "undefined") {
  const sel = document.querySelector("#emb");
  sel.addEventListener("change", () => show(sel.value));
  show(sel.value);
}
if (typeof module !== "undefined") module.exports = { renderRetrieval, renderTransfer, provenance };
