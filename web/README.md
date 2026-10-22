# RECAP replay app

A no-build, static page for exploring recorded RECAP benchmark runs. It never needs a
database, model, or backend. The numbers come from JSON exported from reports, and the
provenance bar states the suite, split, agent mode, exact model, embedder, reranker,
dataset hash, and timeout.

## Serve locally

From the repository root:

```bash
python scripts/export_replay.py \
  --input benchmark/reports/benchmark_test_20261022T061236Z.json \
  --input benchmark/reports/benchmark_dev_20261022T053236Z.json \
  --input benchmark/reports/benchmark_test_20261022T062346Z.json \
  --out web/replay
python -m http.server 8000 --directory web
```

The third input is the live-model run. The exported catalog keeps it selectable but does not
make it the default; the deterministic test artifact remains the headline.

Open <http://localhost:8000>. Serving over HTTP is required: the page uses `fetch()`.
Opening `index.html` from the filesystem will show the explicit “replay data unavailable”
message instead of silent empty charts.

## Deploy

`web/` is plain HTML/CSS/JS plus `web/replay/*.json`. Any static host works. The repository
includes a GitHub Pages workflow that publishes the directory on pushes to `main`.
No build step and no credentials are required.

If the site is deployed under a subpath, keep the relative URLs intact. The app only fetches
`replay/benchmark.json`, `replay/datasets.json`, and `replay/dataset-N.json`.

## What the page shows

- a contrast task where relevance-only retrieval fails and the policy succeeds;
- v3 tier definitions and the no-memory acceptance gate;
- a silent-drift comparison chosen for contrast, not the first row;
- success, harmful-transfer, abstention, and per-family tables from the recorded report;
- a retrieval inspector with rejected candidates and reasons;
- a run timeline and a **replay-only** question panel that cites report fields;
- a dataset selector that keeps the deterministic and live-model artifacts visibly separate.

The question panel is deterministic code, not an LLM. It answers only from the loaded
report and shows the field it read. It does not invent a number when the field is absent.

## Honesty rules

- Every number is copied from a report; nothing is smoothed or averaged in the browser.
- Deterministic runs are labelled “deterministic replay (not live-model evidence)”.
- Live runs record and display the exact model and timeout. A live run does not make the
  deterministic artifact live.
- Missing fields render as `Not in artifact` or `-`, never as a fabricated value.
- `no_safe_memory` is not relabelled as task success. Safe abstention is reported separately.
- A live model can select the correct gold memory and still fail to execute it. The report
  records that failed outcome as harmful transfer; the page does not turn it into a
  wrong-memory claim.
