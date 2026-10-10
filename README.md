# ClaimFlow lab

A small, reproducible reference implementation for **measuring human-in-the-loop AI systems**.
It simulates an insurance-claims document triage agent, the adjusters who review its work, and the
event log in between — with **synthetic data where the true answer is always known**, so every
metric and estimator can be graded against the truth.

It accompanies the post *"Your AI's Override Rate Is Lying to You"* and tests each of its claims.
→ **[results/REPORT.md](results/REPORT.md)** for verdicts, evidence and limitations.

## What's inside

```
claimflow/
  config.py      every assumption, in one dataclass
  data.py        synthetic claim documents — incl. look-alike "twins" and collision-prone filenames
  triage.py      keyword router (match-strength or margin confidence), calibrated classifier, twin verifier
  events.py      append-only decision log (SQLite, ID-keyed, UPDATE/DELETE blocked by triggers)
  review.py      simulated adjusters under two UIs: pick-to-accept vs one-click accept
  metrics.py     the post's SQL: override split, filename-vs-ID pairing, error bounds, confusion
  audit.py       blind stratified audit with optional adjudication + 95% CI
  experiments.py E1–E4 + a 54-cell sensitivity grid
  claims.py      each claim → PASS / QUALIFIED / FAIL, computed from results
tests/           unit + end-to-end tests (stdlib unittest)
results/         REPORT.md, results.json, charts/
```

## Run

```bash
pip install numpy matplotlib          # Python ≥ 3.10
python -m claimflow                   # full run, a few minutes → results/
python -m claimflow --quick           # smoke run
python -m unittest discover -s tests
```

## Design choices worth stealing

- **Every AI decision has an ID; every human action references it.** Metrics are plain SQL over one table.
- **Human actions are typed** — `corrected`, `repick`, `bulk_set`, `accepted` — so UI noise is separable from model error.
- **The oracle is quarantined.** Ground truth lives in `oracle_truth`, used only to grade estimators — the
  production-shaped metrics never read it.
- **Verdicts are computed, not written.** Change an assumption in `config.py` and the report can flip a claim.

## Honest limits

Synthetic data and simulated people validate *mechanisms and estimators*, not the size of an effect in
your system. See the Limitations section of the report.
