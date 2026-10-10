# ClaimFlow lab

A small, reproducible reference implementation for **measuring human-in-the-loop AI systems**.
It simulates an insurance-claims document triage agent, the adjusters who review its work, and the
event log in between — with **synthetic data where the true answer is always known**, so every
metric and estimator can be graded against the truth.

It accompanies the post *"Your AI's Override Rate Is Lying to You"*. Be clear about what it can show:

- **Mechanisms are encoded, not discovered.** Look-alike documents, reviewers who open what looks wrong,
  and labellers who make mistakes are *assumptions* in `config.py`. Claims that follow from them are tagged
  **BY CONSTRUCTION** — the lab shows they're coherent and how big they get under those assumptions.
- **Estimators are tested.** The lower bound and the blind stratified audit are graded against ground truth
  they can't see — tagged **MEASURED**.
- **Comparison harnesses are tested, magnitudes are not.** The workload-vs-errors comparison and the
  ID-vs-filename join are sound methods, but their numbers come from the generator — tagged **HARNESS**.

→ **[results/REPORT.md](results/REPORT.md)** for every claim, its evidence type, verdict and limitations.

## What's inside

```
claimflow/
  config.py      every assumption, in one dataclass
  data.py        synthetic claim documents — incl. look-alike "twins" and collision-prone filenames
  triage.py      keyword router (match-strength or margin confidence), logistic classifier, twin check
  humans.py      reviewer / labeller error model (errors on look-alikes go to the twin)
  events.py      append-only decision log (SQLite, ID-keyed, UPDATE/DELETE blocked by triggers)
  review.py      simulated adjusters on a timeline, three UIs, paired randomness across UIs
  metrics.py     the post's SQL: override split, filename-vs-ID pairing, error bounds, confusion
  audit.py       blind stratified audit: 4 labelling schemes incl. gold-slice Rogan–Gladen correction,
                 trust-vs-reaudit reviewers, stratified Wald interval with Agresti–Coull variance
  experiments.py E1–E4: paired UI test; bounds grid + audit bias across AI error rates; gate comparison at
                 exact budgets over 20 seeds (incl. per-family threshold, label budgets); join test
  claims.py      each claim tagged BY CONSTRUCTION / HARNESS / MEASURED; only the latter two get verdicts
tests/           unit + end-to-end tests (stdlib unittest)
results/         REPORT.md, results.json, charts/
```

## Run

```bash
pip install numpy matplotlib          # Python ≥ 3.10
python -m claimflow                   # full run, ~5–10 minutes → results/
python -m claimflow --quick           # smoke run
python -m unittest discover -s tests
```

## Design choices worth stealing

- **Every AI decision has an ID; every human action references it.** Metrics are plain SQL over one table.
- **Human actions are typed** — `corrected`, `repick`, `bulk_set`, `accepted` — so UI noise is separable from model error.
- **The oracle is quarantined.** Ground truth lives in `oracle_truth`, used only to grade estimators — the
  production-shaped metrics never read it.
- **Verdicts are computed, not written.** Change an assumption in `config.py` and the report can flip a claim.
- **Comparisons hold workload equal** — a randomised gate hits each budget exactly in expectation; the best deterministic
  threshold and the hold rate it actually achieves are reported alongside. Mean ± sd over 20 seeds.

## Honest limits

Synthetic data and simulated people can test *measurement methods*; they can't tell you how big an effect
is in your system. The classifiers get labelled data the rules don't, while the rules and the twin check are
built from the generator's own vocabulary (an upper bound on hand-written rules). Labellers err independently
unless the ambiguity-correlated audit cell is used. See the Limitations section of the report.
