"""Writes results/REPORT.md — what the lab can and cannot show, claim by claim."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import LabConfig

HOW_TO_READ = """\
**How to read this.** The lab is a simulation with known ground truth. Claims are tagged:
**BY CONSTRUCTION** — the assumptions in `config.py` produce them; the lab shows the mechanism is
coherent and its size *under those assumptions*. **HARNESS** — the comparison method is tested,
but magnitudes come from the assumptions. **MEASURED** — an estimator is graded against ground
truth it cannot see. Only HARNESS and MEASURED claims get a verdict.
"""

LIMITATIONS = """\
- **Synthetic data, simulated people.** Effect sizes depend on the assumptions table below.
- **The look-alike problem is generated, not discovered**: twins share vocabulary and lose their
  distinguishing evidence by design.
- **Unequal inputs**: the classifiers are trained on 300 or 3,000 labelled documents with untuned hyper-parameters;
  the routers get no labels but are built from the generator's own vocabulary — an upper bound on hand-written rules.
- **The twin check reads the generator's own distinguishing vocabulary** — a production evidence list
  would be less complete.
- **No drift**: audit "weeks" are drawn from one static population.
- **Labeller errors**: by default labellers err independently of each other and of how ambiguous a document is.
  One audit cell drops accuracy to 75% on ambiguous look-alikes for all labellers (correlated errors).
- **Gold slice**: 100 items, assumed labelled without error by an expert; rates are pooled across strata.
- **Margin works by family, not by ranking**: within look-alike families no score separates right from
  wrong much better than the others (AUROC table). The margin score helps because it pushes whole
  look-alike families below everything else — the same effect as a per-family threshold.
- **Audit interval**: stratified Wald-type with an Agresti–Coull plug-in variance; conservative at small
  samples. The gold-slice correction does not propagate the gold slice's own uncertainty.
- **The classifiers are deliberately simple.** An LLM classifier's confidence has its own behaviour
  and should be measured the same way.
"""


def write_report(results: dict, claims: list[dict], cfg: LabConfig, out: Path) -> Path:
    L = ["# ClaimFlow lab — validation report for post #1", "", HOW_TO_READ,
         "## Claims", "", "| ID | Claim | Evidence | Verdict | Detail | How the post should say it |",
         "|---|---|---|---|---|---|"]
    for c in claims:
        L.append(f"| {c['id']} | {c['claim']} | {c['evidence']} | **{c['verdict']}** | {c['detail']} | {c['post_wording'] or '—'} |")
    ab = results["e3"]["at_budget"]
    L += ["", "## Errors passed automatically at a review budget", "",
          "Share of ALL documents that are wrong and pass automatically, mean ± sd over 20 seeds. **Exact** = a",
          "randomised gate that holds a random fraction of items at the boundary score so every strategy holds exactly",
          "the budget in expectation. **Deterministic** = best single threshold at or under the budget, with the hold rate it achieves.", "",
          "| Strategy | 20% exact | ⅓ exact | 50% exact | ⅓ deterministic (hold achieved) |", "|---|---|---|---|---|"]
    for name, per in ab.items():
        cells = [f"{100*v['exact']['mean']:.1f}% ± {100*v['exact']['sd']:.1f}" if "exact" in v else
                 f"not reachable ({v['reachable_in']})" for v in per.values()]
        d = per["33%"]
        det = (f"{100*d['deterministic']['mean']:.1f}% ({100*d['deterministic_hold']['mean']:.0f}% held)"
               if "deterministic" in d else "—")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {det} |")
    L += ["", "Paired difference vs one global threshold (exact budget, mean ± 95% CI over 20 seeds):", "",
          "| Strategy | 20% | ⅓ | 50% |", "|---|---|---|---|"]
    for name, per in results["e3"]["paired_vs_global"].items():
        cells = [f"{100*v['mean_diff']:+.1f}pp ± {100*v['ci95']:.1f} ({100*v['relative']:+.0f}%)" if v else "—"
                 for v in (per.get(b) for b in ("20%", "33%", "50%"))]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    dg = results["e3"]["diagnostics"]
    L += ["", "| Classifier | Error rate | ECE | Twin errors at ≥0.9 | AUROC, look-alike families pooled |",
          "|---|---|---|---|---|"]
    for clf, d in dg.items():
        L.append(f"| {clf} | {100*d['error_rate']['mean']:.1f}% | {d['ece']['mean']:.3f} | "
                 f"{100*d['twin_errors_conf_ge_0_9']['mean']:.0f}% | {d['pooled_family_auroc']['mean']:.2f} |")
    L += ["", "## Blind audit: bias by AI error rate, labeller quality and labelling scheme", "",
          "Reviewed items are re-audited blind unless marked trusted. 10 seeds × 20 audits per row.", "",
          "| AI error rate | True error | Labellers | Scheme | Reviewed items | Bias | CI coverage | ±CI |",
          "|---|---|---|---|---|---|---|---|"]
    for a in results["e2"]["audit"]:
        lab = f"{a['labeller_accuracy']:.0%}" + (f", {a['ambiguous_accuracy']:.0%} on ambiguous" if a.get("ambiguous_accuracy") else "")
        L.append(f"| {a['ai_error_rate']} | {100*a['true_error']:.1f}% | {lab} | {a['adjudication']} | "
                 f"{'trusted' if a['trust_reviewers'] else 're-audited'} | {a['bias_pp']:+.1f}pp | {a['coverage']:.0%} | "
                 f"±{a['ci_half_width_pp']:.1f}pp |")
    L += ["", "| Audit weeks (20/bin/week, double-labelled) | ±CI | Coverage | Items labelled |", "|---|---|---|---|"]
    for w, v in results["e2"]["audit_by_weeks"].items():
        L.append(f"| {w} | ±{v['ci_half_width_pp']:.1f}pp | {v['coverage']:.0%} | {v['sampled']} |")
    L += ["", "## Assumptions (defaults)", "", "| Parameter | Value |", "|---|---|"]
    for k, v in asdict(cfg).items():
        L.append(f"| `{k}` | {v} |")
    L += ["", "## Limitations", "", LIMITATIONS, "## Reproduce", "", "```bash",
          "python -m claimflow            # all experiments", "python -m unittest discover -s tests", "```", ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(L))
    return path
