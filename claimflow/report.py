"""Writes results/REPORT.md — claims, verdicts, evidence, assumptions, limitations."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .claims import frontier
from .config import LabConfig

LIMITATIONS = """\
- **Synthetic data, simulated people.** The lab validates *mechanisms and estimators* — that an effect
  exists, which way it points and what drives it. It does not tell you how big the effect is in your
  system. Every rate here moves with the assumptions in the table above (see the sensitivity grid).
- **The classifiers are deliberately simple** (keyword router, numpy logistic regression). An LLM
  classifier's confidence (e.g. verbalised confidence or token log-probs) has its own calibration
  behaviour and should be measured the same way before trusting a threshold.
- **The audit assumes blind labellers** who never see the AI's answer. A non-blind audit inherits anchoring bias.
- **Ground truth (`oracle_truth`) exists only in the lab.** Production systems must estimate it — which
  is what the audit is for.
"""


def write_report(results: dict, claims: list[dict], cfg: LabConfig, out: Path) -> Path:
    fr = frontier(results["e3"])
    lines = [
        "# ClaimFlow lab — validation report for post #1",
        "",
        "Claims from *“Your AI's Override Rate Is Lying to You”*, tested against synthetic insurance-claims data",
        "where the true label of every document is known.",
        "",
        "## Verdicts",
        "",
        "| ID | Claim | Verdict | Evidence | Change to the post |",
        "|---|---|---|---|---|",
    ]
    for c in claims:
        lines.append(f"| {c['id']} | {c['claim']} | **{c['verdict']}** | {c['evidence']} | {c['post_change'] or '—'} |")
    lines += ["", "## Errors that pass automatically, at a fixed human-review budget", "",
              "| Strategy | 20% held | 33% held | 50% held |", "|---|---|---|---|"]
    def cell(x: float) -> str:
        return "not reachable¹" if x == float("inf") else f"{x:.1%}"

    for name, v in fr.items():
        lines.append(f"| {name} | {cell(v['20%'])} | {cell(v['33%'])} | {cell(v['50%'])} |")
    lines += ["", "¹ This strategy always holds more than this share — e.g. every unverifiable twin goes to a human."]
    a = results["e2"]["audit_by_weeks"]
    lines += ["", "## Audit precision by weeks of sampling (20 docs per confidence bin per week)", "",
              "| Weeks | 95% CI half-width | CI coverage |", "|---|---|---|"]
    for w, v in a.items():
        lines.append(f"| {w} | ±{v['ci_half_width_pp']:.1f}pp | {v['coverage']:.0%} |")
    lines += ["", "## Assumptions (defaults)", "", "| Parameter | Value |", "|---|---|"]
    for k, v in asdict(cfg).items():
        lines.append(f"| `{k}` | {v} |")
    lines += ["", "## Limitations", "", LIMITATIONS,
              "## Reproduce", "", "```bash", "python -m claimflow            # all experiments, ~2 min",
              "python -m unittest discover -s tests", "```", ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(lines))
    return path
