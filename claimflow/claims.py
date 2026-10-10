"""Post #1's claims, each tagged with what kind of evidence the lab can give:

  BY CONSTRUCTION — the simulation's assumptions produce it; the lab shows the mechanism is
                    coherent and its size *under those assumptions*. No verdict.
  HARNESS         — the comparison method is tested; the magnitudes come from assumptions.
  MEASURED        — an estimator is graded against ground truth it cannot see.

Only HARNESS and MEASURED claims get PASS / QUALIFIED / FAIL.
"""
from __future__ import annotations

BY_CONSTRUCTION, HARNESS, MEASURED = "BY CONSTRUCTION", "HARNESS", "MEASURED"


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def evaluate(r: dict) -> list[dict]:
    e1, e2, e3, e4 = r["e1"], r["e2"], r["e3"], r["e4"]
    out: list[dict] = []

    def claim(cid, text, evidence, verdict, detail, post=""):
        out.append({"id": cid, "claim": text, "evidence": evidence, "verdict": verdict,
                    "detail": detail, "post_wording": post})

    pick = e1["across_seeds"]["pick_to_accept"]
    nac = pick["not_a_correction_share"]
    claim("C1a", "When accepting requires a re-pick, the override rate misstates real corrections.",
          BY_CONSTRUCTION, "—",
          f"pick-to-accept, 20 seeds: override rate {_pct(pick['naive_override_rate']['mean'])}, real corrections "
          f"{_pct(pick['true_correction_rate']['mean'])}; {_pct(nac['mean'],0)} of overrides changed nothing "
          f"(range {_pct(nac['min'],0)}–{_pct(nac['max'],0)}); re-picks alone were "
          f"{_pct(pick['confirm_as_is_share']['mean'],0)} of all overrides. Representative (median) run: seed {e1['representative_seed']}.",
          "Production observation; the lab only illustrates the mechanism. Quote the mean, not one run.")
    g = e2["grid"]
    claim("C1b", "The share of overrides that aren't corrections depends on hold rate, bulk use and review habits.",
          BY_CONSTRUCTION, "—",
          f"Across {len(g)} assumption cells: {_pct(min(x['not_a_correction_share'] for x in g),0)}–"
          f"{_pct(max(x['not_a_correction_share'] for x in g),0)}.", "Measure yours.")
    pb = e1["paired_bulk_effect"]
    claim("C1c", "Bulk 'set category' loses real corrections (stamps AI errors as human-confirmed).",
          BY_CONSTRUCTION, "—",
          f"Paired over {pb['n']} seeds: +{100*pb['mean_diff']:.2f}pp of documents corrected without bulk "
          f"(t = {pb['t']:.1f}). Direction is built into the bulk model; size depends on bulk usage.",
          "Mechanism, not a measured effect size.")
    b = e2["bounds"]
    viol = [x for x in g if x["lower_bound"] > x["oracle_true_error"]]
    gap = min(x["oracle_true_error"] - x["lower_bound"] for x in g)
    worst = min(g, key=lambda x: x["oracle_true_error"] - x["lower_bound"])
    claim("C2a", "corrections ÷ all suggestions is a lower bound on true error (if reviewers are right).",
          MEASURED, "PASS" if not viol else "QUALIFIED",
          f"Holds in {len(g)-len(viol)}/{len(g)} cells: reviewer accuracy 85–97%, wrong items opened 15–80%, "
          f"correct items opened 2–40%, threshold 0.80/0.90, bulk use 0/20%. Tightest cell: gap {100*gap:.1f}pp (reviewers {worst['reviewer_accuracy']:.0%}, "
          f"open-if-wrong {worst['p_open_if_wrong']:.0%}, open-if-right {worst['p_open_if_right']:.0%}). "
          f"Default: {_pct(b['lower_bound'])} vs true {_pct(b['oracle_true_error'])}.",
          "Lower bound if reviewers' corrections are right; can fail with weak reviewers at very high review coverage.")
    over = sum(x["reviewed_subset_error"] > x["oracle_true_error"] for x in g)
    claim("C2b", "Reviewed-subset error is biased; it is not a bound.",
          BY_CONSTRUCTION, "—",
          f"Overstates in {over}/{len(g)} cells, because every cell triggers review by low confidence or breakage. "
          f"Review driven by value or random QA can understate.", "Never call it an upper bound.")
    aud = e2["audit"]

    def row(rate, lab, adj, trust=False):
        return next(a for a in aud if a["ai_error_rate"].startswith(rate) and a["labeller_accuracy"] == lab
                    and a["adjudication"] == adj and a["trust_reviewers"] == trust and not a.get("ambiguous_accuracy"))
    good = [row(rt, 0.98, "double_all") for rt in ("low", "default", "high")]
    amb = {a["adjudication"]: a for a in aud if a.get("ambiguous_accuracy")}
    weak = {adj: [row(rt, 0.90, adj) for rt in ("low", "default", "high")]
            for adj in ("none", "on_disagreement", "double_all", "gold_corrected")}

    def span(rows, key="bias_pp"):
        return f"{min(x[key] for x in rows):+.1f} to {max(x[key] for x in rows):+.1f}pp"
    claim("C2c", "A blind stratified audit recovers the true error rate.",
          MEASURED, "QUALIFIED",
          f"98% labellers, double-labelled, reviewed items re-audited blind: bias {span(good)} across AI error rates "
          f"of ~4/19/37%, coverage {min(x['coverage'] for x in good):.0%}–{max(x['coverage'] for x in good):.0%}. "
          f"90% labellers: single label {span(weak['none'])}; second label on disagreement {span(weak['on_disagreement'])} "
          f"(sign flips with the base rate); double-label everything {span(weak['double_all'])}; gold-slice "
          f"Rogan–Gladen correction {span(weak['gold_corrected'])}, coverage "
          f"{min(x['coverage'] for x in weak['gold_corrected']):.0%}–{max(x['coverage'] for x in weak['gold_corrected']):.0%} "
          f"(gold-slice uncertainty not propagated). If labellers drop to 75% on ambiguous look-alikes (errors correlated "
          f"across labellers): double-labelled bias {amb['double_all']['bias_pp']:+.1f}pp, coverage {amb['double_all']['coverage']:.0%}; "
          f"gold-corrected {amb['gold_corrected']['bias_pp']:+.1f}pp.",
          "Recovers it with good blind labellers. With weaker labellers every shortcut is biased in a base-rate-dependent "
          "way: improve labellers and measure them against a gold slice.")
    dg = e3["diagnostics"]
    claim("C3a", "A match-strength score is confidently wrong on look-alikes.",
          BY_CONSTRUCTION, "—",
          f"Twin errors at ≥0.9: match-strength {_pct(dg['keyword_router']['twin_errors_conf_ge_0_9']['mean'],0)}, "
          f"margin {_pct(dg['keyword_router_margin']['twin_errors_conf_ge_0_9']['mean'],0)} (by arithmetic: margin can't "
          f"exceed {dg['keyword_router_margin']['max_family_conf']['max']:.2f} on look-alike families), classifier "
          f"{_pct(dg['prob_classifier']['twin_errors_conf_ge_0_9']['mean'],0)}. Within look-alike families (pooled) no score "
          f"separates right from wrong much: AUROC match-strength {dg['keyword_router']['pooled_family_auroc']['mean']:.2f}, "
          f"margin {dg['keyword_router_margin']['pooled_family_auroc']['mean']:.2f}, classifier "
          f"{dg['prob_classifier']['pooled_family_auroc']['mean']:.2f}.",
          "Don't claim margin 'ranks twins better' — it works by pushing whole look-alike families below the rest.")
    ab = e3["at_budget"]

    def at(name, b):
        x = ab[name][b]
        return f"{_pct(x['exact']['mean'])}" if "exact" in x else "n/a"
    names = list(ab)
    lines = "; ".join(f"{n}: " + "/".join(at(n, b) for b in ("20%", "33%", "50%")) for n in names)
    base33 = ab["match-strength, one global threshold"]["33%"]["exact"]["mean"]
    better = [n for n in names if "exact" in ab[n]["33%"] and ab[n]["33%"]["exact"]["mean"] < base33]
    claim("C3b", "At equal workload, treating look-alike families differently beats one global threshold.",
          HARNESS, "PASS" if len(better) == len(names) - 1 else "QUALIFIED",
          f"Errors passed automatically at 20/33/50% held, exactly in expectation (randomised gate at the boundary), mean of 20 seeds — "
          f"{lines}. The ranking changes with the budget; magnitudes come from the generator. The rules are built from the "
          f"generator's own vocabulary (an upper bound on hand-written rules); the classifiers are untuned.",
          "Show all budgets; say the ranking depends on budget; disclose label budgets.")
    e4b = e4["across_seeds"]
    claim("C4", "Joining overrides to suggestions by filename loses and misattributes ground truth.",
          HARNESS, "PASS" if e4b["filename_ambiguous_share"]["mean"] > 0 else "FAIL",
          f"Interleaved timeline, 10 seeds: {_pct(e4b['filename_ambiguous_share']['mean'],0)} ambiguous; latest-draft-wins "
          f"wrong {_pct(e4b['filename_latest_wins_wrong_share']['mean'],0)} (grid "
          f"{_pct(min(x['filename_latest_wins_wrong_share'] for x in e4['grid']),0)}–"
          f"{_pct(max(x['filename_latest_wins_wrong_share'] for x in e4['grid']),0)}). Magnitudes depend on the assumed "
          f"filename collision rate; the direction holds whenever names collide. ID join: 100%.",
          "Quote as a range; the size is an assumption.")
    return out
