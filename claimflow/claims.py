"""Post #1's claims, each evaluated against lab results.
Verdicts: PASS (holds under default assumptions and the sensitivity grid),
QUALIFIED (holds only under stated conditions — the post must say so), FAIL."""
from __future__ import annotations

import numpy as np

STRATEGIES = {
    "router: raise threshold": ("keyword_router", "sweep"),
    "router: + twin verifier": ("keyword_router", "sweep_with_verifier"),
    "router: margin confidence": ("keyword_router_margin", "sweep"),
    "calibrated classifier": ("prob_classifier", "sweep"),
}


def error_at_budget(points: list[dict], budget: float) -> float:
    """Lowest auto-passed error rate reachable while holding at most `budget` of docs
    for human review (operating points from a threshold sweep, linearly interpolated)."""
    pts = sorted({(p["hold_rate"], p["auto_passed_error_rate"]) for p in points})
    feasible = [e for h, e in pts if h <= budget]
    best = min(feasible) if feasible else float("inf")
    for (h0, e0), (h1, e1) in zip(pts, pts[1:]):
        if h0 <= budget < h1:
            best = min(best, e0 + (e1 - e0) * (budget - h0) / (h1 - h0))
    return best


def frontier(e3: dict, budgets=(0.2, 0.33, 0.5)) -> dict:
    return {name: {f"{int(b*100)}%": error_at_budget(e3[clf][key], b) for b in budgets}
            for name, (clf, key) in STRATEGIES.items()}


def evaluate(r: dict) -> list[dict]:
    e1, e2, e3, e4, sens = r["e1"], r["e2"], r["e3"], r["e4"], r["sensitivity"]["grid"]
    pick, click = e1["base"]["pick_to_accept"], e1["base"]["one_click_accept"]
    seeds = e1["across_seeds"]["pick_to_accept"]
    cai_range = (min(g["confirm_as_is_share"] for g in sens), max(g["confirm_as_is_share"] for g in sens))
    b = e2["bounds"]
    adj, single = e2["audit"]["adjudicated"], e2["audit"]["single_label"]
    rt, pr = e3["keyword_router"]["at_default_threshold"], e3["prob_classifier"]["at_default_threshold"]
    fr = frontier(e3)
    grid = e4["grid"]
    out = []

    def claim(cid, text, verdict, evidence, post_change=""):
        out.append({"id": cid, "claim": text, "verdict": verdict, "evidence": evidence, "post_change": post_change})

    claim("C1a", "Naive override rate overstates the true correction rate when accepting requires a re-pick.",
          "PASS" if pick["naive_override_rate"] > pick["true_correction_rate"] * 1.2 else "FAIL",
          f"pick-to-accept: override rate {pick['naive_override_rate']:.1%} vs true corrections "
          f"{pick['true_correction_rate']:.1%} (20 seeds: confirm-as-is {seeds['confirm_as_is_share']['mean']:.1%} "
          f"± {seeds['confirm_as_is_share']['sd']:.1%}).")
    claim("C1b", "'A third to half of overrides are confirm-as-is.'",
          "PASS" if 0.33 <= cai_range[0] and cai_range[1] <= 0.5 else "QUALIFIED",
          f"Lab default {pick['confirm_as_is_share']:.1%}; across the sensitivity grid "
          f"{cai_range[0]:.0%}–{cai_range[1]:.0%}. The share depends on hold rate and how often correct items are opened.",
          "Keep 'a third to half' as a production observation; add that the size depends on hold rate and review habits.")
    claim("C1c", "One-click accept removes confirm-as-is noise.",
          "PASS" if click["confirm_as_is"] == 0 else "FAIL",
          f"one-click-accept: confirm-as-is {click['confirm_as_is']} of {click['overrides']} overrides.")
    sc, sp = e1["across_seeds"]["one_click_accept"]["true_correction_rate"], seeds["true_correction_rate"]
    claim("C1d", "NEW — bulk 'set category' hides real errors, not just adds noise.",
          "PASS" if sc["mean"] - sp["mean"] > 2 * max(sc["sd"], sp["sd"]) / np.sqrt(len(sens)) else "FAIL",
          f"true corrections captured (mean over seeds): one-click {sc['mean']:.1%} vs pick-to-accept with "
          f"bulk-set {sp['mean']:.1%} — bulk-set stamped wrong labels as human-confirmed.",
          "Add: bulk-set doesn't just inflate overrides, it stamps wrong labels as human-confirmed.")
    claim("C2a", "The reviewed subset overstates error (people open what looks wrong).",
          "PASS" if b["reviewed_subset_error"] > b["oracle_true_error"] else "FAIL",
          f"reviewed-subset error {b['reviewed_subset_error']:.1%} vs true error {b['oracle_true_error']:.1%}.")
    claim("C2b", "corrections ÷ all drafts is a lower bound.",
          "PASS" if all(g["lower_bound"] <= g["oracle"] for g in sens) else "FAIL",
          f"default {b['lower_bound']:.1%} ≤ true {b['oracle_true_error']:.1%}; holds in all {len(sens)} sensitivity cells.")
    claim("C2c", "A blind stratified audit (20 per bin per week) recovers the true error.",
          "PASS" if adj["coverage_across_seeds"] >= 0.88 else "QUALIFIED",
          f"with adjudication: 95% CI covers truth {adj['coverage_across_seeds']:.0%} of runs, bias "
          f"{adj['bias_pp']:+.1f}pp, ±{adj['ci_half_width_pp']:.1f}pp after 4 weeks. Single-label audit: bias "
          f"{single['bias_pp']:+.1f}pp, coverage {single['coverage_across_seeds']:.0%}.",
          "Add: audit labeller mistakes bias the estimate upward — send disagreements to a second blind labeller.")
    hi_rt, hi_pr = rt["share_of_twin_errors_with_conf_ge_0_9"], pr["share_of_twin_errors_with_conf_ge_0_9"]
    claim("C3a", "On look-alike twins the classifier is confidently wrong.",
          "PASS" if min(hi_rt, hi_pr) >= 0.3 else ("QUALIFIED" if max(hi_rt, hi_pr) >= 0.3 else "FAIL"),
          f"match-strength router: {rt['share_of_twin_errors_with_conf_ge_0_9']:.0%} of twin errors at ≥0.9; "
          f"calibrated classifier: {pr['share_of_twin_errors_with_conf_ge_0_9']:.0%}. Twin pairs are "
          f"{rt['twin_pair_error_share']:.0%} of errors in both.",
          "Rewrite: true when confidence is a match-strength score; a calibrated or margin-based score sees twins.")
    thr_worst = all(fr["router: raise threshold"]["33%"] >= v["33%"] for v in fr.values())
    calib_fixes = fr["calibrated classifier"]["33%"] < fr["router: + twin verifier"]["33%"]
    claim("C3b", "Raising the global threshold is an expensive way to catch twin errors.",
          "QUALIFIED" if thr_worst and calib_fixes else ("PASS" if thr_worst else "FAIL"),
          "Auto-passed error at a 33% human-review budget: " +
          "; ".join(f"{k} {v['33%']:.1%}" for k, v in fr.items()) + ".",
          "Rewrite: a threshold only works if confidence is computed against the runner-up.")
    claim("C3c", "A twin-aware verification step beats raising the threshold.",
          "PASS" if fr["router: + twin verifier"]["33%"] < fr["router: raise threshold"]["33%"] else "FAIL",
          f"at 33% review budget: verifier {fr['router: + twin verifier']['33%']:.1%} vs threshold "
          f"{fr['router: raise threshold']['33%']:.1%}; but margin confidence ({fr['router: margin confidence']['33%']:.1%}) "
          f"beats both.",
          "Lead with 'fix the confidence first', verification second.")
    worst = max(grid, key=lambda g: g["filename_ambiguous_share"])
    claim("C4", "Joining overrides to drafts by filename loses and misattributes ground truth.",
          "PASS" if e4["base"]["filename_ambiguous_share"] > 0 and e4["base"]["id_join_coverage"] == 1 else "FAIL",
          f"default: {e4['base']['filename_ambiguous_share']:.0%} ambiguous, latest-draft-wins picks the wrong draft "
          f"{e4['base']['filename_latest_wins_wrong_share']:.0%} of the time; up to "
          f"{worst['filename_ambiguous_share']:.0%} ambiguous at {worst['generic_filename_share']:.0%} generic names / "
          f"{worst['docs_per_tenant']} docs per tenant. ID join: {e4['base']['id_join_coverage']:.0%}.",
          "Add: the 'latest draft wins' fallback is worse than ambiguity — it is silently wrong.")
    return out
