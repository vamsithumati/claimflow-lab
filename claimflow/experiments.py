"""Experiments. Each returns plain dicts (JSON-serialisable).

E1  override split under three UIs, paired over seeds
E2  error bounds + blind audit under different labeller/reviewer error models
E3  gate operating points — every distinct threshold, real points only, 20 seeds
E4  filename vs ID join on an interleaved timeline
"""
from __future__ import annotations

import itertools

import numpy as np

from . import metrics as m
from .audit import ADJUDICATION, run_audit
from .config import LabConfig
from .data import TWIN_OF, make_corpus
from .pipeline import classify_corpus, run_scenario
from .triage import gate

def _stats(xs) -> dict:
    a = np.asarray(list(xs), dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def _paired(a, b) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return {"mean_diff": float(d.mean()), "sd": float(sd),
            "t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("inf") * np.sign(d.mean() or 1),
            "n": int(len(d))}


# ---------------------------------------------------------------------- E1
def e1_override_split(cfg: LabConfig, seeds: int = 20) -> dict:
    uis = ("pick_to_accept", "pick_to_accept_no_bulk", "one_click_accept")
    per = {ui: [] for ui in uis}
    for s in range(seeds):
        c = cfg.with_(seed=s)
        for ui in uis:
            per[ui].append(m.override_split(run_scenario(c, ui=ui).log))
    keys = ("naive_override_rate", "true_correction_rate", "confirm_as_is_share", "not_a_correction_share")
    shares = [r["not_a_correction_share"] for r in per["pick_to_accept"]]
    rep_seed = int(np.argsort(shares)[len(shares) // 2])          # median run, not the most dramatic one
    return {
        "representative_seed": rep_seed,
        "representative": {ui: per[ui][rep_seed] for ui in uis},
        "base": {ui: m.override_split(run_scenario(cfg, ui=ui).log) for ui in uis},
        "confusion": m.correction_confusion(run_scenario(cfg, ui="one_click_accept").log),
        "across_seeds": {ui: {k: _stats(r[k] for r in per[ui]) for k in keys} for ui in uis},
        "paired_bulk_effect": _paired([r["true_correction_rate"] for r in per["pick_to_accept_no_bulk"]],
                                      [r["true_correction_rate"] for r in per["pick_to_accept"]]),
        "paired_repick_vs_oneclick": _paired([r["true_correction_rate"] for r in per["one_click_accept"]],
                                             [r["true_correction_rate"] for r in per["pick_to_accept_no_bulk"]]),
    }


# ---------------------------------------------------------------------- E2
def e2_bounds_and_audit(cfg: LabConfig, audits: int = 100, seeds: int = 10, per_seed: int = 20) -> dict:
    out: dict = {"bounds": m.error_bounds(run_scenario(cfg).log), "grid": [], "audit": [], "audit_by_weeks": {}}

    # does the lower bound survive noisy reviewers? does reviewed-subset always overstate?
    for acc, p_wrong, p_right, thr, bulk in itertools.product(
            (0.85, 0.90, 0.97), (0.15, 0.35, 0.80), (0.02, 0.10, 0.40), (0.80, 0.90), (0.0, 0.2)):
        c = cfg.with_(reviewer_accuracy=acc, p_open_if_wrong=p_wrong, p_open_if_right=p_right,
                      threshold=thr, p_bulk_batch=bulk)
        r = run_scenario(c)
        b, s = m.error_bounds(r.log), m.override_split(r.log)
        out["grid"].append({"reviewer_accuracy": acc, "p_open_if_wrong": p_wrong, "p_open_if_right": p_right,
                            "threshold": thr, "p_bulk_batch": bulk, **b,
                            "confirm_as_is_share": s["confirm_as_is_share"],
                            "not_a_correction_share": s["not_a_correction_share"]})

    # audit: does each labelling scheme stay unbiased as the AI's error rate changes?
    base_rates = {"low (~4%)": dict(p_distinct_twin=0.97, p_cross_token=0.03),
                  "default (~19%)": {},
                  "high (~37%)": dict(p_distinct_twin=0.40, p_cross_token=0.50)}
    runs = {}
    for name, kw in base_rates.items():
        for s_ in range(seeds):
            r = run_scenario(cfg.with_(seed=100 + s_, **kw))
            runs[(name, s_)] = (r, m.error_bounds(r.log)["oracle_true_error"])
    for name, lab_acc, adj, trust in itertools.chain(
            itertools.product(base_rates, (0.98, 0.90), ADJUDICATION, (False,)),
            [("default (~19%)", 0.98, "double_all", True)],
            [("default (~19%)", "0.98, 0.75 on ambiguous", adj, False) for adj in ADJUDICATION]):
        amb = None
        if isinstance(lab_acc, str):
            lab_acc, amb = 0.98, 0.75
        c = cfg.with_(audit_labeller_accuracy=lab_acc, audit_labeller_ambiguous_accuracy=amb, **base_rates[name])
        cov, ests, hws, truths = [], [], [], []
        for s_ in range(seeds):
            r, truth = runs[(name, s_)]
            truths.append(truth)
            for i in range(per_seed):
                a_ = run_audit(r, c, np.random.default_rng(10_000 * s_ + i), adj, trust)
                cov.append(a_.covers(truth))
                ests.append(a_.estimate - truth)
                hws.append((a_.ci_high - a_.ci_low) / 2)
        out["audit"].append({"ai_error_rate": name, "true_error": float(np.mean(truths)),
                             "labeller_accuracy": lab_acc, "ambiguous_accuracy": amb,
                             "adjudication": adj, "trust_reviewers": trust,
                             "bias_pp": 100 * float(np.mean(ests)), "coverage": float(np.mean(cov)),
                             "ci_half_width_pp": 100 * float(np.mean(hws))})

    run = run_scenario(cfg)
    truth = m.error_bounds(run.log)["oracle_true_error"]
    head = [run_audit(run, cfg, np.random.default_rng(i), "double_all", trust_reviewers=False) for i in range(audits)]
    out["headline_audit"] = {"estimate": float(np.mean([a.estimate for a in head])),
                             "ci_half_width_pp": 100 * float(np.mean([(a.ci_high - a.ci_low) / 2 for a in head])),
                             "coverage": float(np.mean([a.covers(truth) for a in head])), "truth": truth}
    for w in (1, 2, 4, 8):
        res = [run_audit(run, cfg.with_(audit_weeks=w), np.random.default_rng(i), "double_all", trust_reviewers=False)
               for i in range(audits)]
        out["audit_by_weeks"][w] = {"ci_half_width_pp": 100 * float(np.mean([(a.ci_high - a.ci_low) / 2 for a in res])),
                                    "coverage": float(np.mean([a.covers(truth) for a in res])),
                                    "sampled": res[0].sampled}
    return out


# ---------------------------------------------------------------------- E3
STRATEGIES = {
    "match-strength, one global threshold": ("keyword_router", "off", "global"),
    "match-strength, threshold on look-alike families only": ("keyword_router", "off", "family"),
    "match-strength + look-alike evidence check": ("keyword_router", "hold", "global"),
    "margin score, one global threshold": ("keyword_router_margin", "off", "global"),
    "classifier, 300 labels": ("prob_classifier_300", "off", "global"),
    "classifier, 3,000 labels": ("prob_classifier", "off", "global"),
}
BUDGETS = (0.20, 1 / 3, 0.50)


def _bkey(b: float) -> str:
    return f"{round(b * 100)}%"


def operating_points(classified, verify: str, scope: str = "global") -> list[dict]:
    """Every distinct threshold → (share held, share of ALL docs that are errors passed automatically).
    scope='family': the threshold only applies to documents labelled as a look-alike family member."""
    rows = []
    for d, label, conf in classified:
        g = gate(label, conf, d, -1.0, verify, "x")          # threshold −1: only the evidence check can hold
        in_scope = scope == "global" or g.label in TWIN_OF
        rows.append((conf if in_scope else np.inf, g.held, g.label != d.true_label))
    conf = np.array([r[0] for r in rows]); vheld = np.array([r[1] for r in rows]); wrong = np.array([r[2] for r in rows])
    n = len(rows)
    pts = []
    for t in np.unique(np.concatenate([conf[np.isfinite(conf)], [np.inf]])):
        held = vheld | (conf < t)
        pts.append({"threshold": float(t) if np.isfinite(t) else None, "hold_rate": float(held.mean()),
                    "auto_passed_error_rate": float((~held & wrong).sum() / n)})
    return pts


def at_budget(points: list[dict], budget: float) -> dict | None:
    """Hit the budget exactly — in expectation — with a randomised gate: at the boundary score, hold a
    random fraction of items (deployable; the fraction is set from past traffic). Linear interpolation
    between adjacent thresholds is exact for this policy. Also returns the best deterministic point
    at or under the budget and the hold rate it achieves."""
    pts = sorted({(p["hold_rate"], p["auto_passed_error_rate"]) for p in points})
    under = [(h, e) for h, e in pts if h <= budget + 1e-12]
    over = [(h, e) for h, e in pts if h > budget + 1e-12]
    if not under:
        return None
    hd, ed = min(under, key=lambda x: (x[1], -x[0]))
    exact = ed
    if over:
        h0, e0 = max(under)
        h1, e1 = min(over)
        exact = min(ed, e0 + (e1 - e0) * (budget - h0) / (h1 - h0))
    return {"randomised_exact": exact, "deterministic": ed, "deterministic_hold": hd}


def _auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def ece(classified, bins: int = 10) -> float:
    conf = np.array([c for _, _, c in classified]); ok = np.array([lab == d.true_label for d, lab, _ in classified])
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if mask.any():
            e += mask.mean() * abs(conf[mask].mean() - ok[mask].mean())
    return float(e)


def e3_gate(cfg: LabConfig, seeds: int = 20) -> dict:
    budgets: dict = {name: {_bkey(b): [] for b in BUDGETS} for name in STRATEGIES}
    diag: dict = {}
    curves: dict = {}
    seed_list = list(range(seeds)) + ([] if cfg.seed < seeds else [cfg.seed])
    for s in seed_list:
        c = cfg.with_(seed=s)
        docs = make_corpus(c, c.n_docs)
        cache = {clf: classify_corpus(c, clf, docs) for clf in {v[0] for v in STRATEGIES.values()}}
        for name, (clf, verify, scope) in STRATEGIES.items():
            pts = operating_points(cache[clf], verify, scope)
            if s == cfg.seed:
                curves[name] = pts
            if s >= seeds:
                continue
            for b in BUDGETS:
                budgets[name][_bkey(b)].append(at_budget(pts, b))
        if s >= seeds:
            continue
        for clf, cl in cache.items():
            wrong = [(d, lab, cf) for d, lab, cf in cl if lab != d.true_label]
            twin = [w for w in wrong if TWIN_OF.get(w[0].true_label) == w[1]]
            fam = [(cf, lab == d.true_label) for d, lab, cf in cl if lab in TWIN_OF]
            diag.setdefault(clf, []).append({
                "ece": ece(cl), "error_rate": len(wrong) / len(cl),
                "twin_pair_error_share": len(twin) / max(1, len(wrong)),
                "twin_errors_conf_ge_0_9": sum(w[2] >= 0.9 for w in twin) / max(1, len(twin)),
                "pooled_family_auroc": _auroc([c_ for c_, ok in fam if ok], [c_ for c_, ok in fam if not ok]),
                "max_family_conf": max(c_ for c_, _ in fam)})
    base_name = "match-strength, one global threshold"
    paired = {}
    for name, per_b in budgets.items():
        if name == base_name:
            continue
        paired[name] = {}
        for b, vals in per_b.items():
            pairs = [(v["randomised_exact"], g["randomised_exact"]) for v, g in zip(vals, budgets[base_name][b])
                     if v is not None and g is not None]
            if len(pairs) > 1:
                d = np.array([x - y for x, y in pairs])
                paired[name][b] = {"mean_diff": float(d.mean()), "ci95": 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d)),
                                   "relative": float(d.mean() / np.mean([y for _, y in pairs]))}
    summary = {}
    for name, per_b in budgets.items():
        summary[name] = {}
        for b, vals in per_b.items():
            ok = [v for v in vals if v is not None]
            summary[name][b] = ({"exact": _stats(v["randomised_exact"] for v in ok),
                                 "deterministic": _stats(v["deterministic"] for v in ok),
                                 "deterministic_hold": _stats(v["deterministic_hold"] for v in ok),
                                 "reachable_in": f"{len(ok)}/{len(vals)} seeds"} if ok
                                else {"reachable_in": f"0/{len(vals)} seeds"})
    diagnostics = {clf: {k: _stats(r[k] for r in rows) for k in rows[0]} for clf, rows in diag.items()}
    return {"at_budget": summary, "paired_vs_global": paired, "diagnostics": diagnostics,
            "curves_seed": cfg.seed, "curves": curves}


# ---------------------------------------------------------------------- E4
def e4_pairing(cfg: LabConfig, seeds: int = 10) -> dict:
    grid = []
    for share in (0.1, 0.3, 0.45, 0.6):
        for tenants in (50, 20, 5):
            c = cfg.with_(generic_filename_share=share, n_tenants=tenants)
            grid.append({"generic_filename_share": share, "n_tenants": tenants,
                         "docs_per_tenant": c.n_docs // tenants, **m.pairing_coverage(run_scenario(c).log)})
    per = [m.pairing_coverage(run_scenario(cfg.with_(seed=s)).log) for s in range(seeds)]
    return {"base": m.pairing_coverage(run_scenario(cfg).log),
            "across_seeds": {k: _stats(p[k] for p in per) for k in per[0] if k != "overrides"},
            "grid": grid}
