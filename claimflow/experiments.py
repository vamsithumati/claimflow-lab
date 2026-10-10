"""Experiments E1–E4, one per claim in post #1, plus a sensitivity sweep.
Each returns plain dicts so results serialise straight to JSON."""
from __future__ import annotations

import itertools

import numpy as np

from . import metrics as m
from .audit import run_audit
from .config import LabConfig
from .pipeline import run_scenario

THRESHOLDS = (0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98)


def _mean_sd(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def e1_override_split(cfg: LabConfig, seeds: int = 20) -> dict:
    out: dict = {"base": {}, "across_seeds": {}}
    for ui in ("pick_to_accept", "one_click_accept"):
        out["base"][ui] = m.override_split(run_scenario(cfg, ui=ui).log)
        runs = [m.override_split(run_scenario(cfg.with_(seed=s), ui=ui).log) for s in range(seeds)]
        out["across_seeds"][ui] = {k: _mean_sd([r[k] for r in runs])
                                   for k in ("naive_override_rate", "confirm_as_is_share", "true_correction_rate")}
    return out


def e2_bounds_and_audit(cfg: LabConfig, audits: int = 300, seeds: int = 10, audits_per_seed: int = 40) -> dict:
    run = run_scenario(cfg)
    bounds = m.error_bounds(run.log)
    truth = bounds["oracle_true_error"]
    out = {"bounds": bounds, "audit": {}}
    for adjudicate in (True, False):
        res = [run_audit(run, cfg, np.random.default_rng(i), adjudicate) for i in range(audits)]
        across = []
        for s in range(seeds):
            r = run_scenario(cfg.with_(seed=100 + s))
            t = m.error_bounds(r.log)["oracle_true_error"]
            across += [run_audit(r, cfg, np.random.default_rng(1000 * s + i), adjudicate).covers(t)
                       for i in range(audits_per_seed)]
        out["audit"]["adjudicated" if adjudicate else "single_label"] = {
            "estimate": _mean_sd([a.estimate for a in res]),
            "bias_pp": 100 * (float(np.mean([a.estimate for a in res])) - truth),
            "ci_half_width_pp": 100 * float(np.mean([(a.ci_high - a.ci_low) / 2 for a in res])),
            "coverage_same_data": float(np.mean([a.covers(truth) for a in res])),
            "coverage_across_seeds": float(np.mean(across)),
            "sampled_per_audit": res[0].sampled,
        }
    weeks = {}
    for w in (1, 2, 4, 8):
        c = cfg.with_(audit_weeks=w)
        res = [run_audit(run, c, np.random.default_rng(i)) for i in range(100)]
        weeks[w] = {"ci_half_width_pp": 100 * float(np.mean([(a.ci_high - a.ci_low) / 2 for a in res])),
                    "coverage": float(np.mean([a.covers(truth) for a in res]))}
    out["audit_by_weeks"] = weeks
    return out


def e3_twins(cfg: LabConfig) -> dict:
    out: dict = {}
    for clf in ("keyword_router", "keyword_router_margin", "prob_classifier"):
        sweep, sweep_verified = [], []
        for t in THRESHOLDS:
            g = m.oracle_gate_stats(run_scenario(cfg.with_(threshold=t), classifier=clf).log)
            sweep.append({"threshold": t, **g})
            g = m.oracle_gate_stats(run_scenario(cfg.with_(threshold=t), classifier=clf, verify_twins=True).log)
            sweep_verified.append({"threshold": t, **g})
        verified = m.oracle_gate_stats(run_scenario(cfg, classifier=clf, verify_twins=True).log)
        base = next(s for s in sweep if s["threshold"] == cfg.threshold)
        out[clf] = {"at_default_threshold": base, "sweep": sweep,
                    "with_twin_verifier": verified, "sweep_with_verifier": sweep_verified,
                    "confusion": m.correction_confusion(run_scenario(cfg, classifier=clf).log)}
    return out


def e4_pairing(cfg: LabConfig) -> dict:
    grid = []
    for share in (0.1, 0.3, 0.45, 0.6):
        for tenants in (50, 20, 5):
            c = cfg.with_(generic_filename_share=share, n_tenants=tenants)
            grid.append({"generic_filename_share": share, "n_tenants": tenants,
                         "docs_per_tenant": c.n_docs // tenants,
                         **m.pairing_coverage(run_scenario(c).log)})
    return {"base": m.pairing_coverage(run_scenario(cfg).log), "grid": grid}


def sensitivity(cfg: LabConfig) -> dict:
    """54 cells: hold threshold × review habits × twin evidence quality × bulk-action use."""
    rows = []
    grid = itertools.product((0.80, 0.85, 0.90), (0.02, 0.05, 0.10), (0.6, 0.8), (0.0, 0.2, 0.4))
    for threshold, p_open_right, p_distinct, p_bulk in grid:
        c = cfg.with_(threshold=threshold, p_open_if_right=p_open_right, p_distinct_twin=p_distinct,
                      p_bulk_batch=p_bulk)
        r = run_scenario(c)
        s, g, b = m.override_split(r.log), m.oracle_gate_stats(r.log), m.error_bounds(r.log)
        rows.append({"threshold": threshold, "p_open_if_right": p_open_right,
                     "p_distinct_twin": p_distinct, "p_bulk_batch": p_bulk,
                     "confirm_as_is_share": s["confirm_as_is_share"],
                     "naive_override_rate": s["naive_override_rate"],
                     "true_correction_rate": s["true_correction_rate"],
                     "twin_pair_error_share": g["twin_pair_error_share"],
                     "twin_errors_conf_ge_0_9": g["share_of_twin_errors_with_conf_ge_0_9"],
                     "lower_bound": b["lower_bound"], "oracle": b["oracle_true_error"],
                     "reviewed_subset_error": b["reviewed_subset_error"]})
    return {"grid": rows}
