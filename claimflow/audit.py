"""Blind stratified audit of the AI's true error rate.

Population = every AI suggestion. Items a human already looked at individually can be
either *trusted* (their judgment counted as truth — cheap, but those reviewers saw the
AI's answer, so they are not blind) or *re-audited* blind as their own stratum.
Everything else is stratified by confidence bin and sampled blind.

Interval: a stratified Wald-type interval whose per-stratum variance uses the Agresti–Coull
adjusted proportion (x+2)/(n+4), so a stratum that shows zero errors still contributes
uncertainty. It is conservative at small samples. The gold-slice correction's own estimation
error is not propagated (stated limitation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LabConfig
from .humans import noisy_label
from .pipeline import Run

ADJUDICATION = ("none", "on_disagreement", "double_all", "gold_corrected")


@dataclass(frozen=True)
class AuditResult:
    estimate: float
    ci_low: float
    ci_high: float
    sampled: int

    def covers(self, truth: float) -> bool:
        return self.ci_low <= truth <= self.ci_high


def _labeller(rng: np.random.Generator, doc, cfg: LabConfig) -> str:
    u = rng.random(3)
    acc = cfg.audit_labeller_accuracy
    if cfg.audit_labeller_ambiguous_accuracy is not None and doc.is_twin and not doc.distinct_visible:
        acc = cfg.audit_labeller_ambiguous_accuracy   # every labeller struggles on the same ambiguous docs
    return noisy_label(doc.true_label, acc, cfg.audit_labeller_twin_error_share, *u)


def _audit_label(rng: np.random.Generator, it, cfg: LabConfig, adjudication: str) -> str:
    a = _labeller(rng, it.doc, cfg)
    if adjudication == "on_disagreement" and a != it.draft.label:
        a = _labeller(rng, it.doc, cfg)          # second blind labeller decides
    elif adjudication == "double_all":
        b = _labeller(rng, it.doc, cfg)
        if a != b:
            a = _labeller(rng, it.doc, cfg)      # third labeller breaks the tie
    return a


def _gold_rates(run: Run, cfg: LabConfig, rng: np.random.Generator, n_gold: int) -> tuple[float, float]:
    """Labeller sensitivity/specificity for 'the AI is wrong', estimated on a gold slice of n_gold items
    that an expert labelled with the true answer (assumed error-free — an extra labelling cost).
    Rates are pooled across the population but applied per stratum; labeller sensitivity really
    differs by stratum, which is one source of residual bias. Returns (sens, spec)."""
    idx = rng.choice(len(run.items), size=min(n_gold, len(run.items)), replace=False)
    tp = fn = tn = fp = 0
    for i in idx:
        it = run.items[i]
        ai_wrong = it.draft.label != it.doc.true_label
        says_wrong = _labeller(rng, it.doc, cfg) != it.draft.label
        if ai_wrong:
            tp += says_wrong
            fn += not says_wrong
        else:
            fp += says_wrong
            tn += not says_wrong
    sens = (tp + 0.5) / (tp + fn + 1)
    spec = (tn + 0.5) / (tn + fp + 1)
    return sens, spec


def human_judgments(run: Run) -> dict[str, str]:
    """doc_id → the label an individual reviewer left (no override = they kept the AI's)."""
    out: dict[str, str] = {}
    for ev in run.human_events:
        if ev.actor == "adjuster_bulk":
            continue
        did = ev.item.doc.doc_id
        if ev.kind == "human_view":
            out.setdefault(did, ev.item.draft.label)
        elif ev.kind in ("human_override", "human_accept"):
            out[did] = ev.label
    return out


def run_audit(run: Run, cfg: LabConfig, rng: np.random.Generator, adjudication: str = "gold_corrected",
              trust_reviewers: bool = False, n_gold: int = 100) -> AuditResult:
    """adjudication:
    none            one blind label per item
    on_disagreement a second blind label replaces the first when it disagrees with the AI
    double_all      two blind labels per item, a third breaks ties
    gold_corrected  one blind label per item, then a Rogan–Gladen correction using labeller
                    sensitivity/specificity measured on an expert-labelled gold slice (n_gold items)"""
    assert adjudication in ADJUDICATION, adjudication
    sens, spec = _gold_rates(run, cfg, rng, n_gold) if adjudication == "gold_corrected" else (1.0, 1.0)
    youden = max(sens + spec - 1, 0.05)
    label_mode = "none" if adjudication == "gold_corrected" else adjudication
    judged = human_judgments(run)
    known_errors = 0
    strata: list[list] = [[] for _ in range(len(cfg.audit_bins) - 1)]
    reviewed_stratum: list = []
    for it in run.items:
        did = it.doc.doc_id
        if did in judged:
            if trust_reviewers:
                known_errors += judged[did] != it.draft.label
                continue
            reviewed_stratum.append(it)
            continue
        k = int(np.searchsorted(cfg.audit_bins, it.draft.confidence, side="right") - 1)
        strata[min(max(k, 0), len(strata) - 1)].append(it)
    if reviewed_stratum:
        strata.append(reviewed_stratum)

    per_stratum = cfg.audit_per_bin_per_week * cfg.audit_weeks
    est_errors, var, sampled = 0.0, 0.0, 0
    for s in strata:
        N = len(s)
        if N == 0:
            continue
        n = min(per_stratum, N)
        idx = rng.choice(N, size=n, replace=False)
        x = sum(_audit_label(rng, s[i], cfg, label_mode) != s[i].draft.label for i in idx)
        p_obs = x / n
        p_hat = min(max((p_obs + spec - 1) / youden, 0.0), 1.0) if adjudication == "gold_corrected" else p_obs
        est_errors += N * p_hat
        p_ac = (x + 2) / (n + 4)                      # Agresti–Coull plug-in for the variance
        var += N * N * (1 - n / N) * p_ac * (1 - p_ac) / n / youden ** 2
        sampled += n
    total = len(run.items)
    est = (known_errors + est_errors) / total
    half = 1.96 * np.sqrt(var) / total
    return AuditResult(est, est - half, est + half, sampled)
