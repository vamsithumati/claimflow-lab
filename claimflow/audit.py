"""Blind stratified audit: estimate the AI's true error rate from
(a) documents a human genuinely judged, plus (b) a stratified random sample of
everything else, stratified by confidence bin."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LabConfig
from .data import CATEGORIES
from .pipeline import Run

JUDGED_ACTIONS = ("corrected", "repick", "accepted")  # bulk_set is not a judgment


@dataclass(frozen=True)
class AuditResult:
    estimate: float
    ci_low: float
    ci_high: float
    sampled: int

    def covers(self, truth: float) -> bool:
        return self.ci_low <= truth <= self.ci_high


def _blind_label(rng: np.random.Generator, truth: str, acc: float) -> str:
    return truth if rng.random() < acc else str(rng.choice([c for c in CATEGORIES if c != truth]))


def run_audit(run: Run, cfg: LabConfig, rng: np.random.Generator, adjudicate: bool = True) -> AuditResult:
    judged = {
        sid: (ai != human)
        for sid, ai, human in run.log.q(
            f"""SELECT o.subject_id, d.label, o.label FROM decision_events o
                JOIN decision_events d ON d.decision_id = o.draft_decision_id
                WHERE o.kind IN ('human_override','human_accept')
                  AND o.action IN ({",".join("?" * len(JUDGED_ACTIONS))})""", JUDGED_ACTIONS)
    }
    rest = [it for it in run.items if it.doc.doc_id not in judged]
    bins = cfg.audit_bins
    strata: list[list] = [[] for _ in range(len(bins) - 1)]
    for it in rest:
        k = int(np.searchsorted(bins, it.draft.confidence, side="right") - 1)
        strata[min(max(k, 0), len(strata) - 1)].append(it)

    est_errors, var, sampled = 0.0, 0.0, 0
    per_stratum = cfg.audit_per_bin_per_week * cfg.audit_weeks
    for s in strata:
        N = len(s)
        if N == 0:
            continue
        n = min(per_stratum, N)
        idx = rng.choice(N, size=n, replace=False)
        errs = 0
        for i in idx:
            it = s[i]
            lab = _blind_label(rng, it.doc.true_label, cfg.audit_labeller_accuracy)
            if adjudicate and lab != it.draft.label:  # disagreement → independent second label decides
                lab = _blind_label(rng, it.doc.true_label, cfg.audit_labeller_accuracy)
            errs += lab != it.draft.label
        p = errs / n
        est_errors += N * p
        if n > 1:
            var += N * N * (1 - n / N) * p * (1 - p) / (n - 1)
        sampled += n
    total = len(run.items)
    est = (sum(judged.values()) + est_errors) / total
    half = 1.96 * np.sqrt(var) / total
    return AuditResult(est, est - half, est + half, sampled)
