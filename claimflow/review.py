"""Simulated adjusters, on a timeline, under three interface designs.

pick_to_accept          — clearing a held doc requires choosing a category (even the one
                          already selected); some held batches are bulk-set to the AI's label.
pick_to_accept_no_bulk  — same, without the bulk action (isolates the bulk effect).
one_click_accept        — accepting is its own action and logs `human_accept`.

All randomness is pre-drawn per document and per batch, so the three UIs see the same
reviewers making the same choices — differences between UIs are paired, not noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LabConfig
from .data import Document
from .humans import noisy_label
from .triage import Draft

UI_VARIANTS = ("pick_to_accept", "pick_to_accept_no_bulk", "one_click_accept")


@dataclass(frozen=True)
class Item:
    t: int                  # arrival time (document index)
    doc: Document
    draft: Draft
    draft_id: str = ""


@dataclass(frozen=True)
class HumanEvent:
    t: float
    item: Item
    kind: str               # human_view | human_override | human_accept
    label: str | None
    action: str
    actor: str = "adjuster"


def simulate_review(items: list[Item], cfg: LabConfig, ui: str) -> list[HumanEvent]:
    assert ui in UI_VARIANTS, ui
    U = np.random.default_rng([cfg.seed, 1]).random((len(items), 5))  # open, correct, twin, other, delay
    out: list[HumanEvent] = []

    def human(it: Item) -> str:
        u = U[it.t]
        return noisy_label(it.doc.true_label, cfg.reviewer_accuracy, cfg.reviewer_twin_error_share,
                           u[1], u[2], u[3], shown_label=it.draft.label, anchor_share=cfg.reviewer_anchor_share)

    def judge(it: Item, t: float) -> None:
        out.append(HumanEvent(t, it, "human_view", None, "view"))
        h = human(it)
        if h != it.draft.label:
            out.append(HumanEvent(t + 0.1, it, "human_override", h, "corrected"))
        elif ui == "one_click_accept":
            out.append(HumanEvent(t + 0.1, it, "human_accept", h, "accepted"))
        elif it.draft.held:  # pick-to-accept: re-selecting the same label is logged as an override
            out.append(HumanEvent(t + 0.1, it, "human_override", h, "repick"))

    held = [it for it in items if it.draft.held]
    bulk_u = np.random.default_rng([cfg.seed, 2]).random(len(held) // cfg.bulk_batch_size + 1)
    for b, start in enumerate(range(0, len(held), cfg.bulk_batch_size)):
        batch = held[start:start + cfg.bulk_batch_size]
        t = batch[-1].t + cfg.review_lag
        if ui == "pick_to_accept" and bulk_u[b] < cfg.p_bulk_batch:
            for it in batch:  # one click stamps every row with the AI's label — wrong ones included
                out.append(HumanEvent(t, it, "human_view", None, "view", "adjuster_bulk"))
                out.append(HumanEvent(t + 0.1, it, "human_override", it.draft.label, "bulk_set", "adjuster_bulk"))
            continue
        for it in batch:
            judge(it, t)

    lo, hi = cfg.auto_open_delay
    for it in items:
        if it.draft.held:
            continue
        wrong = it.draft.label != it.doc.true_label
        if U[it.t, 0] < (cfg.p_open_if_wrong if wrong else cfg.p_open_if_right):
            judge(it, it.t + lo + U[it.t, 4] * (hi - lo))
    return out
