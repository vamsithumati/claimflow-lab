"""Simulated adjusters working through two interface designs.

pick_to_accept   — clearing a held doc requires choosing a category (even the one already
                   selected); held batches are sometimes bulk-set to the AI's label.
one_click_accept — accepting is its own action and logs `human_accept`, not an override.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LabConfig
from .data import CATEGORIES, Document
from .events import EventLog
from .triage import Draft

UI_VARIANTS = ("pick_to_accept", "one_click_accept")


@dataclass(frozen=True)
class Item:
    doc: Document
    draft: Draft
    draft_id: str


def _human_label(rng: np.random.Generator, cfg: LabConfig, truth: str) -> str:
    if rng.random() < cfg.reviewer_accuracy:
        return truth
    return str(rng.choice([c for c in CATEGORIES if c != truth]))


def simulate_review(items: list[Item], log: EventLog, cfg: LabConfig, ui: str,
                    rng: np.random.Generator) -> set[str]:
    """Returns the set of reviewed document ids."""
    assert ui in UI_VARIANTS, ui
    reviewed: set[str] = set()

    def ev(item: Item, kind: str, label: str | None, action: str, actor: str = "adjuster") -> None:
        log.append(kind, subject_id=item.doc.doc_id, tenant_id=item.doc.tenant_id,
                   filename=item.doc.filename, label=label, draft_decision_id=item.draft_id,
                   action=action, actor=actor)

    held = [it for it in items if it.draft.held]
    auto = [it for it in items if not it.draft.held]

    # Held queue, worked in batches.
    for start in range(0, len(held), cfg.bulk_batch_size):
        batch = held[start:start + cfg.bulk_batch_size]
        bulk = ui == "pick_to_accept" and rng.random() < cfg.p_bulk_batch
        for it in batch:
            reviewed.add(it.doc.doc_id)
            ev(it, "human_view", None, "view")
            if bulk:  # one click sets every row to the AI's label — including the wrong ones
                ev(it, "human_override", it.draft.label, "bulk_set", "adjuster_bulk")
                continue
            human = _human_label(rng, cfg, it.doc.true_label)
            if human != it.draft.label:
                ev(it, "human_override", human, "corrected")
            elif ui == "pick_to_accept":
                ev(it, "human_override", human, "repick")   # same label, logged as an override
            else:
                ev(it, "human_accept", human, "accepted")

    # Auto-processed docs: people open what looks wrong (selection bias).
    for it in auto:
        wrong = it.draft.label != it.doc.true_label
        if rng.random() >= (cfg.p_open_if_wrong if wrong else cfg.p_open_if_right):
            continue
        reviewed.add(it.doc.doc_id)
        ev(it, "human_view", None, "view")
        human = _human_label(rng, cfg, it.doc.true_label)
        if human != it.draft.label:
            ev(it, "human_override", human, "corrected")
        elif ui == "one_click_accept":
            ev(it, "human_accept", human, "accepted")
    return reviewed
