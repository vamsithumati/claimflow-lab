"""All tunable assumptions live here, in one place, so every result can be
re-run under different assumptions (see experiments.sensitivity)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class LabConfig:
    seed: int = 7
    n_docs: int = 5000
    n_train: int = 3000
    n_tenants: int = 20

    # --- document generation -------------------------------------------
    # Probability that the *distinguishing* evidence (e.g. "invoice no", "paid")
    # survives into the text. Twins lose it more often (cropped pages, OCR).
    p_distinct_twin: float = 0.70
    p_distinct_other: float = 0.95
    # Probability a twin doc carries a *misleading* token of its twin
    # ("this is not a tax invoice", "amount paid: 0.00").
    p_cross_token: float = 0.30
    # Probability a scan is badly garbled (few tokens, some from the wrong family)
    p_garbled: float = 0.04
    # Share of uploads with a generic, collision-prone filename (scan.pdf, IMG_1043.jpg)
    generic_filename_share: float = 0.45

    # --- triage gate -----------------------------------------------------
    threshold: float = 0.85

    # --- human review behaviour -------------------------------------------
    p_open_if_wrong: float = 0.35      # an auto-processed wrong doc gets noticed (downstream breaks)
    p_open_if_right: float = 0.05      # a correct auto-processed doc gets opened anyway
    reviewer_accuracy: float = 0.97    # a reviewer who looks picks the true label
    p_bulk_batch: float = 0.20         # pick-to-accept UI: a held batch is bulk-set to the AI label
    bulk_batch_size: int = 8

    # --- audit -------------------------------------------------------------
    audit_per_bin_per_week: int = 20
    audit_weeks: int = 4
    audit_labeller_accuracy: float = 0.98
    audit_bins: tuple[float, ...] = field(default=(0.0, 0.6, 0.8, 0.9, 0.95, 1.0001))

    def with_(self, **kw) -> "LabConfig":
        return replace(self, **kw)
