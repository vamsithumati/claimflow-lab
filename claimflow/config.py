"""All tunable assumptions live here, in one place, so every result can be re-run
under different assumptions (see experiments.sensitivity). Each assumption that
*drives* a claim is marked — those claims are 'by construction', not discoveries."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class LabConfig:
    seed: int = 7
    n_docs: int = 5000
    n_train: int = 3000          # labelled docs available to the probabilistic classifier
    n_train_small: int = 300     # a cheaper labelling budget, for comparison
    n_tenants: int = 20

    # --- document generation (DRIVES: twin errors exist; how often confidence is fooled) ---
    p_distinct_twin: float = 0.70     # distinguishing evidence survives into a twin doc's text
    p_distinct_other: float = 0.95
    p_cross_token: float = 0.30       # a twin doc carries a misleading token of its twin
    p_garbled: float = 0.04           # badly garbled scan
    generic_filename_share: float = 0.45

    # --- triage gate ---------------------------------------------------------------------
    threshold: float = 0.85

    # --- human review (DRIVES: reviewed-subset error > true error) ---------------------------
    p_open_if_wrong: float = 0.35     # an auto-processed wrong doc gets noticed
    p_open_if_right: float = 0.05     # a correct auto-processed doc gets opened anyway
    reviewer_accuracy: float = 0.97   # reviewer picks the true label…
    reviewer_twin_error_share: float = 0.7  # …and when wrong on a twin doc, picks the twin this often
    reviewer_anchor_share: float = 0.5      # …and when wrong, falls back to the AI's answer this often (not blind)
    p_bulk_batch: float = 0.20        # pick-to-accept UI: a held batch is bulk-set to the AI label
    bulk_batch_size: int = 8
    review_lag: int = 40              # held docs are worked this many arrivals after the batch fills
    auto_open_delay: tuple[int, int] = (5, 300)  # wrong auto docs get noticed this long after arrival

    # --- audit ------------------------------------------------------------------------------
    audit_per_bin_per_week: int = 20
    audit_weeks: int = 4
    audit_labeller_accuracy: float = 0.98
    audit_labeller_twin_error_share: float = 0.7  # 0 = independent uniform errors (optimistic)
    # Accuracy on look-alike docs whose distinguishing evidence is missing. None = same as above,
    # i.e. labellers err independently of how ambiguous the document is (optimistic).
    audit_labeller_ambiguous_accuracy: float | None = None
    audit_bins: tuple[float, ...] = field(default=(0.0, 0.6, 0.8, 0.9, 0.95, 1.0001))

    def with_(self, **kw) -> "LabConfig":
        return replace(self, **kw)
