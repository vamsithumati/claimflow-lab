"""How simulated people label things. Errors are not uniform: on a twin document, a
wrong human usually picks the twin — the same confusion the AI makes."""
from __future__ import annotations

from .data import CATEGORIES, TWIN_OF


def noisy_label(truth: str, accuracy: float, twin_error_share: float,
                u_correct: float, u_twin: float, u_other: float,
                shown_label: str | None = None, anchor_share: float = 0.0) -> str:
    """Deterministic given pre-drawn uniforms, so different scenarios can share them.
    shown_label/anchor_share model a NON-blind reviewer: when they get it wrong, they fall
    back to the answer they were shown this often (anchoring)."""
    if u_correct < accuracy:
        return truth
    if shown_label is not None and shown_label != truth and u_other < anchor_share:
        return shown_label
    if truth in TWIN_OF and u_twin < twin_error_share:
        return TWIN_OF[truth]
    others = [c for c in CATEGORIES if c != truth]
    return others[min(int(u_other * len(others)), len(others) - 1)]
