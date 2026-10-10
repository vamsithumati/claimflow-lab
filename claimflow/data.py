"""Synthetic claims documents. Twins share their whole 'family' vocabulary and
differ only in a few distinguishing tokens that are sometimes missing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LabConfig

CATEGORIES: tuple[str, ...] = (
    "repair_estimate", "repair_invoice",
    "medical_invoice", "medical_receipt",
    "police_report", "proof_of_ownership",
)
PRIORS = np.array([0.22, 0.20, 0.15, 0.13, 0.15, 0.15])
TWINS: tuple[tuple[str, str], ...] = (
    ("repair_estimate", "repair_invoice"),
    ("medical_invoice", "medical_receipt"),
)
TWIN_OF = {a: b for a, b in TWINS} | {b: a for a, b in TWINS}

FAMILY_OF = {
    "repair_estimate": "repair", "repair_invoice": "repair",
    "medical_invoice": "medical", "medical_receipt": "medical",
    "police_report": "police", "proof_of_ownership": "ownership",
}
FAMILY_VOCAB = {
    "repair": ["repairer", "workshop", "panel", "labour", "parts", "bumper", "respray",
               "vehicle", "plate", "headlamp", "bonnet", "tax", "total", "reg_no"],
    "medical": ["provider", "patient", "consultation", "clinic", "physio", "item_code",
                "service_date", "referral", "tax", "total", "reg_no", "practitioner"],
    "police": ["incident", "officer", "station", "statement", "witness", "location",
               "event_no", "constable", "offence", "date"],
    "ownership": ["purchase", "serial", "model", "owner", "store", "warranty",
                  "item", "date", "retailer", "sku"],
}
DISTINCT_VOCAB = {
    "repair_estimate": ["estimate", "quote", "valid_30_days", "subject_to_approval"],
    "repair_invoice": ["invoice_no", "tax_invoice", "paid", "amount_due"],
    "medical_invoice": ["invoice_no", "amount_due", "payable_by", "account_no"],
    "medical_receipt": ["receipt", "paid", "payment_received", "thank_you"],
    "police_report": ["report_reference", "lodged", "sworn"],
    "proof_of_ownership": ["proof_of_purchase", "registered_owner", "certificate"],
}
NOISE_VOCAB = ["page", "of", "the", "and", "ref", "claim", "policy", "customer", "address",
               "phone", "email", "please", "note", "copy", "scan", "signature"]
GENERIC_FILENAMES = ["scan.pdf", "document.pdf", "attachment.pdf", "invoice.pdf", "upload.pdf"]


@dataclass(frozen=True)
class Document:
    doc_id: str
    tenant_id: int
    filename: str
    tokens: frozenset[str]
    true_label: str
    distinct_visible: bool

    @property
    def is_twin(self) -> bool:
        return self.true_label in TWIN_OF


def _filename(rng: np.random.Generator, cfg: LabConfig, label: str, i: int) -> str:
    if rng.random() < cfg.generic_filename_share:
        if rng.random() < 0.4:  # phone photos: small counter range, so collisions happen
            return f"IMG_{int(rng.integers(1000, 1150))}.jpg"
        return str(rng.choice(GENERIC_FILENAMES))
    return f"{label}_{i:06d}.pdf"


def make_document(rng: np.random.Generator, cfg: LabConfig, i: int, prefix: str = "doc") -> Document:
    label = str(rng.choice(CATEGORIES, p=PRIORS))
    family = FAMILY_VOCAB[FAMILY_OF[label]]
    tokens = set(rng.choice(family, size=int(rng.integers(5, 10)), replace=False))
    p_distinct = cfg.p_distinct_twin if label in TWIN_OF else cfg.p_distinct_other
    distinct_visible = bool(rng.random() < p_distinct)
    if distinct_visible:
        tokens |= set(rng.choice(DISTINCT_VOCAB[label], size=int(rng.integers(1, 3)), replace=False))
    if label in TWIN_OF and rng.random() < cfg.p_cross_token:
        tokens.add(str(rng.choice(DISTINCT_VOCAB[TWIN_OF[label]])))
    if rng.random() < cfg.p_garbled:  # bad scan: most content lost, stray tokens from elsewhere
        other = FAMILY_VOCAB[str(rng.choice(list(FAMILY_VOCAB)))]
        tokens = set(rng.choice(sorted(tokens), size=2, replace=False)) | set(rng.choice(other, size=3, replace=False))
        distinct_visible = False
    tokens |= set(rng.choice(NOISE_VOCAB, size=int(rng.integers(2, 6)), replace=False))
    return Document(
        doc_id=f"{prefix}_{i:06d}",
        tenant_id=int(rng.integers(0, cfg.n_tenants)),
        filename=_filename(rng, cfg, label, i),
        tokens=frozenset(tokens),
        true_label=label,
        distinct_visible=distinct_visible,
    )


def make_corpus(cfg: LabConfig, n: int, seed_offset: int = 0, prefix: str = "doc") -> list[Document]:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    return [make_document(rng, cfg, i, prefix) for i in range(n)]
