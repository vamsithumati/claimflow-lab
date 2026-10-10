"""Triage classifiers with different *kinds* of confidence, plus a twin verifier.

- KeywordRouter(confidence="strength"): how strongly the top label matched — never
  looks at the runner-up. Common in rule/keyword/filename routers.
- KeywordRouter(confidence="margin"): (top − runner-up) / top. Same rules, no labels needed.
- ProbabilisticClassifier: multinomial logistic regression (numpy). Needs labelled data;
  the lab trains it on 300 or 3,000 synthetic labelled documents, which the routers never get.

Disclosure: the twin verifier reads the generator's own distinguishing vocabulary
(DISTINCT_VOCAB). In production you would hand-write or learn that evidence list, and it
would be less complete than this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import zlib

import numpy as np

from .data import CATEGORIES, DISTINCT_VOCAB, FAMILY_OF, FAMILY_VOCAB, TWIN_OF, Document


@dataclass(frozen=True)
class Draft:
    label: str
    confidence: float
    decided_by: str
    held: bool = False
    hold_reason: str | None = None


class Classifier(Protocol):
    @property
    def name(self) -> str: ...

    def predict(self, doc: Document) -> tuple[str, float]: ...


class KeywordRouter:
    def __init__(self, family_weight: float = 1.0, distinct_weight: float = 2.0, k: float = 3.5,
                 seed: int = 0, confidence: str = "strength") -> None:
        assert confidence in ("strength", "margin"), confidence
        self.confidence_kind = confidence
        self.name = "keyword_router" if confidence == "strength" else "keyword_router_margin"
        self.vocab: dict[str, dict[str, float]] = {}
        for c in CATEGORIES:
            w = {t: family_weight for t in FAMILY_VOCAB[FAMILY_OF[c]]}
            w.update({t: distinct_weight for t in DISTINCT_VOCAB[c]})
            self.vocab[c] = w
        self.k = k
        self.seed = seed

    def predict(self, doc: Document) -> tuple[str, float]:
        scores = np.array([sum(w.get(t, 0.0) for t in doc.tokens) for w in self.vocab.values()])
        best = np.flatnonzero(scores == scores.max())
        # deterministic per-document tie-break, so results don't depend on processing order
        i = int(best[zlib.crc32(f'{self.seed}:{doc.doc_id}'.encode()) % len(best)])
        if self.confidence_kind == "strength":
            return CATEGORIES[i], float(1.0 - np.exp(-scores[i] / self.k))  # saturating match strength
        top = scores[i]
        runner_up = np.sort(scores)[-2]
        return CATEGORIES[i], float((top - runner_up) / top) if top > 0 else 0.0


class ProbabilisticClassifier:
    def __init__(self, l2: float = 1e-3, lr: float = 0.5, epochs: int = 400, name: str = "prob_classifier") -> None:
        self.l2, self.lr, self.epochs, self.name = l2, lr, epochs, name
        self.features: list[str] = []
        self.W: np.ndarray | None = None

    def _x(self, docs: list[Document]) -> np.ndarray:
        idx = {f: j for j, f in enumerate(self.features)}
        X = np.zeros((len(docs), len(self.features) + 1))
        X[:, -1] = 1.0  # bias
        for r, d in enumerate(docs):
            for t in d.tokens:
                if t in idx:
                    X[r, idx[t]] = 1.0
        return X

    def fit(self, docs: list[Document]) -> "ProbabilisticClassifier":
        self.features = sorted({t for d in docs for t in d.tokens})
        X = self._x(docs)
        y = np.array([CATEGORIES.index(d.true_label) for d in docs])
        Y = np.eye(len(CATEGORIES))[y]
        W = np.zeros((X.shape[1], len(CATEGORIES)))
        for _ in range(self.epochs):
            P = _softmax(X @ W)
            W -= self.lr * (X.T @ (P - Y) / len(docs) + self.l2 * W)
        self.W = W
        return self

    def predict(self, doc: Document) -> tuple[str, float]:
        assert self.W is not None, "fit() first"
        p = _softmax(self._x([doc]) @ self.W)[0]
        i = int(p.argmax())
        return CATEGORIES[i], float(p[i])


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _own(label: str) -> set[str]:
    return set(DISTINCT_VOCAB[label]) - set(DISTINCT_VOCAB[TWIN_OF[label]])


def twin_evidence_present(label: str, tokens: frozenset[str]) -> bool:
    """The label's own distinguishing evidence is present AND none of its twin's is."""
    return bool(tokens & _own(label)) and not (tokens & _own(TWIN_OF[label]))


def twin_verdict(label: str, tokens: frozenset[str]) -> tuple[str, str]:
    """Returns (label, verdict) where verdict ∈ verified | relabelled | ambiguous."""
    if twin_evidence_present(label, tokens):
        return label, "verified"
    if twin_evidence_present(TWIN_OF[label], tokens):  # the evidence says it's the twin
        return TWIN_OF[label], "relabelled"
    return label, "ambiguous"


VERIFY_MODES = ("off", "hold", "relabel")


class TriageAgent:
    """Classify, then gate: auto-process at/above threshold, otherwise hold for a human.
    verify='hold'    — a twin label without its own evidence is held.
    verify='relabel' — as above, but flips to the twin when the twin's evidence is unambiguous."""

    def __init__(self, clf: Classifier, threshold: float, verify: str = "off") -> None:
        assert verify in VERIFY_MODES, verify
        self.clf, self.threshold, self.verify = clf, threshold, verify

    def draft(self, doc: Document) -> Draft:
        label, conf = self.clf.predict(doc)
        return gate(label, conf, doc, self.threshold, self.verify, self.clf.name)


def gate(label: str, conf: float, doc: Document, threshold: float, verify: str, name: str) -> Draft:
    if verify != "off" and label in TWIN_OF:
        new_label, verdict = twin_verdict(label, doc.tokens)
        if verdict == "relabelled" and verify == "relabel":
            label = new_label
        elif verdict != "verified":
            return Draft(label, conf, name, held=True, hold_reason="twin_unverified")
    if conf < threshold:
        return Draft(label, conf, name, held=True, hold_reason="low_confidence")
    return Draft(label, conf, name)
