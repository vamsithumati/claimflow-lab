"""Two triage classifiers with two different *kinds* of confidence, plus the
twin-aware verification step.

- KeywordRouter: confidence = how strongly the top label matched (match strength).
  This is what most rule/keyword/filename routers emit. It says nothing about
  how close the runner-up was.
- ProbabilisticClassifier: confidence = calibrated probability across all labels
  (multinomial logistic regression, numpy only). Twins split the probability mass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    """confidence='strength': how strongly the winner matched (what most routers emit).
    confidence='margin': strength × how far the winner is ahead of the runner-up."""

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
        self.rng = np.random.default_rng(seed)

    def predict(self, doc: Document) -> tuple[str, float]:
        scores = np.array([sum(w.get(t, 0.0) for t in doc.tokens) for w in self.vocab.values()])
        best = np.flatnonzero(scores == scores.max())
        i = int(self.rng.choice(best))  # ties broken arbitrarily, as real routers do
        confidence = float(1.0 - np.exp(-scores[i] / self.k))  # saturating match strength
        if self.confidence_kind == "margin" and scores[i] > 0:
            runner_up = np.sort(scores)[-2]
            confidence *= float((scores[i] - runner_up) / scores[i]) ** 0.5
        return CATEGORIES[i], confidence


class ProbabilisticClassifier:
    name = "prob_classifier"

    def __init__(self, l2: float = 1e-3, lr: float = 0.5, epochs: int = 400) -> None:
        self.l2, self.lr, self.epochs = l2, lr, epochs
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


def twin_evidence_present(label: str, tokens: frozenset[str]) -> bool:
    """Deterministic twin check: the label's own distinguishing evidence is present
    AND none of its twin's evidence is. Anything else is ambiguous → a human decides."""
    own = set(DISTINCT_VOCAB[label]) - set(DISTINCT_VOCAB[TWIN_OF[label]])
    other = set(DISTINCT_VOCAB[TWIN_OF[label]]) - set(DISTINCT_VOCAB[label])
    return bool(tokens & own) and not (tokens & other)


class TriageAgent:
    """Classify, then gate: auto-process above threshold, otherwise hold for a human.
    With verify_twins=True, a twin label without its own distinguishing evidence is
    held regardless of confidence."""

    def __init__(self, clf: Classifier, threshold: float, verify_twins: bool = False) -> None:
        self.clf, self.threshold, self.verify_twins = clf, threshold, verify_twins

    def draft(self, doc: Document) -> Draft:
        label, conf = self.clf.predict(doc)
        if conf < self.threshold:
            return Draft(label, conf, self.clf.name, held=True, hold_reason="low_confidence")
        if self.verify_twins and label in TWIN_OF and not twin_evidence_present(label, doc.tokens):
            return Draft(label, conf, self.clf.name, held=True, hold_reason="twin_unverified")
        return Draft(label, conf, self.clf.name)
