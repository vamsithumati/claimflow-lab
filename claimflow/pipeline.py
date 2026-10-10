"""One scenario = generate docs → triage → gate → simulated review, all logged."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import LabConfig
from .data import Document, make_corpus
from .events import EventLog
from .review import Item, simulate_review
from .triage import Classifier, KeywordRouter, ProbabilisticClassifier, TriageAgent

CLASSIFIERS = ("keyword_router", "keyword_router_margin", "prob_classifier")


@dataclass
class Run:
    cfg: LabConfig
    classifier: str
    ui: str
    verify_twins: bool
    log: EventLog
    items: list[Item]
    reviewed: set[str]


@lru_cache(maxsize=32)
def _trained_prob(cfg: LabConfig) -> ProbabilisticClassifier:
    return ProbabilisticClassifier().fit(make_corpus(cfg, cfg.n_train, seed_offset=10_000, prefix="train"))


def build_classifier(name: str, cfg: LabConfig) -> Classifier:
    if name == "keyword_router":
        return KeywordRouter(seed=cfg.seed)
    if name == "keyword_router_margin":
        return KeywordRouter(seed=cfg.seed, confidence="margin")
    if name == "prob_classifier":
        return _trained_prob(cfg)
    raise ValueError(name)


def run_scenario(cfg: LabConfig, classifier: str = "keyword_router", ui: str = "pick_to_accept",
                 verify_twins: bool = False, docs: list[Document] | None = None,
                 db_path: str = ":memory:") -> Run:
    docs = docs if docs is not None else make_corpus(cfg, cfg.n_docs)
    agent = TriageAgent(build_classifier(classifier, cfg), cfg.threshold, verify_twins)
    log = EventLog(db_path)
    items: list[Item] = []
    for d in docs:
        draft = agent.draft(d)
        draft_id = log.append("ai_draft", subject_id=d.doc_id, tenant_id=d.tenant_id, filename=d.filename,
                              label=draft.label, confidence=draft.confidence, held=draft.held,
                              hold_reason=draft.hold_reason, actor=draft.decided_by)
        log.record_truth(d.doc_id, d.true_label, d.is_twin)
        items.append(Item(d, draft, draft_id))
    reviewed = simulate_review(items, log, cfg, ui, np.random.default_rng(cfg.seed + 1))
    log.conn.commit()
    return Run(cfg, classifier, ui, verify_twins, log, items, reviewed)
