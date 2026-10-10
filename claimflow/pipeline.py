"""One scenario = docs arrive → triage → gate → humans review later; every event is
written to the append-only log in time order (drafts and human actions interleaved)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .config import LabConfig
from .data import Document, make_corpus
from .events import EventLog
from .review import HumanEvent, Item, simulate_review
from .triage import Classifier, KeywordRouter, ProbabilisticClassifier, gate

CLASSIFIERS = ("keyword_router", "keyword_router_margin", "prob_classifier", "prob_classifier_300")


@dataclass
class Run:
    cfg: LabConfig
    classifier: str
    ui: str
    verify: str
    log: EventLog
    items: list[Item]
    human_events: list[HumanEvent]


@lru_cache(maxsize=64)
def _trained_prob(cfg_key: tuple, n_train: int, name: str) -> ProbabilisticClassifier:
    cfg = _CFG_CACHE[cfg_key]
    return ProbabilisticClassifier(name=name).fit(
        make_corpus(cfg, n_train, seed_offset=10_000, prefix="train"))


_CFG_CACHE: dict[tuple, LabConfig] = {}


def build_classifier(name: str, cfg: LabConfig) -> Classifier:
    if name == "keyword_router":
        return KeywordRouter(seed=cfg.seed)
    if name == "keyword_router_margin":
        return KeywordRouter(seed=cfg.seed, confidence="margin")
    # the training corpus depends only on the data-generation settings + seed
    key = (cfg.seed, cfg.p_distinct_twin, cfg.p_distinct_other, cfg.p_cross_token, cfg.p_garbled,
           cfg.generic_filename_share, cfg.n_tenants)
    _CFG_CACHE.setdefault(key, cfg)
    if name == "prob_classifier":
        return _trained_prob(key, cfg.n_train, name)
    if name == "prob_classifier_300":
        return _trained_prob(key, cfg.n_train_small, name)
    raise ValueError(name)


def classify_corpus(cfg: LabConfig, classifier: str, docs: list[Document] | None = None
                    ) -> list[tuple[Document, str, float]]:
    """Raw (doc, label, confidence) — enough to evaluate any gate without simulating people."""
    docs = docs if docs is not None else make_corpus(cfg, cfg.n_docs)
    clf = build_classifier(classifier, cfg)
    return [(d, *clf.predict(d)) for d in docs]


def run_scenario(cfg: LabConfig, classifier: str = "keyword_router", ui: str = "pick_to_accept",
                 verify: str = "off", docs: list[Document] | None = None,
                 db_path: str = ":memory:") -> Run:
    clf_name = build_classifier(classifier, cfg).name
    items = [Item(t, d, gate(label, conf, d, cfg.threshold, verify, clf_name))
             for t, (d, label, conf) in enumerate(classify_corpus(cfg, classifier, docs))]
    human = simulate_review(items, cfg, ui)

    # interleave: draft at its arrival time, human actions when they happen
    timeline = [(it.t, 0, i, None) for i, it in enumerate(items)] + \
               [(ev.t, 1, ev.item.t, ev) for ev in human]
    timeline.sort(key=lambda x: (x[0], x[1], x[2]))
    log = EventLog(db_path)
    draft_ids: dict[int, str] = {}
    for _, _, idx, ev in timeline:
        if ev is None:
            it = items[idx]
            d = it.doc
            draft_ids[idx] = log.append(
                "ai_draft", subject_id=d.doc_id, tenant_id=d.tenant_id, filename=d.filename,
                label=it.draft.label, confidence=it.draft.confidence, held=it.draft.held,
                hold_reason=it.draft.hold_reason, actor=it.draft.decided_by)
            log.record_truth(d.doc_id, d.true_label, d.is_twin)
        else:
            d = ev.item.doc
            log.append(ev.kind, subject_id=d.doc_id, tenant_id=d.tenant_id, filename=d.filename,
                       label=ev.label, draft_decision_id=draft_ids[ev.item.t], action=ev.action, actor=ev.actor)
    log.conn.commit()
    items = [replace(it, draft_id=draft_ids[it.t]) for it in items]
    return Run(cfg, classifier, ui, verify, log, items, human)
