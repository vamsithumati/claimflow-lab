"""The metrics from the post, written as SQL over the decision log.
Anything prefixed `oracle_` uses ground truth that only the lab has."""
from __future__ import annotations

from .data import TWIN_OF
from .events import EventLog

# Post, Lie #1: split overrides by comparing the human label with the AI label, joined by ID.
SPLIT_BY_ID = """
WITH paired AS (
  SELECT o.action, d.label AS ai_label, o.label AS human_label
  FROM decision_events o
  JOIN decision_events d
    ON d.decision_id = o.draft_decision_id
   AND d.kind = 'ai_draft'
  WHERE o.kind = 'human_override'
)
SELECT count(*)                                                                    AS overrides,
       count(*) FILTER (WHERE action = 'bulk_set')                                 AS bulk_set,
       count(*) FILTER (WHERE action <> 'bulk_set' AND ai_label = human_label)     AS confirm_as_is,
       count(*) FILTER (WHERE action <> 'bulk_set' AND ai_label <> human_label)    AS true_corrections
FROM paired;
"""

# Post, Lie #4: the naive join on (tenant, filename) to any earlier draft.
FILENAME_JOIN = """
WITH cand AS (
  SELECT o.decision_id AS override_id
  FROM decision_events o
  JOIN decision_events d
    ON d.kind = 'ai_draft' AND d.tenant_id = o.tenant_id AND d.filename = o.filename AND d.seq < o.seq
  WHERE o.kind = 'human_override'
), per AS (
  SELECT override_id, count(*) AS n FROM cand GROUP BY override_id
)
SELECT (SELECT count(*) FROM decision_events WHERE kind = 'human_override') AS overrides,
       count(*) FILTER (WHERE n = 1)  AS clean_pairs,
       count(*) FILTER (WHERE n > 1)  AS ambiguous
FROM per;
"""

# Latest-draft-wins heuristic: how often does it pick the wrong draft?
FILENAME_LATEST_WRONG = """
WITH latest AS (
  SELECT o.decision_id, o.draft_decision_id AS true_draft,
         (SELECT d.decision_id FROM decision_events d
           WHERE d.kind='ai_draft' AND d.tenant_id=o.tenant_id AND d.filename=o.filename AND d.seq<o.seq
           ORDER BY d.seq DESC LIMIT 1) AS picked
  FROM decision_events o WHERE o.kind = 'human_override'
)
SELECT count(*) FILTER (WHERE picked <> true_draft) FROM latest;
"""


def override_split(log: EventLog) -> dict:
    overrides, bulk, cai, corr = log.q(SPLIT_BY_ID)[0]
    drafts = log.one("SELECT count(*) FROM decision_events WHERE kind='ai_draft'")
    return {
        "drafts": drafts,
        "overrides": overrides,
        "naive_override_rate": overrides / drafts,
        "bulk_set": bulk,
        "confirm_as_is": cai,
        "not_a_correction_share": (cai + bulk) / overrides if overrides else 0.0,
        "confirm_as_is_share": cai / overrides if overrides else 0.0,
        "true_corrections": corr,
        "true_correction_rate": corr / drafts,
    }


def pairing_coverage(log: EventLog) -> dict:
    overrides, clean, ambiguous = log.q(FILENAME_JOIN)[0]
    latest_wrong = log.one(FILENAME_LATEST_WRONG)
    by_id = log.one("SELECT count(*) FROM decision_events o JOIN decision_events d "
                    "ON d.decision_id=o.draft_decision_id WHERE o.kind='human_override'")
    return {
        "overrides": overrides,
        "id_join_coverage": by_id / overrides if overrides else 0.0,
        "filename_clean_coverage": clean / overrides if overrides else 0.0,
        "filename_ambiguous_share": ambiguous / overrides if overrides else 0.0,
        "filename_latest_wins_wrong_share": latest_wrong / overrides if overrides else 0.0,
    }


def error_bounds(log: EventLog) -> dict:
    s = override_split(log)
    # reviewed = items a human actually looked at individually (bulk-set rows excluded)
    reviewed = log.one("""SELECT count(DISTINCT subject_id) FROM decision_events
                          WHERE kind = 'human_view' AND actor <> 'adjuster_bulk'""")
    oracle = log.one("""SELECT avg(d.label <> t.true_label) FROM decision_events d
                        JOIN oracle_truth t USING (subject_id) WHERE d.kind='ai_draft'""")
    return {
        "lower_bound": s["true_correction_rate"],
        "reviewed_subset_error": s["true_corrections"] / reviewed if reviewed else 0.0,
        "reviewed_docs": reviewed,
        "oracle_true_error": oracle,
    }


def correction_confusion(log: EventLog) -> dict:
    rows = log.q("""SELECT d.label, o.label, count(*) FROM decision_events o
                    JOIN decision_events d ON d.decision_id = o.draft_decision_id
                    WHERE o.kind='human_override' AND o.label <> d.label GROUP BY 1, 2""")
    total = sum(n for *_, n in rows)
    twin = sum(n for a, b, n in rows if TWIN_OF.get(a) == b)
    return {"pairs": [{"ai": a, "human": b, "n": n} for a, b, n in rows],
            "total": total, "twin_pair_share": twin / total if total else 0.0}


def oracle_gate_stats(log: EventLog) -> dict:
    """Ground-truth view of what the gate let through (lab only)."""
    rows = log.q("""SELECT d.held, d.label, d.confidence, t.true_label
                    FROM decision_events d JOIN oracle_truth t USING (subject_id) WHERE d.kind='ai_draft'""")
    n = len(rows)
    wrong = [(h, lab, c, tr) for h, lab, c, tr in rows if lab != tr]
    auto_wrong = [w for w in wrong if not w[0]]
    twin_err = lambda w: TWIN_OF.get(w[3]) == w[1]  # noqa: E731
    return {
        "hold_rate": sum(r[0] for r in rows) / n,
        "held_but_correct": sum(1 for h, lab, _, tr in rows if h and lab == tr),
        "errors_total": len(wrong),
        "twin_pair_error_share": sum(map(twin_err, wrong)) / len(wrong) if wrong else 0.0,
        "share_of_errors_with_conf_ge_0_9": sum(w[2] >= 0.9 for w in wrong) / len(wrong) if wrong else 0.0,
        "share_of_twin_errors_with_conf_ge_0_9":
            (sum(w[2] >= 0.9 for w in wrong if twin_err(w)) / max(1, sum(map(twin_err, wrong)))),
        "auto_passed_errors": len(auto_wrong),
        "auto_passed_error_rate": len(auto_wrong) / n,
        "auto_passed_twin_errors": sum(map(twin_err, auto_wrong)),
    }
