import sqlite3
import unittest

import numpy as np

from claimflow import metrics as m
from claimflow.audit import run_audit
from claimflow.config import LabConfig
from claimflow.data import TWIN_OF, make_corpus
from claimflow.events import EventLog
from claimflow.experiments import at_budget, operating_points
from claimflow.humans import noisy_label
from claimflow.pipeline import classify_corpus, run_scenario
from claimflow.triage import twin_verdict


def _draft(log, sid, fname="scan.pdf", label="repair_estimate", tenant=1):
    return log.append("ai_draft", subject_id=sid, tenant_id=tenant, filename=fname, label=label,
                      confidence=0.9, held=True)


def _human(log, kind, draft_id, sid, label, fname="scan.pdf", tenant=1, action="corrected"):
    log.append(kind, subject_id=sid, tenant_id=tenant, filename=fname, label=label,
               draft_decision_id=draft_id, action=action)


class EventLogTests(unittest.TestCase):
    def test_append_only(self):
        log = EventLog()
        _draft(log, "a")
        with self.assertRaises(sqlite3.IntegrityError):
            log.conn.execute("UPDATE decision_events SET label='x'")
        with self.assertRaises(sqlite3.IntegrityError):
            log.conn.execute("DELETE FROM decision_events")


class MetricTests(unittest.TestCase):
    def test_split_by_id_separates_bulk(self):
        log = EventLog()
        d1, d2, d3 = _draft(log, "a", "a.pdf"), _draft(log, "b", "b.pdf"), _draft(log, "c", "c.pdf")
        _human(log, "human_override", d1, "a", "repair_estimate", "a.pdf", action="repick")
        _human(log, "human_override", d2, "b", "repair_invoice", "b.pdf")
        _human(log, "human_override", d3, "c", "repair_estimate", "c.pdf", action="bulk_set")
        s = m.override_split(log)
        self.assertEqual((s["overrides"], s["confirm_as_is"], s["bulk_set"], s["true_corrections"]), (3, 1, 1, 1))

    def test_filename_join_is_ambiguous_on_shared_names(self):
        log = EventLog()
        d1 = _draft(log, "a")
        _draft(log, "b")  # same tenant, same 'scan.pdf'
        _human(log, "human_override", d1, "a", "repair_invoice")
        p = m.pairing_coverage(log)
        self.assertEqual(p["filename_ambiguous_share"], 1.0)
        self.assertEqual(p["filename_latest_wins_wrong_share"], 1.0)
        self.assertEqual(p["id_join_coverage"], 1.0)


class TriageTests(unittest.TestCase):
    def test_twin_verdict(self):
        self.assertEqual(twin_verdict("repair_invoice", frozenset({"paid", "invoice_no"})), ("repair_invoice", "verified"))
        self.assertEqual(twin_verdict("repair_invoice", frozenset({"labour"}))[1], "ambiguous")
        self.assertEqual(twin_verdict("repair_invoice", frozenset({"quote"})), ("repair_estimate", "relabelled"))
        self.assertEqual(twin_verdict("repair_invoice", frozenset({"paid", "quote"}))[1], "ambiguous")

    def test_noisy_label_errors_go_to_twin(self):
        self.assertEqual(noisy_label("repair_invoice", 0.9, 1.0, 0.95, 0.0, 0.0), "repair_estimate")
        self.assertEqual(noisy_label("repair_invoice", 0.9, 1.0, 0.10, 0.0, 0.0), "repair_invoice")


class OperatingPointTests(unittest.TestCase):
    def test_budget_exact_and_deterministic(self):
        pts = [{"hold_rate": 0.0, "auto_passed_error_rate": 0.2},
               {"hold_rate": 0.4, "auto_passed_error_rate": 0.0}]
        r = at_budget(pts, 0.2)
        self.assertAlmostEqual(r["randomised_exact"], 0.1)        # half the boundary items held at random
        self.assertEqual((r["deterministic"], r["deterministic_hold"]), (0.2, 0.0))
        self.assertIsNone(at_budget(pts[1:], 0.2))

    def test_family_scope_never_holds_other_labels(self):
        cfg = LabConfig(n_docs=600)
        cl = classify_corpus(cfg, "keyword_router", make_corpus(cfg, 600))
        fam_share = sum(lab in TWIN_OF for _, lab, _ in cl) / len(cl)
        top = max(p["hold_rate"] for p in operating_points(cl, "off", "family"))
        self.assertAlmostEqual(top, fam_share)

    def test_points_are_monotone(self):
        cfg = LabConfig(n_docs=800)
        pts = operating_points(classify_corpus(cfg, "keyword_router_margin", make_corpus(cfg, 800)), "off")
        holds = [p["hold_rate"] for p in pts]
        self.assertEqual(holds, sorted(holds))
        self.assertEqual(pts[-1]["hold_rate"], 1.0)


class OracleQuarantineTests(unittest.TestCase):
    def test_production_metrics_never_read_oracle(self):
        import inspect
        for fn in (m.override_split, m.pairing_coverage, m.correction_confusion):
            self.assertNotIn("oracle_truth", inspect.getsource(fn))
        self.assertNotIn("oracle_truth", m.SPLIT_BY_ID + m.FILENAME_JOIN + m.FILENAME_LATEST_WRONG)


class AuditTests(unittest.TestCase):
    def test_coverage_and_gold_correction(self):
        cfg = LabConfig(n_docs=2000, n_train=1500)
        run = run_scenario(cfg)
        truth = m.error_bounds(run.log)["oracle_true_error"]
        res = [run_audit(run, cfg, np.random.default_rng(i), "double_all") for i in range(40)]
        self.assertGreaterEqual(np.mean([a.covers(truth) for a in res]), 0.85)
        self.assertAlmostEqual(np.mean([a.estimate for a in res]), truth, delta=0.015)
        weak = cfg.with_(audit_labeller_accuracy=0.90)
        biased = np.mean([run_audit(run, weak, np.random.default_rng(i), "none").estimate for i in range(20)])
        fixed = np.mean([run_audit(run, weak, np.random.default_rng(i), "gold_corrected").estimate for i in range(20)])
        self.assertLess(abs(fixed - truth), abs(biased - truth))


class EndToEndTests(unittest.TestCase):
    cfg = LabConfig(n_docs=1500, n_train=1500)

    def test_deterministic(self):
        self.assertEqual(m.override_split(run_scenario(self.cfg).log), m.override_split(run_scenario(self.cfg).log))

    def test_timeline_interleaves(self):
        seqs = run_scenario(self.cfg).log.q("SELECT kind FROM decision_events ORDER BY seq")
        first_human = next(i for i, (k,) in enumerate(seqs) if k != "ai_draft")
        self.assertLess(first_human, 1500)  # humans act before all drafts are in

    def test_paired_uis_share_reviewers(self):
        a = m.override_split(run_scenario(self.cfg, ui="pick_to_accept_no_bulk").log)
        b = m.override_split(run_scenario(self.cfg, ui="one_click_accept").log)
        self.assertEqual(a["true_corrections"], b["true_corrections"])
        self.assertEqual(b["confirm_as_is"], 0)

    def test_bounds_and_audit(self):
        run = run_scenario(self.cfg)
        b = m.error_bounds(run.log)
        self.assertLessEqual(b["lower_bound"], b["oracle_true_error"])
        self.assertGreater(b["reviewed_subset_error"], b["oracle_true_error"])


if __name__ == "__main__":
    unittest.main()
