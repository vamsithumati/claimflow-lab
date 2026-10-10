import sqlite3
import unittest

import numpy as np

from claimflow import metrics as m
from claimflow.audit import run_audit
from claimflow.claims import error_at_budget
from claimflow.config import LabConfig
from claimflow.events import EventLog
from claimflow.pipeline import run_scenario
from claimflow.triage import twin_evidence_present


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
    def test_split_by_id(self):
        log = EventLog()
        d1, d2 = _draft(log, "a", "a.pdf"), _draft(log, "b", "b.pdf")
        _human(log, "human_override", d1, "a", "repair_estimate", "a.pdf", action="repick")
        _human(log, "human_override", d2, "b", "repair_invoice", "b.pdf")
        s = m.override_split(log)
        self.assertEqual((s["overrides"], s["confirm_as_is"], s["true_corrections"]), (2, 1, 1))

    def test_filename_join_is_ambiguous_on_shared_names(self):
        log = EventLog()
        d1 = _draft(log, "a")
        _draft(log, "b")  # same tenant, same 'scan.pdf'
        _human(log, "human_override", d1, "a", "repair_invoice")
        p = m.pairing_coverage(log)
        self.assertEqual(p["filename_ambiguous_share"], 1.0)
        self.assertEqual(p["filename_latest_wins_wrong_share"], 1.0)  # latest is 'b', truth is 'a'
        self.assertEqual(p["id_join_coverage"], 1.0)


class TriageTests(unittest.TestCase):
    def test_twin_verifier(self):
        self.assertTrue(twin_evidence_present("repair_invoice", frozenset({"paid", "invoice_no"})))
        self.assertFalse(twin_evidence_present("repair_invoice", frozenset({"labour", "parts"})))
        # own evidence AND the twin's evidence → ambiguous → not verified
        self.assertFalse(twin_evidence_present("repair_invoice", frozenset({"paid", "quote"})))


class BudgetTests(unittest.TestCase):
    def test_error_at_budget_interpolates(self):
        pts = [{"hold_rate": 0.0, "auto_passed_error_rate": 0.2},
               {"hold_rate": 0.4, "auto_passed_error_rate": 0.0}]
        self.assertAlmostEqual(error_at_budget(pts, 0.2), 0.1)
        self.assertEqual(error_at_budget(pts[1:], 0.2), float("inf"))


class EndToEndTests(unittest.TestCase):
    cfg = LabConfig(n_docs=1500, n_train=1500)

    def test_deterministic(self):
        a = m.override_split(run_scenario(self.cfg).log)
        b = m.override_split(run_scenario(self.cfg).log)
        self.assertEqual(a, b)

    def test_bounds_bracket_truth_and_audit_is_close(self):
        run = run_scenario(self.cfg)
        b = m.error_bounds(run.log)
        self.assertLessEqual(b["lower_bound"], b["oracle_true_error"])
        self.assertGreater(b["reviewed_subset_error"], b["oracle_true_error"])
        est = np.mean([run_audit(run, self.cfg, np.random.default_rng(i)).estimate for i in range(20)])
        self.assertAlmostEqual(est, b["oracle_true_error"], delta=0.03)

    def test_one_click_accept_has_no_confirm_as_is(self):
        s = m.override_split(run_scenario(self.cfg, ui="one_click_accept").log)
        self.assertEqual(s["confirm_as_is"], 0)


if __name__ == "__main__":
    unittest.main()
