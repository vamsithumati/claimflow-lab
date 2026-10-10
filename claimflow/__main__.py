"""Run every experiment, evaluate the claims, write results/ (JSON, REPORT.md, charts)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import experiments as ex
from .charts import all_charts
from .claims import evaluate
from .config import LabConfig
from .report import write_report


def main() -> None:
    ap = argparse.ArgumentParser(prog="claimflow")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="fewer repetitions (smoke run)")
    args = ap.parse_args()
    cfg = LabConfig(seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reps = dict(seeds=3, audits=30, audit_seeds=2) if args.quick else dict(seeds=20, audits=300, audit_seeds=10)
    results = {
        "e1": ex.e1_override_split(cfg, seeds=reps["seeds"]),
        "e2": ex.e2_bounds_and_audit(cfg, audits=reps["audits"], seeds=reps["audit_seeds"]),
        "e3": ex.e3_twins(cfg),
        "e4": ex.e4_pairing(cfg),
        "sensitivity": ex.sensitivity(cfg),
    }
    claims = evaluate(results)
    (out / "results.json").write_text(json.dumps({"results": results, "claims": claims}, indent=2, default=str))
    all_charts(results, out / "charts")
    report = write_report(results, claims, cfg, out)
    for c in claims:
        print(f"{c['id']:<4} {c['verdict']:<10} {c['claim']}")
    print(f"\nwrote {report} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
