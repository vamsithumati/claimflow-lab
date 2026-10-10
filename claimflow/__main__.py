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
    ap.add_argument("--report-only", action="store_true", help="rebuild claims/report/charts from results.json")
    args = ap.parse_args()
    cfg = LabConfig(seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    q = args.quick
    if args.report_only:
        results = json.loads((out / "results.json").read_text())["results"]
        claims = evaluate(results)
        (out / "results.json").write_text(json.dumps({"results": results, "claims": claims}, indent=2, default=str))
        all_charts(results, out / "charts")
        print(f"rebuilt {write_report(results, claims, cfg, out)}")
        return
    results = {
        "e1": ex.e1_override_split(cfg, seeds=3 if q else 20),
        "e2": ex.e2_bounds_and_audit(cfg, audits=20 if q else 100, seeds=2 if q else 10, per_seed=5 if q else 20),
        "e3": ex.e3_gate(cfg, seeds=3 if q else 20),
        "e4": ex.e4_pairing(cfg, seeds=2 if q else 10),
    }
    claims = evaluate(results)
    (out / "results.json").write_text(json.dumps({"results": results, "claims": claims}, indent=2, default=str))
    all_charts(results, out / "charts")
    report = write_report(results, claims, cfg, out)
    for c in claims:
        print(f"{c['id']:<4} {c['evidence']:<16} {c['verdict']:<10} {c['claim']}")
    print(f"\nwrote {report} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
