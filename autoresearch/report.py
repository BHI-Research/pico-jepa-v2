"""CLI: ``python -m autoresearch.report``.

Reads the SQLite ledger and prints summary tables. Supports:
  --last N        last N experiments
  --phase P       filter by phase (pretrain | classify | ensemble | phase4)
  --best          best of each phase
  --hypothesis    Phase 4 results: ensemble vs general gap with bootstrap CI

Designed to be glanceable in a terminal — no fancy formatting, just rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from autoresearch.ledger import Ledger


def _fmt(v: Any, width: int = 10) -> str:
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        return f"{v:.4f}".rjust(width)
    return str(v).rjust(width)


def report_recent(ledger: Ledger, n: int, phase: Optional[str]) -> None:
    rows = ledger.recent_history(limit=n, phase=phase)
    if not rows:
        print("(no experiments)")
        return
    print(f"{'id':>5} {'phase':>10} {'status':>12} {'score':>10} {'wallclock':>10} {'config_hash':>16}")
    print("-" * 80)
    for r in rows:
        print(
            f"{_fmt(r['id'], 5)} {_fmt(r['phase'], 10)} {_fmt(r['status'], 12)} "
            f"{_fmt(r.get('score'), 10)} {_fmt(r.get('wallclock_s'), 10)} "
            f"{_fmt(r.get('config_hash'), 16)}"
        )


def report_best(ledger: Ledger) -> None:
    print(f"{'phase':>10} {'best_id':>8} {'score':>10} {'baseline':>10} {'dirty':>6}")
    print("-" * 60)
    for phase in ("pretrain", "classify", "ensemble", "phase4"):
        rec = ledger.get_ratchet(phase)
        if rec is None:
            print(f"{phase:>10} {'-':>8} {'-':>10} {'-':>10} {'-':>6}")
            continue
        print(
            f"{phase:>10} {_fmt(rec['best_experiment_id'], 8)} {_fmt(rec.get('current_score'), 10)} "
            f"{_fmt(rec.get('baseline_score'), 10)} {_fmt(rec.get('dirty'), 6)}"
        )


def report_hypothesis(ledger: Ledger) -> None:
    rows = ledger.recent_history(limit=200, phase="phase4")
    if not rows:
        print("(no phase 4 evaluations yet)")
        return
    print(f"{'id':>5} {'gap':>8} {'CI_low':>8} {'CI_high':>8} {'ens_top1':>10} {'gen_top1':>10}")
    print("-" * 75)
    for r in rows:
        m: Dict[str, Any] = {}
        try:
            m = json.loads(r.get("metrics_json") or "{}")
        except Exception:
            pass
        print(
            f"{_fmt(r['id'], 5)} {_fmt(m.get('gap'), 8)} {_fmt(m.get('gap_ci_low'), 8)} "
            f"{_fmt(m.get('gap_ci_high'), 8)} {_fmt(m.get('ensemble_top1'), 10)} "
            f"{_fmt(m.get('general_top1'), 10)}"
        )

    latest = rows[0]
    try:
        m = json.loads(latest.get("metrics_json") or "{}")
    except Exception:
        m = {}
    gap = m.get("gap")
    ci_low = m.get("gap_ci_low")
    if gap is not None and ci_low is not None:
        verdict = "supported" if (ci_low > 0) else ("inconclusive" if gap > 0 else "rejected")
        print(f"\nHypothesis (latest phase 4): gap={gap:.4f}, CI low={ci_low:.4f} -> {verdict}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Inspect the autoresearch ledger.")
    p.add_argument("--ledger-path", default=None)
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--phase", default=None, choices=["pretrain", "classify", "ensemble", "phase4"])
    p.add_argument("--best", action="store_true")
    p.add_argument("--hypothesis", action="store_true")
    args = p.parse_args(argv)

    ledger = Ledger(args.ledger_path) if args.ledger_path else Ledger()

    if args.best:
        report_best(ledger)
        return 0
    if args.hypothesis:
        report_hypothesis(ledger)
        return 0
    report_recent(ledger, n=args.last, phase=args.phase)
    return 0


if __name__ == "__main__":
    sys.exit(main())
