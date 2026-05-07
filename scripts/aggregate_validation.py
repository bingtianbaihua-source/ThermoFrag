#!/usr/bin/env python
"""Phase-7 validation aggregator.

Reads every ``results/eval/phase7/AGGREGATE/<NN>_<task>_summary.json`` and
emits a single ``results/eval/phase7/validation_report.md`` listing each
task with its observed-vs-target threshold, plus an overall PASS/FAIL
roll-up.

Spec: ``docs/validation/00_shared_infrastructure.md`` § "Aggregation
contract".

Usage:

    python scripts/aggregate_validation.py [--out PATH]

Idempotent — re-running overwrites the report from current JSON state.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

DEFAULT_AGG_DIR = Path("results/eval/phase7/AGGREGATE")
DEFAULT_OUT = Path("results/eval/phase7/validation_report.md")

TASK_TITLES = {
    "00_pose_extraction": "Task 0 — Pose extraction (shared infra)",
    "01_mm_gbsa":         "Task 1 — MM-GBSA rescoring",
    "02_multi_scoring":   "Task 2 — Smina + GNINA cross-scoring",
    "03_known_actives":   "Task 3 — Known-actives recovery",
    "04_md":              "Task 4 — MD stability",
    "05_prolif":          "Task 5 — ProLIF pose vs cognate",
    "06_admet":           "Task 6 — ADMET",
    "07_mu_crossval":     "Task 7 — μ-matrix cross-validation",
}


def _fmt_threshold(name: str, t: dict) -> str:
    target = t.get("target", "?")
    observed = t.get("observed", "?")
    passed = t.get("pass")
    if passed is True:
        sym = "✅ PASS"
    elif passed is False:
        sym = "❌ FAIL"
    else:
        sym = "—"
    return f"| `{name}` | {target} | {observed} | {sym} |"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agg_dir", type=Path, default=DEFAULT_AGG_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.agg_dir.exists():
        print(f"[err] {args.agg_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    summaries: dict[str, dict] = {}
    for path in sorted(args.agg_dir.glob("*_summary.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"[warn] failed to parse {path}: {e}", file=sys.stderr)
            continue
        task_id = data.get("task_id") or path.stem.replace("_summary", "")
        summaries[task_id] = data

    if not summaries:
        print(f"[err] no summaries found under {args.agg_dir}", file=sys.stderr)
        sys.exit(2)

    lines = ["# Phase 7 — Validation Report",
             "",
             f"Generated: {dt.datetime.utcnow().isoformat(timespec='seconds')}Z",
             f"Sources: `{args.agg_dir}/*_summary.json` ({len(summaries)} tasks).",
             ""]

    overall_pass = True
    overall_eval_count = 0
    total_pass = 0
    total_fail = 0

    for task_id in sorted(summaries.keys()):
        data = summaries[task_id]
        title = TASK_TITLES.get(task_id, f"Task {task_id}")
        lines.append(f"## {title}")
        lines.append("")
        completed = data.get("completed_utc", "—")
        notes = data.get("notes", "")
        lines.append(f"Completed: `{completed}`")
        if notes:
            lines.append("")
            lines.append(f"Notes: {notes}")
        lines.append("")

        thresholds = data.get("thresholds") or {}
        if not thresholds:
            lines.append("_(no thresholds reported in this task summary)_")
            lines.append("")
            continue
        lines.append("| Threshold | Target | Observed | Pass |")
        lines.append("|---|---|---|---|")
        for name, t in thresholds.items():
            lines.append(_fmt_threshold(name, t))
            if t.get("pass") is True:
                total_pass += 1
                overall_eval_count += 1
            elif t.get("pass") is False:
                total_fail += 1
                overall_eval_count += 1
                overall_pass = False
        lines.append("")

    lines.append("## Roll-up")
    lines.append("")
    lines.append(f"* Tasks reported: **{len(summaries)}** of 8 (0 + 1..7).")
    lines.append(f"* Thresholds evaluated: **{overall_eval_count}** "
                 f"({total_pass} PASS, {total_fail} FAIL).")
    if overall_eval_count == 0:
        verdict = "n/a"
    else:
        verdict = "PASS" if overall_pass else "PARTIAL (some thresholds failed — see honest-failure policy)"
    lines.append(f"* Overall verdict: **{verdict}**.")
    lines.append("")
    lines.append("Honest-failure policy: any FAIL line above is reported as-is "
                 "in the manuscript (mirroring the C4 strain reframing). "
                 "See `docs/VALIDATION_PLAN.md` § Success criteria.")
    lines.append("")

    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"[write] {args.out}  ({len(summaries)} tasks, "
          f"{total_pass}/{overall_eval_count} thresholds PASS)")


if __name__ == "__main__":
    main()
