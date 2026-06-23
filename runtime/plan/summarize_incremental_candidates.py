#!/usr/bin/env python3
"""Summarize incremental candidate-select experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCORE_RE = re.compile(
    r"\[elastic-candidate-score\] budget=(\d+) candidate=(\d+) file=([^ ]+) "
    r"steady=([0-9.]+) transition=([0-9.]+) score=([0-9.]+) changed=(\d+) "
    r"load=([0-9.]+)MB prepare=([0-9.]+)MB evict=([0-9.]+)MB"
)
SELECT_RE = re.compile(
    r"\[elastic-candidate\] budget=(\d+) table_budget=(\d+) candidate=(\d+) file=([^ ]+) "
    r"score=([0-9.]+) transition=([0-9.]+) changed=(\d+) "
    r"load=([0-9.]+)MB prepare=([0-9.]+)MB evict=([0-9.]+)MB"
)


def read_result_row(artifact: Path, log_path: Path) -> dict[str, Any]:
    csv_path = artifact / "summary" / "results.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    log_s = str(log_path)
    for row in rows:
        if row.get("log") == log_s or Path(row.get("log", "")).name == log_path.name:
            return row
    return rows[0]


def parse_candidate_log(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    selections: list[dict[str, Any]] = []
    score_groups: list[list[dict[str, Any]]] = []
    current_scores: list[dict[str, Any]] = []

    for line in text.splitlines():
        sm = SCORE_RE.search(line)
        if sm:
            candidate = int(sm.group(2))
            if candidate == 0 and current_scores:
                score_groups.append(current_scores)
                current_scores = []
            current_scores.append({
                "budget": int(sm.group(1)),
                "candidate": candidate,
                "file": sm.group(3),
                "steady": float(sm.group(4)),
                "transition": float(sm.group(5)),
                "score": float(sm.group(6)),
                "changed": int(sm.group(7)),
                "load_mb": float(sm.group(8)),
                "prepare_mb": float(sm.group(9)),
                "evict_mb": float(sm.group(10)),
            })
            continue

        cm = SELECT_RE.search(line)
        if cm:
            selections.append({
                "budget": int(cm.group(1)),
                "table_budget": int(cm.group(2)),
                "candidate": int(cm.group(3)),
                "file": cm.group(4),
                "score": float(cm.group(5)),
                "transition": float(cm.group(6)),
                "changed": int(cm.group(7)),
                "load_mb": float(cm.group(8)),
                "prepare_mb": float(cm.group(9)),
                "evict_mb": float(cm.group(10)),
            })

    if current_scores:
        score_groups.append(current_scores)

    candidate_counts = Counter(sel["candidate"] for sel in selections)
    budget_counts = Counter(sel["budget"] for sel in selections)
    nonzero = [sel for sel in selections if sel["candidate"] != 0]

    margins: list[float] = []
    nonzero_score_events: list[str] = []
    for group in score_groups:
        if not group:
            continue
        by_id = {row["candidate"]: row for row in group}
        best = min(group, key=lambda row: row["score"])
        base = by_id.get(0)
        if base is not None:
            margins.append(float(base["score"]) - float(best["score"]))
        if best["candidate"] != 0:
            nonzero_score_events.append(
                f"B{best['budget']}: cand{best['candidate']} score={best['score']:.3f} "
                f"transition={best['transition']:.3f} changed={best['changed']}"
            )

    def avg(key: str, rows: list[dict[str, Any]]) -> float:
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

    return {
        "selections": len(selections),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "budget_counts": dict(sorted(budget_counts.items())),
        "nonzero_count": len(nonzero),
        "nonzero_events": nonzero_score_events[:12],
        "avg_transition_ms": avg("transition", selections),
        "avg_load_mb": avg("load_mb", selections),
        "avg_evict_mb": avg("evict_mb", selections),
        "max_base_minus_best_score_ms": max(margins) if margins else 0.0,
        "mean_base_minus_best_score_ms": avg("__value__", [{"__value__": v} for v in margins]),
    }


def fmt_float(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v) if v != "" else "n/a"


def summarize_artifact(artifact: Path) -> list[dict[str, Any]]:
    logs = sorted((artifact / "logs").glob("*candidate-select*.log"))
    rows: list[dict[str, Any]] = []
    for log in logs:
        parsed = parse_candidate_log(log)
        result = read_result_row(artifact, log)
        rows.append({
            "artifact": str(artifact),
            "log": log.name,
            "trace": result.get("trace", ""),
            "method": result.get("method", "candidate-select"),
            "raw_ms_per_token": result.get("raw_ms_per_token", ""),
            "exec_ms_per_token": result.get("exec_ms_per_token", ""),
            "provider_get_ms_total": result.get("provider_get_ms_total", ""),
            "apply_ms_total": result.get("apply_ms_total", ""),
            "apply_count": result.get("apply_count", ""),
            "load_planned": result.get("load_planned", ""),
            "xform_planned": result.get("xform_planned", ""),
            "direct_read_ms": result.get("direct_read_ms", ""),
            "direct_read_calls": result.get("direct_read_calls", ""),
            "direct_read_mb": result.get("direct_read_mb", ""),
            "online_calls": result.get("online_calls", ""),
            **parsed,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Incremental Candidate Summary",
        "",
        "| artifact | trace | raw ms/tok | provider ms | apply count | load/xform | direct read ms/calls/MB | online calls | selected candidates | nonzero | avg transition | avg load MB |",
        "|---|---|---:|---:|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        counts = ", ".join(f"c{k}={v}" for k, v in row.get("candidate_counts", {}).items())
        direct = f"{fmt_float(row.get('direct_read_ms'))}/{row.get('direct_read_calls', 'n/a')}/{fmt_float(row.get('direct_read_mb'))}"
        lines.append(
            "| {artifact} | {trace} | {raw} | {provider} | {apply_count} | {load}/{xform} | {direct} | {online} | {counts} | {nonzero} | {avg_transition} | {avg_load} |".format(
                artifact=Path(str(row.get("artifact", ""))).name,
                trace=row.get("trace", ""),
                raw=fmt_float(row.get("raw_ms_per_token", "")),
                provider=fmt_float(row.get("provider_get_ms_total", "")),
                apply_count=row.get("apply_count", ""),
                load=row.get("load_planned", ""),
                xform=row.get("xform_planned", ""),
                direct=direct,
                online=row.get("online_calls", ""),
                counts=counts,
                nonzero=row.get("nonzero_count", 0),
                avg_transition=fmt_float(row.get("avg_transition_ms", 0.0)),
                avg_load=fmt_float(row.get("avg_load_mb", 0.0)),
            )
        )
    lines.append("")
    for row in rows:
        events = row.get("nonzero_events", [])
        if not events:
            continue
        lines.append(f"## {Path(str(row.get('artifact', ''))).name}")
        lines.append("")
        for event in events:
            lines.append(f"- {event}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize candidate-select score logs")
    ap.add_argument("artifacts", type=Path, nargs="+")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for artifact in args.artifacts:
        rows.extend(summarize_artifact(artifact))
    if args.out_csv:
        write_csv(args.out_csv, rows)
    if args.out_md:
        write_markdown(args.out_md, rows)
    if not args.out_csv and not args.out_md:
        write_markdown(Path("/dev/stdout"), rows)


if __name__ == "__main__":
    main()
