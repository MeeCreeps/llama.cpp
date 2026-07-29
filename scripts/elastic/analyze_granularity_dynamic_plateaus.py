#!/usr/bin/env python3
"""Aggregate repeated Elastic dynamic runs by contiguous budget plateau.

The sweep runner keeps transition tokens in the raw per-token CSV.  This
analysis deliberately drops the first one-token decode after every observed
budget change, computes one statistic per process/plateau, and only then
aggregates across processes.  Thus tokens are not incorrectly treated as
independent experimental repeats.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_LABELS = ("90a", "40a", "45a", "50", "45b", "40b", "90b")


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = rank - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    args = parser.parse_args()

    run_path = args.result_dir / "dynamic_runs.csv"
    with run_path.open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    run_meta = {row["run_id"]: row for row in runs}

    plateau_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for token_path in sorted((args.result_dir / "tokens").glob("*.csv")):
        run_id = token_path.stem
        meta = run_meta.get(run_id)
        if meta is None:
            raise SystemExit(f"missing dynamic_runs.csv row for {run_id}")
        with token_path.open(newline="", encoding="utf-8") as handle:
            tokens = list(csv.DictReader(handle))

        groups: list[tuple[str, list[dict[str, str]]]] = []
        for token in tokens:
            budget = token["budget_mib"]
            if not groups or groups[-1][0] != budget:
                groups.append((budget, []))
            groups[-1][1].append(token)
        transient_groups_dropped = 0
        while len(groups) > len(args.labels):
            # A token whose evaluation straddles a BudgetWatcher boundary can
            # observe one intermediate MiB value.  It is the transition token
            # that this analysis excludes, not a real dwell plateau.
            candidates = [
                index for index in range(1, len(groups) - 1)
                if sum(int(token["n_tokens"]) == 1 for token in groups[index][1]) <= 1
            ]
            if not candidates:
                break
            del groups[candidates[0]]
            transient_groups_dropped += 1
        if len(groups) != len(args.labels):
            raise SystemExit(
                f"{run_id}: observed {len(groups)} budget plateaus, "
                f"but {len(args.labels)} labels were supplied")

        resident_rows = [
            token for token in tokens
            if token.get("resident_mib") not in (None, "")
            and token.get("target_mib") not in (None, "")
            and float(token["resident_mib"]) >= 0
            and float(token["target_mib"]) >= 0
        ]
        overshoots = [
            float(token["resident_mib"]) - float(token["target_mib"])
            for token in resident_rows
        ]
        audit_rows.append({
            "run_id": run_id,
            "backend": meta["backend"],
            "repeat": meta["repeat"],
            "mode": meta["mode"],
            "n_token_rows": len(tokens),
            "n_state_rows": len(resident_rows),
            "max_resident_minus_target_mib": max(overshoots) if overshoots else "",
            "n_budget_violations": sum(value > 0.01 for value in overshoots),
            "n_transient_budget_groups_dropped": transient_groups_dropped,
        })

        for ordinal, (label, (budget, group)) in enumerate(zip(args.labels, groups)):
            decodes = [token for token in group if int(token["n_tokens"]) == 1]
            if decodes:
                decodes = decodes[1:]
            if not decodes:
                continue
            latency = [float(token["latency_ms"]) for token in decodes]
            resident = [
                float(token["resident_mib"]) for token in decodes
                if token.get("resident_mib") not in (None, "")
                and float(token["resident_mib"]) >= 0
            ]
            reloads = [
                int(token["reloads_total"]) for token in decodes
                if token.get("reloads_total") not in (None, "")
            ]
            evicts = [
                int(token["evicts_total"]) for token in decodes
                if token.get("evicts_total") not in (None, "")
            ]
            intervals = max(1, len(decodes) - 1)
            plateau_rows.append({
                "run_id": run_id,
                "backend": meta["backend"],
                "repeat": meta["repeat"],
                "mode": meta["mode"],
                "plateau": label,
                "plateau_ordinal": ordinal,
                "budget_mib": budget,
                "n_decode": len(decodes),
                "median_latency_ms": statistics.median(latency),
                "p95_latency_ms": percentile(latency, 95),
                "mean_latency_ms": statistics.mean(latency),
                "max_resident_mib": max(resident) if resident else "",
                "target_mib": decodes[-1].get("target_mib", ""),
                "reloads_per_token": (reloads[-1] - reloads[0]) / intervals if reloads else "",
                "evicts_per_token": (evicts[-1] - evicts[0]) / intervals if evicts else "",
            })

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_repeat: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in plateau_rows:
        key = (str(row["mode"]), str(row["plateau"]))
        grouped[key].append(row)
        by_repeat[(str(row["repeat"]), str(row["plateau"]))][str(row["mode"])] = row

    aggregate_rows: list[dict[str, object]] = []
    label_order = {label: index for index, label in enumerate(args.labels)}
    for (mode, label), values in sorted(
            grouped.items(), key=lambda item: (label_order[item[0][1]], item[0][0])):
        medians = [float(row["median_latency_ms"]) for row in values]
        p95s = [float(row["p95_latency_ms"]) for row in values]
        means = [float(row["mean_latency_ms"]) for row in values]
        def win_count(metric: str) -> int:
            wins = 0
            for (_, repeat_label), mode_values in by_repeat.items():
                if repeat_label != label or not mode_values:
                    continue
                winner = min(
                    mode_values,
                    key=lambda candidate: float(mode_values[candidate][metric]))
                wins += winner == mode
            return wins
        aggregate_rows.append({
            "mode": mode,
            "plateau": label,
            "n_runs": len(values),
            "n_decode_total": sum(int(row["n_decode"]) for row in values),
            "median_of_run_medians_ms": statistics.median(medians),
            "median_of_run_p95_ms": statistics.median(p95s),
            "median_of_run_means_ms": statistics.median(means),
            "min_run_median_ms": min(medians),
            "max_run_median_ms": max(medians),
            "p50_win_count": win_count("median_latency_ms"),
            "mean_win_count": win_count("mean_latency_ms"),
            "p95_win_count": win_count("p95_latency_ms"),
        })

    write_csv(args.result_dir / "dynamic_plateau_runs.csv", plateau_rows)
    write_csv(args.result_dir / "dynamic_plateau_aggregate.csv", aggregate_rows)
    write_csv(args.result_dir / "residency_audit.csv", audit_rows)


if __name__ == "__main__":
    main()
