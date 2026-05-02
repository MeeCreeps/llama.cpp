#!/usr/bin/env python3
"""Aggregate M3 sweep CSVs + JSON sidecars into a single table.

Reads every results/M3/sweep/sweep_*.csv (+ matching .summary.json) and
prints a Markdown table for pasting into M3.md.

Usage: python tools/workload-gen/m3_summarize.py results/M3/sweep
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


BUDGET_RE = re.compile(r"sweep_(\d+)\.csv$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path, help="directory holding sweep_*.csv + .summary.json")
    args = ap.parse_args()

    rows = []
    for csv in sorted(args.dir.glob("sweep_*.csv")):
        m = BUDGET_RE.search(csv.name)
        if not m:
            continue
        budget = int(m.group(1))  # MiB
        sj_path = csv.with_suffix(".csv.summary.json")
        if not sj_path.exists():
            print(f"warn: missing {sj_path}", file=sys.stderr)
            continue
        sj = json.loads(sj_path.read_text())

        df = pd.read_csv(csv)
        df["ttft"] = df.first_token - df.arrival
        df["tpot"] = (df.finish - df.first_token) / (df.n_output_tokens - 1).clip(lower=1)

        rows.append({
            "budget_MiB": budget,
            "n_reqs":     sj["n_requests"],
            "hits":       sj["hits"],
            "misses":     sj["misses"],
            "evicts":     sj["evicts"],
            "hit_rate":   round(sj["hit_rate"], 3),
            "resident_MiB": sj["resident_bytes"] // (1024 * 1024),
            "peak_MiB":   sj["peak_bytes"] // (1024 * 1024),
            "ttft_mean":  round(df.ttft.mean(), 2),
            "ttft_p99":   round(df.ttft.quantile(0.99), 2),
            "tpot_ms":    round(df.tpot.mean() * 1000, 1),
            "wall_s":     round(df.finish.max(), 2),
            "tok_per_s":  round(df.n_output_tokens.sum() / df.finish.max(), 2),
        })

    if not rows:
        print("no data", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame(rows).sort_values("budget_MiB", ascending=False, kind="stable")
    print(out.to_markdown(index=False))


if __name__ == "__main__":
    main()
