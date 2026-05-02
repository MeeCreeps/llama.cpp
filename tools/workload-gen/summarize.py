#!/usr/bin/env python3
"""Print a one-scenario summary from a multi-lora-bench CSV + sidecar.

Usage: python summarize.py <metrics.csv>

Reads <csv> and the matching <csv>.summary.json (written by main.cpp),
emits a human-readable block to stdout.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sj_path = args.csv.with_suffix(".csv.summary.json")
    sj = json.loads(sj_path.read_text()) if sj_path.exists() else {}

    df["ttft"]    = df.first_token - df.arrival
    df["e2e"]     = df.finish - df.arrival
    df["decode"]  = df.finish - df.first_token
    df["tpot_ms"] = (df.decode / (df.n_output_tokens - 1).clip(lower=1)) * 1000

    n        = len(df)
    wall     = df.finish.max() if n else 0.0
    out_tok  = int(df.n_output_tokens.sum())
    in_tok   = int(df.n_input_tokens.sum())

    def pct(x, q):
        return float(x.quantile(q)) if len(x) else float("nan")

    print(f"=== {args.csv.name} ===")
    print(f"  requests:      {n}")
    print(f"  wall time:     {wall:.2f} s")
    print(f"  total in tok:  {in_tok}")
    print(f"  total out tok: {out_tok}")
    print(f"  throughput:    req/s={n/wall:.2f}  tok/s={out_tok/wall:.2f}" if wall > 0 else "")
    print(f"")
    print(f"  TTFT  mean={df.ttft.mean()*1000:.1f}ms  p50={pct(df.ttft, .50)*1000:.1f}ms  p99={pct(df.ttft, .99)*1000:.1f}ms")
    print(f"  TPOT  mean={df.tpot_ms.mean():.1f}ms  p50={pct(df.tpot_ms, .50):.1f}ms  p99={pct(df.tpot_ms, .99):.1f}ms")
    print(f"  e2e   mean={df.e2e.mean()*1000:.1f}ms  p50={pct(df.e2e, .50)*1000:.1f}ms  p99={pct(df.e2e, .99)*1000:.1f}ms")
    print(f"")
    if sj:
        budget_mib = sj.get("budget_bytes", 0) // (1024 * 1024)
        peak_mib   = sj.get("peak_bytes",   0) // (1024 * 1024)
        print(f"  cache  hit={sj.get('hits', 0)}  miss={sj.get('misses', 0)}  evict={sj.get('evicts', 0)}")
        print(f"         hit_rate={sj.get('hit_rate', 0):.4f}")
        print(f"         budget={budget_mib} MiB (0=unbounded)  peak={peak_mib} MiB  resident_count={sj.get('resident_count',0)}")
        print(f"  config n_slots={sj.get('n_slots','?')}  n_ctx={sj.get('n_ctx','?')}")
    print(f"")
    print(f"  per-adapter:")
    if "adapter_id" in df:
        agg = df.groupby("adapter_id").agg(
            n=("id", "count"),
            ttft_mean_ms=("ttft", lambda s: s.mean() * 1000),
            tpot_mean_ms=("tpot_ms", "mean"),
            n_out_mean=("n_output_tokens", "mean"),
        ).round(1)
        for adapter, row in agg.iterrows():
            print(f"    {adapter:<14s} n={int(row.n):3d}  ttft={row.ttft_mean_ms:7.1f}ms  tpot={row.tpot_mean_ms:6.1f}ms  out_tok={row.n_out_mean:5.1f}")


if __name__ == "__main__":
    main()
