#!/usr/bin/env python3
"""Generate the four M4 plots from a multi-lora-bench CSV.

Outputs (all PNG, 150 dpi):
  ttft_cdf.png            — TTFT CDF, one curve per adapter
  throughput_timeline.png — req/s + tok/s over 10-second buckets
  adapter_access.png      — request count per adapter
  ttft_scatter.png        — per-request TTFT vs arrival_time

Usage: python plot_results.py <metrics.csv> --out-dir <dir>
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless on server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ttft_cdf(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for adapter, sub in df.groupby("adapter_id"):
        x = np.sort(sub["ttft"].values)
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=f"{adapter} (n={len(x)})", drawstyle="steps-post")
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("CDF")
    ax.set_title("TTFT CDF by adapter")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_throughput_timeline(df: pd.DataFrame, out: Path, bucket_s: float = 10.0) -> None:
    if len(df) == 0:
        return
    t_max = max(df.finish.max(), df.arrival.max())
    bins  = np.arange(0, t_max + bucket_s, bucket_s)
    finished_per_bucket = np.histogram(df.finish, bins=bins)[0]
    arrived_per_bucket  = np.histogram(df.arrival, bins=bins)[0]

    # Tokens out per bucket (attribute output-token count to finish bucket).
    tok_per_bucket = np.zeros(len(bins) - 1, dtype=float)
    finish_idx = np.clip(np.searchsorted(bins, df.finish, side="right") - 1, 0, len(bins) - 2)
    for i, n in zip(finish_idx, df.n_output_tokens):
        tok_per_bucket[i] += n

    centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax2 = ax.twinx()
    ax.plot(centers, arrived_per_bucket  / bucket_s, label="arrivals (req/s)",  marker="o", color="tab:blue")
    ax.plot(centers, finished_per_bucket / bucket_s, label="completions (req/s)", marker="x", color="tab:green")
    ax2.plot(centers, tok_per_bucket / bucket_s, label="tok/s", marker="s", linestyle="--", color="tab:red")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("req/s")
    ax2.set_ylabel("tok/s")
    ax.set_title(f"Arrivals + completions vs tok/s ({int(bucket_s)} s buckets)")
    ax.grid(True, alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_adapter_access(df: pd.DataFrame, out: Path) -> None:
    counts = df["adapter_id"].value_counts().sort_values(ascending=False)
    if len(counts) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(counts)), counts.values, tick_label=counts.index)
    ax.set_xlabel("adapter")
    ax.set_ylabel("request count")
    ax.set_title("Adapter access pattern")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ttft_scatter(df: pd.DataFrame, out: Path) -> None:
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for adapter, sub in df.groupby("adapter_id"):
        ax.scatter(sub.arrival, sub.ttft, label=adapter, alpha=0.7, s=24)
    ax.set_xlabel("arrival time (s)")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Per-request TTFT vs arrival")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["ttft"] = df.first_token - df.arrival

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_ttft_cdf          (df, args.out_dir / "ttft_cdf.png")
    plot_throughput_timeline(df, args.out_dir / "throughput_timeline.png")
    plot_adapter_access    (df, args.out_dir / "adapter_access.png")
    plot_ttft_scatter      (df, args.out_dir / "ttft_scatter.png")
    print(f"[plot] wrote 4 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
