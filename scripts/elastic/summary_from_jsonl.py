#!/usr/bin/env python3
# scripts/elastic/summary_from_jsonl.py
#
# 把 elastic baseline 写出的 metrics JSONL 折算成单 summary JSON。
# 对齐 docs/elastic/baseline_external_execution.md §5 第 10 条验收标准：
#   "最后输出 JSON summary：总 token 数、TPOT 的 mean/p50/p90/p99、
#    总 flash 字节、合规率（必须 1.0）、GPU buffer churn 速率"
#
# 用法：
#   ./summary_from_jsonl.py metrics.jsonl > summary.json
#
# JSONL 每行 schema：见 runtime/metrics_logger.h。
#   {t, B_t_mb, resident_bytes, n_blocks_resident, token_id,
#    layer_load_latency_ms, decode_latency_ms, flash_bytes_read}

import argparse
import json
import statistics
import sys


def pct(xs, p):
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = max(0, min(len(xs_sorted) - 1, int(round(p / 100.0 * (len(xs_sorted) - 1)))))
    return xs_sorted[k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", help="metrics 文件")
    p.add_argument("--baseline-tokens-file", help="基线运行的 stdout 文件，比较 token 一致性")
    p.add_argument("--elastic-tokens-file", help="elastic 运行的 stdout 文件，比较 token 一致性")
    args = p.parse_args()

    records = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: skip bad line: {e}", file=sys.stderr)

    if not records:
        sys.exit("空 JSONL")

    # 时间相邻 record 之差当作 TPOT（per-token latency）
    ts = [r["t"] for r in records]
    tpot_sec = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    tpot_ms = [x * 1000.0 for x in tpot_sec if x > 0]

    B = [r["B_t_mb"] for r in records]
    R_mb = [r["resident_bytes"] / 1024.0 / 1024.0 for r in records]
    nb = [r["n_blocks_resident"] for r in records]
    flash = [r["flash_bytes_read"] for r in records]
    flash_total = flash[-1] if flash else 0

    # 合规：resident_mb <= B_t_mb（与规范 §5 第 7 条一致）
    violations = [
        (r["t"], r["B_t_mb"], r["resident_bytes"] / 1024.0 / 1024.0)
        for r in records
        if r["resident_bytes"] / 1024.0 / 1024.0 > r["B_t_mb"]
    ]
    compliance_rate = 1.0 - len(violations) / len(records) if records else 0.0

    # buffer churn：用 flash_bytes_read 变化作为 reload 近似（每次 reload 写一段）
    duration_s = ts[-1] - ts[0] if len(ts) >= 2 else 0.0
    flash_rate_mb_per_s = (flash_total / 1024.0 / 1024.0) / duration_s if duration_s > 0 else 0.0

    summary = {
        "n_records": len(records),
        "duration_sec": duration_s,
        "tpot_ms": {
            "n": len(tpot_ms),
            "mean": statistics.fmean(tpot_ms) if tpot_ms else 0.0,
            "p50":  pct(tpot_ms, 50),
            "p90":  pct(tpot_ms, 90),
            "p99":  pct(tpot_ms, 99),
        },
        "B_t_mb":     {"min": min(B), "max": max(B), "mean": statistics.fmean(B)},
        "resident_mb": {"min": min(R_mb), "max": max(R_mb), "mean": statistics.fmean(R_mb)},
        "n_blocks_resident": {"min": min(nb), "max": max(nb)},
        "flash_bytes_read_total":    flash_total,
        "flash_bytes_read_total_mb": flash_total / 1024.0 / 1024.0,
        "flash_rate_mb_per_s":       flash_rate_mb_per_s,
        "compliance_rate":           compliance_rate,
        "violations": violations[:5],  # 前 5 条违规
    }

    # 可选：比较 token 一致性
    if args.baseline_tokens_file and args.elastic_tokens_file:
        with open(args.baseline_tokens_file) as f:
            baseline = f.read().strip()
        with open(args.elastic_tokens_file) as f:
            elastic = f.read().strip()
        summary["token_match"] = baseline == elastic
        if not summary["token_match"]:
            summary["token_diff_preview"] = {
                "baseline_head": baseline[:200],
                "elastic_head":  elastic[:200],
            }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
