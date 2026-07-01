#!/usr/bin/env python3
"""Plot trace memory vs full-residency memory requirement as SVG.

Each output SVG has three panels:
1. raw available memory trace;
2. gap to the full-residency requirement;
3. percentage of the requirement covered by the trace.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path


def read_trace(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            t = float(row.get("t_sec", i) or i)
            m = float(row.get("mem_available_mb", row.get("budget_mib", 0)) or 0)
            rows.append((t, m))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def planned_weight_mib(model_meta: Path) -> float:
    payload = json.loads(model_meta.read_text())
    total = 0
    for w in payload.get("weights", []):
        total += int(w.get("byte_size", w.get("bytes", 0)) or 0)
    return total / 1024.0 / 1024.0


def line_points(xs: list[float], ys: list[float], x0: float, y0: float, w: float, h: float,
                xmin: float, xmax: float, ymin: float, ymax: float) -> str:
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    pts = []
    for x, y in zip(xs, ys):
        px = x0 + (x - xmin) / (xmax - xmin) * w
        py = y0 + h - (y - ymin) / (ymax - ymin) * h
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def nice_bounds(values: list[float], include: list[float] | None = None, pad_frac: float = 0.08) -> tuple[float, float]:
    vals = list(values)
    if include:
        vals.extend(include)
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def svg_text(x: float, y: float, text: str, size: int = 13, weight: str = "400",
             anchor: str = "start", color: str = "#1f2937") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{html.escape(text)}</text>'
    )


def panel(svg: list[str], title: str, xs: list[float], ys: list[float], *,
          x0: float, y0: float, w: float, h: float, xmin: float, xmax: float,
          ymin: float, ymax: float, line_color: str, zero: float | None = None,
          required: float | None = None, unit: str = "MiB") -> None:
    svg.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#ffffff" stroke="#d1d5db"/>')
    svg.append(svg_text(x0, y0 - 10, title, size=14, weight="700"))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = y0 + h - frac * h
        val = ymin + frac * (ymax - ymin)
        svg.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0 + w:.1f}" y2="{yy:.1f}" stroke="#eef2f7"/>')
        svg.append(svg_text(x0 - 8, yy + 4, f"{val:.0f}", size=11, anchor="end", color="#6b7280"))
    if zero is not None and ymin <= zero <= ymax:
        yy = y0 + h - (zero - ymin) / (ymax - ymin) * h
        svg.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0 + w:.1f}" y2="{yy:.1f}" stroke="#111827" stroke-width="1.2" stroke-dasharray="4 4"/>')
    if required is not None and ymin <= required <= ymax:
        yy = y0 + h - (required - ymin) / (ymax - ymin) * h
        svg.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0 + w:.1f}" y2="{yy:.1f}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="6 4"/>')
        svg.append(svg_text(x0 + w - 4, yy - 5, f"required {required:.0f} {unit}", size=11, anchor="end", color="#dc2626"))
    pts = line_points(xs, ys, x0, y0, w, h, xmin, xmax, ymin, ymax)
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{line_color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')
    svg.append(svg_text(x0, y0 + h + 18, f"t={xmin:.0f}s", size=11, color="#6b7280"))
    svg.append(svg_text(x0 + w, y0 + h + 18, f"t={xmax:.0f}s", size=11, anchor="end", color="#6b7280"))


def plot_trace(path: Path, out: Path, *, required_mib: float, model_label: str,
               weight_mib: float, kv_mib: float, misc_mib: float, pinned_mib: float,
               safety_mib: float) -> dict[str, float | str]:
    rows = read_trace(path)
    xs = [t for t, _ in rows]
    mem = [m for _, m in rows]
    gap = [m - required_mib for m in mem]
    pct = [m / required_mib * 100.0 for m in mem]
    xmin, xmax = min(xs), max(xs)

    width = 1120
    height = 910
    left = 95
    right = 35
    panel_w = width - left - right
    panel_h = 190
    y1, y2, y3 = 150, 410, 670

    mem_min, mem_max = nice_bounds(mem, include=[required_mib])
    gap_min, gap_max = nice_bounds(gap, include=[0.0])
    pct_min, pct_max = nice_bounds(pct, include=[100.0])

    deficit = max(0.0, required_mib - min(mem))
    surplus = max(mem) - required_mib

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
    ]
    title = f"{path.name} - {model_label}"
    svg.append(svg_text(left, 46, title, size=22, weight="800", color="#111827"))
    subtitle = (
        f"required={required_mib:.1f} MiB "
        f"(weights={weight_mib:.1f}, KV={kv_mib:.0f}, misc={misc_mib:.0f}, "
        f"pinned={pinned_mib:.0f}, safety={safety_mib:.0f}); "
        f"trace min/mean/max={min(mem):.1f}/{sum(mem)/len(mem):.1f}/{max(mem):.1f} MiB"
    )
    svg.append(svg_text(left, 74, subtitle, size=13, color="#374151"))
    svg.append(svg_text(left, 98, f"minimum deficit={deficit:.1f} MiB; maximum surplus={surplus:.1f} MiB", size=13, color="#374151"))

    panel(svg, "1. Trace available memory", xs, mem, x0=left, y0=y1, w=panel_w, h=panel_h,
          xmin=xmin, xmax=xmax, ymin=mem_min, ymax=mem_max, line_color="#2563eb",
          required=required_mib)
    panel(svg, "2. Gap: available memory - full-residency requirement", xs, gap, x0=left, y0=y2,
          w=panel_w, h=panel_h, xmin=xmin, xmax=xmax, ymin=gap_min, ymax=gap_max,
          line_color="#7c3aed", zero=0.0)
    panel(svg, "3. Requirement coverage percentage", xs, pct, x0=left, y0=y3, w=panel_w,
          h=panel_h, xmin=xmin, xmax=xmax, ymin=pct_min, ymax=pct_max,
          line_color="#059669", zero=100.0)

    svg.append("</svg>\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg))
    return {
        "trace": str(path),
        "plot": str(out),
        "required_mib": required_mib,
        "trace_min_mib": min(mem),
        "trace_mean_mib": sum(mem) / len(mem),
        "trace_max_mib": max(mem),
        "min_gap_mib": min(gap),
        "max_gap_mib": max(gap),
        "min_coverage_pct": min(pct),
        "max_coverage_pct": max(pct),
    }


def slug(path: Path) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return s.strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create trace memory gap SVG plots")
    ap.add_argument("--trace-glob", action="append", required=True,
                    help="trace glob; may be repeated")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--model-meta", type=Path)
    ap.add_argument("--weight-mib", type=float,
                    help="planned weight MiB; if omitted, computed from --model-meta")
    ap.add_argument("--kv-mib", type=float, default=512)
    ap.add_argument("--misc-mib", type=float, default=256)
    ap.add_argument("--pinned-mib", type=float, default=0)
    ap.add_argument("--safety-mib", type=float, default=64)
    args = ap.parse_args()

    if args.weight_mib is None:
        if not args.model_meta:
            raise SystemExit("--weight-mib or --model-meta is required")
        args.weight_mib = planned_weight_mib(args.model_meta)
    required = args.weight_mib + args.kv_mib + args.misc_mib + args.pinned_mib + args.safety_mib

    traces: list[Path] = []
    for pat in args.trace_glob:
        traces.extend(sorted(Path(".").glob(pat)))
    traces = sorted(dict.fromkeys(t.resolve() for t in traces))
    if not traces:
        raise SystemExit("no traces matched")

    rows = []
    for trace in traces:
        out = args.out_dir / f"{slug(trace)}.svg"
        rows.append(plot_trace(
            trace, out, required_mib=required, model_label=args.model_label,
            weight_mib=args.weight_mib, kv_mib=args.kv_mib, misc_mib=args.misc_mib,
            pinned_mib=args.pinned_mib, safety_mib=args.safety_mib,
        ))

    summary = args.out_dir / "SUMMARY.md"
    lines = [
        f"# Trace Memory Gap Plots: {args.model_label}",
        "",
        f"- full-residency requirement: `{required:.1f} MiB`",
        f"- weights/KV/misc/pinned/safety: `{args.weight_mib:.1f}/{args.kv_mib:.0f}/{args.misc_mib:.0f}/{args.pinned_mib:.0f}/{args.safety_mib:.0f} MiB`",
        "",
        "| trace | plot | min/mean/max MiB | min/max gap MiB | coverage % |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        plot_rel = Path(str(r["plot"])).name
        lines.append(
            f"| `{Path(str(r['trace'])).name}` | [{plot_rel}]({plot_rel}) | "
            f"{float(r['trace_min_mib']):.1f}/{float(r['trace_mean_mib']):.1f}/{float(r['trace_max_mib']):.1f} | "
            f"{float(r['min_gap_mib']):.1f}/{float(r['max_gap_mib']):.1f} | "
            f"{float(r['min_coverage_pct']):.1f}-{float(r['max_coverage_pct']):.1f} |"
        )
    summary.write_text("\n".join(lines) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
