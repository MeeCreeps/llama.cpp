#!/usr/bin/env python3
import csv
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

SERIAL = "3C15AU002CL00000"
REMOTE_DIR = "/data/local/tmp/hyzheng/elastic"
MODEL = "OLMoE-1B-7B-0125-Instruct-Q4_0.gguf"
META = REPO / "runtime/plan/model_meta/OLMoE-1B-7B-0125-Instruct-Q4_0_gguf_all.weights_ops.json"
TOP_PS = [0.20, 0.25, 0.30, 0.35, 0.40]
N_TOKENS = 1000
PROMPT = "Write a long technical note about ocean currents, climate, and navigation."

N_LAYERS = 16
N_EXPERTS = 64

COLORS = {
    0.20: "#1565c0",
    0.25: "#00897b",
    0.30: "#ef6c00",
    0.35: "#8e24aa",
    0.40: "#c62828",
}


def shquote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def run_top_p(top_p: float) -> tuple[Path, Path]:
    tag = f"top_p_{top_p:.2f}".replace(".", "p")
    out_path = ROOT / f"{tag}.out"
    err_path = ROOT / f"{tag}.err"
    if err_path.exists():
        prev = err_path.read_text(errors="ignore")
        if (
            "__LLAMA_INNER_RC__=0" in prev
            and "[moe-dyn-ids]" in prev
            and "all_experts=1" in prev
            and "candidates=64" in prev
        ):
            return out_path, err_path
        out_path.unlink(missing_ok=True)
        err_path.unlink(missing_ok=True)

    remote_cmd = (
        f"cd {shquote(REMOTE_DIR)} && "
        "export LD_LIBRARY_PATH=. && "
        f"export LLAMA_MOE_DYNAMIC_TOP_P={top_p:.2f} && "
        "export LLAMA_MOE_DYNAMIC_TOP_P_DEBUG=1 && "
        "export LLAMA_MOE_DYNAMIC_ALL_EXPERTS=1 && "
        "export LLAMA_MOE_DYNAMIC_GPU_DECODE=1 && "
        f"./llama-cli -m {shquote(MODEL)} "
        f"-no-cnv -p {shquote(PROMPT)} "
        f"-n {N_TOKENS} -c 1536 -b 64 -ub 64 -t 8 -ngl 99 "
        "--no-warmup --seed 7 --temp 0 --ignore-eos; "
        "rc=$?; echo __LLAMA_INNER_RC__=$rc >&2; exit $rc"
    )
    with out_path.open("wb") as out_f, err_path.open("wb") as err_f:
        proc = subprocess.run(
            ["adb", "-s", SERIAL, "shell", remote_cmd],
            stdout=out_f,
            stderr=err_f,
            timeout=1200,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"top_p={top_p:.2f} failed rc={proc.returncode}; see {err_path}")
    return out_path, err_path


ACTIVE_RE = re.compile(r"\[moe-dyn\]\s+layer=(\d+).*?active_k=\[([^\]]*)\]")
IDS_RE = re.compile(r"\[moe-dyn-ids\]\s+layer=(\d+).*?ids=\[(.*)\]")


def parse_active_k(err_path: Path) -> list[list[int]]:
    records: list[list[int]] = []
    current: dict[int, list[int]] = {}
    for line in err_path.read_text(errors="ignore").splitlines():
        m = ACTIVE_RE.search(line)
        if not m:
            continue
        layer = int(m.group(1))
        vals = [int(v.strip()) for v in m.group(2).split(",") if v.strip()]
        current[layer] = vals
        if len(current) == N_LAYERS:
            if sorted(current) == list(range(N_LAYERS)) and all(len(current[i]) == 1 for i in range(N_LAYERS)):
                records.append([current[i][0] for i in range(N_LAYERS)])
            current = {}
    return records[:N_TOKENS]


def parse_selected_ids(err_path: Path) -> list[list[list[int]]]:
    records: list[list[list[int]]] = []
    current: dict[int, list[list[int]]] = {}
    for line in err_path.read_text(errors="ignore").splitlines():
        m = IDS_RE.search(line)
        if not m:
            continue
        layer = int(m.group(1))
        payload = m.group(2)
        tokens = []
        for chunk in re.findall(r"\[([^\]]*)\]", payload):
            vals = [int(v.strip()) for v in chunk.split(",") if v.strip()]
            tokens.append(vals)
        current[layer] = tokens
        if len(current) == N_LAYERS:
            if sorted(current) == list(range(N_LAYERS)) and all(len(current[i]) == 1 for i in range(N_LAYERS)):
                records.append([current[i][0] for i in range(N_LAYERS)])
            current = {}
    return records[:N_TOKENS]


def load_weight_sizes() -> tuple[float, float, float, float]:
    mib = 1024 * 1024
    meta = json.loads(META.read_text())
    weights = meta["weights"]
    expert = [w for w in weights if "_exps.weight" in w["name"]]
    non_expert = [w for w in weights if "_exps.weight" not in w["name"]]
    full_model_mib = sum(w["byte_size"] for w in weights) / mib
    full_expert_pool_mib = sum(w["byte_size"] for w in expert) / mib
    fixed_non_expert_mib = sum(w["byte_size"] for w in non_expert) / mib
    per_active_expert_layer_mib = full_expert_pool_mib / (N_LAYERS * N_EXPERTS)
    return full_model_mib, full_expert_pool_mib, fixed_non_expert_mib, per_active_expert_layer_mib


def write_active_csv_and_stats(all_active_k: dict[float, list[list[int]]]) -> list[dict[str, str]]:
    full_model_mib, full_expert_pool_mib, fixed_non_expert_mib, per_expert_mib = load_weight_sizes()
    rows = []
    stats = []
    for top_p, records in all_active_k.items():
        expert_mibs = []
        ratios = []
        active_counts = []
        avg_per_layer_counts = []
        min_layer_counts = []
        max_layer_counts = []
        for token_idx, active_by_layer in enumerate(records):
            total_active = sum(active_by_layer)
            avg_per_layer = total_active / N_LAYERS
            min_layer = min(active_by_layer)
            max_layer = max(active_by_layer)
            active_expert_mib = total_active * per_expert_mib
            total_active_weight_mib = fixed_non_expert_mib + active_expert_mib
            ratio = active_expert_mib / total_active_weight_mib
            active_counts.append(total_active)
            avg_per_layer_counts.append(avg_per_layer)
            min_layer_counts.append(min_layer)
            max_layer_counts.append(max_layer)
            expert_mibs.append(active_expert_mib)
            ratios.append(ratio)
            rows.append(
                {
                    "top_p": f"{top_p:.2f}",
                    "token_idx": token_idx,
                    "total_active_experts": total_active,
                    "avg_active_experts_per_layer": f"{avg_per_layer:.6f}",
                    "min_active_experts_one_layer": min_layer,
                    "max_active_experts_one_layer": max_layer,
                    "active_expert_weight_mib_no_kv": f"{active_expert_mib:.6f}",
                    "fixed_non_expert_weight_mib_no_kv": f"{fixed_non_expert_mib:.6f}",
                    "total_active_weight_working_set_mib_no_kv": f"{total_active_weight_mib:.6f}",
                    "expert_fraction_of_total_active_weight": f"{ratio:.6f}",
                }
            )
        stats.append(
            {
                "top_p": f"{top_p:.2f}",
                "tokens": str(len(records)),
                "active_experts_mean": f"{statistics.mean(active_counts):.2f}",
                "active_experts_min": str(min(active_counts)),
                "active_experts_max": str(max(active_counts)),
                "avg_active_experts_per_layer_mean": f"{statistics.mean(avg_per_layer_counts):.2f}",
                "avg_active_experts_per_layer_min": f"{min(avg_per_layer_counts):.2f}",
                "avg_active_experts_per_layer_max": f"{max(avg_per_layer_counts):.2f}",
                "max_active_experts_one_layer": str(max(max_layer_counts)),
                "expert_weight_mib_mean": f"{statistics.mean(expert_mibs):.2f}",
                "expert_weight_mib_min": f"{min(expert_mibs):.2f}",
                "expert_weight_mib_max": f"{max(expert_mibs):.2f}",
                "expert_fraction_mean": f"{statistics.mean(ratios):.4f}",
                "expert_fraction_min": f"{min(ratios):.4f}",
                "expert_fraction_max": f"{max(ratios):.4f}",
                "fixed_non_expert_weight_mib": f"{fixed_non_expert_mib:.2f}",
                "full_expert_pool_mib": f"{full_expert_pool_mib:.2f}",
                "full_model_weight_mib": f"{full_model_mib:.2f}",
            }
        )

    rows_path = ROOT / "top_p_sweep_active_weight_timeseries.csv"
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats_path = ROOT / "top_p_sweep_summary.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)
    return stats


def write_reload_csv_and_stats(
    all_active_k: dict[float, list[list[int]]],
    all_selected_ids: dict[float, list[list[list[int]]]],
) -> list[dict[str, str]]:
    _, _, _, per_expert_mib = load_weight_sizes()
    rows = []
    stats = []
    for top_p in TOP_PS:
        active_records = all_active_k[top_p]
        id_records = all_selected_ids[top_p]
        n = min(len(active_records), len(id_records))
        prev_sets: list[set[int]] | None = None
        reload_mibs = []
        churn_fracs = []
        for token_idx in range(n):
            active_by_layer = active_records[token_idx]
            ids_by_layer = id_records[token_idx]
            cur_sets = []
            active_total = 0
            reload_count = 0
            for layer, (active_k, ids) in enumerate(zip(active_by_layer, ids_by_layer)):
                if len(ids) != N_EXPERTS:
                    raise RuntimeError(
                        f"top_p={top_p:.2f} token={token_idx} layer={layer}: expected {N_EXPERTS} ids, got {len(ids)}"
                    )
                active_ids = set(ids[:active_k])
                cur_sets.append(active_ids)
                active_total += len(active_ids)
                if prev_sets is None:
                    reload_count += len(active_ids)
                else:
                    reload_count += len(active_ids - prev_sets[layer])
            reload_mib = reload_count * per_expert_mib
            active_mib = active_total * per_expert_mib
            churn_frac = reload_count / active_total if active_total else 0.0
            reload_mibs.append(reload_mib)
            churn_fracs.append(churn_frac)
            rows.append(
                {
                    "top_p": f"{top_p:.2f}",
                    "token_idx": token_idx,
                    "active_experts": active_total,
                    "new_experts_vs_prev_token": reload_count,
                    "reload_demand_mib": f"{reload_mib:.6f}",
                    "active_expert_weight_mib": f"{active_mib:.6f}",
                    "reload_fraction_of_active_experts": f"{churn_frac:.6f}",
                }
            )
            prev_sets = cur_sets
        warm = reload_mibs[1:] if len(reload_mibs) > 1 else reload_mibs
        warm_frac = churn_fracs[1:] if len(churn_fracs) > 1 else churn_fracs
        stats.append(
            {
                "top_p": f"{top_p:.2f}",
                "tokens": str(n),
                "reload_mib_mean_including_cold": f"{statistics.mean(reload_mibs):.2f}",
                "reload_mib_mean_excluding_cold": f"{statistics.mean(warm):.2f}",
                "reload_mib_min_excluding_cold": f"{min(warm):.2f}",
                "reload_mib_max_excluding_cold": f"{max(warm):.2f}",
                "reload_fraction_mean_excluding_cold": f"{statistics.mean(warm_frac):.4f}",
                "reload_fraction_min_excluding_cold": f"{min(warm_frac):.4f}",
                "reload_fraction_max_excluding_cold": f"{max(warm_frac):.4f}",
            }
        )

    rows_path = ROOT / "top_p_sweep_reload_demand_timeseries.csv"
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats_path = ROOT / "top_p_sweep_reload_demand_summary.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)
    return stats


def plot_one(
    out_path: Path,
    title: str,
    top_label: str,
    bottom_label: str,
    x_label: str,
    top_vals: list[float],
    bottom_vals: list[float],
    color: str,
    top_step: float,
    bottom_step: float,
    top_ymax: float | None = None,
    bottom_ymax: float | None = None,
) -> None:
    width, height = 950, 610
    ml, mr, mt, mb = 86, 34, 54, 56
    gap = 54
    panel_h = (height - mt - mb - gap) / 2
    plot_w = width - ml - mr

    if top_ymax is None:
        top_ymax = max(top_step, math.ceil(max(top_vals) / top_step) * top_step)
    if bottom_ymax is None:
        bottom_ymax = max(bottom_step, math.ceil(max(bottom_vals) / bottom_step) * bottom_step)

    def sx(x: float, n: int) -> float:
        return ml + x / max(1, n - 1) * plot_w

    def sy(y: float, y0: float, ymax: float) -> float:
        return y0 + (ymax - y) / ymax * panel_h

    def poly(vals: list[float], y0: float, ymax: float) -> str:
        n_vals = len(vals)
        return " ".join(f"{sx(i, n_vals):.2f},{sy(v, y0, ymax):.2f}" for i, v in enumerate(vals))

    top_y0 = mt
    bot_y0 = mt + panel_h + gap
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:20px;font-weight:700}.label{font-size:14px;fill:#222}.small{font-size:11px;fill:#444}.axis{stroke:#333;stroke-width:1.1}.grid{stroke:#e2e2e2;stroke-width:1}</style>',
        f'<text x="{width/2}" y="29" text-anchor="middle" class="title">{title}</text>',
    ]
    y = 0.0
    while y <= top_ymax + 1e-9:
        yy = sy(y, top_y0, top_ymax)
        label = f"{y:.1f}" if top_step < 1 else f"{int(y)}"
        svg.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{width-mr}" y2="{yy:.2f}" class="grid"/>')
        svg.append(f'<text x="{ml-8}" y="{yy+3:.2f}" text-anchor="end" class="small">{label}</text>')
        y += top_step
    y = 0.0
    while y <= bottom_ymax + 1e-9:
        yy = sy(y, bot_y0, bottom_ymax)
        label = f"{y:.1f}" if bottom_step < 1 else f"{int(y)}"
        svg.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{width-mr}" y2="{yy:.2f}" class="grid"/>')
        svg.append(f'<text x="{ml-8}" y="{yy+3:.2f}" text-anchor="end" class="small">{label}</text>')
        y += bottom_step

    n_plot = len(top_vals)
    x_ticks = [0, 200, 400, 600, 800, n_plot - 1]
    for x in x_ticks:
        if x < 0 or x >= n_plot:
            continue
        xx = sx(float(x), n_plot)
        svg.append(f'<line x1="{xx:.2f}" y1="{top_y0}" x2="{xx:.2f}" y2="{top_y0+panel_h}" class="grid"/>')
        svg.append(f'<line x1="{xx:.2f}" y1="{bot_y0}" x2="{xx:.2f}" y2="{bot_y0+panel_h}" class="grid"/>')
        svg.append(f'<text x="{xx:.2f}" y="{height-mb+20}" text-anchor="middle" class="small">{x}</text>')
    for y0 in [top_y0, bot_y0]:
        svg.append(f'<line x1="{ml}" y1="{y0}" x2="{ml}" y2="{y0+panel_h}" class="axis"/>')
        svg.append(f'<line x1="{ml}" y1="{y0+panel_h}" x2="{width-mr}" y2="{y0+panel_h}" class="axis"/>')

    svg.append(f'<polyline points="{poly(top_vals, top_y0, top_ymax)}" fill="none" stroke="{color}" stroke-width="2.1"/>')
    svg.append(f'<polyline points="{poly(bottom_vals, bot_y0, bottom_ymax)}" fill="none" stroke="{color}" stroke-width="2.1"/>')
    svg.append(f'<text transform="translate(24,{top_y0+panel_h/2}) rotate(-90)" text-anchor="middle" class="label">{top_label}</text>')
    svg.append(f'<text transform="translate(24,{bot_y0+panel_h/2}) rotate(-90)" text-anchor="middle" class="label">{bottom_label}</text>')
    svg.append(f'<text x="{width/2}" y="{height-17}" text-anchor="middle" class="label">{x_label}</text>')
    svg.append(f'<text x="{width-mr-8}" y="{top_y0+16}" text-anchor="end" class="small">mean {statistics.mean(top_vals):.1f}, range {min(top_vals):.1f}-{max(top_vals):.1f}</text>')
    svg.append(f'<text x="{width-mr-8}" y="{bot_y0+16}" text-anchor="end" class="small">mean {statistics.mean(bottom_vals):.3f}, range {min(bottom_vals):.3f}-{max(bottom_vals):.3f}</text>')
    svg.append("</svg>")
    out_path.write_text("\n".join(svg) + "\n")


def make_plots() -> None:
    active_rows = list(csv.DictReader((ROOT / "top_p_sweep_active_weight_timeseries.csv").open()))
    reload_rows = list(csv.DictReader((ROOT / "top_p_sweep_reload_demand_timeseries.csv").open()))

    active_by_p: dict[float, list[dict[str, str]]] = {}
    reload_by_p: dict[float, list[dict[str, str]]] = {}
    for row in active_rows:
        active_by_p.setdefault(float(row["top_p"]), []).append(row)
    for row in reload_rows:
        reload_by_p.setdefault(float(row["top_p"]), []).append(row)

    active_dir = ROOT / "per_top_p_active"
    reload_dir = ROOT / "per_top_p_reload"
    count_dir = ROOT / "per_top_p_expert_count"
    active_dir.mkdir(exist_ok=True)
    reload_dir.mkdir(exist_ok=True)
    count_dir.mkdir(exist_ok=True)

    active_weight_all = [float(r["active_expert_weight_mib_no_kv"]) for r in active_rows]
    active_ratio_all = [float(r["expert_fraction_of_total_active_weight"]) for r in active_rows]
    avg_count_all = [float(r["avg_active_experts_per_layer"]) for r in active_rows]
    spread_count_all = [
        float(r["max_active_experts_one_layer"]) - float(r["min_active_experts_one_layer"])
        for r in active_rows
    ]
    reload_all = [float(r["reload_demand_mib"]) for r in reload_rows if int(r["token_idx"]) > 0]
    reload_frac_all = [float(r["reload_fraction_of_active_experts"]) for r in reload_rows if int(r["token_idx"]) > 0]

    active_ymax = max(100.0, math.ceil(max(active_weight_all) / 100) * 100)
    active_ratio_ymax = max(0.1, math.ceil(max(active_ratio_all) * 10) / 10)
    count_ymax = max(8.0, math.ceil(max(avg_count_all) / 4) * 4)
    count_spread_ymax = max(8.0, math.ceil(max(spread_count_all) / 4) * 4)
    reload_ymax = max(100.0, math.ceil(max(reload_all) / 100) * 100)
    reload_frac_ymax = max(0.1, math.ceil(max(reload_frac_all) * 10) / 10)

    for top_p in TOP_PS:
        tag = f"top_p_{top_p:.2f}".replace(".", "p")
        active_data = active_by_p[top_p]
        weights = [float(r["active_expert_weight_mib_no_kv"]) for r in active_data]
        ratios = [float(r["expert_fraction_of_total_active_weight"]) for r in active_data]
        avg_counts = [float(r["avg_active_experts_per_layer"]) for r in active_data]
        count_spreads = [
            float(r["max_active_experts_one_layer"]) - float(r["min_active_experts_one_layer"])
            for r in active_data
        ]
        plot_one(
            count_dir / f"{tag}_active_expert_count.svg",
            f"All-expert active expert count, top_p={top_p:.2f}",
            "avg experts/layer",
            "max-min experts/layer",
            "decode token index",
            avg_counts,
            count_spreads,
            COLORS[top_p],
            top_step=2.0,
            bottom_step=4.0,
            top_ymax=count_ymax,
            bottom_ymax=count_spread_ymax,
        )

        plot_one(
            active_dir / f"{tag}_active_weight_and_ratio.svg",
            f"All-expert dynamic Top-P, top_p={top_p:.2f}",
            "expert weight MB",
            "expert / active total",
            "decode token index",
            weights,
            ratios,
            COLORS[top_p],
            top_step=100.0,
            bottom_step=0.1,
            top_ymax=active_ymax,
            bottom_ymax=active_ratio_ymax,
        )

        reload_data = reload_by_p[top_p]
        reloads = [float(r["reload_demand_mib"]) for r in reload_data][1:]
        fracs = [float(r["reload_fraction_of_active_experts"]) for r in reload_data][1:]
        plot_one(
            reload_dir / f"{tag}_reload_demand.svg",
            f"All-expert reload demand, top_p={top_p:.2f}",
            "new expert MB/token",
            "new / active experts",
            "decode token index, cold-start token excluded",
            reloads,
            fracs,
            COLORS[top_p],
            top_step=100.0,
            bottom_step=0.1,
            top_ymax=reload_ymax,
            bottom_ymax=reload_frac_ymax,
        )


def write_summary(active_stats: list[dict[str, str]], reload_stats: list[dict[str, str]]) -> None:
    md = []
    md.append("# MoE All-Expert Dynamic Top-p Sweep")
    md.append("")
    md.append(f"Model: `{MODEL}`")
    md.append(f"Tokens per run: `{N_TOKENS}`")
    md.append("Mode: `LLAMA_MOE_DYNAMIC_ALL_EXPERTS=1`, so dynamic Top-P selects from all 64 experts per layer instead of only the original fixed top-8 candidates.")
    md.append("Definition: expert weight working set only. KV cache, activation tensors, allocator overhead, and dense non-expert weights are excluded from the top subplot.")
    md.append("Expert-count columns use the per-token average across MoE layers, so the value is in the `1..64` per-layer expert range.")
    md.append("")
    md.append("| top_p | avg experts/layer mean | avg experts/layer min/max | max experts in one layer | expert MB mean | expert MB min/max | expert / total active mean |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in active_stats:
        md.append(
            f"| {row['top_p']} | {row['avg_active_experts_per_layer_mean']} | "
            f"{row['avg_active_experts_per_layer_min']} / {row['avg_active_experts_per_layer_max']} | "
            f"{row['max_active_experts_one_layer']} | "
            f"{row['expert_weight_mib_mean']} | {row['expert_weight_mib_min']} / {row['expert_weight_mib_max']} | {row['expert_fraction_mean']} |"
        )
    md.append("")
    md.append("## Token-to-token Reload Demand")
    md.append("")
    md.append("Reload demand is computed from expert ids as active experts in token `t` that were not active in token `t-1`, summed across layers. The cold-start token is excluded from the mean/min/max below.")
    md.append("")
    md.append("| top_p | reload MB mean | reload MB min/max | new expert fraction mean |")
    md.append("|---:|---:|---:|---:|")
    for row in reload_stats:
        md.append(
            f"| {row['top_p']} | {row['reload_mib_mean_excluding_cold']} | "
            f"{row['reload_mib_min_excluding_cold']} / {row['reload_mib_max_excluding_cold']} | "
            f"{row['reload_fraction_mean_excluding_cold']} |"
        )
    md.append("")
    md.append("Artifacts:")
    md.append("")
    md.append("- `per_top_p_active/top_p_*_active_weight_and_ratio.svg`")
    md.append("- `per_top_p_active/top_p_*_active_weight_and_ratio.png`")
    md.append("- `per_top_p_expert_count/top_p_*_active_expert_count.svg`")
    md.append("- `per_top_p_expert_count/top_p_*_active_expert_count.png`")
    md.append("- `per_top_p_reload/top_p_*_reload_demand.svg`")
    md.append("- `per_top_p_reload/top_p_*_reload_demand.png`")
    md.append("- `top_p_sweep_active_weight_timeseries.csv`")
    md.append("- `top_p_sweep_summary.csv`")
    md.append("- `top_p_sweep_reload_demand_timeseries.csv`")
    md.append("- `top_p_sweep_reload_demand_summary.csv`")
    md.append("- raw logs: `top_p_*.out`, `top_p_*.err`")
    (ROOT / "SUMMARY.md").write_text("\n".join(md) + "\n")


def main() -> None:
    all_active_k: dict[float, list[list[int]]] = {}
    all_selected_ids: dict[float, list[list[list[int]]]] = {}
    for top_p in TOP_PS:
        print(f"=== run all-expert top_p={top_p:.2f} ===", flush=True)
        _, err_path = run_top_p(top_p)
        active = parse_active_k(err_path)
        selected = parse_selected_ids(err_path)
        if len(active) != len(selected):
            raise RuntimeError(
                f"top_p={top_p:.2f}: active records {len(active)} != selected-id records {len(selected)}"
            )
        if not active:
            raise RuntimeError(f"top_p={top_p:.2f}: no decode records parsed")
        all_active_k[top_p] = active
        all_selected_ids[top_p] = selected
        print(f"parsed {len(active)} decode records for top_p={top_p:.2f}", flush=True)

    active_stats = write_active_csv_and_stats(all_active_k)
    reload_stats = write_reload_csv_and_stats(all_active_k, all_selected_ids)
    make_plots()
    write_summary(active_stats, reload_stats)
    print(ROOT / "SUMMARY.md")


if __name__ == "__main__":
    main()
