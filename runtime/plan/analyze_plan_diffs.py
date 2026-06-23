#!/usr/bin/env python3
"""Summarize placement differences between elastic memory plans.

This is a research/debug helper for comparing offline table plans with online
or incremental generated plans.  It intentionally works only on plan JSON files
and does not require a device.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def component(name: str) -> str:
    if ".attn_" in name:
        return "attn"
    if ".ffn_" in name:
        return "ffn"
    if "output" in name:
        return "output"
    return "other"


def weight_map(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(w["weight_id"]): w for w in plan.get("weights", []) if isinstance(w, dict) and "weight_id" in w}


def op_map(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(o["weight_id"]): o for o in plan.get("ops", []) if isinstance(o, dict) and "weight_id" in o}


def loc(weight: dict[str, Any]) -> str:
    return str(weight.get("location", "disk"))


def backend(op: dict[str, Any] | None) -> str:
    return str(op.get("compute_backend", "cpu")) if op else "cpu"


def summary(plan: dict[str, Any]) -> dict[str, Any]:
    weights = weight_map(plan)
    ops = op_map(plan)
    return {
        "budget": plan.get("budget_mib"),
        "pred": plan.get("pred_per_token_ms"),
        "loc": Counter(loc(w) for w in weights.values()),
        "backend": Counter(backend(ops.get(i)) for i in weights),
        "resident_mb": sum(w.get("byte_size", 0) for w in weights.values() if loc(w) != "disk") / 1024 / 1024,
        "disk_mb": sum(w.get("byte_size", 0) for w in weights.values() if loc(w) == "disk") / 1024 / 1024,
        "timeline": Counter(e.get("kind") for e in plan.get("timeline", []) if isinstance(e, dict)),
    }


def diff(base: dict[str, Any], target: dict[str, Any]):
    base_weights = weight_map(base)
    target_weights = weight_map(target)
    base_ops = op_map(base)
    target_ops = op_map(target)
    counts: Counter[str] = Counter()
    mb: Counter[str] = Counter()
    comps: dict[str, Counter[str]] = defaultdict(Counter)
    layers: dict[tuple[int, str, str], int] = defaultdict(int)

    for weight_id, base_weight in base_weights.items():
        if weight_id not in target_weights:
            continue
        target_weight = target_weights[weight_id]
        base_loc = loc(base_weight)
        target_loc = loc(target_weight)
        base_backend = backend(base_ops.get(weight_id))
        target_backend = backend(target_ops.get(weight_id))
        if base_loc == target_loc and base_backend == target_backend:
            continue
        key = f"{base_loc}/{base_backend}->{target_loc}/{target_backend}"
        size_mb = base_weight.get("byte_size", 0) / 1024 / 1024
        comp = component(str(base_weight.get("name", "")))
        layer = int(base_weight.get("layer", -1))
        counts[key] += 1
        mb[key] += size_mb
        comps[comp][key] += 1
        layers[(layer, comp, key)] += 1

    return counts, mb, comps, layers


def plan_files(plan_dir: Path):
    files = []
    for path in sorted(plan_dir.glob("*_plan.json")):
        match = re.search(r"online_(\d+)_(\d+)_plan\.json$", path.name)
        if match:
            files.append((int(match.group(2)), int(match.group(1)), path))
    return sorted(files)


def offline_plan_for(offline_dir: Path, budget_mib: int) -> Path:
    direct = offline_dir / f"plan_{budget_mib}MiB.json"
    if direct.exists():
        return direct
    budgets = sorted(
        int(m.group(1))
        for p in offline_dir.glob("plan_*MiB.json")
        if (m := re.match(r"plan_(\d+)MiB\.json$", p.name))
    )
    pick = max(b for b in budgets if b <= budget_mib)
    return offline_dir / f"plan_{pick}MiB.json"


def print_plan_summary(label: str, plan: dict[str, Any]) -> None:
    s = summary(plan)
    print(
        f"{label}: B={s['budget']} pred={s['pred']} resident={s['resident_mb']:.1f}MB "
        f"disk={s['disk_mb']:.1f}MB loc={dict(s['loc'])} backend={dict(s['backend'])} "
        f"timeline={dict(s['timeline'])}"
    )


def analyze(label: str, plan_dir: Path, offline_dir: Path) -> None:
    print(f"\n## {label}")
    files = plan_files(plan_dir)
    print(f"plans {len(files)} dir {plan_dir}")
    if not files:
        return

    aggregate: Counter[str] = Counter()
    aggregate_mb: Counter[str] = Counter()
    aggregate_components: dict[str, Counter[str]] = defaultdict(Counter)
    changed_counts: list[int] = []

    for call, budget, path in files:
        online = load_json(path)
        offline = load_json(offline_plan_for(offline_dir, budget))
        counts, mb, comps, layers = diff(offline, online)
        changed_counts.append(sum(counts.values()))
        aggregate.update(counts)
        aggregate_mb.update(mb)
        for comp, comp_counts in comps.items():
            aggregate_components[comp].update(comp_counts)

        if call in (1, 2, 3, len(files)) or sum(counts.values()) > 80:
            print_plan_summary(f"call{call} online", online)
            print_plan_summary(f"call{call} offline", offline)
            print(f"  diff count {sum(counts.values())} types {dict(counts)}")
            print(f"  diff MB {{ {', '.join(f'{k}: {v:.1f}' for k, v in mb.items())} }}")
            top_layers = sorted(layers.items(), key=lambda kv: kv[1], reverse=True)[:8]
            print(f"  top layer/component diffs {top_layers}")

    print(
        "changed count min/median/max",
        min(changed_counts),
        statistics.median(changed_counts),
        max(changed_counts),
    )
    print(f"aggregate diff count {dict(aggregate)}")
    print(f"aggregate diff MB {{ {', '.join(f'{k}: {v:.1f}' for k, v in aggregate_mb.items())} }}")
    print("aggregate by component")
    for comp, counts in aggregate_components.items():
        print(f"  {comp}: {dict(counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-dir", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    args = parser.parse_args()

    for i, plan_dir in enumerate(args.plan_dir):
        label = args.label[i] if i < len(args.label) else plan_dir.name
        analyze(label, plan_dir, args.offline_dir)


if __name__ == "__main__":
    main()
