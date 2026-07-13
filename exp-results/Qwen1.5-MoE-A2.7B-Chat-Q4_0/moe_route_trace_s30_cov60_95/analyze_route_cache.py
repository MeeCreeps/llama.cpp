#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CacheState:
    capacity: int = 0
    resident: set[int] = field(default_factory=set)
    age: dict[int, int] = field(default_factory=dict)
    freq: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    clock: int = 0
    accesses: int = 0

    def resize_like_runtime(self, capacity: int) -> None:
        if self.capacity and abs(self.capacity - capacity) >= 4:
            self.resident.clear()
            self.age.clear()
            self.freq.clear()
            self.accesses = 0
        self.capacity = capacity


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            layer = int(row["weight"].split(".")[1])
            rows.append({
                "layer": layer,
                "capacity": int(row["capacity"]),
                "experts": [int(x) for x in row["expert_ids"].split(";")],
            })
    return rows


def simulate_lru_or_lfu(rows: list[dict], policy: str, decay: int = 1024) -> tuple[int, int]:
    states = defaultdict(CacheState)
    hits = misses = 0
    for row in rows:
        state = states[row["layer"]]
        state.resize_like_runtime(row["capacity"])
        protected = set()
        for expert in row["experts"]:
            state.clock += 1
            state.accesses += 1
            state.freq[expert] += 1
            if decay and state.accesses % decay == 0:
                for key in list(state.freq):
                    state.freq[key] = (state.freq[key] + 1) // 2
            if expert in state.resident:
                hits += 1
            else:
                misses += 1
                if len(state.resident) >= state.capacity:
                    candidates = state.resident - protected
                    if policy == "lru":
                        victim = min(candidates, key=lambda x: state.age[x])
                    else:
                        victim = min(candidates, key=lambda x: (state.freq[x], state.age[x]))
                    state.resident.remove(victim)
                    state.age.pop(victim, None)
                state.resident.add(expert)
            state.age[expert] = state.clock
            protected.add(expert)
    return hits, misses


@dataclass
class TwoQState:
    capacity: int = 0
    probation: deque[int] = field(default_factory=deque)
    protected: deque[int] = field(default_factory=deque)

    def reset_if_needed(self, capacity: int) -> None:
        if self.capacity and abs(self.capacity - capacity) >= 4:
            self.probation.clear()
            self.protected.clear()
        self.capacity = capacity


def remove_value(queue: deque[int], value: int) -> None:
    queue.remove(value)


def simulate_two_q(rows: list[dict], probation_slots: int) -> tuple[int, int]:
    states = defaultdict(TwoQState)
    hits = misses = 0
    for row in rows:
        state = states[row["layer"]]
        state.reset_if_needed(row["capacity"])
        probation_cap = min(probation_slots, state.capacity)
        protected_cap = state.capacity - probation_cap
        for expert in row["experts"]:
            if expert in state.protected:
                hits += 1
                remove_value(state.protected, expert)
                state.protected.append(expert)
            elif expert in state.probation:
                hits += 1
                remove_value(state.probation, expert)
                if protected_cap:
                    if len(state.protected) >= protected_cap:
                        state.probation.append(state.protected.popleft())
                    state.protected.append(expert)
                else:
                    state.probation.append(expert)
            else:
                misses += 1
                state.probation.append(expert)
            while len(state.probation) > probation_cap:
                state.probation.popleft()
            while len(state.protected) > protected_cap:
                state.probation.append(state.protected.popleft())
    return hits, misses


def simulate_belady(rows: list[dict]) -> tuple[int, int]:
    by_layer = defaultdict(list)
    for row_index, row in enumerate(rows):
        for expert in row["experts"]:
            by_layer[row["layer"]].append((row_index, row["capacity"], expert, set(row["experts"])))

    hits = misses = 0
    inf = 1 << 60
    for events in by_layer.values():
        future = defaultdict(deque)
        for pos, (_, _, expert, _) in enumerate(events):
            future[expert].append(pos)
        resident = set()
        old_capacity = 0
        for pos, (_, capacity, expert, protected) in enumerate(events):
            if old_capacity and abs(old_capacity - capacity) >= 4:
                resident.clear()
            old_capacity = capacity
            future[expert].popleft()
            if expert in resident:
                hits += 1
                continue
            misses += 1
            if len(resident) >= capacity:
                candidates = resident - protected
                victim = max(candidates, key=lambda x: future[x][0] if future[x] else inf)
                resident.remove(victim)
            resident.add(expert)
    return hits, misses


def add_result(results: list[dict], name: str, hits: int, misses: int) -> None:
    accesses = hits + misses
    results.append({
        "policy": name,
        "gate_hits": hits,
        "gate_misses": misses,
        "estimated_runtime_misses": misses * 3,
        "hit_rate_pct": 100.0 * hits / accesses if accesses else 0.0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.trace)
    results = []

    add_result(results, "lru", *simulate_lru_or_lfu(rows, "lru"))
    for decay in (32, 64, 128, 256, 512, 1024, 0):
        label = f"lfu_decay_{decay}" if decay else "lfu_no_decay"
        add_result(results, label, *simulate_lru_or_lfu(rows, "lfu", decay))
    for probation in (2, 4, 8, 12, 16):
        add_result(results, f"two_q_probation_{probation}", *simulate_two_q(rows, probation))
    add_result(results, "belady_runtime_resize_lower_bound", *simulate_belady(rows))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for row in sorted(results, key=lambda x: x["gate_misses"]):
        print(f"{row['policy']:36s} misses={row['gate_misses']:5d} "
              f"runtime_est={row['estimated_runtime_misses']:5d} hit={row['hit_rate_pct']:.2f}%")


if __name__ == "__main__":
    main()
