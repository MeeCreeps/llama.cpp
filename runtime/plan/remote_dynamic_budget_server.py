#!/usr/bin/env python3
"""HTTP wrapper for dynamic_budget_solver.py.

The phone posts the current budget and residency state. The server runs the
Python/CP-SAT solver locally and returns the native ExecPlan JSON. This keeps
Python and OR-Tools off the phone while preserving the online state-dependent
planning path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dynamic_budget_solver


class SolverHandler(BaseHTTPRequestHandler):
    server: "SolverHTTPServer"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/solve":
            self.send_error(404)
            return

        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            plan, solve_ms = self.server.solve(req)
            self._send_json({"ok": True, "solve_ms": solve_ms, "plan": plan})
        except Exception as exc:  # keep the phone-side error concrete
            traceback.print_exc(file=sys.stderr)
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"))

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _send_json(self, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SolverHTTPServer(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], args: argparse.Namespace):
        super().__init__(addr, SolverHandler)
        self.args = args
        self.verbose = args.verbose
        self._state_lock = threading.Lock()
        self._last_cpu_resident: set[str] = set()
        self._last_placements: dict[str, str] = {}
        self._last_plan: dict[str, Any] | None = None
        self._last_working_set_counters: dict[str, tuple[int, int]] = {}
        self._run_id = ""
        self._model_meta = json.loads(args.model_meta.read_text())
        self._granularity_model = dynamic_budget_solver.GranularityCostModel.load(
            args.granularity_profile, args.granularity_backend)

    def _reset_for_run(self, run_id: str) -> None:
        with self._state_lock:
            if run_id == self._run_id:
                return
            self._run_id = run_id
            self._last_cpu_resident.clear()
            self._last_placements.clear()
            self._last_plan = None
            self._last_working_set_counters.clear()

    def _augment_cpu_residency(self, state: dict[str, Any]) -> dict[str, Any]:
        """Carry CPU placements returned by the previous solve.

        This is a legacy/debug fallback. Normal online planning should use the
        canonical runtime placement map reported by the phone.
        """
        if not self.args.carry_cpu_residency:
            return state
        with self._state_lock:
            last_cpu = set(self._last_cpu_resident)
        if not last_cpu:
            return state

        rows = state.get("weights", [])
        if not isinstance(rows, list):
            return state
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("name") not in last_cpu:
                continue
            flags = row.setdefault("flags", [])
            if not isinstance(flags, list):
                continue
            has_gpu = any("gpu_compute_resident" == str(f).lower() for f in flags)
            has_cpu = any("cpu_compute_resident" == str(f).lower() for f in flags)
            if not has_gpu and not has_cpu:
                flags.append("cpu_compute_resident")
        return state

    def _remember_cpu_residency(self, plan: dict[str, Any]) -> None:
        if not self.args.carry_cpu_residency:
            return
        cpu = {
            str(w.get("name"))
            for w in plan.get("weights", [])
            if isinstance(w, dict) and str(w.get("location", "")).lower() == "cpu"
        }
        with self._state_lock:
            self._last_cpu_resident = cpu

    def _augment_previous_placements(self, state: dict[str, Any]) -> dict[str, Any]:
        """Attach the previous logical plan without claiming physical residency."""
        with self._state_lock:
            previous = dict(self._last_placements)
            previous_working_unit = (
                self._last_plan.get("working_unit")
                if self._last_plan and
                isinstance(self._last_plan.get("working_unit"), dict)
                else None)
        if previous_working_unit and "working_unit" not in state:
            state["working_unit"] = previous_working_unit
        if not previous:
            return state

        rows = state.get("weights", [])
        if not isinstance(rows, list):
            return state
        for row in rows:
            if not isinstance(row, dict):
                continue
            location = previous.get(str(row.get("name", "")))
            if location in {"cpu", "gpu", "disk"}:
                row["previous_location"] = location
        return state

    def _remember_placements(self, plan: dict[str, Any]) -> None:
        placements = {
            str(w.get("name")): str(w.get("location", "disk")).lower()
            for w in plan.get("weights", [])
            if isinstance(w, dict)
            and str(w.get("name", ""))
            and str(w.get("location", "disk")).lower() in {"cpu", "gpu", "disk"}
        }
        with self._state_lock:
            self._last_placements = placements
            # Returned plans are immutable after solve(). Keep the object
            # directly; copying hundreds of working-unit tiles added material
            # latency to every online callback.
            self._last_plan = plan

    def _stabilize_plan(self, plan: dict[str, Any], req: dict[str, Any]) -> dict[str, Any]:
        with self._state_lock:
            previous = self._last_plan
        if not previous:
            return plan

        budget_mib = int(req["budget_mib"])
        reserved_mib = (
            int(req.get("kv_mib", self.args.kv_mib))
            + int(req.get("misc_mib", self.args.misc_mib))
            + int(req.get("safety_mib", self.args.safety_mib))
            + int(req.get(
                "planner_stream_reserve_mib",
                self.args.planner_stream_reserve_mib))
        )
        usable_bytes = max(0, budget_mib - reserved_mib) * 1024 * 1024
        previous_bytes = sum(
            int(w.get("byte_size", 0))
            for w in previous.get("weights", [])
            if isinstance(w, dict) and str(w.get("location", "disk")).lower() in {"cpu", "gpu"}
        )
        def effective_ms(candidate: dict[str, Any]) -> float:
            # Granularity planning predicts the complete three-stage unit
            # pipeline. A placement-only switch gate used to suppress
            # profitable split/merge edits when placement stayed unchanged.
            if str(req.get("granularity_policy", "none")) != "none":
                working_unit = candidate.get("working_unit", {})
                if isinstance(working_unit, dict):
                    try:
                        value = float(working_unit.get(
                            "predicted_ms", math.inf))
                    except (TypeError, ValueError):
                        value = math.inf
                    if math.isfinite(value):
                        return value
            return float(candidate.get("pred_per_token_ms", math.inf))

        previous_ms = effective_ms(previous)
        proposed_ms = effective_ms(plan)
        predicted_gain_ms = previous_ms - proposed_ms
        if (
            previous_bytes <= usable_bytes
            and math.isfinite(previous_ms)
            and math.isfinite(proposed_ms)
            and predicted_gain_ms < self.args.plan_switch_min_gain_ms
        ):
            previous = copy.deepcopy(previous)
            previous["budget_mib"] = budget_mib
            previous["online_stabilization"] = {
                "mode": "keep_previous",
                "previous_pred_per_token_ms": previous_ms,
                "proposed_pred_per_token_ms": proposed_ms,
                "predicted_gain_ms": predicted_gain_ms,
                "min_gain_ms": self.args.plan_switch_min_gain_ms,
                "resident_bytes": previous_bytes,
                "usable_bytes": usable_bytes,
                "score": (
                    "working_unit_predicted_ms"
                    if str(req.get("granularity_policy", "none")) != "none"
                    else "pred_per_token_ms"),
            }
            return previous
        return plan

    def _adapt_working_sets(
        self,
        plan: dict[str, Any],
        budget_plan: dict[str, Any],
        state: dict[str, Any],
        bootstrap: bool,
    ) -> dict[str, Any]:
        """Patch budget targets using observed working-set pressure.

        The CP plan supplies the hard capacity ceiling for the current budget.
        Runtime state decides whether growing to that ceiling is worth paying
        for now; shrinking for budget compliance is always immediate.
        """
        budget_by_kind = {
            str(row.get("kind", "")): row
            for row in budget_plan.get("working_sets", [])
            if isinstance(row, dict) and str(row.get("kind", ""))
        }
        state_by_kind = {
            str(row.get("kind", "")): row
            for row in state.get("working_sets", [])
            if isinstance(row, dict) and str(row.get("kind", ""))
        }
        if not budget_by_kind:
            return plan

        result = copy.deepcopy(plan)
        output: list[dict[str, Any]] = []
        for kind, budget_ws in budget_by_kind.items():
            row = copy.deepcopy(budget_ws)
            ceiling = int(row.get("budget_capacity", row.get("target_capacity", -1)))
            row["budget_capacity"] = ceiling
            runtime_ws = state_by_kind.get(kind)
            if bootstrap or runtime_ws is None or ceiling < 0:
                output.append(row)
                continue

            active = int(runtime_ws.get("active_capacity", -1))
            observed_required = int(runtime_ws.get("observed_required_capacity", 0))
            accesses = int(runtime_ws.get("accesses", 0))
            misses = int(runtime_ws.get("misses", 0))
            with self._state_lock:
                prev_accesses, prev_misses = self._last_working_set_counters.get(kind, (accesses, misses))
                self._last_working_set_counters[kind] = (accesses, misses)
            delta_accesses = max(0, accesses - prev_accesses)
            delta_misses = max(0, misses - prev_misses)
            miss_rate = delta_misses / delta_accesses if delta_accesses else 0.0

            if active < 0:
                target = ceiling
                reason = "runtime_uninitialized"
            elif active > ceiling:
                target = ceiling
                reason = "budget_shrink"
            elif observed_required > active and active < ceiling:
                target = min(ceiling, max(observed_required, active + self.args.working_set_grow_step))
                reason = "observed_high_water_grow"
            elif miss_rate >= self.args.working_set_grow_miss_rate and active < ceiling:
                target = min(ceiling, active + self.args.working_set_grow_step)
                reason = "miss_pressure_grow"
            else:
                target = active
                reason = "keep_current"

            row.update({
                "target_capacity": target,
                "policy": "runtime_pressure",
                "state_aware": True,
                "runtime_active_capacity": active,
                "runtime_observed_required_capacity": observed_required,
                "runtime_delta_accesses": delta_accesses,
                "runtime_delta_misses": delta_misses,
                "runtime_miss_rate": miss_rate,
                "budget_capacity_ceiling": ceiling,
                "adjustment_reason": reason,
            })
            output.append(row)
        result["working_sets"] = output
        return result

    def solve(self, req: dict[str, Any]) -> tuple[dict[str, Any], float]:
        solve_t0 = time.perf_counter()
        self._reset_for_run(str(req.get("run_id", "")))
        with self._state_lock:
            bootstrap = self._last_plan is None
            cached_plan = self._last_plan
        budget_mib = int(req["budget_mib"])
        if (
            self.args.same_budget_cache
            and not bootstrap
            and cached_plan is not None
            and int(cached_plan.get("budget_mib", -1)) == budget_mib
        ):
            solve_ms = (time.perf_counter() - solve_t0) * 1000.0
            working_unit = cached_plan.get("working_unit", {})
            print(json.dumps({
                "event": "solve-cache",
                "run_id": str(req.get("run_id", "")),
                "budget_mib": budget_mib,
                "solve_ms": solve_ms,
                "working_unit_policy": (
                    working_unit.get("policy")
                    if isinstance(working_unit, dict) else None),
                "working_unit_predicted_ms": (
                    working_unit.get("predicted_ms")
                    if isinstance(working_unit, dict) else None),
            }, sort_keys=True), flush=True)
            return cached_plan, solve_ms
        # Before the first decode, lazy materialization makes the phone's
        # residency snapshot incomplete. Bootstrap from the state-unaware CP
        # optimum, then use physical state plus the previous logical target for
        # every incremental adjustment.
        state = {"weights": []} if self.args.ignore_state or bootstrap else self._augment_cpu_residency(req.get("state", {"weights": []}))
        state = self._augment_previous_placements(state)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(state, f)
            state_path = Path(f.name)
        try:
            granularity_offline_dir = getattr(
                self.args, "granularity_offline_dir", None)
            granularity_offline_target_plan = None
            if granularity_offline_dir is not None:
                candidate = (
                    Path(granularity_offline_dir) /
                    f"plan_{int(req['budget_mib'])}MiB.json"
                )
                if candidate.exists():
                    granularity_offline_target_plan = candidate
            ns = SimpleNamespace(
                model_meta=self.args.model_meta,
                cost_dir=self.args.cost_dir,
                state=state_path,
                budget_mib=int(req["budget_mib"]),
                kv_mib=int(req.get("kv_mib", self.args.kv_mib)),
                misc_mib=int(req.get("misc_mib", self.args.misc_mib)),
                safety_mib=int(req.get("safety_mib", self.args.safety_mib)),
                planner_stream_reserve_mib=int(req.get(
                    "planner_stream_reserve_mib",
                    self.args.planner_stream_reserve_mib)),
                prefetch_distance=int(req.get("prefetch_distance", self.args.prefetch_distance)),
                time_limit_ms=int(req.get("time_limit_ms", self.args.time_limit_ms)),
                allow_cpu_fallback=bool(req.get("allow_cpu_fallback", self.args.allow_cpu_fallback)),
                transition_weight=float(req.get("transition_weight", self.args.transition_weight)),
                transition_horizon_tokens=float(req.get("transition_horizon_tokens", self.args.transition_horizon_tokens)),
                disk_reload_multiplier=float(req.get("disk_reload_multiplier", self.args.disk_reload_multiplier)),
                disk_gpu_reload_multiplier=float(req.get("disk_gpu_reload_multiplier", self.args.disk_gpu_reload_multiplier)),
                overlap_model=str(req.get("overlap_model", self.args.overlap_model)),
                cp_objective=str(req.get("cp_objective", self.args.cp_objective)),
                allowed_placements=str(req.get("allowed_placements", self.args.allowed_placements)),
                allow_output_cpu=bool(
                    req.get("allow_output_cpu", self.args.allow_output_cpu)),
                allow_output_disk=bool(
                    req.get("allow_output_disk", self.args.allow_output_disk)),
                dynamic_active_experts=float(req.get("dynamic_active_experts", self.args.dynamic_active_experts)),
                dynamic_total_experts=float(req.get("dynamic_total_experts", self.args.dynamic_total_experts)),
                dynamic_weight_pattern=str(req.get("dynamic_weight_pattern", self.args.dynamic_weight_pattern)),
                force_weight_placement=list(
                    self.args.force_weight_placement),
                granularity_policy=str(
                    req.get("granularity_policy",
                            self.args.granularity_policy)),
                granularity_backend=str(
                    req.get("granularity_backend",
                            self.args.granularity_backend)),
                granularity_profile=self.args.granularity_profile,
                granularity_current_plan=None,
                granularity_offline_target_plan=(
                    granularity_offline_target_plan),
                granularity_horizon_tokens=float(
                    req.get("granularity_horizon_tokens",
                            self.args.granularity_horizon_tokens)),
                granularity_min_gain_ms=float(
                    req.get("granularity_min_gain_ms",
                            self.args.granularity_min_gain_ms)),
                granularity_max_edits=int(
                    req.get("granularity_max_edits",
                            self.args.granularity_max_edits)),
                granularity_beam_width=int(
                    req.get("granularity_beam_width",
                            self.args.granularity_beam_width)),
                out=Path("/dev/null"),
            )
            t0 = time.perf_counter()
            if (
                self.args.granularity_placement_source == "offline-table"
                and granularity_offline_target_plan is not None
            ):
                # Granularity is the controlled variable in this comparison.
                # Reuse the exact offline placement for the current budget so
                # Online/Diff cannot win or lose merely because CP-SAT chose a
                # different equal-byte residency set. Online may replace the
                # complete frontier; Diff-tree may apply only bounded edits.
                budget_plan = json.loads(
                    granularity_offline_target_plan.read_text())
                plan = copy.deepcopy(budget_plan)
                offline_working_unit = budget_plan.get("working_unit")
                current_working_unit = state.get("working_unit")
                dynamic_budget_solver.attach_working_unit(
                    self._model_meta,
                    plan,
                    self._granularity_model,
                    policy=ns.granularity_policy,
                    current_working_unit=(
                        current_working_unit
                        if isinstance(current_working_unit, dict)
                        else None),
                    offline_working_unit=(
                        offline_working_unit
                        if isinstance(offline_working_unit, dict)
                        else None),
                    telemetry=state.get("granularity"),
                    horizon_tokens=ns.granularity_horizon_tokens,
                    min_gain_ms=ns.granularity_min_gain_ms,
                    max_edits=ns.granularity_max_edits,
                    beam_width=ns.granularity_beam_width,
                )
                plan.setdefault("cost_model", {})[
                    "online_placement_source"
                ] = "precomputed-offline-budget-table"
                budget_plan = plan
            else:
                budget_plan = dynamic_budget_solver.build_plan(ns)
                budget_plan.setdefault("cost_model", {})[
                    "online_placement_source"
                ] = "stateful-runtime-cp"
            plan = self._stabilize_plan(budget_plan, req)
            plan = self._adapt_working_sets(plan, budget_plan, state, bootstrap)
            solve_ms = (time.perf_counter() - t0) * 1000.0
            working_unit = plan.get("working_unit", {})
            units = (
                working_unit.get("units", [])
                if isinstance(working_unit, dict) else [])
            unit_modes = {"multi": 0, "tensor": 0, "cut": 0}
            weight_unit_counts: dict[int, int] = {}
            for unit in units if isinstance(units, list) else []:
                if not isinstance(unit, dict):
                    continue
                weight_ids = {
                    int(tile.get("weight_id", -1))
                    for tile in unit.get("tiles", [])
                    if isinstance(tile, dict) and
                    int(tile.get("weight_id", -1)) >= 0
                }
                if len(weight_ids) > 1:
                    unit_modes["multi"] += 1
                else:
                    for weight_id in weight_ids:
                        weight_unit_counts[weight_id] = (
                            weight_unit_counts.get(weight_id, 0) + 1)
            for count in weight_unit_counts.values():
                if count > 1:
                    unit_modes["cut"] += count
                else:
                    unit_modes["tensor"] += 1
            diagnostics = (
                plan.get("cost_model", {}).get("working_unit", {})
                if isinstance(plan.get("cost_model"), dict) else {})
            print(json.dumps({
                "event": "solve",
                "run_id": str(req.get("run_id", "")),
                "budget_mib": int(req["budget_mib"]),
                "solve_ms": solve_ms,
                "pred_per_token_ms": plan.get("pred_per_token_ms"),
                "working_unit_predicted_ms": (
                    working_unit.get("predicted_ms")
                    if isinstance(working_unit, dict) else None),
                "working_unit_policy": (
                    working_unit.get("policy")
                    if isinstance(working_unit, dict) else None),
                "unit_modes": unit_modes,
                "bootstrap_offline": diagnostics.get("bootstrap_offline"),
                "offline_target_source": diagnostics.get(
                    "offline_target_source"),
                "online_placement_source": (
                    plan.get("cost_model", {}).get(
                        "online_placement_source")
                    if isinstance(plan.get("cost_model"), dict)
                    else None),
                "wait_ratio": diagnostics.get("wait_ratio"),
                "edit_count": diagnostics.get("edit_count"),
                "edits_truncated": diagnostics.get(
                    "edits_truncated", False),
                "edits": diagnostics.get("edits", []),
                "cut_boundaries": diagnostics.get("cut_boundaries"),
                "merge_boundaries": diagnostics.get(
                    "merge_boundaries"),
                "mode_boundary_cost_ms": diagnostics.get(
                    "mode_boundary_cost_ms"),
                "constant_switch_fast_path": diagnostics.get(
                    "constant_switch_fast_path", False),
                "accepted": diagnostics.get("accepted"),
                "steady_gain_ms": diagnostics.get("steady_gain_ms"),
                "horizon_net_gain_ms": diagnostics.get(
                    "horizon_net_gain_ms"),
                "offline_distance_before": diagnostics.get(
                    "offline_distance_before"),
                "offline_distance_after": diagnostics.get(
                    "offline_distance_after"),
                "stabilization": plan.get("online_stabilization"),
            }, sort_keys=True), flush=True)
            self._remember_cpu_residency(plan)
            self._remember_placements(plan)
            return plan, solve_ms
        finally:
            try:
                state_path.unlink()
            except OSError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Remote dynamic budget solver server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model-meta", type=Path, required=True)
    ap.add_argument("--cost-dir", type=Path, required=True)
    ap.add_argument("--kv-mib", type=int, default=128)
    ap.add_argument("--misc-mib", type=int, default=256)
    ap.add_argument("--safety-mib", type=int, default=64)
    ap.add_argument("--planner-stream-reserve-mib", type=int, default=0)
    ap.add_argument("--prefetch-distance", type=int, default=1)
    ap.add_argument("--time-limit-ms", type=int, default=20)
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--transition-weight", type=float, default=0.1)
    ap.add_argument("--transition-horizon-tokens", type=float, default=1.0)
    ap.add_argument("--disk-reload-multiplier", type=float, default=1.0)
    ap.add_argument("--disk-gpu-reload-multiplier", type=float, default=4.0)
    ap.add_argument("--overlap-model", choices=("pipeline", "none"), default="pipeline")
    ap.add_argument("--cp-objective", choices=("resource_makespan", "interval_makespan", "sum"), default="resource_makespan")
    ap.add_argument("--allowed-placements", default="cpu,gpu,disk_cpu,disk_gpu")
    ap.add_argument(
        "--force-weight-placement", action="append", default=[],
        metavar="WEIGHT=PLACEMENT")
    ap.add_argument("--allow-output-cpu", action="store_true")
    ap.add_argument("--allow-output-disk", action="store_true")
    ap.add_argument("--dynamic-active-experts", type=float, default=0.0)
    ap.add_argument("--dynamic-total-experts", type=float, default=0.0)
    ap.add_argument("--dynamic-weight-pattern", default="_exps.weight")
    ap.add_argument(
        "--granularity-policy",
        choices=(
            "none", "fixed-multi", "fixed-tensor", "fixed-cut",
            "offline", "online", "diff-tree"),
        default="none")
    ap.add_argument(
        "--granularity-backend", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--granularity-profile", type=Path)
    ap.add_argument(
        "--granularity-offline-dir", type=Path,
        help="precomputed Offline-Mixed budget table for Diff-tree targets")
    ap.add_argument(
        "--granularity-placement-source",
        choices=("stateful-cp", "offline-table"),
        default="stateful-cp",
        help=(
            "placement source for online granularity policies; use "
            "offline-table for a controlled granularity-only mechanism test "
            "or stateful-cp for the full dynamic-planner comparison"))
    ap.add_argument("--granularity-horizon-tokens", type=float, default=8.0)
    ap.add_argument("--granularity-min-gain-ms", type=float, default=0.0)
    ap.add_argument("--granularity-max-edits", type=int, default=4)
    ap.add_argument("--granularity-beam-width", type=int, default=128)
    ap.add_argument("--working-set-grow-miss-rate", type=float, default=0.15)
    ap.add_argument("--working-set-grow-step", type=int, default=4)
    ap.add_argument("--plan-switch-min-gain-ms", type=float, default=0.1,
                    help="keep a feasible previous plan unless the proposed steady-state gain reaches this threshold")
    ap.add_argument("--carry-cpu-residency", action="store_true",
                    help="legacy debug fallback: infer CPU residency from the previous returned plan")
    ap.add_argument("--ignore-state", action="store_true",
                    help="solve online requests as stateless budget-only CP plans; useful for correctness baselines")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--same-budget-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reuse the last immutable plan within one quantized budget "
            "bucket; disable only for per-token planner diagnostics"))
    args = ap.parse_args()

    server = SolverHTTPServer((args.host, args.port), args)
    print(json.dumps({"host": args.host, "port": args.port, "model_meta": str(args.model_meta), "cost_dir": str(args.cost_dir)}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
