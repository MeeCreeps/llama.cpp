#!/usr/bin/env python3
"""HTTP wrapper for dynamic_budget_solver.py.

The phone posts the current budget and residency state. The server runs the
Python/CP-SAT solver locally and returns the native ExecPlan JSON. This keeps
Python and OR-Tools off the phone while preserving the online state-dependent
planning path.
"""

from __future__ import annotations

import argparse
import json
import threading
import tempfile
import time
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

    def solve(self, req: dict[str, Any]) -> tuple[dict[str, Any], float]:
        state = self._augment_cpu_residency(req.get("state", {"weights": []}))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(state, f)
            state_path = Path(f.name)
        try:
            ns = SimpleNamespace(
                model_meta=self.args.model_meta,
                cost_dir=self.args.cost_dir,
                state=state_path,
                budget_mib=int(req["budget_mib"]),
                kv_mib=int(req.get("kv_mib", self.args.kv_mib)),
                misc_mib=int(req.get("misc_mib", self.args.misc_mib)),
                safety_mib=int(req.get("safety_mib", self.args.safety_mib)),
                prefetch_distance=int(req.get("prefetch_distance", self.args.prefetch_distance)),
                time_limit_ms=int(req.get("time_limit_ms", self.args.time_limit_ms)),
                allow_cpu_fallback=bool(req.get("allow_cpu_fallback", self.args.allow_cpu_fallback)),
                transition_weight=float(req.get("transition_weight", self.args.transition_weight)),
                disk_reload_multiplier=float(req.get("disk_reload_multiplier", self.args.disk_reload_multiplier)),
                disk_gpu_reload_multiplier=float(req.get("disk_gpu_reload_multiplier", self.args.disk_gpu_reload_multiplier)),
                overlap_model=str(req.get("overlap_model", self.args.overlap_model)),
                cp_objective=str(req.get("cp_objective", self.args.cp_objective)),
                allowed_placements=str(req.get("allowed_placements", self.args.allowed_placements)),
                out=Path("/dev/null"),
            )
            t0 = time.perf_counter()
            plan = dynamic_budget_solver.build_plan(ns)
            solve_ms = (time.perf_counter() - t0) * 1000.0
            self._remember_cpu_residency(plan)
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
    ap.add_argument("--prefetch-distance", type=int, default=1)
    ap.add_argument("--time-limit-ms", type=int, default=20)
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--transition-weight", type=float, default=0.1)
    ap.add_argument("--disk-reload-multiplier", type=float, default=1.0)
    ap.add_argument("--disk-gpu-reload-multiplier", type=float, default=4.0)
    ap.add_argument("--overlap-model", choices=("pipeline", "none"), default="pipeline")
    ap.add_argument("--cp-objective", choices=("resource_makespan", "interval_makespan", "sum"), default="resource_makespan")
    ap.add_argument("--allowed-placements", default="cpu,gpu,disk_cpu,disk_gpu")
    ap.add_argument("--carry-cpu-residency", action="store_true",
                    help="legacy debug fallback: infer CPU residency from the previous returned plan")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    server = SolverHTTPServer((args.host, args.port), args)
    print(json.dumps({"host": args.host, "port": args.port, "model_meta": str(args.model_meta), "cost_dir": str(args.cost_dir)}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
