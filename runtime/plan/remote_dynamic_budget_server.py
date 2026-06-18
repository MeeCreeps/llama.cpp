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

    def solve(self, req: dict[str, Any]) -> tuple[dict[str, Any], float]:
        state = req.get("state", {"weights": []})
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
                out=Path("/dev/null"),
            )
            t0 = time.perf_counter()
            plan = dynamic_budget_solver.build_plan(ns)
            solve_ms = (time.perf_counter() - t0) * 1000.0
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
    ap.add_argument("--transition-weight", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    server = SolverHTTPServer((args.host, args.port), args)
    print(json.dumps({"host": args.host, "port": args.port, "model_meta": str(args.model_meta), "cost_dir": str(args.cost_dir)}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
