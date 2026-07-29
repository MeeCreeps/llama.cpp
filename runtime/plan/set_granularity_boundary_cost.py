#!/usr/bin/env python3
"""Derive a mixed-frontier calibration profile without refitting pure modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--cut-boundary-ms", type=float, required=True)
    parser.add_argument("--merge-boundary-ms", type=float, default=0.0)
    parser.add_argument("--source-suffix", default="mixed-boundary-calibration")
    args = parser.parse_args()

    if args.cut_boundary_ms < 0.0:
        parser.error("--cut-boundary-ms must be non-negative")
    if args.merge_boundary_ms < 0.0:
        parser.error("--merge-boundary-ms must be non-negative")

    root = json.loads(args.base.read_text())
    profiles = root.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"{args.base} has no profiles object")
    selected = profiles.get(args.backend)
    if not isinstance(selected, dict):
        raise ValueError(
            f"{args.base} has no {args.backend} profile")
    selected["cut_boundary_ms"] = args.cut_boundary_ms
    selected["merge_boundary_ms"] = args.merge_boundary_ms
    root["source"] = (
        f"{root.get('source', args.base.stem)}-"
        f"{args.source_suffix}-cut{args.cut_boundary_ms:g}-"
        f"merge{args.merge_boundary_ms:g}"
    )
    calibration = root.setdefault(
        "calibration", {}).setdefault(args.backend, {})
    calibration["mixed_boundary"] = {
        "cut_boundary_ms": args.cut_boundary_ms,
        "merge_boundary_ms": args.merge_boundary_ms,
        "status": "candidate",
        "selection_data": "short fixed-node trace; full trace held out",
    }
    root["held_out"] = (
        "the original 600-second dynamic trace is not used for boundary "
        "candidate selection"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(root, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
