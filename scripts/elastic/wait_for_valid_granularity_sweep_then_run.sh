#!/usr/bin/env bash
set -euo pipefail

if (( $# < 9 )); then
    echo "usage: $0 SERVICE RUNS_CSV BACKEND RATIOS MODES PROFILE PHASE_PROFILE COST_DIR COMMAND [ARGS...]" >&2
    exit 2
fi

service="$1"
runs_csv="$2"
backend="$3"
ratios="$4"
modes="$5"
profile="$6"
phase_profile="$7"
cost_dir="$8"
shift 8
python_bin="${ELASTIC_PYTHON:-python3}"

while systemctl --user is-active --quiet "$service"; do
    sleep 30
done

"$python_bin" - \
    "$runs_csv" "$backend" "$ratios" "$modes" \
    "$profile" "$phase_profile" "$cost_dir" <<'PY'
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

runs_path = Path(sys.argv[1])
backend = sys.argv[2]
ratios = [int(value) for value in sys.argv[3].split(",") if value]
modes = [value for value in sys.argv[4].split(",") if value]
profile = Path(sys.argv[5])
phase_profile = Path(sys.argv[6])
cost_dir = Path(sys.argv[7])

required_outputs = (
    runs_path,
    profile,
    phase_profile,
    cost_dir / "calibration.json",
)
missing = [str(path) for path in required_outputs if not path.is_file()]
if missing:
    raise SystemExit("missing calibrated outputs: " + ", ".join(missing))

rows = list(csv.DictReader(
    runs_path.open(newline="", encoding="utf-8")))
expected = set(itertools.product(ratios, modes))
actual = {
    (int(row.get("ratio_pct") or -1), row.get("mode", ""))
    for row in rows
}
if len(rows) != len(expected) or actual != expected:
    raise SystemExit(
        f"incomplete sweep: rows={len(rows)} "
        f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}")

def truth(row, key):
    return row.get(key, "").lower() == "true"

by_ratio = defaultdict(list)
for row in rows:
    label = f"{row.get('ratio_pct')}%/{row.get('mode')}"
    checks = {
        "backend": row.get("backend") == backend,
        "pipeline": row.get("pipeline") == "on",
        "returncode": row.get("returncode") == "0",
        "runtime": truth(row, "valid_runtime"),
        "idle": truth(row, "valid_device_idle"),
        "token": truth(row, "valid_token_sequence"),
        "pipeline_budget":
            int(float(row.get("pipeline_budget_violations") or -1)) == 0,
        "steady_tokens":
            int(float(row.get("steady_decode_tokens") or 0)) >= 9,
    }
    if backend == "gpu":
        checks["gpu_sync"] = (
            int(float(row.get("gpu_unit_sync_errors") or 0)) == 0)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"invalid sweep row {label}: {', '.join(failed)}")
    by_ratio[int(row["ratio_pct"])].append(row)

wbm_totals = {round(float(row["wbm_total_mib"]), 3) for row in rows}
if len(wbm_totals) != 1:
    raise SystemExit(f"mismatched WBM totals: {sorted(wbm_totals)}")

for ratio, group in by_ratio.items():
    budgets = {round(float(row["budget_mib"]), 3) for row in group}
    targets = {round(float(row["weight_target_mib"]), 3) for row in group}
    hashes = {row.get("token_prefix_sha256", "") for row in group}
    if len(budgets) != 1 or len(targets) != 1:
        raise SystemExit(
            f"{ratio}% modes use different budget/weight targets")
    if "" in hashes or len(hashes) != 1:
        raise SystemExit(f"{ratio}% token-prefix mismatch")
    direct_mib = [float(row.get("direct_read_mib") or 0) for row in group]
    if max(direct_mib, default=0) > 0:
        spread = (max(direct_mib) - min(direct_mib)) / max(direct_mib)
        if spread > 0.02:
            raise SystemExit(
                f"{ratio}% direct-read byte spread {spread:.2%} exceeds 2%")

if backend == "gpu":
    cut_rows = [row for row in rows if row.get("mode") == "cut"]
    if any(int(float(row.get("cut_dual_calls") or 0)) <= 0
           for row in cut_rows):
        raise SystemExit("GPU Cut dual-kernel path was not exercised")
    if any(int(float(row.get("cut_dual_queue_errors") or 0)) != 0
           for row in cut_rows):
        raise SystemExit("GPU Cut dual-kernel queue errors detected")
    profile_root = json.loads(profile.read_text())
    selected = profile_root.get("profiles", {}).get("gpu", {})
    compute_rate = float(selected.get("compute_ms_per_mib") or 0.0)
    compute_fit = (
        profile_root.get("calibration", {}).get("gpu", {})
        .get("compute_fit", {})
    )
    compute_details = compute_fit.get("details") or {}
    if compute_rate <= 0.001:
        raise SystemExit(
            "GPU compute rate is implausibly small; CPU fallback "
            "compute_pt may have been miscalibrated as GPU compute")
    if "highest-residency" not in str(compute_details.get("source", "")):
        raise SystemExit(
            "GPU profile lacks undisturbed resident-tail compute fit")
PY

exec "$@"
