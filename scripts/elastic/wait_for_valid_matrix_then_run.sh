#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
    echo "usage: $0 SERVICE RESULTS_CSV EXPECTED_ROWS COMMAND [ARGS...]" >&2
    exit 2
fi

service="$1"
results_csv="$2"
expected_rows="$3"
shift 3
python_bin="${ELASTIC_PYTHON:-python3}"

while systemctl --user is-active --quiet "$service"; do
    sleep 30
done

"$python_bin" - "$results_csv" "$expected_rows" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"missing prerequisite results: {path}")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
if len(rows) != expected:
    raise SystemExit(
        f"prerequisite matrix has {len(rows)} rows, expected {expected}")
for row in rows:
    method = row.get("method", "<unknown>")
    checks = {
        "status": row.get("status") == "ok",
        "rc": row.get("rc") == "0",
        "token_trace": row.get("valid_token_trace", "").lower() == "true",
        "isolation":
            row.get("valid_measurement_isolation", "").lower() == "true",
        "pipeline_budget":
            int(float(row.get("pipeline_budget_violations") or -1)) == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(
            f"invalid prerequisite row {method}: {', '.join(failed)}")
PY

exec "$@"
