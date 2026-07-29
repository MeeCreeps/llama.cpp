#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 UPSTREAM_USER_UNIT COMMAND [ARG ...]" >&2
  exit 2
fi

upstream="$1"
shift

while true; do
  active="$(systemctl --user show "$upstream" -p ActiveState --value 2>/dev/null || true)"
  case "$active" in
    active|activating|reloading)
      sleep 30
      ;;
    *)
      break
      ;;
  esac
done

result="$(systemctl --user show "$upstream" -p Result --value 2>/dev/null || true)"
status="$(systemctl --user show "$upstream" -p ExecMainStatus --value 2>/dev/null || true)"
if [[ "$result" != "success" || "$status" != "0" ]]; then
  echo "upstream $upstream failed: Result=$result ExecMainStatus=$status" >&2
  exit 1
fi

exec "$@"
