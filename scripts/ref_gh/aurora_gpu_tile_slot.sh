#!/usr/bin/env bash
# Bind one independent Aurora process to an explicitly selected PVC tile.
set -euo pipefail

if (($# < 2)); then
  echo "usage: $0 <gpu.tile> <command> [args...]" >&2
  exit 2
fi

case "$1" in
  0.0|0.1|1.0|1.1|2.0|2.1|3.0|3.1|4.0|4.1|5.0|5.1)
    export ZE_AFFINITY_MASK="$1"
    ;;
  *)
    echo "invalid Aurora GPU tile: $1" >&2
    exit 2
    ;;
esac
shift
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
ulimit -c 0
echo "host=$(hostname) ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}" >&2
exec "$@"
