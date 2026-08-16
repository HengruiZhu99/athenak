#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${root}"

sha256sum -c SHA256SUMS
printf '%s  %s\n' "$(sha256sum SHA256SUMS | awk '{print $1}')" SHA256SUMS \
  | sha256sum -c -

