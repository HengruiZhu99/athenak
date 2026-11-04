#!/bin/bash -l
set -euo pipefail
cd "$(dirname "$0")"
module load oneapi/release/2025.3.1
module list > build-modules.txt 2>&1
icpx --version > build-compiler.txt
exec /usr/bin/python3.11 build.py
