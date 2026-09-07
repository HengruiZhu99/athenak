#!/bin/bash
source /home/hz0693/athenak_env
set -eo pipefail
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-transfer-001
trap 'printf "%s\n" "$?" > build-accessfix.exit' EXIT
cmake --build build -j4 > build-accessfix.log 2>&1
