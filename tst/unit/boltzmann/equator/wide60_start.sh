#!/bin/bash
# Start the 288-unit 1.0600 evolution from the fastest completed wide pilot.
set -eo pipefail
module purge
module load anaconda3/2025.12
set -u
suite=$SLURM_SUBMIT_DIR/tst/unit/boltzmann/equator
bench=$SLURM_SUBMIT_DIR/../measure-$1
export PYTHONPATH="$suite${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
export EQUATOR_PARENT
EQUATOR_PARENT=$(python - "$bench" <<'PY'
from pathlib import Path
import sys
from performance import summarize

bench=Path(sys.argv[1])
candidates=[]
for name in ("wide288-b36", "wide288-72x36", "wide288-b72"):
    run=bench/name
    if (run/"wall.seconds").exists() and (run/"wall.seconds").stat().st_size and (
            "Terminating on time limit" in (run/"stdout").read_text()):
        candidates.append(summarize(run))
if not candidates or len({(v["host"],tuple(v["pci_devices"]),v["executable_sha256"],
                           v["initial_data_sha256"],v["start_time"],v["end_time"])
                          for v in candidates}) != 1:
    raise SystemExit("missing or unmatched completed large-domain benchmarks")
best=min(candidates,key=lambda v:v["GPU_hours"])
if best["start_time"] != 0 or abs(best["end_time"]-.5)>1.e-12 or best["GPUs"]!=4:
    raise SystemExit("unexpected benchmark interval or hardware")
print((bench/best["layout"]).resolve())
PY
)
echo "Continuing $EQUATOR_PARENT to $2 code units"
exec bash "$suite/stellar.sh" wide-j10600-to60 j10600_refined "$2"
