#!/bin/bash
# Finish the last restart segment, then release the GPUs and analyze on a CPU node.
set -eu
suite=$SLURM_SUBMIT_DIR/tst/unit/boltzmann/equator
base=$SLURM_SUBMIT_DIR/..
bash "$suite/stellar.sh" wide-j10600-to60 j10600_refined "$1"
data=${EQUATOR_DATA:-/scratch/gpfs/FPRETORI/hzhu/photon-boundary-20261120/compact_spin_survey}
export WIDE_RUN="$base/measure-$SLURM_JOB_ID/wide-j10600-to60"
export WIDE_METADATA="$data/j10600_refined/ring.adm.json" WIDE_REVIEW="$2/review"
export WIDE_SEED="$base/j09508-trapped.json"
sbatch --parsable --export=ALL -A fpretori -p cpu -N1 -n1 -c2 --mem=16G -t 06:00:00 \
  -J wide106-review -o wide60-review-%j.log <<'POST'
#!/bin/bash
set -eo pipefail
module purge
module load anaconda3/2025.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tst/unit/boltzmann/equator/wide60_review.py "$WIDE_RUN" "$WIDE_METADATA" "$WIDE_REVIEW" "$WIDE_SEED"
POST
