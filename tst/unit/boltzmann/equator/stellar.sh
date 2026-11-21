#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --job-name=eq-mesh
#SBATCH --output=equator-%j.log
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
set -u
source_dir=$SLURM_SUBMIT_DIR
exe=${EQUATOR_ATHENA:-$source_dir/build/src/athena}
tests=$source_dir/tst/unit/boltzmann
data=${EQUATOR_DATA:-/scratch/gpfs/FPRETORI/hzhu/photon-boundary-20261120/compact_spin_survey}
output="$source_dir/../measure-$SLURM_JOB_ID"
mkdir "$output"
export CUDA_DEVICE_ORDER=PCI_BUS_ID KOKKOS_DEVICE_ID=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ucx OMPI_MCA_coll=^ucc,hcoll UCX_TLS=self,sm,cuda_copy,cuda_ipc
measure() {
    local tag=$1 family=$2 end=$3
    shift 3
    if [[ -n "${EQUATOR_COMPLETED:-}" ]] &&
       grep -q '^Terminating on time limit' "$EQUATOR_COMPLETED/$tag/stdout" 2>/dev/null; then
        return
    fi
    mkdir "$output/$tag"
    cd "$output/$tag"
    ln -s "$data/$family/ring.adm" ring.adm
    sha256sum ring.adm "$exe" > hashes
    local launch=(-i input)
    if [[ -n "${EQUATOR_PARENT:-}" ]]; then
        grep -q '^Terminating on time limit' "$EQUATOR_PARENT/stdout"
        cmp hashes "$EQUATOR_PARENT/hashes"
        cp "$EQUATOR_PARENT/input" input
        local checkpoints=("$EQUATOR_PARENT"/rst/*.rst)
        local checkpoint="${checkpoints[-1]}"
        test -s "$checkpoint"
        printf '%s\n' "$checkpoint" > restart.path
        launch=(-r "$checkpoint")
    else
        python3 "$tests/equator/layout.py" input "$@"
    fi
    nvidia-smi --query-gpu=timestamp,uuid,memory.used --format=csv,noheader,nounits -l 1 > gpu-memory.csv &
    local monitor=$!
    trap 'kill "$monitor" 2>/dev/null || true' RETURN
    local ranks=$SLURM_NTASKS
    local bind=(--gpu-bind=mask_gpu:0xff) mapper=()
    if [[ "$ranks" -lt 8 ]]; then
        bind=(--gpus-per-task=1 --gpu-bind=single:1)
        mapper=(--rank-bound)
    fi
    /usr/bin/time -f '%e' -o wall.seconds srun --exact -N1 -n"$ranks" -c4 --gres=gpu:"$ranks" \
        "${bind[@]}" --cpu-bind=none --mpi=pmix --kill-on-bad-exit=1 \
        python3 "$tests/stellar_photon_rank.py" "${mapper[@]}" "$exe" "${launch[@]}" time/tlim="$end" > stdout 2>&1
    grep -q '^Terminating on time limit' stdout
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    trap - RETURN
}
if [[ $# -eq 0 ]]; then
    # Identical physical times, diagnostics, rank count and B200s for all four layouts.
    measure square-full j09508 1 --square
    measure square-half j09508 1 --square --half
    measure coarse-full j09508 1 --square --coarse
    measure coarse-half j09508 1 --square --coarse --half
elif [[ $1 == --pair ]]; then
    family=$2
    end=$3
    shift 3
    measure "$family-full" "$family" "$end" "$@"
    measure "$family-half" "$family" "$end" "$@" --half
else
    measure "$@"
fi
