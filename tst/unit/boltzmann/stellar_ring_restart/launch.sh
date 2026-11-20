#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --job-name=ring09508
#SBATCH --output=slurm-%j.log
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx anaconda3/2025.12
set -u
root=$SLURM_SUBMIT_DIR
base=$(dirname "$root")
read -ra original < "$base/runs/52549_0/command"
exe=${original[0]}
args=("${original[@]:1}")
export CUDA_DEVICE_ORDER=PCI_BUS_ID KOKKOS_DEVICE_ID=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ucx OMPI_MCA_coll=^ucc,hcoll UCX_TLS=self,sm,cuda_copy,cuda_ipc
launch=(srun --exact -N1 -n8 -c4 --gres=gpu:8 --gpu-bind=mask_gpu:0xff --cpu-bind=none --mpi=pmix --kill-on-bad-exit=1 python3 "$base/stellar_photon_rank.py" "$exe")
deck() {
    cp "$base/runs/52549_0/input" input
    printf '\n<output5>\nfile_type = rst\ndcycle = %s\n' "$1" >> input
}
if [[ ! -f "$root/test.passed" ]]; then
    testdir="$root/test-$SLURM_JOB_ID"
    mkdir -p "$testdir/direct" "$testdir/resumed"
    cd "$testdir/direct"
    deck 2
    "${launch[@]}" "${args[@]}" time/nlim=4 >stdout 2>&1
    cd ../resumed
    "${launch[@]}" -r ../direct/rst/torus.00001.rst >stdout 2>&1
    cd ..
    python "$root/check.py" | tee "$root/test.passed.tmp"
    mv "$root/test.passed.tmp" "$root/test.passed"
fi
cd "$root"
segment="$root/segment-$SLURM_JOB_ID"
mkdir "$segment"
cd "$segment"
if [[ $# -eq 0 ]]; then
    deck 2400
    start=("${args[@]}")
else
    start=(-r "$1")
fi
# Leave wall-clock margin for a clean final restart write and scheduler handoff.
"${launch[@]}" "${start[@]}" time/tlim=84 -t 02:45:00 >stdout 2>&1
if grep -q '^Terminating on time limit' stdout && grep -Eq '^time=8\.400000e\+01 cycle=|^time=84(\.0*)? cycle=' stdout; then
    touch "$root/complete"
elif ! grep -q '^Terminating on wall clock limit' stdout; then
    echo 'Non-wall-clock stop before target' >&2
    exit 1
fi
