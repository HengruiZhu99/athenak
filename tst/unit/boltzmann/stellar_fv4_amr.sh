#!/bin/bash
# From a configured StellarAI CUDA checkout, allocate one node, two tasks and two GPUs.
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
set -u
export OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID KOKKOS_DEVICE_ID=0
export OMPI_MCA_pml=ucx OMPI_MCA_coll=^ucc,hcoll UCX_TLS=self,sm,cuda_copy
case ${1:-prescribed} in
  prescribed) backreaction=false ;;
  coupled) backreaction=true ;;
  *) echo "Usage: $0 [prescribed|coupled]" >&2; exit 2 ;;
esac
root=$PWD
work="$root/qualification/amr-$SLURM_JOB_ID"
mkdir -p "$work"
cmake -S . -B build
cmake --build build --target athena_boltzmann_fourth_amr --parallel "$SLURM_CPUS_PER_TASK"
sha256sum build/src/athena build/athena_boltzmann_fourth_amr
rank="$root/tst/unit/boltzmann/stellar_photon_rank.py"
launch() {
  local count=$1
  shift
  srun --exact -N1 -n "$count" -c "$SLURM_CPUS_PER_TASK" --gpus-per-task=1 \
    --gpu-bind=single:1 --cpu-bind=none --mpi=pmix --kill-on-bad-exit=1 \
    python3 "$rank" --rank-bound "$@"
}
launch 1 "$root/build/athena_boltzmann_fourth_amr" > "$work/operator.csv"
awk '{print} /^<problem>$/{print "exercise_deterministic_amr = true"} /^<output2>$/{print "dcycle = 3"}' \
  "$root/inputs/boltzmann/ring.athinput" > "$work/input"
settings=(problem/ring_radius=15 problem/angular_width=0.65
  boltzmann/sign=both "boltzmann/backreaction=$backreaction" boltzmann/nmu_half=6 boltzmann/transport_order=4
  time/integrator=ssprk104 time/nlim=6 fastflow/num_horizons=0 amr_criterion0/method=user
  mesh_refinement/ncycle_check=1 mesh_refinement/refinement_interval=1
  mesh_refinement/max_nmb_per_rank=8 output1/dcycle=1)
for count in 1 2; do
  mkdir "$work/ranks-$count"
  launch "$count" "$root/build/src/athena" -i "$work/input" \
    -d "$work/ranks-$count" "${settings[@]}" > "$work/ranks-$count/run.log" 2>&1
done
mkdir "$work/resumed"
launch 1 "$root/build/src/athena" -r "$work/ranks-2/rst/ring.00001.rst" \
  -d "$work/resumed" > "$work/resumed/run.log" 2>&1
