#!/bin/bash
# One B200 per array task. Submit from the source checkout containing flat.adm.
set -eo pipefail
module purge
module load cudatoolkit/13.2 hpcx-latest/hpcx
set -u
root=$PWD
exe=$root/build/src/athena
input=$root/inputs/boltzmann/photon_torus.athinput
index=$SLURM_ARRAY_TASK_ID
nx=(48 72 96 72 72 48 96 48 96 48)
mu=(24 24 24 12 36 24 36 36 48 12)
sectors=(32 32 32 16 48 32 48 48 64 16)
n=${nx[index]}
dir=runs/${SLURM_ARRAY_JOB_ID}/$index
mkdir -p "$(dirname "$dir")"
mkdir "$dir"
cd "$dir"
hostname
nvidia-smi -L
sha256sum "$exe"
if [[ $index == 5 ]]; then
  set -x
  "$exe" -i "$input" problem/torus_file="$root/strong.adm"
else
  cp "$input" flat.athinput
  cat >> flat.athinput <<'EOF'

<output2>
file_type = bin
variable = tmunu
dt = 0.5
EOF
  set -x
  "$exe" -i flat.athinput problem/torus_file="$root/flat.adm" \
    boltzmann/fixed_metric=true boltzmann/backreaction=false \
    mesh/x1max=12 mesh/x2min=-3 mesh/x2max=3 mesh/nx1="$n" mesh/nx2="$((n/2))" \
    meshblock/nx1="$((n/2))" meshblock/nx2="$((n/2))" \
    boltzmann/nmu_half="${mu[index]}" boltzmann/npsi="${sectors[index]}" \
    time/tlim=6.5 output1/dcycle=200 time/ndiag=500
fi
