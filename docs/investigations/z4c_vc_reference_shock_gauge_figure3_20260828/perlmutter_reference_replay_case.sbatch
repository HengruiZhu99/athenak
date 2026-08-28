#!/usr/bin/env bash
#SBATCH --account=m3328_g
#SBATCH --qos=shared_interactive
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=z4c-shock-replay
#SBATCH --output=/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828/slurm/replay-%j.out
#SBATCH --error=/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828/slurm/replay-%j.err

set -euo pipefail
: "${CASE_LABEL:?n128 or n512}"
: "${RUN_TLIM:?matched N256 endpoint}"
: "${SEGMENT:?three-digit segment identifier}"

campaign=/pscratch/sd/h/hzhu/z4c-vc-reference-shock-gauge-figure3-20260828
gate=${campaign}/evidence/N256_FIGURE3_GATE_PASS
test -f "${gate}"
grep -Fxq N256_FIGURE3_GATE_PASS "${gate}"

export RUN_ROOT=${campaign}/runs/${CASE_LABEL}_reference_shock_seg${SEGMENT}
export HISTORY_FILE=${campaign}/authority/n256_reference_shock_authority.jsonl
if [[ -n "${RESTART_FILE:-}" ]]; then export RESTART_FILE; fi
exec "${campaign}/perlmutter_run_reference_replay.sh"
