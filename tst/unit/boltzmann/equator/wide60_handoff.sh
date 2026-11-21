#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=wide106-handoff
# Submit the final GPU segment after its predecessor releases a GPU queue slot.
set -euo pipefail
record=$1
exec 9>"$record/handoff.lock"
flock -x 9
if awk '$2==60 {found=1} END {exit !found}' "$record/segments.txt"; then exit 0; fi
parent=$(awk '$2==40 {print $1}' "$record/segments.txt")
test -n "$parent"
base=$(cd "$SLURM_SUBMIT_DIR/.." && pwd)
export EQUATOR_PARENT="$base/measure-$parent/wide-j10600-to60"
grep -q '^Terminating on time limit' "$EQUATOR_PARENT/stdout"
end=$2
id=$(sbatch --parsable -A fpretori -p gpu -N1 --ntasks-per-node=8 --cpus-per-task=4 \
     --gres=gpu:8 --mem=240G -t 24:00:00 --export=ALL -o wide60-%j.log -J wide106-60M \
     tst/unit/boltzmann/equator/wide60_finish.sh "$end" "$record")
echo "$id 60 $end" | tee -a "$record/segments.txt"
