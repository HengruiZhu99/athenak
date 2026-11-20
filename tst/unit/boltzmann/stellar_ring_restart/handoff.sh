#!/bin/bash
#SBATCH --account=fpretori
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --job-name=ring-handoff
#SBATCH --output=handoff-%j.log
set -euo pipefail
prev=$1
[[ ! -f complete ]] || exit 0
segment="segment-$prev"
state=$(sacct -X -j "$prev" -n -P -o State | head -1 | cut -d'|' -f1)
# A successful wall stop closes its final checkpoint. After an allocation loss,
# use the preceding checkpoint, since the last file could have been interrupted.
files=("$segment"/rst/*.rst)
case "$state" in
  COMPLETED) grep -q '^Terminating on wall clock limit' "$segment/stdout"; idx=$((${#files[@]}-1));;
  TIMEOUT*|NODE_FAIL*|PREEMPTED*) idx=$((${#files[@]}-2));;
  *) echo "Stopped for inspection: $state"; exit 1;;
esac
((idx>=0)) && [[ -s "${files[$idx]}" ]] || { echo 'No closed checkpoint'; exit 1; }
next=$(sbatch --parsable launch.sh "$PWD/${files[$idx]}")
echo "Continuing job $prev from ${files[$idx]} in job $next"
sbatch --dependency=afterany:"$next" handoff.sh "$next"
