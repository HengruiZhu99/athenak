#!/bin/bash -l
set -euo pipefail
launch=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side/validation/perf-two-node-v1/ipc-cache-v2/launch
case_dir=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side/runs/native-n12-ipc-cache-abba-v2
cd "$launch"
sha256sum -c launchers.sha256
bash run-native-ipc-cache.sh
gate=$(/usr/bin/python3.11 "$launch/native_cache_qualification.py" gate)
if [[ "$gate" == run ]]; then
 bash run-native-cache-qualification.sh
else
 [[ "$gate" == skip ]]
 printf 'Qualification skipped: screen lacks full-range separation or full clean ABBA\n' > "$case_dir/qualification-skipped.txt"
fi
sha256sum -c launchers.sha256 > "$case_dir/combined-launchers-after.txt"
printf 'Combined allocation finished; independent review pending\n' > "$case_dir/combined-complete.txt"
