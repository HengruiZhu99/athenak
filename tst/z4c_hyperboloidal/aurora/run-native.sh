#!/bin/bash -l
# Fresh fixed N12 correctness run; never reads a production restart.
set -euo pipefail
ulimit -c 0
nodes=${1:?Pass 2 or 8 nodes}
steps=${NATIVE_STEPS:-8}
case "$nodes" in 2|8) ;; *) exit 2 ;; esac
case "$steps" in 1|8|3551) ;; *) exit 2 ;; esac
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
build_dir=${NATIVE_BUILD:-$task_root/build-sycl}
case_name=${NATIVE_CASE:-n12-nodes${nodes}-steps${steps}}
[[ "$case_name" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
case_dir=$task_root/runs/$case_name
build_dir=$(realpath "$build_dir")
case "$build_dir" in "$task_root"/build-*) ;; *) exit 2 ;; esac
case "$case_dir" in *..*) exit 2 ;; esac
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
allocated=$(sort -u "$PBS_NODEFILE" | awk 'END {print NR}')
[[ "$allocated" == "$nodes" ]]
[[ ! -e "$case_dir" ]] || { printf 'Run directory already exists\n' >&2; exit 2; }
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
module list > "$case_dir/modules.txt" 2>&1
qstat -u hzhu > "$case_dir/queue-at-start.txt"
sha256sum -c "$build_dir/executable.sha256" > "$case_dir/executable-check.txt"
/usr/bin/python3.11 "$task_root/source/tst/z4c_hyperboloidal/aurora/source_manifest.py" \
  "$task_root/source" "$build_dir/source-sha256.json" --check > "$case_dir/source-check.txt"
input=$task_root/source/tst/z4c_hyperboloidal/aurora/n12.athinput
cp "$input" "$case_dir/input.athinput"
sha256sum "$input" > "$case_dir/input.sha256"
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export MPIR_CVAR_ENABLE_GPU=1
export NEO_CACHE_DIR="$case_dir/cache/neo" SYCL_CACHE_DIR="$case_dir/cache/sycl"
mkdir -p "$NEO_CACHE_DIR" "$SYCL_CACHE_DIR"
unset ZE_AFFINITY_MASK
wrapper=$(command -v gpu_tile_compact.sh)
printf '%s  %s\n' 4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd "$wrapper" \
  | sha256sum -c - > "$case_dir/gpu-wrapper-check.txt"
ranks=$((12*nodes))
mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /bin/bash -c \
  'printf "%s local_rank=%s tile=%s implicit_scaling=%s\n" "$(hostname)" "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-unknown}}" "$ZE_AFFINITY_MASK" "$EnableImplicitScaling"' \
  > "$case_dir/binding.txt"
/usr/bin/python3.11 - "$case_dir/binding.txt" "$nodes" <<'PY'
import collections,pathlib,re,sys
hosts=collections.defaultdict(dict)
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    match=re.fullmatch(r'(\S+) local_rank=(\d+) tile=(\d+\.\d+) implicit_scaling=0',line)
    assert match, ('unexpected binding',line)
    host,rank,tile=match.groups();rank=int(rank)
    assert rank not in hosts[host], ('duplicate local rank',line)
    assert tile==f'{rank//2}.{rank%2}', ('wrong tile',line)
    hosts[host][rank]=tile
assert len(hosts)==int(sys.argv[2])
assert all(set(ranks)==set(range(12)) for ranks in hosts.values())
print('Verified twelve unique GPU tile ranks on each allocated node')
PY
command=(mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper"
  "$build_dir/src/athena" -i "$case_dir/input.athinput" -d "$case_dir" -t 00:50:00 "time/nlim=$steps")
printf '%q ' "${command[@]}" > "$case_dir/command.txt"
printf '\n' >> "$case_dir/command.txt"
timeout 3420 "${command[@]}" > "$case_dir/run.log" 2>&1
sha256sum -c "$build_dir/executable.sha256" >> "$case_dir/executable-check.txt"
/usr/bin/python3.11 "$task_root/source/tst/z4c_hyperboloidal/aurora/source_manifest.py" \
  "$task_root/source" "$build_dir/source-sha256.json" --check >> "$case_dir/source-check.txt"
/usr/bin/python3.11 - "$case_dir/run.log" "$steps" <<'PY'
import pathlib,re,sys
text=pathlib.Path(sys.argv[1]).read_text();steps=int(sys.argv[2])
assert 'Terminating on cycle limit' in text, 'incomplete run (wall/time limit is not success)'
assert re.search(r'^time=\S+ cycle='+str(steps)+r'$',text,re.M), 'wrong final cycle'
rows=re.findall(r'^elapsed=\S+ cycle=(\d+) time=(\S+) dt=(\S+)$',text,re.M)
assert rows and all(float(dt)==0.00046944203735617445 for _,_,dt in rows), 'dt changed'
assert {int(cycle) for cycle,_,_ in rows}==set(range(steps+1)), 'cycle coverage missing'
print('Fresh native run completed',steps,'whole RK4 steps')
PY
