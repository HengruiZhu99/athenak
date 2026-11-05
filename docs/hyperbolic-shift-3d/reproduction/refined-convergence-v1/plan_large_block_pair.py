"""Bind a fresh, same-allocation block12/block24 performance/physics pair."""
from pathlib import Path
import json, sys
root=Path(sys.argv[1]).resolve();package=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'launcher-boosted-transition-v1'))
from control import read,write,digest
from storage import capacity
assert not (root/'allocation-lock.json').exists()
assert read(root/'build-result.json')['status']=='completed'
science=read(root/'science.json')
assert digest(root/science['binary'])==science['binary_sha256']
assert digest(root/'source-manifest.json')==science['source_manifest_sha256']
launcher=root/'launcher-boosted-transition-v1'
for name,record in read(launcher/'manifest.json').items():
    assert digest(launcher/name)==record['sha256']
base=read(launcher/'qualified-identity.json');large=read(package/'qualified-identity.json')
assert base['source_commit']==large['source_commit']==science['source_commit']
assert digest(root/'candidate.cfg')==base['config_sha256']
assert digest(package/'candidate.cfg')==large['config_sha256']
rel='campaigns/boosted-block12-block24-pair-v1';directory=root/rel
directory.mkdir(parents=True,exist_ok=False)
package_rel=str(package.relative_to(root))
pbs=f'''#!/bin/bash -l
#PBS -N rf2-block-pair
#PBS -A CompactBinaryMerger
#PBS -q debug-scaling
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
set -euo pipefail
ulimit -c 0
cd {root}
module load oneapi/release/2025.3.1
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
export NEO_CACHE_DIR="$PWD/runtime-cache/neo" SYCL_CACHE_DIR="$PWD/runtime-cache/sycl"
unset ZE_AFFINITY_MASK
mkdir -p "$NEO_CACHE_DIR" "$SYCL_CACHE_DIR"
module list > {rel}/modules.txt 2>&1
exec /usr/bin/python3.11 {package_rel}/worker.py "$PWD" "{rel}/plan.json"
'''
(directory/'job.pbs').write_text(pbs)
files=['science.json','source-manifest.json',science['binary'],'candidate.cfg',rel+'/job.pbs',
       'launcher-boosted-transition-v1/manifest.json']
files += ['launcher-boosted-transition-v1/'+f for f in read(launcher/'manifest.json')]
files += [str(f.relative_to(root)) for f in package.iterdir() if f.is_file()]
bound={f:digest(root/f) for f in sorted(set(files))}
plan=dict(plan=rel+'/plan.json',purpose='large_block_pair',nodes=4,ranks=48,
 queue='debug-scaling',output=rel+'/output',measured_steps=600,
 planned_output_bytes=3_500_000_000,atomic_reserve_bytes=2_450_000_000,
 source_commit=science['source_commit'],bindings=bound,cases=[],
 qualification='Fresh exact v=.1 geometry, same gauge/RHS and dt1/24000 through.075M. Mesh-block size/SMR topology and shell layout change; not a pure block repartition. Short physics/cost control only, no automatic physics restart or long-run adoption.')
for name,n,identity,config in [('block12',12,base['identity'],'candidate.cfg'),
                              ('block24',24,large['identity'],package_rel+'/candidate.cfg')]:
    plan['cases'].append(dict(name=name,plan=plan['plan'],n=n,kind='boost',boost=.1,
       identity=identity,config=config,binary=science['binary'],source_commit=science['source_commit'],
       nodes=4,ranks=48,dt=1/24000,target_time=.025,diagnostic_interval=.0125,
       restart=None,start_time=0.,start_steps=0,output=rel+'/output/'+name,bindings=bound))
write(directory/'binding-storage.json',capacity(root,planned=plan['planned_output_bytes'],reserve=plan['atomic_reserve_bytes']))
write(directory/'plan.json',plan)
print(json.dumps(dict(plan=plan['plan'],sha256=digest(directory/'plan.json'),submitted=False)))
