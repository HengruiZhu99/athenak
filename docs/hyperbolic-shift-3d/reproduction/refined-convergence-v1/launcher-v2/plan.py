#!/usr/bin/env python3
"""Create a new immutable RF2 stationary SMR plan; never submit here."""
import argparse,json,pathlib
from control import read,write,digest,bindings,verify
from storage import capacity
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--segment',type=int,default=1);p.add_argument('--previous');a=p.parse_args();root=pathlib.Path(a.root).resolve()
science=read(root/'science.json');assert read(root/'build-result.json')['status']=='completed'
launcher=read(root/'launcher-v2/manifest.json')
for f,v in launcher.items():assert digest(root/'launcher-v2'/f)==v['sha256']
config='source/inputs/hyperbolic_shift_3d/refined_convergence/central6_stationary_matched_plateau.cfg'
identity=read(root/'launcher-v2/qualified-identity.json')['identity']
rel=f'campaigns/stationary6-n16-v2-segment{a.segment:02d}'
d=root/rel;d.mkdir(parents=True,exist_ok=False)
start_time=0.;start_steps=0;restart=None
if a.previous:
 previous=read(root/a.previous);terminal=read((root/a.previous).parent/'terminal-verification.json')
 if not terminal['admitted'] or terminal['checkpoint']['status']!='checkpointed':raise RuntimeError('previous not healthy walltime')
 if a.segment!=previous['segment']+1:raise RuntimeError('segment sequence')
 v=verify(root,previous,root/previous['output']);start_time=v['time'];start_steps=v['steps'];restart=previous['output']+'/state.checkpoint'
 if (root/'allocation-lock.json').exists():raise RuntimeError('previous lock not released')
else:assert a.segment==1
pbs=f'''#!/bin/bash -l
#PBS -N rf2-stat6-n16
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
exec /usr/bin/python3.11 launcher-v2/worker.py "$PWD" "$PWD/{rel}/plan.json"
'''
(d/'job.pbs').write_text(pbs)
files=['science.json','source-manifest.json',science['binary'],config,rel+'/job.pbs','launcher-v2/manifest.json']+['launcher-v2/'+f for f in launcher]
if restart:files.append(restart)
plan=dict(plan=rel+'/plan.json',segment=a.segment,source_commit=science['source_commit'],binary=science['binary'],identity=identity,n=16,kind='boost',boost=0,dt=.00025,target_time=10/3,diagnostic_interval=.05,nodes=4,ranks=48,queue='debug-scaling',config=config,restart=restart,start_time=start_time,start_steps=start_steps,output=rel+'/output',planned_output_bytes=2_000_000_000,atomic_reserve_bytes=800_000_000,bindings={f:digest(root/f) for f in files})
assert plan['bindings'][science['binary']]==science['binary_sha256'] and plan['bindings']['source-manifest.json']==science['source_manifest_sha256']
write(d/'binding-storage.json',capacity(root,planned=plan['planned_output_bytes'],reserve=plan['atomic_reserve_bytes']));write(d/'plan.json',plan);print(json.dumps(dict(plan=plan['plan'],sha256=digest(d/'plan.json'))))
