"""Fresh boosted full-SMR runs and verified same-case walltime continuations.

Prepare only after reconciling the older stationary campaign. Does not submit.
The actual GPU binary/science must already have been successfully built/frozen.
"""
from pathlib import Path
import argparse,json,math,sys
LAUNCHER='launcher-boosted-transition-v1'
a=argparse.ArgumentParser();a.add_argument('root');a.add_argument('--n',type=int,choices=[12,14,16],required=True)
a.add_argument('--segment',type=int,default=1);a.add_argument('--previous');args=a.parse_args()
root=Path(args.root).resolve();sys.path.insert(0,str(root/LAUNCHER))
from control import read,write,digest,verify
from storage import capacity
assert not (root/'allocation-lock.json').exists(),'Reconcile existing allocation first'
science=read(root/'science.json');assert read(root/'build-result.json')['status']=='completed'
assert digest(root/science['binary'])==science['binary_sha256']
assert digest(root/'source-manifest.json')==science['source_manifest_sha256']
launcher=read(root/LAUNCHER/'manifest.json')
for f,v in launcher.items():assert digest(root/LAUNCHER/f)==v['sha256']
qualification=read(root/LAUNCHER/'qualified-identity.json')
identity=qualification['identity'];config='candidate.cfg';dt=1/(2000*args.n)
assert digest(root/config)==qualification['config_sha256']
assert science['source_commit']==qualification['source_commit']
assert (root/LAUNCHER/'verify-checkpoint').stat().st_mode&0o111
restart=None;start_time=0.;start_steps=0
if args.previous:
 assert args.segment>1
 parent=(root/args.previous).resolve();assert parent.is_relative_to(root)
 old=read(parent);terminal=read(parent.parent/'terminal-verification.json');release=read(parent.parent/'allocation-released.json')
 assert terminal['admitted'] and terminal['checkpoint']['status']=='checkpointed'
 assert release['terminal_verification_sha256']==digest(parent.parent/'terminal-verification.json')
 assert release['plan_sha256']==digest(parent) and release['job_id']==terminal['job_id']
 assert args.segment==old['segment']+1 and old['variant']=='boosted_transition'
 assert old['n']==args.n and old['dt']==dt and old['identity']==identity and old['config']==config
 assert old['source_commit']==science['source_commit'] and old['target_time']==10/3 and old['boost']==.1
 v=verify(root,old,root/old['output']);assert v['time']<10/3
 restart=old['output']+'/state.checkpoint';start_time=v['time'];start_steps=v['steps']
else:assert args.segment==1
# Positive-volume/surface/overlap terms scale no faster than N^3 for this fixed
# topology. Add64MB rather than treating this estimate as an exact byte schema.
checkpoint_budget=math.ceil(qualification['checkpoint_bytes_n12']*(args.n/12)**3)+64_000_000
planned=2*checkpoint_budget+256_000_000;atomic=checkpoint_budget+64_000_000
storage=capacity(root,planned=planned,reserve=atomic)
rel=f'campaigns/boosted-transition-n{args.n}-segment{args.segment:02d}'
d=root/rel;d.mkdir(parents=True,exist_ok=False)
pbs=f'''#!/bin/bash -l
#PBS -N rf2-boost-n{args.n}
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
exec /usr/bin/python3.11 {LAUNCHER}/worker.py "$PWD" "$PWD/{rel}/plan.json"
'''
(d/'job.pbs').write_text(pbs)
files=['science.json','source-manifest.json',science['binary'],config,rel+'/job.pbs',LAUNCHER+'/manifest.json']+[LAUNCHER+'/'+f for f in launcher]
if restart:
 files.extend([restart,str(parent.relative_to(root)),str((parent.parent/'terminal-verification.json').relative_to(root)),str((parent.parent/'allocation-released.json').relative_to(root))])
plan=dict(plan=rel+'/plan.json',variant='boosted_transition',segment=args.segment,source_commit=science['source_commit'],binary=science['binary'],identity=identity,n=args.n,kind='boost',boost=.1,dt=dt,target_time=10/3,diagnostic_interval=.025,nodes=4,ranks=48,queue='debug-scaling',config=config,restart=restart,start_time=start_time,start_steps=start_steps,output=rel+'/output',planned_output_bytes=planned,atomic_reserve_bytes=atomic,parent_plan=args.previous,bindings={f:digest(root/f) for f in files},qualification='Exact global v=.1, full400-leaf SMR tube, radial24/angular2. Same configured resolution family with dt=1/(2000*N). Checkpoint embeds finder seed, tracker and topology. No GPU smoke gate; no long-time/convergence claim before measured results.')
assert plan['bindings'][science['binary']]==science['binary_sha256']
write(d/'binding-storage.json',storage);write(d/'plan.json',plan)
print(json.dumps(dict(plan=plan['plan'],sha256=digest(d/'plan.json'),submitted=False,checkpoint_budget=checkpoint_budget,storage=storage)))
