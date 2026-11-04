"""Prepare a stationary radial18 resolution run using the frozen RF2 worker.

Fresh N12/N14 states; N16 may explicitly extend its admitted .6M control.
Automatic continuations require a fully admitted same-case walltime state.
No equation, compiler, binary, floor, or boundary-penalty change.
"""
from pathlib import Path
import argparse
import json
import sys

a=argparse.ArgumentParser()
a.add_argument('root');a.add_argument('--n',type=int,choices=[12,14,16],required=True)
a.add_argument('--segment',type=int,default=1);a.add_argument('--previous')
a.add_argument('--from-short',action='store_true')
args=a.parse_args();root=Path(args.root).resolve()
sys.path.insert(0,str(root/'launcher-v7'))
from control import read,write,digest,verify
from storage import capacity

assert not (root/'allocation-lock.json').exists(), 'Existing allocation: reconcile first'
science=read(root/'science.json')
assert read(root/'build-result.json')['status']=='completed'
assert digest(root/science['binary'])==science['binary_sha256']
assert digest(root/'source-manifest.json')==science['source_manifest_sha256']
launcher=read(root/'launcher-v7/manifest.json')
for f,v in launcher.items():assert digest(root/'launcher-v7'/f)==v['sha256']
identity=read(root/'launcher-v7/qualified-identity-radial.json')['identity']
config='radial-controls-v1/central20_stationary_smr025_radial18_reference.cfg'
dt=1/(2000*args.n);restart=None;start_time=0.;start_steps=0;parent=None
if args.from_short:
    assert args.n==16 and args.segment==1 and not args.previous
    parent='campaigns/stationary20-reference-radial-dt4-n16-segment01/plan.json'
elif args.previous:
    assert args.segment>1
    parent=args.previous
else:
    assert args.segment==1
    # N16 already has a valid fresh-to-.6M trajectory; do not repeat it.
    assert args.n!=16, 'Use explicit --from-short for N16'
if parent:
    old=read(root/parent);directory=(root/parent).parent
    terminal=read(directory/'terminal-verification.json')
    release=read(directory/'allocation-released.json')
    assert terminal['admitted']
    assert release['terminal_verification_sha256']==digest(directory/'terminal-verification.json')
    assert old['n']==args.n and old['dt']==dt and old['identity']==identity
    assert old['config']==config and old['source_commit']==science['source_commit']
    if args.from_short:
        assert old['variant']=='radial' and terminal['checkpoint']['status']=='completed'
        assert abs(terminal['checkpoint']['time']-.2)<1e-12
    else:
        assert old['variant']=='radial_resolution' and args.segment==old['segment']+1
        assert old['target_time']==10/3 and terminal['checkpoint']['status']=='checkpointed'
    v=verify(root,old,root/old['output'])
    assert v['time']<10/3
    restart=old['output']+'/state.checkpoint';start_time=v['time'];start_steps=v['steps']
rel=f'campaigns/stationary20-radial18-resolution-n{args.n}-segment{args.segment:02d}'
directory=root/rel;directory.mkdir(exist_ok=False)
template=(root/'campaigns/stationary20-reference-radial-dt4-n16-segment01/job.pbs').read_text()
template=template.replace('#PBS -N rf2-dt4-radial',f'#PBS -N rf2-r18-n{args.n}')
template=template.replace('campaigns/stationary20-reference-radial-dt4-n16-segment01',rel)
(directory/'job.pbs').write_text(template)
files=['science.json','source-manifest.json',science['binary'],config,rel+'/job.pbs',
       'launcher-v7/manifest.json','radial-controls-v1/provenance.json',
       str(Path(__file__).resolve().relative_to(root))]+['launcher-v7/'+f for f in launcher]
if restart:
    files.extend([restart,parent,str(((root/parent).parent/'terminal-verification.json').relative_to(root)),
                  str(((root/parent).parent/'allocation-released.json').relative_to(root))])
plan=dict(plan=rel+'/plan.json',variant='radial_resolution',segment=args.segment,
          source_commit=science['source_commit'],binary=science['binary'],identity=identity,
          n=args.n,kind='boost',boost=0,dt=dt,target_time=10/3,diagnostic_interval=.025,
          nodes=4,ranks=48,queue='debug-scaling',config=config,restart=restart,
          start_time=start_time,start_steps=start_steps,output=rel+'/output',
          planned_output_bytes=2_000_000_000,atomic_reserve_bytes=1_000_000_000,
          parent_plan=parent,from_completed_short_control=args.from_short,
          bindings={f:digest(root/f) for f in files},
          qualification='Stationary radial18 global-N resolution family with dt=1/(2000*N); same fixed SMR geometry, initial data, gauge, angles-per-N and boundary penalty. Space/time refined together; no convergence proof from survival.')
write(directory/'binding-storage.json',capacity(root,planned=plan['planned_output_bytes'],reserve=plan['atomic_reserve_bytes']))
write(directory/'plan.json',plan)
print(json.dumps(dict(plan=plan['plan'],sha256=digest(directory/'plan.json'),submitted=False)))
