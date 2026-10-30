#!/usr/bin/env python3
"""Read-only, common-domain comparison of the four admitted 3M checkpoints."""
from pathlib import Path
import json,hashlib,subprocess,time,math
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
RAW=ROOT/'output/hyperbolic-core-shell/smooth-gauge-v1/aurora'
receipt=json.loads((RAW/'local-verification.json').read_text());assert receipt['verified']
oldroot=HERE.parent/'resolution-20m-v1'
reports={('old',16):json.loads((oldroot/'candidate-history/n16.report.json').read_text()),('old',24):json.loads((oldroot/'candidate-history/n24.report.json').read_text())}
for n in [16,24]:reports['new',n]=json.loads((RAW/f'output/n{n}.report.json').read_text())
for (variant,n),r in reports.items():
 assert r['status']=='completed' and r['t_over_M']==3 and r['checkpoint_admitted']
 if variant=='new':assert r['gauge_blend']=='smoothstep9_w_056_080' and r['checkpoint_schema']=='HYP3D_MPI_GW1'
radii=[r['horizon_trial_radius'] for r in reports.values()];assert all(math.isfinite(r) and 0<r<.16 for r in radii)
cut=max(.0753,math.ceil(max(radii)*10000)/10000+.0001)
bands=sorted(set([0,.03,.06,cut,.1,.13,.16,.18,.22,.3,.4,.6,.8,.9,.95,.98,1]))
paths={('old',16):ROOT/'output/hyperbolic-core-shell/n16-timestep-to3m-v1/n16-original-dt-reproduction-v1/output/puncture.checkpoint',('old',24):ROOT/'output/hyperbolic-core-shell/resolution-20m-v1/n24-dt1-segment01/output/puncture.checkpoint'}
for n in [16,24]:paths['new',n]=RAW/f'output/n{n}.checkpoint'
out=HERE/'radial';out.mkdir(exist_ok=True);records=[]
for key,p in paths.items():
 variant,n=key;exe=ROOT/'build-hyperbolic-mpi/tst/hyperbolic_shift_3d'/('hyperbolic_shift_3d_checkpoint_constraints_smooth' if variant=='new' else 'hyperbolic_shift_3d_checkpoint_constraints')
 before=hashlib.sha256(p.read_bytes()).hexdigest();command=[str(exe),str(n),str(p),str(.0005*16/n),str(cut),str(out/f'{variant}-n{n}.json'),','.join(map(str,bands))];start=time.monotonic()
 with (out/f'{variant}-n{n}.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0 and hashlib.sha256(p.read_bytes()).hexdigest()==before
 records.append(dict(variant=variant,n=n,command=command,seconds=time.monotonic()-start,checkpoint_sha256=before,checkpoint_unchanged=True,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()))
 (out/'execution.json').write_text(json.dumps(dict(common_cutoff=cut,horizon_candidates=radii,runs=records),indent=2)+'\n')
 print(variant,n,'complete',flush=True)
