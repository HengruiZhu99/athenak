from pathlib import Path
import json,hashlib,subprocess,time
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
cases={
 'DO1':('direct-outer-v1','n16-direct-outer-fresh10m-v1-segment01','direct_outer','edc58b0df7ef0391567221d8b95990e82ab05aad6264bc87d7e4a49ff2449c5c'),
 'IT1':('inner-telegrapher-v1','n16-inner-telegrapher-fresh10m-v1-segment01','inner_telegrapher','37665a7936afc1e404c653a1ea8fc1051a437a31f3bc907dc5cadec76c483d9f')}
bands=[0,.03,.06,.1,.13,.16,.18,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.675,.7,.725,.75,.775,.8,.85,.9,.95,.98,1]
records=[]
for label,(folder,campaign,target,sha) in cases.items():
 cp=ROOT/'output/hyperbolic-core-shell'/folder/'aurora/campaigns'/campaign/'output/puncture.checkpoint'
 assert hashlib.sha256(cp.read_bytes()).hexdigest()==sha
 exe=ROOT/'build-hyperbolic-mpi/tst/hyperbolic_shift_3d'/('hyperbolic_shift_3d_checkpoint_constraints_'+target)
 command=[str(exe),'16',str(cp),'.0005','.1',str(HERE/f'{label}-10m-radial.json'),','.join(map(str,bands))]
 start=time.monotonic()
 with (HERE/f'{label}-10m-radial.log').open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0 and hashlib.sha256(cp.read_bytes()).hexdigest()==sha
 records.append(dict(label=label,command=command,checkpoint_sha256=sha,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),seconds=time.monotonic()-start))
 (HERE/'execution.json').write_text(json.dumps(records,indent=2)+'\n');print(label,'done',flush=True)
