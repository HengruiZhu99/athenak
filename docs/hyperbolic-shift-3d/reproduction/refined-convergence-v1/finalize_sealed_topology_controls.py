"""Read-only completion check after the recorded coordinator field-name typo."""
from pathlib import Path
import json,hashlib,subprocess,importlib.util,shutil
import numpy as np
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
E=D/'sealed-topology-controls';B=R/'output/refined-convergence/sealed-topology-candidate-v1';O=B/'physics'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
commands=json.loads((E/'commands.json').read_text());assert len(commands)==4 and all(x['returncode']==0 and x['resource_abort'] is None for x in commands)
assert all(json.loads((E/f'topology-mpi{i}.json').read_text())['passed'] for i in [1,2])
assert sha(O/'uniform-parity.checkpoint')==sha(R/'output/refined-convergence/boosted-radial24-control/state.checkpoint')
spec=importlib.util.spec_from_file_location('reader',D/'check_smr_restart.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
x,xt,xa,xf=reader.read(str(R/'output/refined-convergence/smr-tube-candidate-v1/checks/boost-mpi1-restart-step4'));y,yt,ya,yf=reader.read(str(O/'smr-restart-parity'))
assert x['identity']==y['identity'] and x['config']==y['config'] and xt==yt and xa==ya and x['time']==y['time'] and x['dt']==y['dt']
assert all(np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(a,b) for a,b in zip(xf,yf));del xf,yf
prefix=O/'full-smr-radial24-angular2';report=json.loads(Path(str(prefix)+'.report.json').read_text())
assert report['checkpoint_admitted'] and report['status']=='completed' and report['steps']==2 and report['dt']==1/24000 and report['leaves']==400
v=json.loads(subprocess.check_output([str(R/'output/refined-convergence/verify-checkpoint'),str(prefix)+'.checkpoint'],text=True));assert v['identity']==report['identity'] and v['n']==12 and v['leaves']==400
uniform_identity=json.loads((E/'uniform-parity-verification.json').read_text())['identity']
box=x['identity'].split(';smr-box-origin-v1=')[1].split(';')[0]
expected=uniform_identity.replace(';smr=0,',';smr=3,').replace(';shell-angular-subdivisions-v1=',';smr-box-origin-v1='+box+';shell-angular-subdivisions-v1=').replace(';amr_policy=uniform',';amr_policy=smr')
assert v['identity']==expected and v['kind']==2 and v['subdivisions']==4 and v['steps']==2 and v['regrids']==0 and v['dt']==1/24000 and v['time']==2/24000
build=json.loads((E/'build.json').read_text())
assert sha(B/'evolve')==next(z['binary_sha256'] for z in build['targets'] if z['name']=='evolve')
assert all(sha(R/path)==digest for path,digest in build['headers'].items())
assert sha(R/'build-refined-convergence/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_refined_amr')==build['production_binary_sha256_unchanged']
v.update(sha256=sha(str(prefix)+'.checkpoint'),bytes=Path(str(prefix)+'.checkpoint').stat().st_size)
rows=[json.loads(x) for x in Path(str(prefix)+'.history.jsonl').read_text().splitlines()]
assert len(rows)==3 and rows[-1]['step']==2 and abs(rows[-1]['t_over_M']-.00025)<1e-15
assert all(x['finest_level']==3 and x['puncture_finest_margin']>0 and x['candidate_finest_margin']>0 and x['horizon_complete'] and x['horizon_found'] for x in rows)
assert max(x['cfl'] for x in rows)<=.2+1e-12
assert max(max(x['null'],x['ghost_null']) for x in rows)<1e-12
for name in ['uniform-parity','smr-restart-parity','full-smr-radial24-angular2']:
 generations=[json.loads(x) for x in (O/(name+'.rank0.generations.jsonl')).read_text().splitlines()]
 assert all(x['memory_bytes']['native_state_compensation']==0 for x in generations)
config=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_radial24_angular2_reference.cfg'
result=dict(passed=True,uniform_checkpoint_byte_equal=True,smr_restart_all8_arrays_byte_equal=True,full_smr_checkpoint=v,full_smr_initial=rows[0],full_smr_final=rows[-1],candidate_binary_sha256=sha(B/'evolve'),config_sha256=sha(config),qualification='Full400-leaf boosted SMR plus radial24/angular2 combined local two-step check. No GPU, long-time, dynamic AMR or convergence claim. Memory retirement changes no numerical identity. Native output untouched; this completion check fixes a coordinator-only history-key typo.')
(E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('verified full combined SMR, checkpoint and parity',flush=True)
