"""Preserve and reconstruct one fully admitted short radial/time control."""
from pathlib import Path
import argparse,json,hashlib,subprocess,shutil,gzip,time,math
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['time','radial'],required=True);p.add_argument('--segment',type=int,default=1);a=p.parse_args()
rel=f'campaigns/stationary20-reference-{a.variant}-dt4-n16-segment{a.segment:02d}'
raw=R/'output/refined-convergence/stationary-reference-v2-aurora'/rel;O=D/f'{a.variant}-control-segment{a.segment:02d}';O.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
term=json.loads((raw/'terminal-verification.json').read_text());release=json.loads((raw/'allocation-released.json').read_text());plan=json.loads((raw/'plan.json').read_text());worker=json.loads((raw/'output/worker.json').read_text())
assert term['admitted'] and term['checkpoint']['status']=='completed'
assert plan['variant']==a.variant and plan['dt']==.00003125 and plan['target_time']==.2
assert term['checkpoint']['plan_sha256']==sha(raw/'plan.json') and release['terminal_verification_sha256']==sha(raw/'terminal-verification.json')
assert term['checkpoint']['identity']==plan['identity'] and abs(term['checkpoint']['time']-.2)<1e-12
for name,v in worker['artifacts'].items():
    f=raw/'output'/name;assert f.stat().st_size==v['bytes'] and sha(f)==v['sha256'],name
cp=raw/'output/state.checkpoint';assert sha(cp)==term['checkpoint']['sha256']
verification=dict(all_worker_artifacts_verified=True,artifacts=worker['artifacts'],checkpoint_sha256=sha(cp),terminal_verification_sha256=sha(raw/'terminal-verification.json'),matching_lock_release=True)
(O/'local-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
for f in raw.iterdir():
    if f.is_file():shutil.copy2(f,O/f.name)
(O/'output').mkdir(exist_ok=True)
for f in (raw/'output').iterdir():
    if f.suffix not in ('.json','.jsonl','.log'):continue
    dest=O/'output'/f.name
    if f.name.endswith('.samples.jsonl'):
        with f.open('rb') as src,gzip.open(str(dest)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
    else:shutil.copy2(f,dest)
manifest=D/'stationary-reference-v2-build/source-manifest.json';m=json.loads(manifest.read_text());count=0
for name,v in m['files'].items():
    if name.startswith(('src/','tst/hyperbolic_shift_3d/')):assert sha(R/name)==v['sha256'],name;count+=1
which='reference' if a.variant=='time' else 'radial';reader=R/f'output/refined-convergence/{which}-endpoint-reader/reader'
prefix=reader.parent/f'{a.variant}-control-segment{a.segment:02d}';assert not prefix.with_suffix('.json').exists()
cfg=R/f'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial{12 if a.variant=="time" else 18}_reference.cfg'
assert sha(cfg)==plan['bindings'][plan['config']]
cmd=['mpiexec','-n','2',str(reader),'16',str(cp),'.00003125',str(cfg),'smr',str(prefix),'.3']
record=dict(command=cmd,source_manifest_sha256=sha(manifest),matching_production_files=count,checkpoint_sha256=sha(cp),analysis_binary_sha256=sha(reader),started=time.time(),qualification='Exact admitted short endpoint and unchanged physical configuration; dedicated explicit known Aurora k/gamma identity calibration. Compare native report norms before interpreting spatial diagnostics; no campaign binary or input checkpoint change.')
(O/'analysis-command.json').write_text(json.dumps(record,indent=2)+'\n')
with (O/'analysis.log').open('w') as f:run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
record.update(returncode=run.returncode,elapsed=time.time()-record['started'],checkpoint_unchanged=sha(cp)==record['checkpoint_sha256']);assert run.returncode==0 and record['checkpoint_unchanged']
x=json.loads(prefix.with_suffix('.json').read_text());report=json.loads((raw/'output/state.report.json').read_text());history=[json.loads(s) for s in (raw/'output/state.history.jsonl').read_text().splitlines()]
norms=[[math.sqrt(v/b[0]) for v in b[1:]] for b in x['integrals']];expected=[[report['H'],report['M']],[history[-1]['fixed'][1]['H'],history[-1]['fixed'][1]['M']]]
errors=[[abs(norms[k][f]-expected[k][f])/(1+abs(expected[k][f])) for f in range(2)] for k in range(2)];assert max(v for row in errors for v in row)<1e-9
record.update(native_report_norms_match=errors,full_norms=norms[0],fixed03_norms=norms[1]);(O/'analysis-command.json').write_text(json.dumps(record,indent=2)+'\n');(O/'spatial-native.json').write_text(json.dumps(x,indent=2)+'\n')
for f in prefix.parent.glob(prefix.name+'.rank*.csv'):(O/(f.name+'.gz')).write_bytes(gzip.compress(f.read_bytes(),mtime=0))
print(json.dumps(record,indent=2))
