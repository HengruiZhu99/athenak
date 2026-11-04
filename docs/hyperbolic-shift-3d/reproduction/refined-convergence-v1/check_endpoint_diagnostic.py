"""Check the separate read-only diagnostic against preserved production evidence."""
from pathlib import Path
import hashlib,json,math,shutil
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
RAW=ROOT/'output/refined-convergence/endpoint-diagnostic'
DEST=HERE/'endpoint-reader-check';DEST.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((HERE/'production-controls-verification.json').read_text())
cp=next(r['checkpoint'] for r in old['records'] if r['name']=='production-cadence')
p=ROOT/cp['path'];assert p.stat().st_size==cp['bytes'] and sha(p)==cp['sha256']
q=json.loads((RAW/'check-current-boundary-n12.json').read_text())
assert q['identity']==cp['identity'] and q['n']==12 and q['steps']==2
assert q['native_volume_integral_relative_difference']<1e-12
history=[json.loads(l) for l in (ROOT/'output/refined-convergence/production-cadence.history.jsonl').read_text().splitlines()]
h=history[-1];assert q['t_over_M']==h['t_over_M']
norms=[[math.sqrt(v/a[0]) for v in a[1:]] for a in q['integrals']]
expected=[[h[k] for k in ['H','M','Mr']],next([r[k] for k in ['H','M','Mr']] for r in h['fixed'] if r['cut']==.3)]
error=max(abs(a-b)/max(abs(b),1e-30) for row,ref in zip(norms,expected) for a,b in zip(row,ref))
assert error<1e-10
log=(RAW/'reject-wrong-config.log').read_text()
assert 'checkpoint formulation identity mismatch' in log
assert not (RAW/'reject-wrong-config.json').exists()
files=['build.json','check-current-boundary-n12.json','check-current-boundary-n12.log',
       'check-current-boundary-n12.rank0.csv','check-current-boundary-n12.rank1.csv','reject-wrong-config.log']
artifacts=[]
for f in files:
 src=RAW/f;shutil.copyfile(src,DEST/f);artifacts.append(dict(file=f,sha256=sha(src),bytes=src.stat().st_size))
result=dict(passed=True,checkpoint=cp,checkpoint_unchanged=True,norms=norms,history_norms=expected,
            maximum_relative_history_norm_difference=error,wrong_configuration_rejected=True,artifacts=artifacts,
            qualification='Local N12 stationary two-step admitted fixture; analysis-only CPU constraint integrals agree with production device kernel and saved history. No long-run, resolution convergence or independent PDE accuracy claim.')
(DEST/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['passed','norms','maximum_relative_history_norm_difference','wrong_configuration_rejected']},indent=2))
