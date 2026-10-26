#!/usr/bin/env python3
"""Verify a collected read-only allocation before assessing its saved RHS."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';P=A/'cpu-memory-rhs-audit-v1';C=A/'cpu-memory-rhs-audit-v1-collected';scripts=Path(__file__).resolve().parent
while not (C/'collection-receipt.json').exists():time.sleep(15)
receipt=json.loads((C/'collection-receipt.json').read_text());manifest=json.loads((C/'remote-manifest.json').read_text());expected=json.loads((P/'portable-source-manifest.json').read_text());plan=json.loads((P/'cpu-plan.json').read_text());scheduler=manifest['scheduler']['Jobs'][receipt['job']];R=C/'result'
if not receipt['passed'] or manifest['source_mismatches'] or manifest['source_files']!=expected or scheduler.get('Exit_status')!=0:raise RuntimeError('Audit collection/source/scheduler integrity failed')
for name in ['audit_ringdown_rhs','benchmark_frozen_rhs']:
 build=json.loads((R/(name+'.build.json')).read_text());file='source/tst/test_scalar/'+name+'.cpp'
 if build['source_sha256']!=expected[file] or build['binary_sha256']!=sha(R/name):raise RuntimeError('Audit executable build/source binding failed')
for name,digest in plan['source_sha256'].items():
 if sha(Path('src/test_scalar')/name)!=digest:raise RuntimeError('Frozen scalar delivery source changed')
memory=json.loads((R/'memory-audit-receipt.json').read_text());records=memory['records']
if not memory['passed'] or memory['plan_sha256']!=sha(P/'cpu-plan.json') or len(records)!=2 or records[0]['rhs_sha256']!=records[1]['rhs_sha256'] or any(sha(R/('memory-'+r['label'])/'active-rhs.f64')!=r['rhs_sha256'] for r in records):raise RuntimeError('Full RHS memory-placement parity failed')
primary=E/'cpu-schwarzschild-commensurate-finer/primary-resolved-reference-review.json';proof=json.loads(primary.read_text());runs=proof['runs'];audits=[R/f'actual-rhs-n{n}' for n in proof['resolutions']];oracle=E/'radial-oracle/smooth/waveform-upwind-n2400.csv'
subprocess.run([sys.executable,str(scripts/'assess_ringdown_rhs.py'),'--runs',*runs,'--audits',*[str(d) for d in audits],'--oracle-waveform',str(oracle),'--reference-sampler','lagrange6','--source','src/test_scalar','--output',str(E/'cpu-schwarzschild-commensurate-finer/actual-resolved-rhs-review.json')],check=True)
review=E/'cpu-memory-rhs-audit-review.json'
if review.exists():raise RuntimeError('Existing read-only audit review is immutable')
review.write_text(json.dumps(dict(passed=True,admission=False,collection_sha256=sha(C/'collection-receipt.json'),plan_sha256=sha(P/'cpu-plan.json'),full_active_RHS_bitwise_identical=True,source_sha256=plan['source_sha256'],default_over_hbm_speedup=memory['default_over_hbm_speedup'],selected_placement='default',actual_rhs_review_sha256=sha(E/'cpu-schwarzschild-commensurate-finer/actual-resolved-rhs-review.json'),scope='Read-only full scalar operator parity and native-state actual time-derivative convergence. Primary late field convergence remains failed; this is not stage admission.'),indent=2)+'\n')
print(review,flush=True)
