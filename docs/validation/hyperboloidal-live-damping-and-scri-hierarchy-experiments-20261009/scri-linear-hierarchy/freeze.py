from pathlib import Path
import json,hashlib,shutil
p=Path(__file__).resolve().parent;repo=p.parents[2];dest=p/'immutable-linear-scri-hierarchy-20261009';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert not dest.exists();r=json.loads((p/'receipt.json').read_text());assert r['passed_actual_kernel_linear_hierarchy_gate'] and not r['PDE_finite_Q_amplitude_blowup_claim']
assert all(sha(repo/k)==v for k,v in r['sources'].items());assert all(x['returncode']==0 and not x['stderr'] for x in r['commands'])
dest.mkdir();names=['taylor_kernel.cpp','firstjet_gate.cpp','run_audit.py','check_hierarchy.py','finalize.py','freeze.py','DERIVATION.md','REPORT.md','receipt.json','compile-receipt.json','kernel.json','firstjet.json','firstjet-debug.json','check-report.json','check.log','check.stderr','check-final.log','finalize.log','finalize.stderr','first-pass-two-families','second-pass-extended-seed','third-pass-general-corner','failed-json-serialization','failed-Theta-corner-without-null-firstjet','failed-mixed-auto-declaration']
for name in names:
 f=p/name
 if f.is_dir():shutil.copytree(f,dest/name)
 else:shutil.copy2(f,dest/name)
index={'scope':'Actual linear outer CMC reference R0 firstjet map, necessary hierarchy and frozen-normal amplitude distinction. No closed scri PDE, native evolution or finite-Q amplitude blowup claim.','files':{str(f.relative_to(dest)):sha(f) for f in sorted(dest.rglob('*')) if f.is_file()},'source_count':r['source_count'],'commands':len(r['commands']),'receipt_sha256':sha(dest/'receipt.json'),'binary_records':{str((p/name).relative_to(repo)):{'sha256':sha(p/name),'bytes':(p/name).stat().st_size} for name in r['binary_sha256']},'all_frozen_hashes_verified':True}
(dest/'index.json').write_text(json.dumps(index,indent=2)+'\n');assert all(sha(dest/k)==v for k,v in index['files'].items());print('FROZEN',len(index['files']),'files',sha(dest/'index.json'));print('receipt',sha(dest/'receipt.json'))
