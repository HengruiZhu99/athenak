from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent;R=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'receipt.json').read_text());assert r['sources_unchanged'];assert all(sha(R/f)==v for f,v in r['source_before'].items());assert all(c['returncode']==0 and not(P/c['stderr']).read_bytes()for c in r['commands']);assert (P/'actual-release.json').read_bytes()==(P/'actual-debug.json').read_bytes();c=json.loads((P/'check-report.json').read_text());assert c['passed_initial_Einstein_and_finite_conformal_regularity_checks']and not c['sigma5_quadratic_null_ideal_invariant']and not c['sigma3_is_a_candidate_admission']
O=P/'immutable-Q-sigma5-Einstein-null-noninvariance-20261009';assert not O.exists();O.mkdir();files={}
for f in sorted(P.rglob('*')):
 if not f.is_file()or O in f.parents or f.name in ['invariant-release','invariant-debug']:continue
 rel=f.relative_to(P);q=O/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,q);files[str(rel)]=sha(q)
idx={'scope':'Actual linear Einstein-compatible initial gauge counterexample to sigma5 quadratic-null time-jet invariance. Initial4D conformal regularity finite; no amplitude-blowup/evolution/sigma3-admission claim.','files':files,'source_input_count':len(r['source_before']),'command_count':len(r['commands']),'actual_rows':480,'receipt_sha256':sha(P/'receipt.json'),'firstjet_index_sha256':'5490057dfc04e060bca65ec7ee1a3bb363a34cab0fe5c890f23d477a7bda7ec1','outside_binaries':{f:{'sha256':sha(P/f),'bytes':(P/f).stat().st_size}for f in ['invariant-release','invariant-debug']},'file_count':len(files),'bytes':sum((O/f).stat().st_size for f in files)}
(O/'index.json').write_text(json.dumps(idx,indent=2)+'\n');assert all(sha(O/f)==v for f,v in files.items());print('index',sha(O/'index.json'),'receipt',idx['receipt_sha256'],'files',len(files),'bytes',idx['bytes'])
