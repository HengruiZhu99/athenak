from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent;R=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'receipt.json').read_text());assert r['sources_unchanged'];assert all(sha(R/f)==v for f,v in r['source_before'].items());assert all(c['returncode']==0 and not(P/c['stderr']).read_bytes()for c in r['commands']);assert (P/'actual.json').read_bytes()==(P/'actual-debug.json').read_bytes();assert json.loads((P/'check-report.json').read_text())['passed']
O=P/'immutable-Q-null-firstjet-map-20261009';assert not O.exists();O.mkdir();files={}
for f in sorted(P.rglob('*')):
 if not f.is_file()or O in f.parents or f.name in ['jet-map','jet-map-debug']:continue
 rel=f.relative_to(P);q=O/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,q);files[str(rel)]=sha(q)
idx={'scope':'Actual linear analytic-reference Q/null first-jet singular and leading-constraint maps; necessary compatibility only, no exact Einstein/nonlinear closure or evolution acceptance.','files':files,'source_input_count':len(r['source_before']),'command_count':len(r['commands']),'actual_basis_columns':320,'nonlinear_FD_directions':960,'production_inputs_unchanged':365,'receipt_sha256':sha(P/'receipt.json'),'core_index_sha256':'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96','outside_binaries':{f:{'sha256':sha(P/f),'bytes':(P/f).stat().st_size}for f in ['jet-map','jet-map-debug']},'file_count':len(files),'bytes':sum((O/f).stat().st_size for f in files)}
(O/'index.json').write_text(json.dumps(idx,indent=2)+'\n');assert all(sha(O/f)==v for f,v in files.items());print('index',sha(O/'index.json'),'receipt',idx['receipt_sha256'],'files',len(files),'bytes',idx['bytes'])
