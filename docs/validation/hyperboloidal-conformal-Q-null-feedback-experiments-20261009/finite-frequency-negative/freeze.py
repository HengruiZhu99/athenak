from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'receipt.json').read_text());assert r['sources_unchanged'] and all(x['returncode']==0 and not(P/x['stderr']).read_text()for x in r['commands'])
for name,digest in r['source_before'].items():assert sha(ROOT/name)==digest,name
F=P/'immutable-Q-null-finite-frequency-negative-20261009';assert not F.exists();paths=[x for x in P.rglob('*')if x.is_file()and x.suffix in ['.cpp','.hpp','.py','.json','.stdout','.stderr','.md']];F.mkdir();files={}
for p in sorted(paths):name=p.relative_to(P);q=F/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);files[str(name)]=sha(q)
index={'scope':'Negative local finite-Omega reference primitive full20 Fourier screen; no global/native/subsidiary classification or candidate acceptance. Core gate untouched.','files':files,'input_count':len(r['source_before']),'command_count':len(r['commands']),'matrix_count':1120,'production_files_unchanged':365,'core_index_sha256':'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96','large_outputs_outside_snapshot':{'fourier':{'sha256':sha(P/'fourier'),'bytes':(P/'fourier').stat().st_size,'original_repo_path':str((P/'fourier').relative_to(ROOT))}},'receipt_sha256':sha(F/'receipt.json'),'file_count':len(files),'bytes':sum((F/n).stat().st_size for n in files)}
(F/'index.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps({'path':str(F.relative_to(ROOT)),'index_sha256':sha(F/'index.json'),'receipt_sha256':index['receipt_sha256'],'files':len(files),'bytes':index['bytes'],'inputs':index['input_count'],'commands':index['command_count']},indent=2))
