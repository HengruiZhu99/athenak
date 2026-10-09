from pathlib import Path
import hashlib,json,shutil,subprocess
P=Path(__file__).resolve().parent;F=P/'immutable-Einstein-coordinate-local-attempts-20261009';assert not F.exists();F.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[];large=[]
for p in sorted(P.rglob('*')):
 if not p.is_file()or F in p.parents or '__pycache__'in p.parts:continue
 relative=str(p.relative_to(P));record={'path':relative,'sha256':sha(p),'bytes':p.stat().st_size,'original_path':str(p)}
 if p.name=='probe'or p.stat().st_size>1048576:record['metadata_only_reason']='as-built executable retained locally'if p.name=='probe'else'large scientific stdout retained locally';large.append(record);continue
 dest=F/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha(dest)==record['sha256'];files.append(record)
json_count=0
for record in files:
 if record['path'].endswith('.json'):
  obj=json.loads((F/record['path']).read_text());json.dumps(obj,allow_nan=False);json_count+=1
index={'kind':'frozen complete-reference plus failed broad-coordinate local attempts','launch_HEAD_at_capture':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'runtime_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2','files':files,'large_payloads_metadata_only':large,'small_file_count':len(files),'small_bytes':sum(x['bytes']for x in files),'finite_json_count':json_count,'all_copied_sha256_reverified':True,'complete_cartesian_coordinate_gate_passed':False,'operator_or_spectrum_or_evolution_admitted':False,'report_sha256':sha(F/'REPORT.md'),'summary_sha256':sha(F/'summary.json')}
(F/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(F),'index_sha256':sha(F/'index.json'),'files':len(files),'bytes':index['small_bytes'],'finite_json':json_count,'large_records':len(large)},indent=2))
