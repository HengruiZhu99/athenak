"""Outer command/log receipt for root-authorized pinned saved-data figure."""
from pathlib import Path
import hashlib,json,os,subprocess,time
P=Path(__file__).resolve().parent;R=P.parents[2]
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
D=P/'invocation001';D.mkdir(exist_ok=False);s=pin(P/'plot_saved.py');i=pin(P/'source-index.json');assert s['sha256']=='66f23acd462f949d36f08a24f772ae7342b709e560922e48b3e0c04cbc0333ec';assert i['sha256']=='e3e2cb6aa1e5a82c193250f2afbaacfdceea0b5ecb6d3345de85fe78091da73c'
q=json.loads((P/'recipe.json').read_text());envs=q['environment'];env=os.environ.copy();env.update(envs)
command=[q['python'],'-B',str(P/'plot_saved.py')];record={'command':command,'cwd':str(R),'environment_overrides':envs,'source_before':s,'index_before':i,'runner_before':pin(__file__),'returncode':None,'root_scope':'SavedJSON scientific figure after pins; no arrays/native/probe calls.'};dump(D/'receipt.json',record)
started=time.monotonic()
with (D/'stdout').open('wb') as so,(D/'stderr').open('wb') as se:r=subprocess.run(command,cwd=R,env=env,stdout=so,stderr=se)
record.update(returncode=r.returncode,seconds=time.monotonic()-started,stdout=pin(D/'stdout'),stderr=pin(D/'stderr'),source_unchanged=pin(P/'plot_saved.py')==s,index_unchanged=pin(P/'source-index.json')==i);dump(D/'receipt.json',record);print(json.dumps(record,indent=2));assert r.returncode==0
