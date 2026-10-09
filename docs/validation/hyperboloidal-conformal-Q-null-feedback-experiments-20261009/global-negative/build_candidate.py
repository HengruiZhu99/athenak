"""Build exact admitted Q/null full22 and native20 sources with dependencies."""
from pathlib import Path
import hashlib,json,shlex,subprocess,time
w=Path(__file__).resolve().parent;root=w.parents[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
auth=json.loads((w/'gate-authorization.json').read_text());assert auth['scientific_gate_authorized_before_compile']
for key in ['gate','independent']:
 p=Path(auth[key+'_index_path']);assert sha(p)==auth[key+'_index_sha256']
 for name,row in json.loads(p.read_text())['files'].items():assert sha(p.parent/name)==(row if isinstance(row,str) else row['sha256'])
r={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','compiler_version':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'variant':'conformal-Q physical-inner lapse blend plus preferred-Q shift and sigma5 null feedback, explicit false/true/xi2; C0 kappa2zero','authorization':auth,'builds':{}}
for label,folder in [('full22',w/'full22-candidate'),('native20',w)]:
 cmd=json.loads((folder/'build-spatialnorm.json').read_text());start=time.monotonic();proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(folder/'build-spatialnorm.log').write_text(proc.stdout);assert proc.returncode==0,proc.stdout
 dc=[];skip=False
 for x in cmd:
  if skip:skip=False;continue
  if x=='-o':skip=True;continue
  if x.endswith('.a'):continue
  dc.append(x)
 dc+=['-M','-MT','candidate'];out=subprocess.check_output(dc,text=True);(folder/'dependencies-spatialnorm.make').write_text(out);deps=sorted(set(str(Path(x).resolve()) for x in shlex.split(out.replace('\\\n',' ').split(':',1)[1])));hashes={p:sha(p) for p in deps};mismatch=[]
 for p,h in hashes.items():
  q=Path(p)
  if q.is_relative_to(root/'src'):
   rel=str(q.relative_to(root));expected=subprocess.check_output(['git','show',r['runtime_implementation']+':'+rel],cwd=root)
   if hashlib.sha256(expected).hexdigest()!=h:mismatch.append(rel)
 assert not mismatch
 r['builds'][label]={'command':cmd,'seconds':time.monotonic()-start,'executable_sha256':sha(folder/'server-spatialnorm'),'dependency_command':dc,'compiler_dependency_hashes':hashes,'link_archive_hashes':{p:sha(p) for p in cmd if p.endswith('.a')},'production_header_mismatches_vs27c':mismatch,'exit':proc.returncode}
 print(label,r['builds'][label]['seconds'],r['builds'][label]['executable_sha256'],flush=True)
r['no_production_edits']=subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)=='';assert r['no_production_edits'];(w/'build-provenance.json').write_text(json.dumps(r,indent=2)+'\n')
