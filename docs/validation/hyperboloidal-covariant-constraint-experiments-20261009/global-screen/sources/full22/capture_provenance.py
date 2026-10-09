"""Capture compiler dependency bytes plus exact build/run source provenance."""
from pathlib import Path
import hashlib,json,shlex,subprocess
w=Path(__file__).resolve().parent;root=w.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'implementation_reference':'27c19d20696ea6dd4704032c51dfd026218f64f2','git_status_porcelain':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),'compiler_version':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'builds':{}}
for g in ['production','spatialnorm']:
 cmd=json.loads((w/f'build-{g}.json').read_text());dc=[];skip=False
 for x in cmd:
  if skip:skip=False;continue
  if x=='-o':skip=True;continue
  if x.endswith('.a'):continue
  dc.append(x)
 dc+=['-M','-MT','audit'];out=subprocess.check_output(dc,text=True);(w/f'dependencies-{g}.make').write_text(out);deps=shlex.split(out.replace('\\\n',' ').split(':',1)[1]);paths=sorted(set(str(Path(p).resolve()) for p in deps));prod={};mismatch=[]
 for path in paths:
  q=Path(path)
  if q.is_relative_to(root/'src'):
   rel=str(q.relative_to(root));expected=subprocess.check_output(['git','show',result['implementation_reference']+':'+rel],cwd=root);expectedsha=hashlib.sha256(expected).hexdigest();prod[rel]=expectedsha
   if expectedsha!=sha(q):mismatch.append(rel)
 result['builds'][g]={'command':cmd,'dependency_command':dc,'compiler_reported_dependency_hashes':{p:sha(Path(p)) for p in paths},'archive_hashes':{x:sha(Path(x)) for x in cmd if x.endswith('.a')},'executable_sha256':sha(w/f'server-{g}'),'production_header_reference_hashes':prod,'production_header_mismatches':mismatch}
result['scratch_sources']={p.name:sha(p) for p in w.iterdir() if p.is_file() and p.suffix in ['.cpp','.hpp','.py']};result['frozen_spatialnorm_wrapper_sha256']=sha(w/'native_injection.hpp');result['frozen_spatialnorm_math_sha256']=sha(w/'spatial_norm_control.hpp');(w/'build-provenance.json').write_text(json.dumps(result,indent=2)+'\n');print('dependency counts',{k:len(v['compiler_reported_dependency_hashes']) for k,v in result['builds'].items()},'production mismatches',{k:v['production_header_mismatches'] for k,v in result['builds'].items()},flush=True)
