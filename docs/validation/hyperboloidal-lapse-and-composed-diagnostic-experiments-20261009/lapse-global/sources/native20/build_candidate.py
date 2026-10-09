"""Compile gate-authorized private inner-lapse-advection sources with exact dependencies."""
from pathlib import Path
import hashlib,json,shlex,subprocess,time
here=Path(__file__).resolve().parent;root=here.parents[2];v2=here/'full22-candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
auth=json.loads((here/'gate-authorization.json').read_text())
gate=Path(auth['gate_receipt_path']);gd=json.loads(gate.read_text())
assert sha(gate)==auth['gate_receipt_sha256']
indexpath=Path(auth['gate_index_path']);assert sha(indexpath)==auth['gate_index_sha256']
index=json.loads(indexpath.read_text())
for name,row in index['files'].items():
 p=indexpath.parent/name;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
assert gd['passed_lower_order_lapse_local_gates'] and gd['sources_unchanged']
assert auth['scientific_gate_authorized_before_compile']
result={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','continuum_gate_receipt_path':str(gate),'continuum_gate_receipt_sha256':sha(gate),'compiler_version':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'variant':'regular inner relative-lapse-advection source alone on C0 spatialnorm; unchanged physicalP pole and shift','builds':{}}
for label,folder in [('full22',v2),('native20',here)]:
 for g in ['spatialnorm']:
  cmd=json.loads((folder/f'build-{g}.json').read_text());start=time.monotonic()
  proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  (folder/f'build-{g}.log').write_text(proc.stdout)
  if proc.returncode:raise RuntimeError(proc.stdout)
  dc=[];skip=False
  for x in cmd:
   if skip:skip=False;continue
   if x=='-o':skip=True;continue
   if x.endswith('.a'):continue
   dc.append(x)
  dc+=['-M','-MT','candidate'];out=subprocess.check_output(dc,text=True)
  (folder/f'dependencies-{g}.make').write_text(out)
  deps=sorted(set(str(Path(x).resolve()) for x in shlex.split(out.replace('\\\n',' ').split(':',1)[1])))
  mismatches=[];hashes={p:sha(Path(p)) for p in deps}
  for p,s in hashes.items():
   q=Path(p)
   if q.is_relative_to(root/'src'):
    rel=str(q.relative_to(root));expected=subprocess.check_output(['git','show',result['runtime_implementation']+':'+rel],cwd=root)
    if hashlib.sha256(expected).hexdigest()!=s:mismatches.append(rel)
  result['builds'][label+'-'+g]={'command':cmd,'seconds':time.monotonic()-start,'executable_sha256':sha(folder/f'server-{g}'),'dependency_command':dc,'compiler_dependency_hashes':hashes,'link_archive_hashes':{p:sha(Path(p)) for p in cmd if p.endswith('.a')},'production_header_mismatches_vs27c':mismatches,'exit':proc.returncode}
  assert not mismatches
  print(label,g,'built',result['builds'][label+'-'+g]['seconds'],result['builds'][label+'-'+g]['executable_sha256'],flush=True)
result['explicit_overlay_hashes']={str(p):sha(p) for p in (here/'overlay').rglob('*') if p.is_file()}
result['lapse_helper_sha256']=sha(v2/'inner_lapse_advection.hpp')
assert result['lapse_helper_sha256']=='39f125347e050bbf662ce3dc1e354791b37cafa7949fdf1ff19cedda3d8d85e1'
result['frozen_scientific_gate_index']=auth
result['no_production_edits']=subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''
assert result['no_production_edits']
(here/'build-provenance.json').write_text(json.dumps(result,indent=2)+'\n')
print('DONE',sha(here/'build-provenance.json'),flush=True)
