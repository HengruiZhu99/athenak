"""Compile exact private C1 tangent sources and capture compiler dependencies."""
from pathlib import Path
import hashlib,json,shlex,subprocess,time
here=Path(__file__).resolve().parent;root=here.parents[2];v2=here/'full22-candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gate=root/'build-layer-research/continuum/covariant-constraint-propagation/immutable-C1-blend-constraint-20261009/receipt.json'
gd=json.loads(gate.read_text());assert gd['passed_reference_tangent_and_blend_gates']
result={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','continuum_gate_receipt_path':str(gate),'continuum_gate_receipt_sha256':sha(gate),'compiler_version':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'variant':'prescribed1−Wgauge bulk C1+covector repair, no C1 reference subtraction','builds':{}}
for label,folder in [('full22',v2),('native20',here)]:
 for g in ['production','spatialnorm']:
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
result['C1_math_sha256']=sha(v2/'c1_additions.hpp');result['C1_wrapper_sha256']=sha(v2/'bulk_injection.hpp')
assert result['C1_math_sha256']=='908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7'
result['bulk_helper_sha256']=sha(v2/'bulk_c1_additions.hpp')
assert result['bulk_helper_sha256']=='4c7b6637fc9c38339134d5d5824e589986110edc9efa55489ebe0fc941bafbe2'
result['no_production_edits']=subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''
assert result['no_production_edits']
(here/'build-provenance.json').write_text(json.dumps(result,indent=2)+'\n')
print('DONE',sha(here/'build-provenance.json'),flush=True)
