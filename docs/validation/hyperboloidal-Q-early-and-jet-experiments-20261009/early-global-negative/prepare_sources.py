"""Copy frozen late-Q templates only; early helper binding/compile remain held."""
from pathlib import Path
import hashlib,json,shutil,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];v=w/'full22-candidate';late=w.parent/'full-tensor-conformal-q-null-feedback';frozen=late/'immutable-Q-null-global-screen-20261009';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(frozen/'index.json')=='d12e8e4da86f0918cfc7ad741c61a4cff3214494dc808f90df9eb061f0918214'
idx=json.loads((frozen/'index.json').read_text())
for n,h in idx['files'].items():assert sha(frozen/n)==h['sha256'] and (frozen/n).stat().st_size==h['bytes']
v.mkdir(exist_ok=True);copied={}
groups=[(w,'',['tangent_server.cpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','validate.py']),
 (v,'full22-candidate',['projected_base.hpp','full22_server.cpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','diagnostic_fields.cpp','diagnostic_constraint_norms.cpp','validate22.py','assemble_projected.py','krylov_propagate.py','short_canonical.py','analyze_history.py','analyze_fields.py','build_fields.py','build_reference.py','reference_coefficients.cpp','compare_controls.py'])]
for dst,sub,names in groups:
 for n in names:
  p=frozen/sub/n;q=dst/n;assert not q.exists();shutil.copy2(p,q);copied[str(q)]={'frozen_source':str(p),'sha256':sha(q)}
seed=late/'spatialnorm-validation-vectors.npz';meta=json.loads((frozen/'large-artifacts-metadata-only.json').read_text());assert sha(seed)==meta['local_large_artifacts']['spatialnorm-validation-vectors.npz']['sha256'];shutil.copy2(seed,w/seed.name)
for dst,sub in [(w,''),(v,'full22-candidate')]:
 command=json.loads((frozen/sub/'build-spatialnorm.json').read_text());command=[s.replace(str(late),str(w)) for s in command];assert command[command.index('-o')+1]==str(dst/'server-spatialnorm');(dst/'HELD-build-spatialnorm.json').write_text(json.dumps(command,indent=2)+'\n')
r={'status':'HELD: earlier mode0=Wgauge helper/index and independent frozen review not yet bound','launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'late_frozen_index_sha256':sha(frozen/'index.json'),'late_107_small_files_reverified':True,'copied_templates':copied,'seed_sha256':sha(seed),'scientific_compilation_authorized':False,'helper_bound':False,'matrix_or_propagation_performed':False,'planned_difference_only':'beta null feedback weight Vlate(.85,.95) replaced by Wgauge(.45,.85), sigma5; same physical-inner alpha blend, preferred regular shift, C0 geometry/damping and explicitfalse/true/xi2','late_sources_or_outputs_modified':False,'root_owns_native':True,'no_automatic_t6_or_long_native':True}
(w/'SOURCE_PREPARATION_HOLD.json').write_text(json.dumps(r,indent=2)+'\n');print('Fresh early-weight source templates prepared. No mathematical helper copied, no compile/matrix/propagation.')
