"""Fresh C0 spatial-norm templates only; do not compile before frozen admission."""
from pathlib import Path
import hashlib,json,shutil
w=Path(__file__).resolve().parent;root=w.parents[2];v=w/'full22-candidate'
original=w.parent/'full-tensor-propagator';v.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();copied={}
for name in ['tangent_server.cpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','validate.py','spatialnorm-validation-vectors.npz']:
 p=original/'projected-v1'/name;q=w/name;assert not q.exists();shutil.copy2(p,q);copied[str(q)]={'source':str(p),'sha256':sha(q)}
for name in ['projected_base.hpp','full22_server.cpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','diagnostic_fields.cpp','diagnostic_constraint_norms.cpp','krylov_propagate.py','expm_propagate.py']:
 p=original/'full22-v2'/name;q=v/name;assert not q.exists();shutil.copy2(p,q);copied[str(q)]={'source':str(p),'sha256':sha(q)}
prior=w.parent/'full-tensor-kappa2-profile/full22-candidate'
for name in ['validate22.py','assemble_projected.py','validate_krylov_pilot.py','analyze_history.py','analyze_fields.py','compare_controls.py','build_fields.py']:
 p=prior/name;q=v/name;assert not q.exists();shutil.copy2(p,q)
 s=q.read_text().replace('C0 prescribed kappa2(analytic Omega), finiteΩ strict interior','regular inner relative-lapse-advection source alone, C0 spatialnorm finiteΩ strict interior').replace('C0 prescribed kappa2 damping profile','regular inner relative-lapse-advection source alone, C0 spatialnorm').replace("for g in ['production','spatialnorm']:","for g in ['spatialnorm']:").replace("'profile_","'lapse_")
 if name=='build_fields.py':s=s.replace("build-production.json","build-spatialnorm.json").replace("server-production","server-spatialnorm")
 q.write_text(s);copied[str(q)]={'source':str(p),'source_sha256':sha(p),'prepared_sha256':sha(q)}
for folder,base in [(w,original/'projected-v1'),(v,original/'full22-v2')]:
 cmd=json.loads((base/'build-spatialnorm.json').read_text())
 sourceprefix=str(original/'full22-v2') if folder==v else str(original)
 cmd=[x.replace(sourceprefix,str(folder)) for x in cmd]
 cmd[1:1]=['-I'+str(w/'overlay'),'-I'+str(v)]
 assert cmd[cmd.index('-o')+1]==str(folder/'server-spatialnorm')
 assert next(x for x in cmd if x.endswith('.cpp')).startswith(str(folder))
 (folder/'build-spatialnorm.json').write_text(json.dumps(cmd,indent=2)+'\n')
r={'status':'fresh C0 source templates only; helper unbound; no compile/run before frozen gate',
   'candidate':'regular inner relative-lapse-advection source alone on C0 spatialnorm',
   'no_kappa2_C1_or_other_source_combination':True,'copied_source_identities':copied}
(w/'SOURCE_PREPARATION_HOLD.json').write_text(json.dumps(r,indent=2)+'\n')
print('Prepared fresh source-only spatialnorm templates and fixed-output commands.')
