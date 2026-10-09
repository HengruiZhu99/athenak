"""Verify immutable lapse-only admission and bind exact helper to fresh sources."""
from pathlib import Path
import difflib,hashlib,json,shutil,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];v=w/'full22-candidate'
gate=root/'build-layer-research/continuum/inner-lapse-advection-control/immutable-inner-lapse-advection-local-20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(gate/'index.json')=='4ff923dfabbae02c509b1cfeecbe18aa173bdcd5640e6d562d1d04934c8a7e8c'
index=json.loads((gate/'index.json').read_text())
for n,row in index['files'].items():
 p=gate/n;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],n
r=json.loads((gate/'receipt.json').read_text())
assert sha(gate/'receipt.json')=='b85ec14d091969f798ea8d58287bfc8b9dd0262cc147db42a630b79e6d1e4721'
assert r['passed_lower_order_lapse_local_gates'] and not r['native_global_or_scri_stability_accepted'] and r['sources_unchanged']
assert r['source_before']==r['source_after'] and len(r['source_before'])==376
assert len(r['commands'])==10 and all(c['returncode']==0 and not c['stderr'] for c in r['commands'])
for n,s in r['source_before'].items():
 p=Path(n);p=p if p.is_absolute() else root/p;assert sha(p)==s,n
shutil.copy2(gate/'inner_lapse_advection.hpp',v/'inner_lapse_advection.hpp')
assert sha(v/'inner_lapse_advection.hpp')=='39f125347e050bbf662ce3dc1e354791b37cafa7949fdf1ff19cedda3d8d85e1'
src=root/'src/z4c/hyperboloidal/cartesian_patch.hpp';s=src.read_text();base=s
mark='#include "z4c/hyperboloidal/spherical_ghosts.hpp"';assert s.count(mark)==1
s=s.replace(mark,mark+'\n#include "inner_lapse_advection.hpp"')
mark='      AddMeshUpwindAdvectionWithVelocity<3>(udev,full.beta_u,idx,0,k,j,i,rhs,gauge_rhs);';assert s.count(mark)==1
s=s.replace(mark,'      gauge_rhs.alpha += ResearchInnerLapseAdvection(p,u,lg);\n'+mark)
dst=w/'overlay/z4c/hyperboloidal/cartesian_patch.hpp';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(s)
(w/'cartesian-patch.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile=str(src),tofile=str(dst))))
mark='Pack(r,g,out);}'
for p in [v/'projected_base.hpp',w/'tangent_server.cpp']:
 before=p.read_text();assert before.count(mark)==1
 after=before.replace(mark,'g.alpha+=hyp::ResearchInnerLapseAdvection(p,u,patch.layer_gauge);'+mark)
 p.write_text(after)
 (w/(p.stem+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='fresh-C0/'+p.name,tofile=str(p))))
auth={'gate_index_path':str(gate/'index.json'),'gate_index_sha256':sha(gate/'index.json'),'gate_receipt_path':str(gate/'receipt.json'),'gate_receipt_sha256':sha(gate/'receipt.json'),'verified_index_files':len(index['files']),'verified_unchanged_recorded_inputs':376,'verified_successful_commands':10,'helper_sha256':sha(v/'inner_lapse_advection.hpp'),'scientific_gate_authorized_before_compile':True,'scope':'finiteOmega exploratory lapse-only source on C0 spatialnorm; no stabilization/scri/collapsed-lapse/global acceptance','source_hold_superseded':True,'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'overlay_sha256':sha(dst),'cached_Point_sha256':sha(v/'projected_base.hpp'),'native20_Point_sha256':sha(w/'tangent_server.cpp'),'no_kappa2_C1_or_other_source_combination':True}
(w/'gate-authorization.json').write_text(json.dumps(auth,indent=2)+'\n');print(json.dumps(auth,indent=2))
