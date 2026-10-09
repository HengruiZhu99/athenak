"""Bind the byte-frozen shared bulk C1 helper to private native tangent sources."""
from pathlib import Path
import hashlib,json,shutil,difflib
here=Path(__file__).resolve().parent;root=here.parents[2];v2=here/'full22-candidate';prior=here.parent/'full-tensor-covariant-c1';original=here.parent/'full-tensor-propagator'
gate=root/'build-layer-research/continuum/covariant-constraint-propagation/immutable-C1-blend-constraint-20261009';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(gate/'index.json')=='1bb5f697691404c7a8ed19c20fc77aa79c5aa3fa31a81248d10c7e958d3ac2f0'
index=json.loads((gate/'index.json').read_text())
for name,row in index['files'].items():
 p=gate/name;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
for name in ['bulk_c1_additions.hpp','c1_additions.hpp']:shutil.copy2(gate/name,v2/name)
assert sha(v2/'bulk_c1_additions.hpp')=='4c7b6637fc9c38339134d5d5824e589986110edc9efa55489ebe0fc941bafbe2'
helper='''// SCRATCH ONLY: prescribed bulk C1 additions, no fully covariant-system claim.
#ifndef SCRATCH_GLOBAL_BULK_C1_INJECTION_HPP_
#define SCRATCH_GLOBAL_BULK_C1_INJECTION_HPP_
#include "bulk_c1_additions.hpp"
namespace z4c { namespace hyperboloidal {
template <typename T>
KOKKOS_INLINE_FUNCTION bool ResearchAddBulkC1(
    const Z4cJet<T>&u,const OmegaJet<T>&o,T radius,
    const LayerGaugeParameters&g,Z4cRHS<T>&rhs) {
  Z4cRHS<T> delta;
  if(!AssembleC1Interior(BulkC1Additions(u,o,radius,g),o.omega,delta))return false;
  rhs.chi+=delta.chi;rhs.trace+=delta.trace;rhs.theta+=delta.theta;
  for(int i=0;i<3;++i){rhs.lambda[i]+=delta.lambda[i];
    for(int j=0;j<3;++j){rhs.metric[i][j]+=delta.metric[i][j];rhs.a[i][j]+=delta.a[i][j];}}
  return true;
}
}}
#endif
'''
(v2/'bulk_injection.hpp').write_text(helper)
src=root/'src/z4c/hyperboloidal/cartesian_patch.hpp';s=src.read_text();base=s;mark='#include "z4c/hyperboloidal/spherical_ghosts.hpp"';assert s.count(mark)==1;s=s.replace(mark,mark+'\n#include "bulk_injection.hpp"');mark='      AddMeshUpwindAdvectionWithVelocity<3>(udev,full.beta_u,idx,0,k,j,i,rhs,gauge_rhs);';assert s.count(mark)==1;s=s.replace(mark,'      // Private prescribed coefficient; no C1 reference subtraction.\n      if (!ResearchAddBulkC1(u,omega,p.radius,lg,rhs)) { ++bad; return; }\n'+mark);dst=here/'overlay/z4c/hyperboloidal/cartesian_patch.hpp';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(s);(here/'cartesian-patch.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile=str(src),tofile=str(dst))))
mark='Pack(r,g,out);}'
for p,orig in [(v2/'projected_base.hpp',original/'full22-v2/projected_base.hpp'),(here/'tangent_server.cpp',original/'projected-v1/tangent_server.cpp')]:
 s=p.read_text();assert s.count(mark)==1;s=s.replace(mark,'if(!hyp::ResearchAddBulkC1(u,hyp::CartesianOmega(u,p),p.radius,patch.layer_gauge,r))throw std::runtime_error("invalid local bulk C1 addition");'+mark);p.write_text(s);(here/(p.stem+'.diff')).write_text(''.join(difflib.unified_diff(orig.read_text().splitlines(True),s.splitlines(True),fromfile=str(orig),tofile=str(p))))
# Reuse only corrected candidate drivers, each in the new fixed directory.
for name in ['validate22.py','assemble_projected.py','validate_krylov_pilot.py','analyze_history.py','analyze_fields.py']:
 shutil.copy2(prior/'full22-candidate'/name,v2/name)
p=v2/'assemble_projected.py';s=p.read_text().replace("C1 plus derived covariant Lambda connection, finiteΩ strict interior", "prescribed bulk C1+covector repair, coefficient1−Wgauge, finiteΩ strict interior");p.write_text(s)
for g in ['production','spatialnorm']:
 for folder in [here,v2]:
  cmd=json.loads((folder/f'build-{g}.json').read_text());out=cmd[cmd.index('-o')+1];cpp=next(x for x in cmd if x.endswith('.cpp'));assert str(here) in out and str(here) in cpp
status={'gate_index_path':str(gate/'index.json'),'gate_index_sha256':sha(gate/'index.json'),'gate_all27_files_verified':True,'gate_receipt_sha256':sha(gate/'receipt.json'),'bulk_helper_sha256':sha(v2/'bulk_c1_additions.hpp'),'C1_math_sha256':sha(v2/'c1_additions.hpp'),'wrapper_sha256':sha(v2/'bulk_injection.hpp'),'overlay_sha256':sha(dst),'cached_Point_sha256':sha(v2/'projected_base.hpp'),'variant':'prescribed bulk1−Wgauge multiplier on all C1+covector additions; not full covariant equations','no_C1_reference_subtraction':True,'unchanged_gauge_stencils_ghosts_projector':True}
(here/'gate-bound-sources.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
