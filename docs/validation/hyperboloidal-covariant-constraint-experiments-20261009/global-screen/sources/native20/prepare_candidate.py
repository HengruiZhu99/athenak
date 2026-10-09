"""Prepare private C1 source overlay only. Do not compile/run before pole gates."""
from pathlib import Path
import hashlib,json,shutil,difflib
here=Path(__file__).resolve().parent;root=here.parents[2]
old=here.parent/'full-tensor-propagator';v1=old/'projected-v1';v2=old/'full22-v2'
new=here/'full22-candidate';new.mkdir(exist_ok=True)
overlay=here/'overlay';patch_rel=Path('z4c/hyperboloidal/cartesian_patch.hpp')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def cp(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
header=root/'build-layer-research/continuum/covariant-z4/immutable-C1-tensor-identity-20261009/c1_additions.hpp'
assert sha(header)=='908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7'
cp(header,new/'c1_additions.hpp')
helper='''// SCRATCH ONLY: C1 additions after existing C0 analytic-reference subtraction.
#ifndef SCRATCH_GLOBAL_C1_INJECTION_HPP_
#define SCRATCH_GLOBAL_C1_INJECTION_HPP_
#include "c1_additions.hpp"
namespace z4c { namespace hyperboloidal {
template <typename T>
KOKKOS_INLINE_FUNCTION bool ResearchAddCovariantC1(
    const Z4cJet<T>&u,const OmegaJet<T>&o,Z4cRHS<T>&rhs) {
  Z4cRHS<T> delta;
  if(!AssembleC1Interior(TensorC1Additions(u,o,T(1),true),o.omega,delta))return false;
  rhs.chi+=delta.chi;rhs.trace+=delta.trace;rhs.theta+=delta.theta;
  for(int i=0;i<3;++i){rhs.lambda[i]+=delta.lambda[i];
    for(int j=0;j<3;++j){rhs.metric[i][j]+=delta.metric[i][j];rhs.a[i][j]+=delta.a[i][j];}}
  return true;
}
}}
#endif
'''
(new/'c1_injection.hpp').write_text(helper)
src=root/'src'/patch_rel;text=src.read_text();original=text
marker='#include "z4c/hyperboloidal/spherical_ghosts.hpp"'
assert text.count(marker)==1;text=text.replace(marker,marker+'\n#include "c1_injection.hpp"')
marker='      AddMeshUpwindAdvectionWithVelocity<3>(udev,full.beta_u,idx,0,k,j,i,rhs,gauge_rhs);'
assert text.count(marker)==1
text=text.replace(marker,'      // Private candidate: no C1 reference subtraction, no gauge/source change.\n      if (!ResearchAddCovariantC1(u,omega,rhs)) { ++bad; return; }\n'+marker)
dst=overlay/patch_rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(text)
(here/'cartesian-patch.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(src),tofile=str(dst))))
for name in ['full22_server.cpp','projected_base.hpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','validate22.py','expm_propagate.py','krylov_propagate.py','analyze_history.py','analyze_fields.py','diagnostic_fields.cpp','diagnostic_constraint_norms.cpp','capture_provenance.py']:
 cp(v2/name,new/name)
# Point and actual CartesianPatch must add exactly the same live C1 delta.
p=new/'projected_base.hpp';oldtext=p.read_text();marker='Pack(r,g,out);}'
assert oldtext.count(marker)==1
p.write_text(oldtext.replace(marker,'if(!hyp::ResearchAddCovariantC1(u,hyp::CartesianOmega(u,p),r))throw std::runtime_error("invalid local covariant C1 addition");'+marker))
(here/'cached-point.diff').write_text(''.join(difflib.unified_diff(oldtext.splitlines(True),p.read_text().splitlines(True),fromfile=str(v2/p.name),tofile=str(p))))
for name in ['tangent_server.cpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','validate.py']:
 cp(v1/name,here/name)
p=here/'tangent_server.cpp';oldtext=p.read_text();assert oldtext.count(marker)==1
p.write_text(oldtext.replace(marker,'if(!hyp::ResearchAddCovariantC1(u,hyp::CartesianOmega(u,p),r))throw std::runtime_error("invalid local covariant C1 addition");'+marker))
for g in ['production','spatialnorm']:
 cp(v1/f'{g}-validation-vectors.npz',here/f'{g}-validation-vectors.npz')
 for folder,target in [(v1,here),(v2,new)]:
  cmd=json.loads((folder/f'build-{g}.json').read_text())
  origin=old if folder==v1 else folder
  cmd=[x.replace(str(origin),str(target)) for x in cmd]
  cmd[1:1]=['-I'+str(overlay),'-I'+str(new)]
  (target/f'build-{g}.json').write_text(json.dumps(cmd,indent=2)+'\n')
status={'status':'SOURCE PREPARATION ONLY; compilation and propagation held pending continuum smallOmega negative/stiffness gate','C1_math_sha256':sha(new/'c1_additions.hpp'),'original_cartesian_patch_sha256':sha(src),'overlay_cartesian_patch_sha256':sha(dst),'original_cached_point_sha256':sha(v2/'projected_base.hpp'),'candidate_cached_point_sha256':sha(new/'projected_base.hpp'),'no_C1_reference_subtraction':True,'unchanged_gauge_sources':True,'unchanged_ghosts_and_derivative_stencils':True,'global_gate_frozen_manifest_sha256':sha(here.parent/'full-tensor-global-final/manifest.json'),'source_hashes':{str(p.relative_to(here)):sha(p) for p in sorted(here.rglob('*')) if p.is_file() and p.suffix in ['.cpp','.hpp','.py','.diff']}}
(here/'source-preparation.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
