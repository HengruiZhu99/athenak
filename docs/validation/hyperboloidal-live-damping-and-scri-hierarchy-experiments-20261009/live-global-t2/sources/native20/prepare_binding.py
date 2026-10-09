"""Live-profile source binding only; final frozen actual gate still required."""
from pathlib import Path
import difflib,hashlib,json,shutil
w=Path(__file__).resolve().parent;root=w.parents[2];v=w/'full22-candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
helper=root/'build-layer-research/continuum/live-damping-control/live_damping_profile.hpp'
assert sha(helper)=='69bbbc137486eb3398f94ed50a8b04583c375da049372b82b19330d8432fc153'
shutil.copy2(helper,v/helper.name)
src=root/'src/z4c/hyperboloidal/cartesian_patch.hpp';s=src.read_text();base=s
mark='#include "z4c/hyperboloidal/spherical_ghosts.hpp"';assert s.count(mark)==1
s=s.replace(mark,mark+'\n#include "live_damping_profile.hpp"')
replacements={
 'ConformalRHS(u,omega,damping/u.alpha.value,Real(0))':'ConformalRHS(u,omega,damping/u.alpha.value,ResearchLiveKappa2Profile(u,omega,p.radius,damping))',
 'ConformalRHS(background,omega0,damping,Real(0))':'ConformalRHS(background,omega0,damping,ResearchLiveKappa2Profile(background,omega0,p.radius,damping))',
 'damping/u.alpha.value,Real(0));':'damping/u.alpha.value,ResearchLiveKappa2Profile(u,CartesianOmega(u,p),p.radius,damping));',
 'damping,Real(0)).pole;':'damping,ResearchLiveKappa2Profile(background,CartesianOmega(background,p),p.radius,damping)).pole;',
}
for before,after in replacements.items():assert s.count(before)==1,before;s=s.replace(before,after)
dst=w/'overlay/z4c/hyperboloidal/cartesian_patch.hpp';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(s)
(w/'cartesian-patch.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile=str(src),tofile=str(dst))))
for p in [v/'projected_base.hpp',w/'tangent_server.cpp']:
 before=p.read_text();s=before
 mark='hyp::GaugeRHS<Real>g{};if(!hyp::AssembleInterior(hyp::ConformalRHS(u,hyp::CartesianOmega(u,p),patch.kappa1/u.alpha.value,Real(0))'
 assert s.count(mark)==1
 s=s.replace(mark,'hyp::GaugeRHS<Real>g{};const auto omega=hyp::CartesianOmega(u,p);if(!hyp::AssembleInterior(hyp::ConformalRHS(u,omega,patch.kappa1/u.alpha.value,hyp::ResearchLiveKappa2Profile(u,omega,p.radius,patch.kappa1))')
 p.write_text(s);(w/(p.stem+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='fresh-C0/'+p.name,tofile=str(p))))
r={'status':'source binding only, no compile/run until final actual frozen index and parent admission',
 'helper_sha256':sha(v/'live_damping_profile.hpp'),'overlay_sha256':sha(dst),'cached_Point_sha256':sha(v/'projected_base.hpp'),
 'candidate':'live kappa2=V(r)*[Omega−1+2 beta.grad Omega/kinput], V(.15,.3), source alone on C0 spatialnorm',
 'actual_native_changes':'one include plus four kappa2 ConformalRHS args; no new derivative terms in evolution',
 'reference_linear_change':'P/Theta<-Theta only, delta kappa2*Theta_ref=0; finite-Theta beta cross-couplings remain nonlinear',
 'no_fixed_profile_lapse_C1_or_other_source_combination':True}
(w/'SOURCE_BINDING_HOLD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
