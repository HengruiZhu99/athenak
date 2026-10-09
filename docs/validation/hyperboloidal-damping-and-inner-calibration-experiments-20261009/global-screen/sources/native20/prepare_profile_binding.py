"""Prepare a private prescribed-kappa2 binding; compiling remains gate-held."""
from pathlib import Path
import difflib, hashlib, json, shutil

here = Path(__file__).resolve().parent
root = here.parents[2]
v2 = here / 'full22-candidate'
original = here.parent / 'full-tensor-propagator'
helper = root / 'build-layer-research/continuum/damping-profile-control/damping_profile.hpp'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected = '64ba382f509e81fe347b96e915d3933ee7188c97fd6a054524b1aa187a891532'
assert sha(helper) == expected
shutil.copy2(helper, v2 / helper.name)

src = root / 'src/z4c/hyperboloidal/cartesian_patch.hpp'
s = src.read_text()
base = s
mark = '#include "z4c/hyperboloidal/spherical_ghosts.hpp"'
assert s.count(mark) == 1
s = s.replace(mark, mark + '\n#include "damping_profile.hpp"')
replacements = {
 'ConformalRHS(u,omega,damping/u.alpha.value,Real(0))':
 'ConformalRHS(u,omega,damping/u.alpha.value,ResearchKappa2Profile(omega.omega,ref.scri_radius,ref.curvature_radius,damping))',
 'ConformalRHS(background,omega0,damping,Real(0))':
 'ConformalRHS(background,omega0,damping,ResearchKappa2Profile(omega0.omega,ref.scri_radius,ref.curvature_radius,damping))',
 'damping/u.alpha.value,Real(0));':
 'damping/u.alpha.value,ResearchKappa2Profile(p.omega,ref.scri_radius,ref.curvature_radius,damping));',
 'damping,Real(0)).pole;':
 'damping,ResearchKappa2Profile(p.omega,ref.scri_radius,ref.curvature_radius,damping)).pole;',
}
for before, after in replacements.items():
 assert s.count(before) == 1, before
 s = s.replace(before, after)
dst = here / 'overlay/z4c/hyperboloidal/cartesian_patch.hpp'
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(s)
(here / 'cartesian-patch.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True), s.splitlines(True), fromfile=str(src), tofile=str(dst))))

before = 'hyp::CartesianOmega(u,p),patch.kappa1/u.alpha.value,Real(0))'
after = 'hyp::CartesianOmega(u,p),patch.kappa1/u.alpha.value,hyp::ResearchKappa2Profile(p.omega,patch.reference.scri_radius,patch.reference.curvature_radius,patch.kappa1))'
for p, orig in [(v2 / 'projected_base.hpp', original / 'full22-v2/projected_base.hpp'),
                (here / 'tangent_server.cpp', original / 'projected-v1/tangent_server.cpp')]:
 s = p.read_text()
 assert s.count(before) == 1
 s = s.replace(before, after)
 p.write_text(s)
 (here / (p.stem + '.diff')).write_text(''.join(difflib.unified_diff(orig.read_text().splitlines(True), s.splitlines(True), fromfile=str(orig), tofile=str(p))))

for folder in [here, v2]:
 for gauge in ['production', 'spatialnorm']:
  p = folder / ('build-' + gauge + '.json')
  cmd = json.loads(p.read_text())
  assert str(here) in cmd[cmd.index('-o') + 1]
  assert str(here) in next(x for x in cmd if x.endswith('.cpp'))
  cmd.insert(1, '-I' + str(here / 'overlay'))
  p.write_text(json.dumps(cmd, indent=2) + '\n')

status = {
 'status': 'SOURCE BINDING ONLY; no compile/run until final continuum frozen index is verified',
 'shared_helper_source': str(helper), 'shared_helper_sha256': expected,
 'overlay_sha256': sha(dst), 'cached_Point_sha256': sha(v2 / 'projected_base.hpp'),
 'native20_Point_sha256': sha(here / 'tangent_server.cpp'),
 'variant': 'C0 kappa1=input/alpha; prescribed kappa2=(2*S/a^2/input-1)*(1-Omega)',
 'actual_native_changes': 'include plus four kappa2 ConformalRHS arguments (two evolution/reference, two pole diagnostics)',
 'reference_subtraction': 'existing C0 reference residual only, using same prescribed profile',
 'unchanged_gauge_stencils_ghosts_projector': True,
 'profile_depends_on_fixed_analytic_Omega': True,
}
(here / 'SOURCE_BINDING_HOLD.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status, indent=2))
