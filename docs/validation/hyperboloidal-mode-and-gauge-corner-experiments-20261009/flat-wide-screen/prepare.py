"""Fresh exponent-one overlay reusing the frozen previously tested formulas."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess
root=Path(__file__).resolve().parents[3];w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=root/'build-layer-research/flat-height/flat_power.hpp'
assert sha(old)=='7d3c22bb5e23627a5da83d542fd4100d109fad43612ee3f995c223bd50df1135'
public=root/'src/z4c/hyperboloidal/layer_reference.hpp'
original=re.sub(r'\bLayerReference\b','OriginalLayerReference',public.read_text())
flat=old.read_text().replace('#include "z4c/hyperboloidal/layer_reference.hpp"','')
flat=flat.replace('int omega_power = 2;','int omega_power = 1;').replace('int exponent=2','int exponent=1')
flat=re.sub(r'\bLayerReference\b','OriginalLayerReference',flat)
overlay=w/'overlay/z4c/hyperboloidal/layer_reference.hpp';overlay.parent.mkdir(parents=True,exist_ok=True)
overlay.write_text('#ifndef SCRATCH_WIDE_FLAT_PENROSE_OVERLAY_HPP_\n#define SCRATCH_WIDE_FLAT_PENROSE_OVERLAY_HPP_\n'+original+'\n'+flat+'\nnamespace z4c {namespace hyperboloidal {template<typename T> using LayerReference=FlatPowerReference<T>;}}\n#endif\n')
src=root/'tst/hyperboloidal/test_layer_constraint_tangent.cpp';text=src.read_text()
text=text.replace('#include "z4c/hyperboloidal/cartesian_patch.hpp"','#include "native_injection.hpp"\n#include "z4c/hyperboloidal/cartesian_patch.hpp"')
text=text.replace('5 / u.alpha.value','10 / u.alpha.value')
needle='  hyp::CartesianConformalPatch patch(grid, a, degree, lp, gauge, symmetric);'
assert needle in text;text=text.replace(needle,needle+'\n  patch.kappa1=10; patch.dissipation=.1;\n  constexpr double probe_eps=1e-4;')
needle='  patch.InitializeReference(initial);';assert needle in text
text=text.replace(needle,needle+'\n  auto reference_state=patch.Allocate("reference initial");\n  auto minus_rhs=patch.Allocate("minus gauge RHS");\n  Kokkos::deep_copy(reference_state,initial);')
text=text.replace('+= .1 * shape *','+= probe_eps * .1 * shape *').replace('+= .02 * shape *','+= probe_eps * .02 * shape *')
needle='  patch.RHS(q, rhs);\n  const auto ref = patch.reference;'
assert needle in text
text=text.replace(needle,'''  patch.RHS(q, rhs);
  Kokkos::parallel_for("minus gauge seed",q.size(),KOKKOS_LAMBDA(const int s) {
    q.data()[s]=2*reference_state.data()[s]-initial.data()[s];
  });
  patch.RHS(q,minus_rhs);
  Kokkos::parallel_for("actual centered gauge Jv",rhs.size(),KOKKOS_LAMBDA(const int s) {
    rhs.data()[s]=(rhs.data()[s]-minus_rhs.data()[s])/(2*probe_eps);
  });
  Kokkos::deep_copy(initial,reference_state);
  const auto ref = patch.reference;''')
needle='    std::cout << "{\\"kind\\":\\"native\\",\\"n\\":" << n'
assert needle in text
text=text.replace(needle,'''    double max_m=0,max_z=0,m_radius=0,z_radius=0;
    for(size_t p=0;p<host.extent(0);++p){
      if(host(p,2)>max_m){max_m=host(p,2);m_radius=host(p,0);}
      if(host(p,3)>max_z){max_z=host(p,3);z_radius=host(p,0);}
    }
'''+needle)
needle='              << ",\\"initial_H\\":" << first.h_l2';assert needle in text
text=text.replace(needle,'''              << ",\\"generator_eps\\":" << probe_eps
              << ",\\"Omega_min\\":" << patch.min_omega
              << ",\\"Mdot_max\\":" << std::sqrt(max_m)
              << ",\\"Zdot_max\\":" << std::sqrt(max_z)
              << ",\\"Mmax_r\\":" << m_radius << ",\\"Zmax_r\\":" << z_radius
'''+needle)
(w/'constraint_tangent.cpp').write_text(text)
original_local=root/'build-layer-research/flat-height/power_extended.cpp'
local=original_local.read_text().replace('#include "flat_power.hpp"','')
local=local.replace('#include "z4c/hyperboloidal/cartesian_patch.hpp"','#include "native_injection.hpp"\n#include "z4c/hyperboloidal/cartesian_patch.hpp"')
local=local.replace('5/u.alpha.value','10/u.alpha.value').replace('5/baseline.alpha','10/baseline.alpha')
local=local.replace('hyp::FlatPowerReference<double>','hyp::LayerReference<double>')
local=local.replace('hyp::LayerReference<double> old','hyp::OriginalLayerReference<double> old')
start=local.index('const std::vector<std::array<double,4>> configs')
end=local.index(';',start)
local=local[:start]+'const std::vector<std::array<double,4>> configs{{1,.5,.05,.95}}'+local[end:]
(w/'local_gate.cpp').write_text(local)
wrapper=root/'build-layer-research/boundary/full-tensor-propagator/full22-v2'
for name in ['native_injection.hpp','spatial_norm_control.hpp']:shutil.copy2(wrapper/name,w/name)
shutil.copy2(old,w/'frozen-flat_power.hpp')
receipt={'scope':'Existing frozen exponent-one flat-height family; fresh wide supplement only; no novel geometry claim',
 'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'inputs':{str(p.relative_to(root)):sha(p) for p in [old,public,src,original_local,wrapper/'native_injection.hpp',wrapper/'spatial_norm_control.hpp']},
 'generated':{str(p.relative_to(w)):sha(p) for p in [overlay,w/'constraint_tangent.cpp',w/'local_gate.cpp',w/'native_injection.hpp',w/'spatial_norm_control.hpp']},
 'changes':'LayerReference base renamed only, frozen formula copied exactly, default exponent2->1. Native tangent source binds original spatialnorm wrapper/kappa10 and actual centered pure-gauge Jv at reference; added peak/minOmega diagnostics only.',
 'long_double_mantissa_bits':53,'long_double_wider_than_double':False}
(w/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
