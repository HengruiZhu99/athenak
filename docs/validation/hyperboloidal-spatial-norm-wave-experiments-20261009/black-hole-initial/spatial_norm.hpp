#ifndef SCRATCH_SPATIAL_NORM_GAUGE_HPP_
#define SCRATCH_SPATIAL_NORM_GAUGE_HPP_
#include "z4c/hyperboloidal/layer_gauge.hpp"
namespace z4c { namespace hyperboloidal {
// Scratch algebraic source, live geometry only. Outward n=-dOmega/|dOmega|
// uses the admitted monotone radial compactification where W>0.
template<typename T> KOKKOS_INLINE_FUNCTION
GaugeRHSParts<T> SpatialNormGauge(const LayerPoint<T>&p,const Z4cJet<T>&u,
    const LayerGaugeParameters&g,T S,T a,T eta) {
 auto out=InteriorLayerGauge(p,u,g);if(!out.valid)return out;
 const T W=LayerCoefficients(p.radius,u.alpha.value,g).weight;
 if(W==0)return out;
 if(!(eta>0)||!g.physical_trace_lapse||g.preferred_source){out.valid=false;return out;}
 const auto geo=Geometry(u.metric),ref=Geometry(p.state.metric);
 if(!geo.valid||!ref.valid){out.valid=false;return out;}
 T Ghat=0,deltaG=0,norm=0;
 for(int i=0;i<3;++i) {
  norm+=p.domega[i]*p.domega[i];
  for(int j=0;j<3;++j) {
   Ghat+=p.state.chi.value*ref.inverse[i][j]*p.domega[i]*p.domega[j];
   deltaG+=((u.chi.value-p.state.chi.value)*ref.inverse[i][j]
            +u.chi.value*(geo.inverse[i][j]-ref.inverse[i][j]))*p.domega[i]*p.domega[j];
  }
 }
 if(!(Ghat>0&&norm>0)){out.valid=false;return out;}
 const T C=S/a*(1-S/(eta*a*a));
 for(int i=0;i<3;++i){const T ni=-p.domega[i]/Kokkos::sqrt(norm);
  out.pole.beta[i]-=eta*W*(u.beta.value[i]-p.beta[i]+C*ni*deltaG/Ghat);
 }
 return out;
}
template<typename T> KOKKOS_INLINE_FUNCTION
bool AssembleSpatialNorm(const GaugeRHSParts<T>&parts,T O,GaugeRHS<T>&rhs) {
 if(!AssembleGaugeInterior(parts,O,rhs))return false;
 for(int i=0;i<3;++i){rhs.beta[i]+=parts.pole.beta[i]/O;if(!Kokkos::isfinite(rhs.beta[i]))return false;}
 return true;
}
}}
#endif
